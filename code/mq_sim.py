################################################################################
# mq_sim.py
#
# Simulate production of metagenomic sequence reads from MetaQuant
# generative model
#
# Kevin McLoughlin
# 11 Apr 2014
################################################################################

import sys
import re
import pdb
from math import log, exp
import yaml

import refseq_db as ref
import numpy as np


################################################################################
# constants

bases = ['A', 'C', 'G', 'T']
base_set = set(bases)

################################################################################
def has_ambiguity(seq, offset, length):
  """
  Report whether subsequence of seq contains Ns or any other ambiguity
  character
  """
  uniq_chars = set(seq[offset:offset+length])
  return (len(uniq_chars - base_set) > 0)

################################################################################
class BiasModel:
  """
  Generates probability that a read begins at a given position, given
  a set of parameters for a zero-order Markov bias model (derived from the
  probabilities of observing each base at each offset relative to the read
  start, independent of preceding bases).
  """

  def __init__(self, read_len=50, max_offset=10):
    self.bias_params = {}
    self.read_len = read_len
    self.max_offset = max_offset
    self.window = 2*max_offset + 1

  def load(self, param_file='../data/zero_order_bias_params.txt'):
    par_in = open(param_file, 'r')
    hdrs = par_in.next().rstrip('\n').split('\t')
    for base in bases:
      self.bias_params[base] = np.zeros(self.window)
    col = {}
    for colno, colname in enumerate(hdrs):
      col[colname] = colno
    for win_pos, line in enumerate(par_in):
      flds = line.rstrip('\n').split('\t')
      for base in bases:
        self.bias_params.setdefault(base,[])[win_pos] = log(float(flds[col[base]]))
    par_in.close()

  def compute_probs(self, targ_seq):
    """
    Generate a vector of probabilities for sampling reads at each usable
    position in the given target sequence, of length equal to that of the
    sequence. Positions that would result in the read or the half-window
    preceding it containing ambiguity characters get zero probabilities, 
    as do positions too close to the beginning or end of the sequence.
    """
    npos = len(targ_seq)
    wts = np.zeros(npos)
    for read_pos in range(self.max_offset, npos-self.read_len):
      if not has_ambiguity(targ_seq, read_pos-self.max_offset, 
         self.read_len+self.max_offset):
        log_bias = 0.0
        for j in range(-self.max_offset, self.max_offset+1):
          log_bias += self.bias_params[targ_seq[read_pos+j:read_pos+j+1]][j]
        wts[read_pos] = exp(log_bias)
    probs = wts/sum(wts)
    return probs

################################################################################
class ThirdOrderBiasModel:
  """
  Generates probability that a read begins at a given position, given
  a set of parameters for a third-order Markov bias model
  """

  def __init__(self, read_len=50, max_offset=10):
    self.bias_params = {}
    self.read_len = read_len
    self.max_offset = max_offset
    self.window = 2*max_offset + 1

  def load(self, param_file='../data/input_bias_params.txt'):
    par_in = open(param_file, 'r')
    hdrs = par_in.next().rstrip('\n').split('\t')
    tetrads = hdrs[1:]
    for tetrad in tetrads:
      self.bias_params[tetrad] = np.zeros(self.window)
    col = {}
    for colno, colname in enumerate(hdrs):
      col[colname] = colno
    for offset, line in enumerate(par_in):
      flds = line.rstrip('\n').split('\t')
      for tetrad in tetrads:
        self.bias_params.setdefault(tetrad,[])[offset] = log(float(flds[col[tetrad]]))
    par_in.close()

  def compute_probs(self, targ_seq):
    """
    Generate a vector of probabilities for sampling reads at each usable
    position in the given target sequence, of length equal to that of the
    sequence. Positions that would result in the read or the half-window
    preceding it containing ambiguity characters get zero probabilities, 
    as do positions too close to the beginning or end of the sequence.
    """
    npos = len(targ_seq)
    wts = np.zeros(npos)
    for read_pos in range(self.max_offset+3, npos-self.read_len):
      if not has_ambiguity(targ_seq, read_pos-self.max_offset-3, 
         self.read_len+self.max_offset+3):
        log_bias = 0.0
        for j in range(-self.max_offset, self.max_offset+1):
          log_bias += self.bias_params[targ_seq[read_pos+j-3:read_pos+j+1]][j]
        wts[read_pos] = exp(log_bias)
    probs = wts/sum(wts)
    return probs

################################################################################

def crp_abund_prior(beta, ntargs):
  """
  Construct a vector of target relative abundances using the 
  stick-breaking construction of the Chinese restaurant process.
  beta is the concentration parameter of the CRP.
  """

  # Sample stick length fractions from Beta(1, beta)
  v = np.random.beta(a=1.0, b=beta, size=ntargs)
  omv = 1.0 - v
  theta = v * np.concatenate(([1.0], 
    np.cumprod(omv[0:-1])))
  return theta

################################################################################

def compute_targ_samp_probs(targ_abundances, targ_lengths):
  """
  Compute the probabilities of sampling a read from each target, given their
  abundances and sequence lengths
  """
  wts = targ_abundances * targ_lengths
  probs = wts/sum(wts)
  return probs

################################################################################
class Selector:
  """
  Maintains selection probabilities for targets, and keeps track of targets
  selected for read generation. For each selected target, holds a vector 
  of position probablities.
  """

  #-----------------------------------------------------------------------------
  def __init__(self, db_name, beta=1.4, read_len=50, max_offset=10):
    self.db_name = db_name
    self.beta = beta
    self.read_len = read_len
    self.max_offset = max_offset
    self.ref_db = None
    self.target_gis = None
    self.targ_abundances = None
    self.target_probs = None
    self.targ_lengths = None
    self.ntargs = 0
    self.sel_targ_abundances = {}
    self.sel_targ_samp_probs = {}
    self.sel_targ_pos_probs = {}
    self.bias_model = None

  #-----------------------------------------------------------------------------
  def setup(self):
    self.ref_db = ref.RefDb(self.db_name)
    self.ref_db.load()
    catalog = self.ref_db.catalog
    self.target_gis = catalog.keys()
    # Randomize the order of targets
    np.random.shuffle(self.target_gis)
    # Compute target abundances and sampling probabilities using 
    # the stick-breaking CRP prior
    self.ntargs = len(self.target_gis)
    self.targ_abundances = crp_abund_prior(self.beta, self.ntargs)
    self.targ_lengths = np.array([catalog[gi].size for gi in self.target_gis])
    self.target_probs = compute_targ_samp_probs(self.targ_abundances,
       self.targ_lengths)
    # Initialize a bias model object
    self.bias_model = BiasModel(self.read_len, self.max_offset)
    self.bias_model.load()


  #-----------------------------------------------------------------------------
  def select_targets(self, count=1):
    """
    Select count targets from the reference DB according to the CRP prior
    probabilities. If a target hasn't been sampled before, compute
    the sample probabilities for each position in the target sequence.
    Returns a dict of counts for each sampled target, keyed by gi strings.
    """
    targ_count = np.random.multinomial(count, pvals=self.target_probs)
    targ_gi_count = {}
    for targ_ind in xrange(self.ntargs):
      if targ_count[targ_ind] > 0:
        gi = self.target_gis[targ_ind]
        if not self.sel_targ_pos_probs.has_key(gi):
          targ_seq = self.ref_db.retrieve(gi)
          self.sel_targ_pos_probs[gi] = self.bias_model.compute_probs(targ_seq)
          self.sel_targ_samp_probs[gi] = self.target_probs[targ_ind]
          self.sel_targ_abundances[gi] = self.targ_abundances[targ_ind]
        targ_gi_count[gi] = targ_count[targ_ind]
    return targ_gi_count

  #-----------------------------------------------------------------------------
  def sample_reads(self, count=1):
    """
    Sample count reads from the targets in the reference database,
    with target sampling probabilities based on the CRP prior and target
    length, and reads within each target sampled according to the bias model.
    Returns a list of (gi, position) tuples.
    """

    targ_gi_count = self.select_targets(count)
    reads = []
    for gi, n_occur in targ_gi_count.iteritems():
      pos_freq = np.random.multinomial(n_occur, 
        pvals=self.sel_targ_pos_probs[gi])
      for pos in xrange(len(self.sel_targ_pos_probs[gi])):
        for i in xrange(pos_freq[pos]):
          reads.append((gi,pos))
    np.random.shuffle(reads)
    return reads

  #-----------------------------------------------------------------------------
  def dump_reads_table(self, reads, table_file):
    """
    Output the read GI numbers and positions generated by sample_reads()
    to a tabular file
    """
    tab_out = open(table_file, 'w')
    for read in reads:
      print >> tab_out, '%s\t%d' % read
    tab_out.close()
    print >> sys.stderr, 'Wrote read position table to %s' % table_file


  #-----------------------------------------------------------------------------
  def dump_reads_fasta(self, reads, fasta_file):
    """
    Output the read sequences corresponding to the output of
    sample_reads() to a FASTA file
    """
    targ_seqs = {}
    fas_out = open(fasta_file, 'w')
    for rnum, read in enumerate(reads):
      (gi,pos) = read
      try:
        targ_seq = targ_seqs[gi]
      except KeyError:
        targ_seq = self.ref_db.retrieve(gi)
        targ_seqs[gi] = targ_seq
      read_seq = targ_seq[pos:pos+self.read_len]
      print >> fas_out, '>read_%d\n%s' % (rnum, read_seq)
    fas_out.close()
    print >> sys.stderr, 'Wrote read FASTA file %s' % fasta_file

  #-----------------------------------------------------------------------------
  def dump_params(self, param_file):
    """
    Save the parameters used to generate a read set in a YAML file
    """
    par_out = open(param_file, 'w')
    target_abund = {}
    bias_params = {}
    for gi, prob in self.sel_targ_abundances.iteritems():
      target_abund[gi] = float(prob)
    for base, biases in self.bias_model.bias_params.iteritems():
      bias_params[base] = [ float(bias) for bias in biases ]
    par_dict = dict(db_name=self.db_name,
                    beta=self.beta,
                    read_len=self.read_len,
                    max_offset=self.max_offset,
                    ntargs=self.ntargs,
                    target_abund=target_abund,
                    bias_params=bias_params)
    yaml.dump(par_dict, stream=par_out)
    par_out.close()
    print >> sys.stderr, 'Wrote params to %s' % param_file

################################################################################
# Test read simulation using viral RefSeq DB

def test_viral():
  selector = Selector(db_name='RefViral')
  selector.setup()
  reads = selector.sample_reads(10)
  for read in reads:
    print >> sys.stderr, '%s\t%d' % read

################################################################################
# Generate a simulated read set. Save the reads (represented as gi, pos pairs
# and as sequences in FASTA format) and the parameters used to generate 
# them in files with the specified prefix.

def simulate_reads(count=1000000, db_name='RefViral', beta=1.4,
     read_len=50, max_offset=10):
  prefix = '%s_ZOM_beta_%.1f_%d_mers_%d' % (db_name, beta, read_len, count)
  selector = Selector(db_name, beta, read_len, max_offset)
  selector.setup()
  reads = selector.sample_reads(count)
  param_file = '../sim_data/%s_params.yaml' % prefix
  read_file = '../sim_data/%s_reads.txt' % prefix
  fasta_file = '../sim_data/%s_reads.fas' % prefix
  selector.dump_params(param_file)
  selector.dump_reads_table(reads, read_file)
  selector.dump_reads_fasta(reads, fasta_file)

################################################################################
# Run Bowtie2 to find alignments of simulated reads against targets in
# reference DB

def align_sim_reads(count=1000000, db_name='RefViral', beta=1.4, read_len=50):
  prefix = '%s_ZOM_beta_%.1f_%d_mers_%d' % (db_name, beta, read_len, count)
  fasta_file = '../sim_data/%s_reads.fas' % prefix
  sam_file = ref.map_reads_bowtie(fasta_file, db_name)
  print >> sys.stderr, '\nAlignments written to %s' % sam_file

################################################################################
# Generate simulated reads, and then immediately align them to the reference
# DB used to generate them

def sim_and_align(count=1000000, db_name='RefViral', beta=1.4,
     read_len=50, max_offset=10):
  simulate_reads(count, db_name, beta, read_len, max_offset)
  align_sim_reads(count, db_name, beta, read_len)
