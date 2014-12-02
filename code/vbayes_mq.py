################################################################################
# vbayes_mq.py
#
# Use variational Bayes method to fit abundances and position bias parameters
# to simplified MetaQuant model
#
# Kevin McLoughlin
# 16 Apr 2014
################################################################################

import sys
import re
import pdb
from math import log, exp, sqrt
import yaml
import datetime as dt

import refseq_db as ref
import sam_tools as sam

import numpy as np
from scipy.special import psi, polygamma
from scipy.stats import gaussian_kde


################################################################################
# constants

bases = ['A', 'C', 'G', 'T']
ACGT = bases
ACG = bases[0:3]
base_set = set(bases)
trimers = []
tetramers = []
for a in bases:
  for b in bases:
    for c in bases:
      trimers.append(a+b+c)
      for d in bases:
        tetramers.append(a+b+c+d)

sam_dir = '../sam_files'
sim_dir = '../sim_data'
res_dir = '../results'

# Standard deviations from the mean to produce a 95% confidence interval
ci95sd = 1.959964

################################################################################
class VBayesParams:
  """
  Stores the variational posterior distribution parameters. Implements
  methods for the VBayes updates.
  """

  def __init__(self, sam_data, param_dict, max_offset=10):
    """
    Set the parameters to the default initial state
    """
    self.sam_data = sam_data
    self.ref_db = sam_data.ref_db
    self.param_dict = param_dict
    # Alignments for each read are sets of (gi,pos,strand) tuples
    self.alignments = sam_data.alignments
    self.nreads = len(sam_data.alignments)
    self.target_gi_ind = sam_data.target_gi_ind
    self.target_gis = sam_data.target_gis
    self.ntargs = len(self.target_gis)
    self.targ_len = np.zeros(self.ntargs)
    self.eff_len = np.zeros(self.ntargs)
    self.num_align = np.zeros(self.ntargs)
    self.frac_align = np.zeros(self.ntargs)

    self.max_offset = max_offset
    self.window = 2*max_offset + 1

    # Store the variational categorical probabilities for
    # each read, for each mapped target 
    self.targ_theta = []
    for n in xrange(self.nreads):
      self.targ_theta.append({})
    #self.log_targ_pos_phi = []
    # Parameters for beta distributions in stick-breaking construction
    self.alpha_prior = np.ones(self.ntargs)
    self.beta_prior = np.ones(self.ntargs)
    self.alpha = np.ones(self.ntargs)
    self.beta = np.ones(self.ntargs)
    # Target sequence lengths and alignment counts
    for i, gi in enumerate(self.target_gis):
      self.targ_len[i] = self.ref_db.get_length(gi)
      self.eff_len[i] = self.targ_len[i] - self.sam_data.read_len + 1
      self.num_align[i] = sam_data.targ_align_cnt[gi]
    self.frac_align = self.num_align/sum(self.num_align)
    # Target sampling probability estimates
    self.log_theta = np.zeros(self.ntargs)
    self.sd_log_theta = np.zeros(self.ntargs)
    self.targ_samp_prob = np.zeros(self.ntargs)
    self.est_samp_prob = np.zeros(self.ntargs)
    # Target abundance estimates
    self.rho = np.zeros(self.ntargs)
    self.rho_ci_low = np.zeros(self.ntargs)
    self.rho_ci_hi = np.zeros(self.ntargs)

  def set_targ_theta(self, n, t, theta):
    """
    Set log theta_{nt} to the given value for read n, target t.
    Add an entry for the given target to the dict for the read, if
    necessary.
    """
    self.targ_theta[n][t] = theta

  def init_targ_thetas(self):
    """
    Initialize per-read target assignment proportions to be uniform over targets
    hit by read
    """
    read_ids = sorted(self.alignments.keys())
    for n in xrange(self.nreads):
      read_id = read_ids[n]
      alignments = self.alignments[read_id]

      if len(alignments) == 1:
        # First deal with the case when the read hits only one target; then we
        # assign that target a probability of one.
        (gi, p, strand) = alignments.copy().pop()
        t = self.target_gi_ind[gi]
        self.set_targ_theta(n, t, 1.0)
      else:
        # Otherwise update the categorical probabilities for the read to come
        # from each target that it hits
        targ_indices = set()
        for (gi, p, strand) in alignments:
          targ_indices.add(self.target_gi_ind[gi])
        ntargs = len(targ_indices)
        for t in targ_indices:
          self.set_targ_theta(n, t, 1.0/ntargs)

  def count_targ_sets(self, out_file):
    """
    Count occurrences of each set of targets aligned by reads
    """
    read_ids = sorted(self.alignments.keys())
    targ_set_count = {}
    for n in xrange(self.nreads):
      read_id = read_ids[n]
      alignments = self.alignments[read_id]
      targ_indices = set()
      for (gi, p, strand) in alignments:
        targ_indices.add(self.target_gi_ind[gi])
      targ_tuple = tuple(sorted(targ_indices))
      if len(targ_tuple) == 0:
        raise Exception('Empty tuple')
      targ_set_count[targ_tuple] = targ_set_count.get(targ_tuple,0) + 1
    out = open(out_file, 'w')
    print >> out, 'targ_indices\tcount'
    for targ_tuple, count in targ_set_count.iteritems():
      print >> out, '%s\t%d' % (','.join(['%d' % t for t in targ_tuple]), count)
    out.close()
    return targ_set_count



  #def add_targ_pos_phi(self, n, t, p, delta):
  #  """
  #  Increment log phi_{ntp} by the given amount for read n, target t, position p.
  #  Add entries for the given target and position for the read, if necessary.
  #  """
  #  if not self.log_targ_pos_phi[n].has_key(t):
  #    self.log_targ_pos_phi[n][t] = {}
  #  self.log_targ_pos_phi[n][t][p] = self.log_targ_pos_phi[n][t].get(p,0.0) + delta

  def vbe_step(self):
    """
    Perform the "expectation" step of the variational Bayes algorithm.
    Here we compute expectations for the sufficient statistics of the
    hidden variables.
    """
    read_ids = sorted(self.alignments.keys())
    self.est_samp_prob = np.zeros(self.ntargs)
    for n in xrange(self.nreads):
      read_id = read_ids[n]
      alignments = self.alignments[read_id]

      # Add a term to log theta for each unique target
      # matched by the read. Note that psi() is the digamma function.

      if len(alignments) == 1:
        # First deal with the case when the read hits only one target; then we
        # assign that target a probability of one.
        (gi, p, strand) = alignments.copy().pop()
        t = self.target_gi_ind[gi]
        self.set_targ_theta(n, t, 1.0)
        self.est_samp_prob[t] += 1.0
      else:
        # Otherwise update the categorical probabilities for the read to come
        # from each target that it hits
        targ_indices = set()
        for (gi, p, strand) in alignments:
          targ_indices.add(self.target_gi_ind[gi])
        sum_theta = 0.0
        for t in targ_indices:
          log_theta = psi(self.alpha[t]) - psi(self.alpha[t]+self.beta[t])
          for j in xrange(t):
            log_theta += psi(self.beta[j]) - psi(self.alpha[j]+self.beta[j])
          log_theta += log(self.eff_len[t])
          theta = exp(log_theta)
          sum_theta += theta
          self.set_targ_theta(n, t, theta)
        # Normalize the target probabilities to add to 1
        for t in targ_indices:
          self.targ_theta[n][t] /= sum_theta
          self.est_samp_prob[t] += self.targ_theta[n][t]
    # Estimate target sampling probabilities as the total fraction of reads assigned to 
    # each target
    for t in xrange(self.ntargs):
      self.est_samp_prob[t] /= self.nreads


  def vbm_step(self):
    """
    Perform the "maximization" step of the variational Bayes algorithm,
    by updating the hyperparameters of the variational distributions for
    the parameters.
    """
    for t in xrange(self.ntargs):
      self.alpha[t] = self.alpha_prior[t]
      self.beta[t] = self.beta_prior[t]
    for n in xrange(self.nreads):
      targ_theta = self.targ_theta[n]
      nread_targ = len(targ_theta)
      read_targ_indices = sorted(targ_theta.keys())
      for t in read_targ_indices:
        self.alpha[t] += targ_theta[t]
        for j in range(t):
          self.beta[j] += targ_theta[t]

  def estimate_abundances(self):
    """
    Compute expectations and variances of the log relative abundances (log rho)
    of each target. Use these to compute 95% confidence intervals of the relative
    abundances themselves.
    """
    log_theta = np.zeros(self.ntargs)
    sd_log_theta = np.zeros(self.ntargs)
    for t in xrange(self.ntargs):
      log_theta[t] = psi(self.alpha[t]) - psi(self.alpha[t]+self.beta[t])
      var_log_theta = polygamma(1,self.alpha[t]) - polygamma(1,
         self.alpha[t]+self.beta[t])
      for j in xrange(t):
        log_theta[t] += psi(self.beta[j]) - psi(self.alpha[j]+self.beta[j])
        var_log_theta += polygamma(1,self.beta[j]) - polygamma(1,
           self.alpha[j]+self.beta[j])
      sd_log_theta[t] = sqrt(var_log_theta)
    self.log_theta = log_theta
    self.sd_log_theta = sd_log_theta
    theta_ci_low = np.zeros(self.ntargs)
    theta_ci_hi = np.zeros(self.ntargs)
    for t in xrange(self.ntargs):
      self.targ_samp_prob[t] = exp(log_theta[t])
      theta_ci_low[t] = exp(log_theta[t] - ci95sd * sd_log_theta[t])
      theta_ci_hi[t] = exp(log_theta[t] + ci95sd * sd_log_theta[t])

    # Compute relative abundances and confidence limits
    w = self.targ_samp_prob / self.eff_len
    self.rho = w / sum(w)
    w_low = theta_ci_low / self.eff_len
    self.rho_ci_low = w_low / sum(w_low)
    w_hi = theta_ci_hi / self.eff_len
    self.rho_ci_hi = w_hi / sum(w_hi)


  def write_abundances(self, out_file):
    """
    Write a table of abundance estimates
    """
    sim_abund = self.param_dict['target_abund']
    out = open(out_file, 'w')
    print >> out, '\t'.join([
      'targ_ind', 'target_gi', 'length', 
      'num_align', 
      'exp_num_align', 
      'frac_align',
      'exp_frac_align',
      'rho', 'rho_ci_low', 'rho_ci_hi', 'input_rho',
      'targ_samp_prob', 'est_samp_prob', 'alpha', 'beta', 'alpha+beta', 'log_theta', 'sd_log_theta'])
    # Compute expected number and fraction of reads with alignments to each target
    targ_wt = np.zeros(self.ntargs)
    sum_wts = 0.0
    for t in xrange(self.ntargs):
      input_rho = sim_abund.get(self.target_gis[t], 0.0)
      targ_wt[t] = input_rho * self.eff_len[t]
      sum_wts += targ_wt[t]
    exp_frac_align = targ_wt / sum_wts
    exp_num_align = exp_frac_align * self.nreads
    for t in xrange(self.ntargs):
      input_rho = sim_abund.get(self.target_gis[t], 0.0)
      print >> out, '\t'.join([
         '%d' % t,
         self.target_gis[t],
         '%d' % self.eff_len[t],
         '%.0f' % self.num_align[t],
         '%.0f' % exp_num_align[t],
         '%.6f' % self.frac_align[t],
         '%.6f' % exp_frac_align[t],
         '%.6f' % self.rho[t],
         '%.6f' % self.rho_ci_low[t],
         '%.6f' % self.rho_ci_hi[t],
         '%.6f' % input_rho,
         '%.6f' % self.targ_samp_prob[t],
         '%.6f' % self.est_samp_prob[t],
         '%.3f' % self.alpha[t],
         '%.3f' % self.beta[t],
         '%.3f' % (self.alpha[t] + self.beta[t]),
         '%.3f' % self.log_theta[t],
         '%.3f' % self.sd_log_theta[t] ])
    out.close()

################################################################################
def elapsed(tdelta):
  """
  Convert a datetime.timedelta object to floating point elapsed time
  in seconds
  """
  return 86400.*tdelta.days + tdelta.seconds + 1e-6 * tdelta.microseconds

################################################################################

def init_all(sam_files, db_name='RefViral'):
  """
  Load SAM file(s) from simulated read alignments to ref genomes, and load
  the reference DB.
  """
  ref_db = ref.RefDb(db_name)
  ref_db.load()
  sam_data = sam.SamData()
  sam_data.set_ref_db(ref_db)
  for sam_file in sam_files:
    sam_data = sam.perfect_read_hits(sam_file, sam_data)
  return sam_data


################################################################################
def vb_convergence(iters=10, prefix = 'RefViral_ZOM_beta_1.4_50_mers',
  db_name='RefViral', nreads=10000):
  """
  Main function for testing VB algorithm against simulated data.
  Assess number of VB iterations required for convergence of alpha and beta
  parameters, as well as time for each iteration.
  """
  sam_file = '%s/%s_%d_reads-%s.sam' % (sam_dir, prefix, nreads, db_name)
  sam_data = init_all([sam_file], db_name)

  # Get dict of input parameters used to generate simulated data
  param_file = '%s/%s_%d_params.yaml' % (sim_dir, prefix, nreads)
  par_in = open(param_file, 'r')
  param_dict = yaml.load(par_in)
  par_in.close()

  vb = VBayesParams(sam_data, param_dict)
  out = open('%s/%s_%d_rds_%s_vb_convergence.txt' % (res_dir, prefix, nreads,
     db_name), 'w')
  alpha_hdrs = ['alpha_%s' % vb.target_gis[i] for i in range(vb.ntargs)]
  beta_hdrs = ['beta_%s' % vb.target_gis[i] for i in range(vb.ntargs)]
  log_theta_hdrs = []
  targ_samp_hdrs = []
  vb.init_targ_thetas()
  for i in range(vb.ntargs):
    log_theta_hdrs = log_theta_hdrs + ['log_theta_%s' % vb.target_gis[i],
                                   'sd_log_theta_%s' % vb.target_gis[i] ]
    targ_samp_hdrs = targ_samp_hdrs + ['targ_samp_prob_%s' % vb.target_gis[i],
                                   'est_samp_prob_%s' % vb.target_gis[i] ]
  print >> out, '\t'.join([
    'nreads', 'iter', 'vbe_time', 'vbm_time', 'est_time'] + alpha_hdrs + beta_hdrs
      + log_theta_hdrs + targ_samp_hdrs)
  for i in xrange(iters):
    vbm_start = dt.datetime.now()
    vb.vbm_step()
    vbm_end = dt.datetime.now()
    vbm_time = elapsed(vbm_end - vbm_start)
    vb.vbe_step()
    vbe_end = dt.datetime.now()
    vbe_time = elapsed(vbe_end - vbm_end)
    vb.estimate_abundances()
    est_end = dt.datetime.now()
    est_time = elapsed(est_end - vbe_end)
    print >> sys.stderr, 'Iteration %d' % i
    log_theta_vals = []
    targ_samp_vals = []
    for j in range(vb.ntargs):
      log_theta_vals = log_theta_vals + ['%.3f' % vb.log_theta[j], 
                                     '%.3f' % vb.sd_log_theta[j] ]
      targ_samp_vals = targ_samp_vals + ['%.5f' % vb.targ_samp_prob[j], 
                                     '%.5f' % vb.est_samp_prob[j] ]
    print >> out, '\t'.join([
       '%d' % nreads,
       '%d' % i,
       '%f' % vbe_time,
       '%f' % vbm_time,
       '%f' % est_time] + 
       ['%f' % vb.alpha[j] for j in xrange(vb.ntargs)] +
       ['%f' % vb.beta[j] for j in xrange(vb.ntargs)] + log_theta_vals + targ_samp_vals )
  out.close()
  abund_file = '%s/%s_%d_rds_%s_abund_estimates.txt' % (res_dir, prefix, nreads,
     db_name)
  vb.write_abundances(abund_file)
  return vb
      

################################################################################
def old_vb_convergence(iters=20, nreads=50000):
  """
  DEPRECATED
  Assess number of VB iterations required for convergence of alpha and beta
  parameters, as well as time for each iteration.
  """
  prefix = 'RefViral_beta_1.4_50_mers'
  read_set = ReadSet.load_sim_reads(prefix, nreads)
  vb = VBayesParams(read_set)
  out = open('../results/%s_%d_rds_vb_convergence.txt' % (prefix,nreads), 'w')
  alpha_hdrs = ['alpha_%d' % i for i in range(vb.ntargs)]
  beta_hdrs = ['beta_%d' % i for i in range(vb.ntargs)]
  gamma_hdrs = ['gamma_%d_%s' % (x, tet) for x in xrange(vb.window) 
    for tet in tetramers ]
  print >> out, '\t'.join([
    'nreads', 'iter', 'vbe_time', 'vbm_time'] + alpha_hdrs + beta_hdrs
      + gamma_hdrs)
  for i in xrange(iters):
    vbe_start = dt.datetime.now()
    vb.vbe_step()
    vbe_end = dt.datetime.now()
    vbe_time = elapsed(vbe_end - vbe_start)
    vb.vbm_step()
    vbm_end = dt.datetime.now()
    vbm_time = elapsed(vbm_end - vbe_end)
    print >> sys.stderr, 'Iteration %d' % i
    print >> out, '\t'.join([
       '%d' % nreads,
       '%d' % i,
       '%f' % vbe_time,
       '%f' % vbm_time] + 
       ['%f' % vb.alpha[j] for j in xrange(vb.ntargs)] +
       ['%f' % vb.beta[j] for j in xrange(vb.ntargs)] +
       ['%f' % vb.gamma[x][tet] for x in xrange(vb.window) for tet in tetramers])
  out.close()
      

################################################################################
def zero_order_bias(prefix = 'RefViral_beta_1.4_50_mers', nreads=100000,
 max_offset=10):
  """
  Compute the base frequencies at each offset relative to the starts of the simulated
  reads, as parameters of a zero-order Markov model. Output a table of frequencies
  that we can plot or use as parameters for simulating a new read set.
  """
  read_set = ReadSet.load_sim_reads(prefix, nreads)
  read_set = read_set
  nreads = read_set.size()
  target_dict = read_set.target_dict
  target_gis = read_set.target_gis
  target_ind = read_set.target_ind
  ntargs = len(target_dict)

  max_offset = max_offset
  window = 2*max_offset + 1
  base_count = {}
  for b in bases:
    base_count[b] = np.zeros(window)
  for n in xrange(nreads):
    (t, p) = read_set.reads[n]
    gi = target_gis[t]
    targ_data = target_dict[gi]
    for x in xrange(window):
      b = targ_data.targ_seq[p+x-max_offset]
      base_count[b][x] += 1
  base_freq = {}
  for b in bases:
    base_freq[b] = base_count[b]/nreads
  out = open('../results/%s_zero_order_bias.txt' % prefix, 'w')
  print >> out, '\t'.join(bases)
  for x in xrange(window):
    print >> out, '%d\t%s' % (x-max_offset, '\t'.join(
       ['%.7f' % base_freq[b][x] for b in bases]))
  out.close()

################################################################################
class ReadSet:
  """
  DEPRECATED: Represents a read data set from the old simulation where we generated
  (target,position) tuples only. 
  """

  def __init__(self, prefix):
    self.prefix = prefix
    self.target_dict = {}
    self.reads = []

  def size(self):
    return len(self.reads)

  def load(self, max_reads=100000000):
    """
    Read target gi numbers and read positions from file, set up TargetData
    structures
    """
    param_file = '../sim_data/%s_params.yaml' % self.prefix
    par_in = open(param_file, 'r')
    param_dict = yaml.load(par_in)
    par_in.close()
    db_name = param_dict['db_name']
    ref_db = ref.RefDb(db_name)
    ref_db.load()
  
    self.target_dict = {}
  
    read_file = '../sim_data/%s_reads.txt' % self.prefix
    read_in = open(read_file, 'r')
    for read_cnt, line in enumerate(read_in):
      if read_cnt > max_reads:
        break
      (gi, pos_str) = line.rstrip('\n').split('\t')
      pos = int(pos_str)
      self.reads.append((gi,pos))
      try:
        target_data = self.target_dict[gi]
      except KeyError:
        target_data = TargetData(gi)
        target_data.targ_seq = ref_db.retrieve(gi)
        self.target_dict[gi] = target_data
      target_data.read_pos_counts[pos] = target_data.read_pos_counts.get(pos, 0) + 1
    read_in.close()
    # Convert read GI numbers to target indices.
    # The ordering of the targets shouldn't matter, since (I think) the
    # stick-breaking construction of the CRP is exchangeable. Therefore,
    # we assign indices to the targets based on the gi order.
    self.target_gis = sorted(self.target_dict.keys())
    self.target_ind = {}
    for ind, gi in enumerate(self.target_gis):
      self.target_ind[gi] = ind
    for i in xrange(len(self.reads)):
      self.reads[i] = (self.target_ind[self.reads[i][0]], self.reads[i][1])


  @staticmethod
  def load_sim_reads(prefix, max_reads=100000000):
    read_set = ReadSet(prefix)
    read_set.load(max_reads)
    return read_set


################################################################################
class TargetData:
  """
  DEPRECATED
  Represents a target with sequence reads aligned to it
  """

  def __init__(self, gi):
    self.gi = gi
    self.targ_seq = None
    # Store numbers of reads aligned to each position in target
    self.read_pos_counts = {}

################################################################################
class ThirdOrderVBayesParams:
  """
  DEPRECATED
  Stores the variational posterior distribution parameters. Implements
  methods for the VBayes updates for the MetaQuant model with a 3rd order
  Markov model for position bias.
  """

  def __init__(self, read_set, max_offset=10):
    """
    Set the parameters to the default initial state
    """
    self.read_set = read_set
    self.nreads = read_set.size()
    self.target_dict = read_set.target_dict
    self.target_gis = read_set.target_gis
    self.target_ind = read_set.target_ind
    self.ntargs = len(self.target_dict)

    self.max_offset = max_offset
    self.window = 2*max_offset + 1

    # In our simplified model, each read maps to a unique (target,pos) tuple.
    # So we only need to store the variational categorical probabilities for
    # each mapped target or (target,pos) pair for each read.
    self.targ_theta = np.zeros(self.nreads)
    self.targ_pos_phi = np.zeros(self.nreads)
    # Parameters for beta distributions in stick-breaking construction
    self.alpha_prior = np.ones(self.ntargs)
    self.beta_prior = np.ones(self.ntargs)
    self.alpha = np.ones(self.ntargs)
    self.beta = np.ones(self.ntargs)
    self.targ_len = np.zeros(self.ntargs)
    # Target sequence lengths
    for i in xrange(self.ntargs):
      gi = self.target_gis[i]
      target_data = self.target_dict[gi]
      self.targ_len[i] = len(target_data.targ_seq)
    # Parameters for Dirichlet distributions used to generate position bias
    # probabilities
    self.gamma = []
    self.gamma_prior = []
    self.gamma0 = []
    self.gamma0_prior = []
    for x in xrange(self.window):
      gamma_x = {}
      for tetra in tetramers:
        gamma_x[tetra] = 1.0
      self.gamma_prior.append(gamma_x)
      self.gamma.append(gamma_x)
      gamma0_x = {}
      for trimer in trimers:
        gamma0_x[trimer] = 0.0
      self.gamma0_prior.append(gamma0_x)
      self.gamma0.append(gamma0_x)

  def vbe_step(self):
    """
    Perform the "expectation" step of the variational Bayes algorithm.
    Here we compute expectations for the sufficient statistics of the
    hidden variables.
    """
    for n in xrange(self.nreads):
      (t, p) = self.read_set.reads[n]
      log_theta = psi(self.alpha[t]) - psi(self.alpha[t]+self.beta[t])
      for j in xrange(t-1):
        log_theta += psi(self.beta[j]) - psi(self.alpha[j]+self.beta[j])
      self.targ_theta[n] = exp(log_theta)

      gi = self.target_gis[t]
      targ_data = self.target_dict[gi]
      log_phi = 0.0
      for x in xrange(self.window):
	offset = p+x-self.max_offset-3
        tetramer = targ_data.targ_seq[offset:offset+4]
	log_phi += psi(self.gamma[x][tetramer])
	trimer = tetramer[0:3]
	log_phi -= psi(sum([self.gamma[x][trimer+b] for b in bases]))
      self.targ_pos_phi[n] = exp(log_phi)

  def vbm_step(self):
    """
    Perform the "maximization" step of the variational Bayes algorithm,
    by updating the hyperparameters of the variational distributions for
    the parameters.
    """
    for t in xrange(self.ntargs):
      self.alpha[t] = self.alpha_prior[t]
      self.beta[t] = self.beta_prior[t]
    for x in xrange(self.window):
      for trimer in trimers:
	self.gamma0[x][trimer] = self.gamma0_prior[x][trimer]
        for b in ACG:
	  tetra = trimer+b
	  self.gamma[x][tetra] = self.gamma_prior[x][tetra]
    for n in xrange(self.nreads):
      (t, p) = self.read_set.reads[n]
      self.alpha[t] += self.targ_theta[n]
      for j in xrange(t-1):
        self.beta[j] += self.targ_theta[n]
      gi = self.target_gis[t]
      targ_data = self.target_dict[gi]
      for x in xrange(self.window):
	offset = p+x-self.max_offset-3
        tetramer = targ_data.targ_seq[offset:offset+4]
	trimer = tetramer[0:3]
	self.gamma[x][tetramer] += self.targ_pos_phi[n]
	self.gamma0[x][trimer] += self.targ_pos_phi[n]


