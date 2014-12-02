################################################################################
# online_vbayes.py
#
# Use online variational Bayes method to fit abundances and position bias parameters
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
class OnlineVBayesParams:
  """
  Stores the variational posterior distribution parameters. Implements
  methods for the VBayes updates.
  """

  def __init__(self, sam_data, param_dict, kappa, tau0, max_offset=10):
    """
    Set the parameters to the default initial state
    """
    self.sam_data = sam_data
    self.ref_db = sam_data.ref_db
    self.param_dict = param_dict
    # Alignments for each read are sets of (gi,pos,strand) tuples
    self.alignments = sam_data.alignments
    self.read_ids = sorted(self.alignments.keys())
    self.nreads = len(self.read_ids)
    self.target_gi_ind = sam_data.target_gi_ind
    self.target_gis = sam_data.target_gis
    self.ntargs = len(self.target_gis)
    self.targ_len = np.zeros(self.ntargs)
    self.eff_len = np.zeros(self.ntargs)
    self.num_align = np.zeros(self.ntargs)
    self.frac_align = np.zeros(self.ntargs)

    self.max_offset = max_offset
    self.window = 2*max_offset + 1

    self.kappa = kappa
    self.tau0 = tau0

    # Store the variational categorical probabilities for
    # the read currently being processed, for each mapped target 
    self.targ_theta = {}
    #self.log_targ_pos_phi = []
    # Parameters for beta distributions in stick-breaking construction
    self.alpha_prior = np.ones(self.ntargs)
    self.beta_prior = np.ones(self.ntargs)
    self.alpha = self.alpha_prior
    self.beta = self.beta_prior
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
    # Target abundance estimates
    self.rho = np.zeros(self.ntargs)
    self.rho_ci_low = np.zeros(self.ntargs)
    self.rho_ci_hi = np.zeros(self.ntargs)

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



  def vbe_step(self, n):
    """
    Perform the "expectation" step of the online variational Bayes algorithm.
    Here we compute expectations for the sufficient statistics of the
    hidden variables.
    """
    read_id = self.read_ids[n]
    alignments = self.alignments[read_id]
    self.targ_theta = {}

    # Add a term to log theta for each unique target
    # matched by the read. Note that psi() is the digamma function.

    if len(alignments) == 1:
      # First deal with the case when the read hits only one target; then we
      # assign that target a probability of one.
      (gi, p, strand) = alignments.copy().pop()
      t = self.target_gi_ind[gi]
      self.targ_theta[t] = 1.0
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
        #log_theta += log(self.eff_len[t])
        theta = exp(log_theta)
        sum_theta += theta
        self.targ_theta[t] = theta
      # Normalize the target probabilities to add to 1
      for t in targ_indices:
        self.targ_theta[t] /= sum_theta


  def vbm_step(self, n):
    """
    Perform the "maximization" step of the variational Bayes algorithm,
    by updating the hyperparameters of the variational distributions for
    the parameters.
    """
    lrate = 1/(self.tau0 + n)**self.kappa

    targ_theta = self.targ_theta
    nread_targ = len(targ_theta)
    read_targ_indices = sorted(targ_theta.keys())
    #for t in xrange(self.ntargs):
    for t in read_targ_indices:
      self.alpha[t] = (1.0-lrate) * self.alpha[t]
      self.beta[t] = (1.0-lrate) * self.beta[t]
    for t in read_targ_indices:
      self.alpha[t] += lrate * (self.alpha_prior[t] + self.nreads*targ_theta[t])
      for j in range(t):
        self.beta[j] += lrate * (self.beta_prior[t] + self.nreads*targ_theta[t])

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
      try:
        self.targ_samp_prob[t] = exp(log_theta[t])
      except OverflowError:
        self.targ_samp_prob[t] = 0.0
      try:
        theta_ci_low[t] = exp(log_theta[t] - ci95sd * sd_log_theta[t])
      except OverflowError:
        theta_ci_low[t] = 0.0
      try:
        theta_ci_hi[t] = exp(log_theta[t] + ci95sd * sd_log_theta[t])
      except OverflowError:
        theta_ci_hi[t] = 0.0

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
      'targ_samp_prob', 'alpha', 'beta', 'alpha+beta', 'log_theta', 'sd_log_theta'])
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
def online_vb(kappa, tau0, sam_data=None, prefix = 'RefViral_ZOM_beta_1.4_50_mers',
  db_name='RefViral', nreads=10000):
  """
  Main function for testing online VB algorithm against simulated data.
  """
  if not sam_data:
    sam_file = '%s/%s_%d_reads-%s.sam' % (sam_dir, prefix, nreads, db_name)
    sam_data = init_all([sam_file], db_name)

  # Get dict of input parameters used to generate simulated data
  param_file = '%s/%s_%d_params.yaml' % (sim_dir, prefix, nreads)
  par_in = open(param_file, 'r')
  param_dict = yaml.load(par_in)
  par_in.close()

  vb = OnlineVBayesParams(sam_data, param_dict, kappa, tau0)
  for n in xrange(vb.nreads):
    if (n-1) % 10000 == 0:
      print >> sys.stderr, 'Processed %d reads' % n
    vb.vbe_step(n)
    vb.vbm_step(n)
  vb.estimate_abundances()
  abund_file = '%s/%s_%d_rds_%s_online_abund_estimates.txt' % (res_dir, prefix, nreads,
     db_name)
  vb.write_abundances(abund_file)
  return vb
      

