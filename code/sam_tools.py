###############################################################################
# sam_tools.py
#
# Functions for processing SAM file output from Bowtie2
#
# Kevin McLoughlin
# 05 Jun 2014
###############################################################################

import sys
import os
import re
import pdb


bt2_ind_dir = '../bt2_indices'
bt2_sam_dir = '../bt2_samfiles'
res_dir = '../results'

# Bit flags in SAM records
multi_segment           = 0x001
pair_mapped             = 0x002
query_unmapped          = 0x004
mate_unmapped           = 0x008
rev_comp                = 0x010
next_seg_reversed       = 0x020
first_read              = 0x040
second_read             = 0x080
secondary_alignment     = 0x100
qc_failed               = 0x200
dup_read                = 0x400
supplementary_alignment = 0x800

bad_read = ( multi_segment | query_unmapped | mate_unmapped | qc_failed | 
  dup_read | supplementary_alignment )
read_num_mask = first_read | second_read

###############################################################################
def read_strand(flags):
  """
  Return 0 or 1 according to whether the read aligns to the foward or reverse
  strand, respectively
  """
  return (flags & rev_comp)/rev_comp

###############################################################################
def read_number(flags):
  """
  Parse the flag bits and return 1 or 2 if alignment is for the 1st or 2nd
  read in a read pair, or 0 for an unpaired read
  """
  return (flags & read_num_mask)/first_read

###############################################################################
def parse_cigar(cigar_str):
  """
  Parse an extended CIGAR description of an alignment into a list of
  (operation, count) tuples
  """
  ops = 'MDISHNP'
  descr = []
  op_pat = re.compile(r'(\d+[MDISH])')
  descr = [ ( comp[-1], int(comp[0:-1]) ) for comp in op_pat.split(cigar_str)
    if len(comp) > 0]
  return descr

###############################################################################
class SamData:
  """
  Represents alignment data from one or more SAM files from a single sequencing
  run
  """
  def __init__(self):
    self.read_len = None
    self.target_gi_ind = {}   # Map from gi's to indices
    self.target_gis = []      # Map from indices to gi's
    self.targ_seq = {}        # Cache for target sequences encountered in alignments
    self.alignments = {}
    self.targ_align_cnt = {}  # Number of alignments to each target
    self.ref_db = None

  def add_target(self, gi):
    try:
      targ_ind = self.target_gi_ind[gi]
    except KeyError:
      self.target_gis.append(gi)
      self.target_gi_ind[gi] = len(self.target_gis)-1
      self.targ_seq[gi] = self.ref_db.retrieve(gi)

  def set_read_length(self, read_len):
    if not self.read_len:
      self.read_len = read_len
    else:
      # Sanity check - all reads should be same length
      if read_len != self.read_len:
        raise Exception('Read length varies')

  def set_ref_db(self, ref_db):
    self.ref_db = ref_db

  def add_alignment(self, read_id, target_gi, pos, strand):
    """
    Add the given (gi,position,strand) tuple to the set of alignments for
    read_id. If alignment is on the reverse strand, adjust the start position
    to be the end of the read in SAM coordinates.
    """
    if strand == 1:
      pos += self.read_len - 1
    self.alignments.setdefault(read_id, set()).add((target_gi, pos, strand))
    self.targ_align_cnt[target_gi] = self.targ_align_cnt.get(target_gi,0) + 1

###############################################################################
def mismatch_count(sam_flds):
  """
  Extract the mismatch count from the optional tag-value fields in a SAM record
  """
  for i in xrange(11,len(sam_flds)):
    (tag,datatype,value) = sam_flds[i].split(':')
    if tag == 'NM':
      return int(value)
  raise Exception('Mismatch count missing')

###############################################################################
def perfect_read_hits(sam_file, sam_data = None):
  """
  Read the given SAM file and collect the perfect match alignments for each read. 
  Return them as a dict of (gi,pos,strand,count) tuples keyed by read ID (and 
  read number, if paired). Here gi is the gi number of the target, pos is the
  starting position of the alignment, strand is 0 or 1 for the forward or 
  reverse strand, and count is the number of perfect match alignments with the same 
  gi, pos and strand. 
  If the read_hits dict is passed as a parameter, add the alignments
  to this dict and return the modified dict.
  """
  if not sam_data:
    sam_data = SamData()

  gi_pat = re.compile(r'gi\|(\d+)\|')
  align_count = 0
  print >> sys.stderr, 'Scanning SAM file %s to find alignments for each read' % sam_file
  sam_in = open(sam_file, 'r')
  for line in sam_in:
    #if line.startswith('@SQ'):
    #  ref_fld = line.rstrip('\n').split('\t')[1]
    #  gi_match = gi_pat.search(ref_fld)
    #  if gi_match:
    #    gi = gi_match.group(1)
    #    sam_data.add_target(gi)
    if not line.startswith('@'):
      flds = line.rstrip('\n').split('\t')
      align_count += 1
      if align_count % 100000 == 0:
        print >> sys.stderr, 'Processed %d alignments' % align_count
      flags = int(flds[1])
      if (flags & bad_read) == 0:
        cigar = flds[5]
        cigar_ops = parse_cigar(cigar)
        if len(cigar_ops) == 1 and cigar_ops[0][0] == 'M':
          # Read aligns to target along its entire length. 
          # Set the read length if we don't know it already.
          read_len = cigar_ops[0][1]
          sam_data.set_read_length(read_len)
          # Only count reads with zero mismatches
          if mismatch_count(flds) == 0:
            read_id = flds[0]
            target_gi = gi_pat.match(flds[2]).group(1)
            pos = int(flds[3])
            strand = read_strand(flags)
            sam_data.add_target(target_gi)
            sam_data.add_alignment(read_id, target_gi, pos, strand)
  sam_in.close()
  return sam_data

