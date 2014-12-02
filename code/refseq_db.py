################################################################################
# refseq_db.py
#
# Code to process RefSeq data into easily accessible form for metagenomic
# data simulation and analysis
#
# Kevin McLoughlin
# 11 Apr 2014
################################################################################

import sys
import re
import os
import pdb

bowtie_dir = '/usr/local/bowtie2'

################################################################################
def catalog_db(db_name = 'RefViral'):
  """
  Read a RefSeq FASTA file, parse the headers, and accumulate sequence
  data for each target therein. Create two files: a data file containing
  all the sequence data without line breaks or other delimiters; and
  a catalog file listing the gi, accession, description, length and 
  offset of the data for each target sequence.
  """
  db_file = '../ref_dbs/RefSeq/%s.fna' % db_name
  cat_file = '../ref_dbs/RefSeq/%s_catalog.txt' % db_name
  seq_file = '../ref_dbs/RefSeq/%s_seq.dat' % db_name
  db_in = open(db_file, 'r')
  cat_out = open(cat_file, 'w')
  seq_out = open(seq_file, 'wb')
  hdr_pat = re.compile(r'>gi\|(\d+)\|ref\|([^|]+)\| (.*)')
  gi = None
  acc = ''
  desc = ''
  offset = seq_out.tell()
  seq = ''
  seq_count = 0
  for line in db_in:
    line = line.rstrip('\n')
    hdr_match = hdr_pat.match(line)
    if hdr_match:
      if gi:
        print >> cat_out, '%s\t%s\t%d\t%d\t%s' % (gi, acc, offset, 
           len(seq), desc)
        seq_out.write(seq)
      offset = seq_out.tell()
      seq = ''
      seq_count += 1
      if seq_count % 1000 == 0:
        print >> sys.stderr, 'Processed %d sequences' % seq_count
      gi = hdr_match.group(1)
      acc = hdr_match.group(2)
      desc = hdr_match.group(3)
    else:
      seq = seq + line
  db_in.close()
  if gi:
    print >> cat_out, '%s\t%s\t%d\t%d\t%s' % (gi, acc, offset, len(seq), desc)
    seq_out.write(seq)
  cat_out.close()
  seq_out.close()

################################################################################
def index_db_bowtie2(db_name = 'RefViral'):
  """
  Build index files for Bowtie2
  """
  db_file = '../ref_dbs/RefSeq/%s.fna' % db_name

  cmd = '%s/bowtie2-build -o 3 %s ../bt2_indices/%s' % (bowtie_dir,
       db_file, db_name)
  print >> sys.stderr, 'Running %s' % cmd
  status = os.system(cmd)
  if status != 0:
    raise Exception('%s failed with status %d' % (cmd, status))

###############################################################################
def map_reads_bowtie(read_fasta_file, db_name):
  """
  Find all Bowtie2 alignments of single-end reads against all targets in the 
  specified RefSeq DB
  """
  (dirname, filename) = os.path.split(read_fasta_file)
  (prefix, ext) = os.path.splitext(filename)
  sam_file = '../sam_files/%s-%s.sam' % (prefix, db_name)
  index_prefix = '../bt2_indices/%s' % db_name

  # Bowtie2 command options:
  # -a : show all alignments, not just best one
  # -t : output wall time for each step
  # -p 3 : use 3 cores
  # -f : reads are FASTA format
  cmd = '%s/bowtie2 --end-to-end --very-fast -a -f -t -p 3 -x %s -U %s -S %s' % (
     bowtie_dir, index_prefix, read_fasta_file, sam_file)
  print >> sys.stderr, 'Running %s' % cmd
  status = os.system(cmd)
  if status != 0:
    raise Exception('%s failed with status %d' % (cmd, status))
  return sam_file


################################################################################
class CatEntry:
  """
  Represents catalog info for a reference sequence
  """
  def __init__(self, gi, acc, offset, size, desc):
    self.gi = gi
    self.acc = acc
    self.offset = offset
    self.size = size
    self.desc = desc


################################################################################
class RefDb:
  """
  Stores catalog entries and file handles for a reference sequence database
  """
  def __init__(self, db_name):
    self.db_name = db_name
    self.catalog = {}
    self.seq_in = None

  def load(self):
    """
    Load the catalog data for this reference DB
    """
    seq_file = '../ref_dbs/RefSeq/%s_seq.dat' % self.db_name
    self.seq_in = open(seq_file, 'r')
    cat_file = '../ref_dbs/RefSeq/%s_catalog.txt' % self.db_name
    cat_in = open(cat_file, 'r')
    self.catalog = {}
    for line in cat_in:
      (gi, acc, offset_s, size_s, desc) = line.rstrip('\n').split('\t')
      offset = int(offset_s)
      size = int(size_s)
      cat_entry = CatEntry(gi, acc, offset, size, desc)
      self.catalog[gi] = cat_entry
    cat_in.close()

  def retrieve(self, gi):
    """
    Retrieve the sequence data for the given gi number
    """
    cat_entry = self.catalog[gi]
    self.seq_in.seek(cat_entry.offset)
    seq_data = self.seq_in.read(cat_entry.size)
    return seq_data

  def get_length(self, gi):
    """
    Return the target sequence length for the given gi number
    """
    return self.catalog[gi].size

################################################################################
def test_retrieve(refdb, gi):
  """
  Retrieve the sequence data for the given GI, and show the first and last
  bytes of it
  """
  seq = refdb.retrieve(gi)
  print >> sys.stderr, '>gi|%s\n%s .... %s' % (gi, seq[0:20], seq[-20:])
