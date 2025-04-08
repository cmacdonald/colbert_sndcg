import os
import sys
from checkpoints import CHECKPOINTS
print(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyterrier as pt
pt.init()
import pyterrier_colbert.indexing
from pyterrier_colbert.indexing import ColBERTIndexer
import colbert.indexing.faiss

#code for FAISS indexing adapted from https://github.com/Xiao0728/ColStar_VirtualAppendix/blob/main/ColStar_models/index.md

def make_faiss_index(checkpoint, index_root, index_name):
    colbert.indexing.faiss.SPAN = 1
    dataset = pt.get_dataset("irds:msmarco-passage")
    indexer = ColBERTIndexer(checkpoint,index_root, index_name, chunksize=3)
    indexer.index(dataset.get_corpus_iter())

if __name__=='__main__':
    #it does not make a difference for our purposes, which model the index is based on
    make_faiss_index(CHECKPOINTS['colbert_150k'], "/root/nfs/colbert/evaluation/faiss_index/colbert",
                     "msmarco_passage_index_colbert_150k")
