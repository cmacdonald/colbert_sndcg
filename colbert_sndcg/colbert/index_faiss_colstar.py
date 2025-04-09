import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import faiss
import pyterrier as pt
pt.init()
import pyterrier_colbert.indexing
from pyterrier_colbert.indexing import ColBERTIndexer
import colbert.indexing.faiss
colbert.indexing.faiss.SPAN = 1
index_root="datasets/faiss_index"
index_name="msmarco_passage_index_colstar"
dataset = pt.get_dataset("irds:msmarco-passage")
indexer = ColBERTIndexer("experiments/dirty/train.py/2024-06-25_17.33.15/checkpoints/colbert-100000.dnn", index_root,  index_name, chunksize=3)
indexer.index(dataset.get_corpus_iter())