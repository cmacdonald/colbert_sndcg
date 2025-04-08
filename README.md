# Semantically Proportioned nDCG for Explaining ColBERT's Learning Process

This repository provides functionality to measure the Semantic Matching Proportion (SMP) [1] of a ColBERT-style model 
as well as the semantically proportioned effectiveness of the model (S-nDCG) on all tokens or a specific token type, throughout fine-tuning.

## Installation

Clone this repository
```bash
git clone https://github.com/cmacdonald/colbert_sndcg
```
*or*\
install via pip
```bash
pip install git+https://github.com/cmacdonald/colbert_sndcg
```

## Usage

Given an existing ColBERT checkpoint and a corresponding faiss index, the S-nDCG of the model can be measured as follows:
```python
import pyterrier as pt
if not pt.started():
    pt.init()
from evaluation.sdcg import report_scdg_ldcg

model = 'https://some-checkpoint-name.dnn'

#evaluation on TREC DL 2020
qrels = pt.get_dataset("trec-deep-learning-passages").get_qrels('test-2020')
report_scdg_ldcg(qrels, cutoff, model, dataset='2020',index_root = None, index_name = None, add_all=True)
```

**Note:** For ease of use we provide several different pre-trained checkpoints from different stages of training (after 0, 1k, 2k, 5k, 10k, 20k, 25k, 30k, 40k, 50k, 60k, 70k, 75k, 80k, 90k, 100k, 150k and 200k training steps).
Please note that these checkpoints do NOT include the linear layer (*Lin*) like the original model, as further explained in our paper. Functionality to evaluate your own model (throughout training) will be added shortly.

The pre-trained models (e.g. to reproduce results from the paper) can be evaluated as follows:
```python
from evaluation.sdcg import sdcg_throughout_training

model = 'll_none' #pre-trained ColBERT checkpoints WITHOUT linear layer
dataset = '2020-short' #a shorter (but equivalent) version of TREC DL 2020
sdcg_throughout_training(dataset,model,linear_layer='none',dim=768) #the kwargs linear_layer and dim are necessary here to account for the missing linear layer
```

The notebook SnDCG_Demo.ipynb demonstrates the creation of the corresponding S-nDCG plots as well as the usage of the other metrics including token type specific S-nDCG and SMP throughout training.
**Note:** Due to technical difficulties the pre-trained checkpoints are not uploaded yet. Hence, some parts of the demo notebook will not work yet.

## References
- [1] Wang, X., Macdonald, C., Tonellotto, N., & Ounis, I. (2023, July). Reproducibility, replicability, and insights into dense multi-representation retrieval models: from colbert to col. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 2552-2561).

## Citation
```bibtex
@InProceedings{10.1007/978-3-031-88708-6_22,
author="Mueller, Ariane
and Macdonald, Craig",
title="Semantically Proportioned nDCG for Explaining ColBERT's Learning Process",
booktitle="Advances in Information Retrieval",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="341--356",
isbn="978-3-031-88708-6"
}
```