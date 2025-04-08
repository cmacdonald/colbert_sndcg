import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from checkpoints import CHECKPOINTS

import warnings
import pyterrier as pt
pt.init()

from pyterrier_colbert.ranking import ColBERTModelOnlyFactory

from pyterrier.measures import *

# user-specified TOST
# TOST will omit warnings here, due to low numbers of topics
import statsmodels.stats.weightstats
fn = lambda X ,Y: (0, statsmodels.stats.weightstats.ttost_paired(X, Y, -0.05, 0.05)[0])

dataset = 'test-2019'
#dataset = 'test-2020'

official_bm25 = pt.BatchRetrieve.from_dataset(
            'msmarco_passage',
            'terrier_stemmed_text',
            wmodel='BM25',
            metadata=['docno', 'text'],
            num_results=1000)

for steps in ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k', '150k', '200k']:
#for steps in ['40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
#                  '150k', '200k']:
    colbert = official_bm25 >> ColBERTModelOnlyFactory(CHECKPOINTS['colbert_'+steps]).text_scorer()
    ll_none = official_bm25 >> ColBERTModelOnlyFactory(CHECKPOINTS['ll_none_'+steps],dim=768, linear_layer='none').text_scorer()
    ll_static_mapping = official_bm25 >> ColBERTModelOnlyFactory(CHECKPOINTS['ll_static_mapping_'+steps],linear_layer='static_mapping').text_scorer()
    ll_trained_mapping = official_bm25 >> ColBERTModelOnlyFactory(CHECKPOINTS['ll_trained_mapping_'+steps],linear_layer='trained_mapping').text_scorer()
    ll_static_random = official_bm25 >> ColBERTModelOnlyFactory(CHECKPOINTS['ll_static_random_'+steps], linear_layer='static_random').text_scorer()

    # This filter doesnt work
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings("always")
        df = pt.Experiment(
            [colbert, ll_none, ll_static_mapping, ll_trained_mapping, ll_static_random],
            pt.get_dataset("trec-deep-learning-passages").get_topics(dataset),
            pt.get_dataset("trec-deep-learning-passages").get_qrels(dataset),
            batch_size=100,
            verbose=True,
            eval_metrics=[RR@10, nDCG@10, R@50, R@200],
            test=fn,
            baseline=0,
            names=['colbert', 'll_none', 'll_static_mapping', 'll_trained_mapping', 'll_static_random'])

    with open('evaluation/tost_equivalence/equivalence_'+steps+'_'+dataset+'_new.csv','w') as f:
        df.to_csv(f)
    print(df)