import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

import pandas as pd
import numpy as np
import json

"""
Tests the significance of the difference in measure (e.g. S-nDCG) between the trained and the untrained model
"""

def significance_test(token_type, steps, dataset, value='nsdcg', correction=None):
    df0 = pd.read_csv('evaluation/sdcg/colbert_0k_' + token_type + '_sdcg_ldcg_'+dataset+'_short.csv')
    df_trained = pd.read_csv('evaluation/sdcg/colbert_'+steps+'_'+token_type+'_sdcg_ldcg_'+dataset+'_short.csv')

    #for evaluating ll_none against ColBERT at 200k steps
    #df0 = pd.read_csv('evaluation/sdcg/ll_none_200k_' + token_type + '_sdcg_ldcg_'+dataset+'_short.csv')


    grouped_df0 = df0.groupby('qid').sum(numeric_only=True)
    grouped_df_trained = df_trained.groupby('qid').sum(numeric_only=True)

    if len(grouped_df0)<len(grouped_df_trained):
        for qid in grouped_df_trained.index:
            if qid not in grouped_df0.index:
                grouped_df0.loc[qid] = [0 for i in range(len(grouped_df0.columns))]
    elif len(grouped_df_trained)<len(grouped_df0):
        for qid in grouped_df0.index:
            if qid not in grouped_df_trained.index:
                grouped_df_trained.loc[qid] = [0 for i in range(len(grouped_df_trained.columns))]

    grouped_df0 = grouped_df0.sort_index()
    grouped_df_trained = grouped_df_trained.sort_index()

    result = ttest_rel(np.array(grouped_df0[value]), np.array(grouped_df_trained[value]))

    print('p-value:',result[1])

    if correction:
        corrected_result = multipletests(result[0],method=correction)
        print(corrected_result)
        return result, corrected_result

    return result


if __name__=='__main__':
    #For evaluating untrained vs fully trained ColBERT model
    steps = '200k'
    for dataset in ['2019','2020']:
        p_values = {}
        for token_type in ['subword','stopword', 'numeric', 'stem', 'question', 'low', 'med', 'high']:
            print(token_type)
            result = significance_test(token_type,steps,dataset)
            p_values[token_type] = result[1]

        with open('evaluation/significance/'+steps+'_results_'+dataset+'_short.json', 'w') as f:
            f.write(json.dumps(p_values))

    #for evaluating ll_none vs ColBERT (200k steps)
    p_values = {}
    token_type='all'
    for dataset in ['2019','2020']:
        print(dataset)
        result = significance_test(token_type,steps,dataset, value='ndcg')
        print(result)
        p_values[dataset] = result

    with open('evaluation/significance/colbert_ll_none' + steps + '_results_' + dataset + '_short.json', 'w') as f:
        f.write(json.dumps(p_values))