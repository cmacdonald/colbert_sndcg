import numpy as np
import json
import sys
import os

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyterrier as pt
#pt.init()

from smp import get_smp_df

def report_scdg_ldcg(qrels, cutoff, model, dataset='2020',index_root = None, index_name = None,add_subword=False, add_stop=False, add_numeric=False,
                                   add_low=False, add_med=False, add_high=False, add_all=True, add_stem=False,
                                   add_question=False, **kwargs):
    # Note: as a by-product, this function also computes the nDCG (in smp_df['ndcg'])

    # get dataframe reporting semantic_pct and exact_pct
    # (equivalent to semantic_pct in Equation 3.5 and lexical_pct in Equation 3.7)
    smp_df = get_smp_df(cutoff, model, index_root, index_name,dataset_name=dataset,add_all=add_all, add_subword=add_subword, add_stop=add_stop, add_numeric=add_numeric,
                          add_low=add_low, add_med=add_med, add_high=add_high, add_stem=add_stem, add_question=add_question, **kwargs) #this gives semantic_pct like in equation 3.5

    # filter out documents that do not appear in the qrels as we require a
    # relevance judgement to compute the DCG
    # documents without a relevance judgement are assumed to have label 0 and thus
    # also a DCG of 0
    smp_df = smp_df.merge(qrels, on=['qid','docno'])

    # compute the DCG contribution for each document (i.e. (rel(r)/(log_2(r+1)) in Equation 3.4)
    # we use +2 instead of +1 as the ranks start at 0
    smp_df['dcg'] = smp_df['label']/(np.log2((smp_df['rank']+1)+1))

    #compute idcg for each query
    smp_df['idcg'] = [0 for i in range(len(smp_df))]#0 for error detection
    for qid in smp_df['qid'].unique():
        #this computes the iDCG@k in Equation 3.4
        idcg = get_idcg(qrels,qid,cutoff)
        #set idcg for each document in the query's ranking
        smp_df.loc[smp_df['qid']==qid,'idcg'] = idcg

    # compute each document's nDCG contribution
    # this value is NOT used in S-nDCG/L-nDCG as the contributions for those
    # metrics are normalised with the iDCG in lines 50 and 54 respectively
    smp_df['ndcg'] = smp_df['dcg']/smp_df['idcg']

    # compute each document's S-DCG and L-DCG (i.e. non-normalised S-nDCG and L-nDCG) contribution
    # for each document d_r this computes (semantic_pct(q,d_r)*(rel(r)/log_2(r+1))) in Equation 3.4
    # i.e. each document's S-DCG contribution
    smp_df['sdcg'] = smp_df['dcg']*smp_df['semantic_pct']

    #for each document d_r this computes (semantic_pct(q,d_r)*(rel(r)/log_2(r+1))) in Equation 3.6
    # i.e. each document's L-DCG contribution
    smp_df['ldcg'] = smp_df['dcg']*smp_df['exact_pct']

    #normalise with idcg to obtain the S-nDCG and L-nDCG contributions
    # this computes (semantic_pct(q,d_r)*(1/iDCG@k)*(rel(r)/log_2(r+1))) in Equation 3.4
    smp_df['nsdcg'] = smp_df['sdcg'] / smp_df['idcg']

    # this computes (lexical_pct(q,d_r)*(1/iDCG@k)*(rel(r)/log_2(r+1))) in Equation 3.6
    smp_df['nldcg'] = smp_df['ldcg'] / smp_df['idcg']

    print(smp_df)

    measures = {}
    # this computes the sum over the documents d_r in R_k from equation 3.4
    # we sum and don't average (i.e. no factor 1/k) because the sdcg is the ndcg contribution multiplied with the semantic matching percentage,
    # i.e. the portion of this ndcg contribution that can be attributed to semantic matching (nsdcg+nldcg=ndcg for each document)
    # to form the ndcg, the contributions have to be summed not averaged, which is why we sum and not average here
    grouped_df=smp_df.groupby('qid').sum(numeric_only=True)

    #average S-nDCG and L-nDCG (and nDCG) across all queries
    measures['ndcg'] = grouped_df['ndcg'].mean()
    measures['nsdcg'] = grouped_df['nsdcg'].mean() #this averages the S-nDCG across the different queries
    measures['nldcg'] = grouped_df['nldcg'].mean()

    return smp_df, measures

def get_idcg(qrels,qid,cutoff):
    # computes the iDCG (DCG of ideal ranking) at cutoff for query with the query-id qid
    # filter relevance judgements for the given quer
    df = qrels[qrels['qid']==qid]

    # to simulate an ideal ranking sort qrels by relevance label descending
    df = df.sort_values(by='label',axis='index',ascending=False)
    idcg=0
    for i,item in enumerate(df['label'].items()):
        # i denotes the rank
        # item[1] gives the relevance label
        # use i+2 as ranks start at 0
        idcg += item[1] / (np.log2((i + 1) + 1))
        # apply cutoff (again ranks start at 0 so we use cutoff-1 instead of cutoff)
        if i>=cutoff-1:
            break
    print(idcg)
    return idcg

def sdcg_throughout_training(dataset='2020-short', model='colbert',**kwargs):
    """
    Measures the model's S-nDCG on all token types (including all) throughout training
    Make sure to set the appropriate kwargs (e.g. aggregation, replication, linear_layer, dim) when
    applying with other models.
    """
    print(dataset)
    if dataset in ['2020','2020-short']:
        qrels = pt.get_dataset("trec-deep-learning-passages").get_qrels('test-2020')
    else:
        qrels = pt.get_dataset("trec-deep-learning-passages").get_qrels('test-2019')


    for token_type in ['all','subword', 'stopword', 'numeric', 'stem', 'question','low', 'med', 'high']:
        results = {}
        for steps in ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k',
                      '100k', '150k', '200k']:
            if token_type == 'low':
                print('low')
                sdcg, measures = report_scdg_ldcg(qrels, 10, model + '_' + steps, dataset, add_all=False,
                                                  add_low=True,**kwargs)
            elif token_type == 'med':
                print('med')
                sdcg, measures = report_scdg_ldcg(qrels, 10, model + '_' + steps, dataset, add_all=False,
                                                  add_med=True,**kwargs)
            elif token_type == 'high':
                print('high')
                sdcg, measures = report_scdg_ldcg(qrels, 10, model + '_' + steps, dataset, add_all=False,
                                                  add_high=True,**kwargs)
            elif token_type == 'subword':
                print('subword')
                sdcg, measures = report_scdg_ldcg(qrels, 10, model + '_' + steps, dataset, add_all=False,
                                                  add_subword=True,**kwargs)
            elif token_type == 'stopword':
                print('stopword')
                sdcg, measures = report_scdg_ldcg(qrels, 10, model + '_' + steps, dataset, add_all=False,
                                                  add_stop=True,**kwargs)
            elif token_type == 'numeric':
                print('numeric')
                sdcg, measures = report_scdg_ldcg(qrels, 10, model + '_' + steps, dataset, add_all=False,
                                                  add_numeric=True,**kwargs)
            elif token_type == 'stem':
                print('stem')
                sdcg, measures = report_scdg_ldcg(qrels, 10, model + '_' + steps, dataset, add_all=False,
                                                  add_stem=True,**kwargs)
            elif token_type == 'question':
                print('question')
                sdcg, measures = report_scdg_ldcg(qrels, 10, model + '_' + steps, dataset, add_all=False,
                                                  add_question=True,**kwargs)
            else:
                print('all')
                sdcg, measures = report_scdg_ldcg(qrels, 10, model + '_' + steps, dataset,**kwargs)
            results[steps] = measures
            sdcg.to_csv('evaluation/sdcg/' + model + '_' + steps + '_' + token_type + '_sdcg_ldcg_'+dataset.replace('-','_')+'.csv')
            print(sdcg)
            print(measures)
        with open('evaluation/sdcg/' + model + '_' + token_type + '_results_'+dataset.replace('-','_')+'.json', 'w') as f:
            f.write(json.dumps(results))


if __name__=='__main__':
    sdcg_throughout_training('2020-short','colbert')
    sdcg_throughout_training('2019-short', 'colbert')
