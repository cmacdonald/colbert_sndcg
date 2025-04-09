import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import pearsonr
import numpy as np


def plot_smp_ndcg(model, token_type, dataset):
    """
    Plots a scatter plot with the SMP and nDCG per query for different model checkpoints
    """
    ndcg = {}
    smp = {}
    """for steps in ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k',
                          '100k', '150k', '200k']:"""
    for steps in ['0k','50k','100k','150k','200k']:

        if steps=='0k':
            ndcg_file = 'evaluation/sdcg/' + model + '_' + steps + '_' + token_type + '_sdcg_ldcg_'+dataset.replace('-','_')+'_3090.csv'
        else:
            ndcg_file = 'evaluation/sdcg/' + model + '_' + steps + 'sdcg_ldcg_short_' + dataset[:4]+ '_3090.csv'
        df_ndcg = pd.read_csv(ndcg_file)
        grouped_df_ndcg = df_ndcg.groupby('qid').sum(numeric_only=True)
        #print(len(grouped_df_ndcg))
        ndcg_ser = grouped_df_ndcg['ndcg']

        smp_file = 'evaluation/smp/' + model + '_' + steps + 'smp_'+dataset.replace('-','_')+'.csv'
        df_smp = pd.read_csv(smp_file)
        grouped_df_smp = df_smp.groupby('qid').mean(numeric_only=True)
        smp_ser = grouped_df_smp['semantic_pct']

        #for the 0k models it can be the case that not all qids are included because of how we save the csv
        #(we only save the docid-qid pairs that appear in the qrels, so if the top 10 do not contain any documents
        #in the qrels, the qid will not be included). For those qids we set the ndcg to 0
        if len(smp_ser)>len(ndcg_ser):
            for qid in smp_ser.index:
                if qid not in ndcg_ser.index:
                    ndcg_ser[qid] = 0
        smp_ser = smp_ser.sort_index()
        ndcg_ser = ndcg_ser.sort_index()

        ndcg[steps] = list(ndcg_ser)
        smp[steps] = list(smp_ser)

    fig, ax = plt.subplots()

    if dataset in ['2019','2019-short']:
        name = 'TREC 2019'
    else:
        name = 'TREC 2020'
    ax.set_title('SMP to nDCG per query for '+model+' throughout training for '+name)
    ax.set_xlabel('SMP@10')
    ax.set_ylabel('nDCG@10')
    """for steps in ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k',
                          '100k', '150k', '200k']:"""
    for steps in ['0k','50k','100k','150k','200k']:
        ax.scatter(smp[steps],ndcg[steps],label=steps, alpha=0.5)
        ax.legend()
    plt.savefig('evaluation/rq2_2_graphs/'+model+'_'+'smp_ndcg_graph'+dataset.replace('-','_')+'.png')

def plot_smp_ndcg_graph(model, token_type, dataset):
    """
    Produces the graph relating SMP to nDCG at different points in training, shown in Figure 4.6
    Also computes the Pearson correlation between SMP and nDCG
    """
    points = {}
    for steps in ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k',
                          '100k', '150k', '200k']:

        if steps=='0k':
            ndcg_file = 'evaluation/sdcg/' + model + '_' + steps + '_' + token_type + '_sdcg_ldcg_'+dataset.replace('-','_')+'_3090.csv'
        else:
            ndcg_file = 'evaluation/sdcg/' + model + '_' + steps + 'sdcg_ldcg_short_' + dataset[:4]+ '_3090.csv'
        df_ndcg = pd.read_csv(ndcg_file)
        grouped_df_ndcg = df_ndcg.groupby('qid').sum(numeric_only=True)
        ndcg_ser = grouped_df_ndcg['ndcg']

        smp_file = 'evaluation/smp/' + model + '_' + steps + 'smp_'+dataset.replace('-','_')+'.csv'
        df_smp = pd.read_csv(smp_file)
        grouped_df_smp = df_smp.groupby('qid').mean(numeric_only=True)
        smp_ser = grouped_df_smp['semantic_pct']

        #for the 0k models it can be the case that not all qids are included because of how we save the csv
        #(we only save the docid-qid pairs that appear in the qrels, so if the top 10 do not contain any documents
        #in the qrels, the qid will not be included). For those qids we set the ndcg to 0
        if len(smp_ser)>len(ndcg_ser):
            for qid in smp_ser.index:
                if qid not in ndcg_ser.index:
                    ndcg_ser[qid] = 0
        smp_ser = smp_ser.sort_index()
        ndcg_ser = ndcg_ser.sort_index()

        print(len(ndcg_ser))
        print(len(smp_ser))

        points[steps] = (sum(list(ndcg_ser))/len(list(ndcg_ser)),sum(list(smp_ser))/(len(list(smp_ser))))


    df = pd.DataFrame(points, index=['ndcg', 'smp'])

    df = df.transpose()

    x = list(df['smp'])
    y = list(df['ndcg'])
    t = list(df.index)

    if dataset in ['2019','2019-short']:
        name = 'TREC 2019'
    else:
        name = 'TREC 2020'

    fig, ax = plt.subplots()
    ax.set_title('SMP to nDCG for '+model+' throughout training for '+name)
    ax.set_xlabel('SMP@10')
    ax.set_ylabel('nDCG@10')
    ax.scatter(x, y)
    ax.plot(x,y)
    #ax.scatter(x,y)
    for s in ['0k','10k','50k','100k','200k']:
        ax.scatter(df.loc[s,'smp'], df.loc[s,'ndcg'], label=s)

    ax.legend()

    ax.legend()
    plt.savefig('evaluation/rq2_2_graphs/'+model+'_'+'smp_ndcg_line_graph_'+dataset.replace('-','_')+'_with_0.png')

    print(x,y)

    print(pearsonr(x,y))
    print(np.corrcoef(x,y))

if __name__=='__main__':
    model = 'colbert'
    token_type = 'all'
    plot_smp_ndcg(model, token_type, '2019-short')
    plot_smp_ndcg(model, token_type, '2020-short')
    plot_smp_ndcg_graph(model, token_type, '2019-short')
    plot_smp_ndcg_graph(model, token_type, '2020-short')