import os
import sys
print(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#os.chdir("/root/nfs/colbert")
#print(os.getcwd())

from checkpoints import CHECKPOINTS
from typing import Union
import pyterrier as pt
pt.init()

import matplotlib.pyplot as plt

from pyterrier.measures import *
from pyterrier_colbert.ranking import ColBERTModelOnlyFactory
import pandas as pd

#Partly adapted from https://github.com/Xiao0728/ColStar_VirtualAppendix/blob/main/Reproducibility%20(RQ1%20Res)/Reproducibility_Demo%20(RQ1%20results).ipynb

def evaluate_reranking_effectiveness(dataset:str, models: Union[str,list], checkpoints: Union[str,list], save_path=None, baseline = None, per_query=False,ndcg_only=False):
    """
    datasets: dataset to use for evaluation (options: dev.small, test-2019, test-2020)
    models: the names of the models to be evaluated (options: colbert, ag_first, ag_avg, ag_last,
        ll_none, ll_static_mapping, ll_trained_mapping, ll_static_random
    checkpoints: the checkpoints the models are evaluated on - all models are evaluated on all checkpoints
        (specified by the number of steps after which the checkpoint was created)
        (options: 1k, 2k, 5k, 10k, 20k, 25k, 30k, 40k, 44k, 50k, 60k, 70k, 75k, 80k, 90k, 100k, 150k, 200k)
    """
    if dataset=='dev.small':
        official_bm25 = pd.read_csv(
            "/root/nfs/ColStar_VirtualAppendix/Reproducibility (RQ1 Res)/MSMARCO.Dev.Small/top1000.dev.tar.gz",
            sep="\t", names=['qid', 'docno', 'query', 'text'])
        official_bm25.qid = official_bm25.qid.astype(str)
        official_bm25.docno = official_bm25.docno.astype(str)
    else:
        print("TREC dataset")
        #TREC 2019 or 2020 dataset
        official_bm25 = pt.BatchRetrieve.from_dataset(
            'msmarco_passage',
            'terrier_stemmed_text',
            wmodel='BM25',
            metadata=['docno', 'text'],
            num_results=1000)

    if models.__class__==str:
        models = [models]

    rerankers = []
    names = []
    for model in models:
        for checkpoint in checkpoints:
            if model=='ll_none':
                factory = ColBERTModelOnlyFactory(CHECKPOINTS[model + '_' + checkpoint],dim=768, linear_layer='none')
            elif model=='ll_static_mapping':
                factory = ColBERTModelOnlyFactory(CHECKPOINTS[model + '_' + checkpoint], linear_layer='static_mapping')
            elif model=='ll_trained_mapping':
                factory = ColBERTModelOnlyFactory(CHECKPOINTS[model + '_' + checkpoint], linear_layer='trained_mapping')
            elif model=='ll_static_random':
                factory = ColBERTModelOnlyFactory(CHECKPOINTS[model + '_' + checkpoint], linear_layer='static_random')
            elif model=='ag_first':
                factory = ColBERTModelOnlyFactory(CHECKPOINTS[model + '_' + checkpoint], aggregation='first')
            elif model == 'ag_avg':
                factory = ColBERTModelOnlyFactory(CHECKPOINTS[model + '_' + checkpoint], aggregation='avg')
            elif model=='ag_last':
                factory = ColBERTModelOnlyFactory(CHECKPOINTS[model + '_' + checkpoint], aggregation='last')
            elif model=='ag_avg_pad':
                factory = ColBERTModelOnlyFactory(CHECKPOINTS[model + '_' + checkpoint], aggregation='avg', replication='pad')
            else:
                factory = ColBERTModelOnlyFactory(CHECKPOINTS[model+'_'+checkpoint])
            if dataset=='dev.small':
                rerankers.append(pt.transformer.SourceTransformer(official_bm25)>>factory.text_scorer())
            else:
                rerankers.append(official_bm25 >>factory.text_scorer())

            names.append(model + '_' + checkpoint)

    if ndcg_only:
        df = pt.Experiment(rerankers,
        pt.get_dataset("trec-deep-learning-passages").get_topics(dataset),
        pt.get_dataset("trec-deep-learning-passages").get_qrels(dataset),
        batch_size=100,
        verbose=True,
        save_dir = "./",
        filter_by_qrels=True,
        baseline=baseline,
        eval_metrics=[nDCG@10],
        names=names,
        perquery=per_query
        )
    else:

        df = pt.Experiment(rerankers,
        pt.get_dataset("trec-deep-learning-passages").get_topics(dataset),
        pt.get_dataset("trec-deep-learning-passages").get_qrels(dataset),
        batch_size=100,
        verbose=True,
        save_dir = "./",
        filter_by_qrels=True,
        baseline=baseline,
        eval_metrics=[RR@10, nDCG@10, R@50, R@200, R@1000],
        names=names,
        perquery=per_query
        )

    if save_path:
        df.to_csv(save_path)

    return df

def effectiveness_throughout_training(dataset, models, save_path=None):
    df = evaluate_reranking_effectiveness(dataset, models, ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k', '150k', '200k'],save_path)
    print(df)
    return df, save_path

def effectiveness_44k(dataset, models, save_path=None, baseline=None):
    df = evaluate_reranking_effectiveness(dataset, models,['44k'], save_path, baseline)
    print(df)
    return df, save_path

def effectiveness_200k(dataset, models, save_path=None, baseline=None):
    df = evaluate_reranking_effectiveness(dataset, models,['200k'], save_path, baseline)
    print(df)
    return df, save_path

def make_graph(dataframes: Union[list[str],list[pd.DataFrame]], metric, model_names, checkpoints, title, save_path=None):
    assert len(dataframes)>0

    for i,df in enumerate(dataframes):
        if df.__class__==str:
            #remove 0
            dataframes[i]=pd.read_csv(df)[1:]

    combined_df = pd.concat([df[metric] for df in dataframes], axis=1)
    combined_df.columns = model_names
    combined_df.index = [int(checkpoint[:-1])*1000 for checkpoint in checkpoints]
    ax = combined_df.plot(title=title)

    """
    #for marking the 1k point for linear layer plots
    colours=['blue','orange','green','red','purple']
    for df,c in zip(dataframes,colours):
        first_df = pd.DataFrame(df.iloc[0]).transpose()
        first_df['Training Steps'] = [1000]
        #print(first_df)
        ax = first_df.plot.scatter(x='Training Steps', y=metric, marker='o', ax=ax, c=c)"""

    ax.set_xlabel('Training steps')
    ax.set_ylabel(metric)

    if save_path:
        plt.savefig(save_path)

    #graph.show()

    return ax

if __name__=='__main__':
    #Please only run either TREC 2019 or TREC 2020 at a time as the rankings will be saved and then
    #used for the other dataset leading to incorrect evaluation

    effectiveness_throughout_training('test-2019',['colbert'],"/root/nfs/colbert/evaluation/results/2019_colbert_df_new.csv")
    effectiveness_throughout_training('test-2019', ['ag_first'], "/root/nfs/colbert/evaluation/results/2019_ag_first_df.csv")
    effectiveness_throughout_training('test-2019', ['ag_avg'], "/root/nfs/colbert/evaluation/results/2019_ag_avg_df.csv")
    effectiveness_throughout_training('test-2019', ['ag_last'], "/root/nfs/colbert/evaluation/results/2019_ag_last_df.csv")
    effectiveness_throughout_training('test-2019', ['ll_none'], "/root/nfs/colbert/evaluation/results/2019_ll_none_df_new.csv")
    effectiveness_throughout_training('test-2019', ['ll_static_mapping'], "/root/nfs/colbert/evaluation/results/2019_ll_static_mapping_df.csv")
    effectiveness_throughout_training('test-2019', ['ll_trained_mapping'], "/root/nfs/colbert/evaluation/results/2019_ll_trained_mapping_df.csv")
    effectiveness_throughout_training('test-2019', ['ll_static_random'], "/root/nfs/colbert/evaluation/results/2019_ll_static_random_df.csv")
    effectiveness_throughout_training('test-2019', ['ag_avg_pad'],"/root/nfs/colbert/evaluation/results/2019_ag_avg_pad_df.csv")

    #aggregation
    make_graph(["/root/nfs/colbert/evaluation/results/2019_colbert_df.csv","/root/nfs/colbert/evaluation/results/2019_ag_first_df.csv","/root/nfs/colbert/evaluation/results/2019_ag_avg_df.csv","/root/nfs/colbert/evaluation/results/2019_ag_last_df.csv","/root/nfs/colbert/evaluation/results/2019_ag_avg_pad_df.csv"],
               'RR@10',
               ['colbert','first','avg','last', 'avg pad'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k', '150k', '200k'],
               'Effectiveness throughout training for subword aggregation, RR@10',
               "/root/nfs/colbert/evaluation/results/graph_rr10_2019_aggregation_throughout_training.png")

    make_graph(["/root/nfs/colbert/evaluation/results/2019_colbert_df.csv", "/root/nfs/colbert/evaluation/results/2019_ag_first_df.csv",
                "/root/nfs/colbert/evaluation/results/2019_ag_avg_df.csv", "/root/nfs/colbert/evaluation/results/2019_ag_last_df.csv",
                "/root/nfs/colbert/evaluation/results/2019_ag_avg_pad_df.csv"],
               'nDCG@10',
               ['colbert', 'first', 'avg', 'last', 'avg pad'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for subword aggregation, nDCG@10',
               "/root/nfs/colbert/evaluation/results/graph_ndcg10_2019_aggregation_throughout_training.png")

    make_graph(["/root/nfs/colbert/evaluation/results/2019_colbert_df.csv", "/root/nfs/colbert/evaluation/results/2019_ag_first_df.csv",
                "/root/nfs/colbert/evaluation/results/2019_ag_avg_df.csv", "/root/nfs/colbert/evaluation/results/2019_ag_last_df.csv",
                "/root/nfs/colbert/evaluation/results/2019_ag_avg_pad_df.csv"],
               'R@50',
               ['colbert', 'first', 'avg', 'last', 'avg pad'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for subword aggregation, R@50',
               "/root/nfs/colbert/evaluation/results/graph_r50_2019_aggregation_throughout_training.png")
    make_graph(["/root/nfs/colbert/evaluation/results/2019_colbert_df.csv", "/root/nfs/colbert/evaluation/results/2019_ag_first_df.csv",
                "/root/nfs/colbert/evaluation/results/2019_ag_avg_df.csv", "/root/nfs/colbert/evaluation/results/2019_ag_last_df.csv",
                "/root/nfs/colbert/evaluation/results/2019_ag_avg_pad_df.csv"],
               'R@200',
               ['colbert', 'first', 'avg', 'last', 'avg pad'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for subword aggregation, R@200',
               "/root/nfs/colbert/evaluation/results/graph_r200_2019_aggregation_throughout_training.png")

    make_graph(["/root/nfs/colbert/evaluation/results/2019_colbert_df.csv", "/root/nfs/colbert/evaluation/results/2019_ag_first_df.csv",
                "/root/nfs/colbert/evaluation/results/2019_ag_avg_df.csv", "/root/nfs/colbert/evaluation/results/2019_ag_last_df.csv",
                "/root/nfs/colbert/evaluation/results/2019_ag_avg_pad_df.csv"],
               'R@1000',
               ['colbert', 'first', 'avg', 'last', 'avg pad'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for subword aggregation, R@1000',
               "/root/nfs/colbert/evaluation/results/graph_r1000_2019_aggregation_throughout_training.png")
    
    #linear_layer
    make_graph(["/root/nfs/colbert/evaluation/results/2019_colbert_df.csv", "/root/nfs/colbert/evaluation/results/2019_ll_none_df.csv",
                "/root/nfs/colbert/evaluation/results/2019_ll_static_mapping_df.csv", "/root/nfs/colbert/evaluation/results/2019_ll_trained_mapping_df.csv", "/root/nfs/colbert/evaluation/results/2019_ll_static_random_df.csv"],
               'RR@10',
               ['colbert', 'none', 'static mapping', 'trained mapping', 'static random'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for different linear layers, RR@10',
               "/root/nfs/colbert/evaluation/results/graph_rr10_2019_linear_layer_throughout_training.png")

    make_graph(["/root/nfs/colbert/evaluation/results/2019_colbert_df.csv", "/root/nfs/colbert/evaluation/results/2019_ll_none_df.csv",
                "/root/nfs/colbert/evaluation/results/2019_ll_static_mapping_df.csv", "/root/nfs/colbert/evaluation/results/2019_ll_trained_mapping_df.csv", "/root/nfs/colbert/evaluation/results/2019_ll_static_random_df.csv"],
               'nDCG@10',
               ['colbert', 'none', 'static mapping', 'trained mapping', 'static random'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for different linear layers, nDCG@10',
               "/root/nfs/colbert/evaluation/results/graph_ndcg10_2019_linear_layer_throughout_training.png")

    make_graph(["/root/nfs/colbert/evaluation/results/2019_colbert_df.csv", "/root/nfs/colbert/evaluation/results/2019_ll_none_df.csv",
                "/root/nfs/colbert/evaluation/results/2019_ll_static_mapping_df.csv", "/root/nfs/colbert/evaluation/results/2019_ll_trained_mapping_df.csv", "/root/nfs/colbert/evaluation/results/2019_ll_static_random_df.csv"],
               'R@50',
               ['colbert', 'none', 'static mapping', 'trained mapping', 'static random'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for different linear layers, R@50',
               "/root/nfs/colbert/evaluation/results/graph_r50_2019_linear_layer_throughout_training.png")

    make_graph(["/root/nfs/colbert/evaluation/results/2019_colbert_df.csv", "/root/nfs/colbert/evaluation/results/2019_ll_none_df.csv",
                "/root/nfs/colbert/evaluation/results/2019_ll_static_mapping_df.csv", "/root/nfs/colbert/evaluation/results/2019_ll_trained_mapping_df.csv", "/root/nfs/colbert/evaluation/results/2019_ll_static_random_df.csv"],
               'R@200',
               ['colbert', 'none', 'static mapping', 'trained mapping', 'static random'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for different linear layers, R@200',
               "/root/nfs/colbert/evaluation/results/graph_r200_2019_linear_layer_throughout_training.png")

    make_graph(["/root/nfs/colbert/evaluation/results/2019_colbert_df.csv", "/root/nfs/colbert/evaluation/results/2019_ll_none_df.csv",
                "/root/nfs/colbert/evaluation/results/2019_ll_static_mapping_df.csv", "/root/nfs/colbert/evaluation/results/2019_ll_trained_mapping_df.csv", "/root/nfs/colbert/evaluation/results/2019_ll_static_random_df.csv"],
               'R@1000',
               ['colbert', 'none', 'static mapping', 'trained mapping', 'static random'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for different linear layers, R@1000',
               "/root/nfs/colbert/evaluation/results/graph_r1000_2019_linear_layer_throughout_training.png")
    
    
    #overall effectiveness
    effectiveness_44k('test-2019', ['colbert', 'ag_first', 'ag_avg', 'ag_last','ag_avg_pad',
        'll_none', 'll_static_mapping', 'll_trained_mapping', 'll_static_random'],
                      "/root/nfs/colbert/evaluation/results/2019_44k_overall.csv", 0)

    effectiveness_200k('test-2019', ['colbert', 'ag_first', 'ag_avg', 'ag_last','ag_avg_pad',
        'll_none', 'll_static_mapping', 'll_trained_mapping', 'll_static_random'],
                      "/root/nfs/colbert/evaluation/results/2019_200k_overall.csv", 0)
    
    #2020
    effectiveness_throughout_training('test-2020', ['colbert'],
                                      "/root/nfs/colbert/evaluation/results/2020_colbert_df.csv")

    effectiveness_throughout_training('test-2020', ['ag_first'],
                                      "/root/nfs/colbert/evaluation/results/2020_ag_first_df.csv")
    effectiveness_throughout_training('test-2020', ['ag_avg'],
                                      "/root/nfs/colbert/evaluation/results/2020_ag_avg_df.csv")
    effectiveness_throughout_training('test-2020', ['ag_last'],
                                      "/root/nfs/colbert/evaluation/results/2020_ag_last_df.csv")
    effectiveness_throughout_training('test-2020', ['ll_none'],
                                      "/root/nfs/colbert/evaluation/results/2020_ll_none_df.csv")
    effectiveness_throughout_training('test-2020', ['ll_static_mapping'],
                                      "/root/nfs/colbert/evaluation/results/2020_ll_static_mapping_df.csv")
    effectiveness_throughout_training('test-2020', ['ll_trained_mapping'],
                                      "/root/nfs/colbert/evaluation/results/2020_ll_trained_mapping_df.csv")
    effectiveness_throughout_training('test-2020', ['ll_static_random'],
                                      "/root/nfs/colbert/evaluation/results/2020_ll_static_random_df.csv")
    effectiveness_throughout_training('test-2020', ['ag_avg_pad'],
                                      "/root/nfs/colbert/evaluation/results/2020_ag_avg_pad_df.csv")
    
    # aggregation
    make_graph(["/root/nfs/colbert/evaluation/results/2020_colbert_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_first_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_avg_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_last_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_avg_pad_df.csv"],
               'RR@10',
               ['colbert', 'first', 'avg', 'last', 'avg pad'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for subword aggregation, RR@10',
               "/root/nfs/colbert/evaluation/results/graph_rr10_2020_aggregation_throughout_training.png")

    make_graph(["/root/nfs/colbert/evaluation/results/2020_colbert_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_first_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_avg_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_last_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_avg_pad_df.csv"],
               'nDCG@10',
               ['colbert', 'first', 'avg', 'last', 'avg pad'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for subword aggregation, nDCG@10',
               "/root/nfs/colbert/evaluation/results/graph_ndcg10_2020_aggregation_throughout_training.png")

    make_graph(["/root/nfs/colbert/evaluation/results/2020_colbert_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_first_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_avg_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_last_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_avg_pad_df.csv"],
               'R@50',
               ['colbert', 'first', 'avg', 'last','avg pad'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for subword aggregation, R@50',
               "/root/nfs/colbert/evaluation/results/graph_r50_2020_aggregation_throughout_training.png")
    make_graph(["/root/nfs/colbert/evaluation/results/2020_colbert_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_first_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_avg_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_last_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_avg_pad_df.csv"],
               'R@200',
               ['colbert', 'first', 'avg', 'last', 'avg pad'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for subword aggregation, R@200',
               "/root/nfs/colbert/evaluation/results/graph_r200_2020_aggregation_throughout_training.png")

    make_graph(["/root/nfs/colbert/evaluation/results/2020_colbert_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_first_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_avg_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_last_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ag_avg_pad_df.csv"],
               'R@1000',
               ['colbert', 'first', 'avg', 'last', 'avg pad'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for subword aggregation, R@1000',
               "/root/nfs/colbert/evaluation/results/graph_r1000_2020_aggregation_throughout_training.png")

    # linear_layer
    make_graph(["/root/nfs/colbert/evaluation/results/2020_colbert_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_none_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_static_mapping_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_trained_mapping_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_static_random_df.csv"],
               'RR@10',
               ['colbert', 'none', 'static mapping', 'trained mapping', 'static random'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for different linear layers, RR@10',
               "/root/nfs/colbert/evaluation/results/graph_rr10_2020_linear_layer_throughout_training.png")

    make_graph(["/root/nfs/colbert/evaluation/results/2020_colbert_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_none_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_static_mapping_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_trained_mapping_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_static_random_df.csv"],
               'nDCG@10',
               ['colbert', 'none', 'static mapping', 'trained mapping', 'static random'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for different linear layers, nDCG@10',
               "/root/nfs/colbert/evaluation/results/graph_ndcg10_2020_linear_layer_throughout_training.png")

    make_graph(["/root/nfs/colbert/evaluation/results/2020_colbert_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_none_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_static_mapping_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_trained_mapping_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_static_random_df.csv"],
               'R@50',
               ['colbert', 'none', 'static mapping', 'trained mapping', 'static random'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for different linear layers, R@50',
               "/root/nfs/colbert/evaluation/results/graph_r50_2020_linear_layer_throughout_training.png")

    make_graph(["/root/nfs/colbert/evaluation/results/2020_colbert_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_none_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_static_mapping_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_trained_mapping_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_static_random_df.csv"],
               'R@200',
               ['colbert', 'none', 'static mapping', 'trained mapping', 'static random'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for different linear layers, R@200',
               "/root/nfs/colbert/evaluation/results/graph_r200_2020_linear_layer_throughout_training.png")

    make_graph(["/root/nfs/colbert/evaluation/results/2020_colbert_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_none_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_static_mapping_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_trained_mapping_df.csv",
                "/root/nfs/colbert/evaluation/results/2020_ll_static_random_df.csv"],
               'R@1000',
               ['colbert', 'none', 'static mapping', 'trained mapping', 'static random'],
               ['0k','1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k', '100k',
                '150k', '200k'],
               'Effectiveness throughout training for different linear layers, R@1000',
               "/root/nfs/colbert/evaluation/results/graph_r1000_2020_linear_layer_throughout_training.png")
    
    # overall effectiveness
    effectiveness_44k('test-2020', ['colbert', 'ag_first', 'ag_avg', 'ag_last', 'ag_avg_pad',
                                    'll_none', 'll_static_mapping', 'll_trained_mapping', 'll_static_random'],
                      "/root/nfs/colbert/evaluation/results/2020_44k_overall.csv", 0)
    effectiveness_200k('test-2020', ['colbert', 'ag_first', 'ag_avg', 'ag_last', 'ag_avg_pad',
                                     'll_none', 'll_static_mapping', 'll_trained_mapping', 'll_static_random'],
                       "/root/nfs/colbert/evaluation/results/2020_200k_overall.csv", 0)
