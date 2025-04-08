import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import matplotlib.pyplot as plt

from checkpoints import CHECKPOINTS

import pyterrier as pt
pt.init()
from pyterrier.measures import *
import pandas as pd
from tqdm import tqdm

from colbert.parameters import DEVICE
from colbert.modeling.colbert import ColBERT

"""
This code is mainly adapted from https://github.com/Xiao0728/ColStar_VirtualAppendix/blob/main/Insights%20(RQ3%20Res)/ColStar_SMP_Demo%20(RQ3%20Res).ipynb
However, we make some changes for better modularity and to support our modified ColBERT models
It measures the SMP (report_smp) and provides the basic dataframe for our computation of S-/L-nDCG
"""

qrels2019 = pt.get_dataset("trec-deep-learning-passages").get_qrels('test-2019')
topics2019 = pt.get_dataset("trec-deep-learning-passages").get_topics('test-2019')

topics2020 = pt.get_dataset("trec-deep-learning-passages").get_topics('test-2020')
qrels2020 = pt.get_dataset("trec-deep-learning-passages").get_qrels('test-2020')

from pyterrier_colbert.ranking import ColBERTFactory, ColBERTModelOnlyFactory

index_root = "/root/nfs/colbert/evaluation/faiss_index/colbert"
index_name = "msmarco_passage_index_colbert_150k"

bm25_terrier_stemmed_text = pt.BatchRetrieve.from_dataset(
    'msmarco_passage',
    'terrier_stemmed_text',
    wmodel='BM25',
    metadata=['docno', 'text'],
    num_results=1000)

import torch
import numpy as np


cuda0 = torch.device('cuda:0')
stops = []
with open("evaluation/stopword-list.txt") as f:
    for l in f:
        stops.append(l.strip())
Tokeniser = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
PorterStemmer = pt.autoclass("org.terrier.terms.PorterStemmer")()


def _get_doc_maxsim_tid_remarkable(doc_maxsim_tid, token, q_token, idfdict, idflist, add_subword=False, add_stop=False, add_numeric=False,
                                   add_low=False, add_med=False, add_high=False, add_all=False, add_stem=False,
                                   add_question=False):
    if add_subword:
        if token.startswith("##"):
            return True
        else:
            return False
    elif add_stop:
        token = token.replace("##", "")
        if token in stops:
            return True
        else:
            return False
    elif add_numeric:
        token = token.replace("##", "")
        if token.isnumeric():
            return True
        else:
            return False

    elif add_low:
        if (idfdict[int(doc_maxsim_tid)]) < np.percentile(idflist, 25):
            return True
        else:
            return False
    elif add_stem:
        q_token = q_token.replace('##', '')
        token = token.replace('##', '')
        if PorterStemmer.stem(q_token) == PorterStemmer.stem(token):
            return True
        else:
            return False
    elif add_med:
        if (np.percentile(idflist, 25) < (idfdict[int(doc_maxsim_tid)])) & (
                (idfdict[int(doc_maxsim_tid)]) < np.percentile(idflist, 75)):
            return True
        else:
            return False
    elif add_high:
        if (idfdict[int(doc_maxsim_tid)]) > np.percentile(idflist, 75):
            return True
        else:
            return False
    elif add_all:
        return True


def _get_match_matrix_remarkable(factory, idfdict, idflist, maxsim, idx, idsQ, idsD, add_subword=False, add_stop=False, add_numeric=False,
                                 add_low=False, add_med=False, add_high=False, add_all=False, add_stem=False,
                                 add_question=False):
    exact_match = torch.zeros_like(maxsim)
    semantic_match = torch.zeros_like(maxsim)
    for didx in range(len(idx)):
        for qidx in range(len(idsQ[0])):
            q_tid = idsQ[0][qidx]
            max_dtok_index = idx[didx][qidx]
            doc_maxsim_tid = idsD[didx][max_dtok_index]
            token = factory.args.inference.doc_tokenizer.tok.convert_ids_to_tokens([doc_maxsim_tid])[0]
            q_token = factory.args.inference.query_tokenizer.tok.convert_ids_to_tokens([int(q_tid)])[0]

            current_special_match = _get_doc_maxsim_tid_remarkable(doc_maxsim_tid, token, q_token, idfdict, idflist,
                                                                   add_subword=add_subword, add_stop=add_stop,
                                                                   add_numeric=add_numeric,
                                                                   add_low=add_low, add_med=add_med, add_high=add_high,
                                                                   add_all=add_all,
                                                                   add_stem=add_stem)

            if (q_tid == doc_maxsim_tid) & current_special_match:
                d_token = factory.args.inference.doc_tokenizer.tok.convert_ids_to_tokens([doc_maxsim_tid])[0]
                exact_match[didx][qidx] = 1
            if (q_tid != doc_maxsim_tid) & current_special_match:
                semantic_match[didx][qidx] = 1
    return exact_match, semantic_match


from pyterrier.transformer import TransformerBase


def scorer_smp(factory, idfdict, idflist, verbose=False, gpu=True,
               add_subword=False, add_stop=False, add_numeric=False,
               add_low=False, add_med=False, add_high=False,
               add_all=False, add_stem=False, add_question=False) -> TransformerBase:
    """
    Calculates the ColBERT max_sim operator using previous encodings of queries and documents
    input: qid, query_embs, [query_weights], docno, doc_embs
    output: ditto + score, [+ contributions]
    """

    import torch
    import pyterrier as pt
    assert pt.started(), 'PyTerrier must be started'
    cuda0 = torch.device('cuda') if gpu else torch.device('cpu')

    def _build_interaction(row, D):
        doc_embs = row.doc_embs
        doc_len = doc_embs.shape[0]
        D[row.row_index, 0:doc_len, :] = doc_embs

    def _build_toks(row, idsD):
        doc_toks = row.doc_toks
        doc_len = doc_toks.shape[0]
        idsD[row.row_index, 0:doc_len] = doc_toks

    def _score_query(df):
        with torch.no_grad():
            weightsQ = None
            Q = torch.cat([df.iloc[0].query_embs])
            if "query_weights" in df.columns:
                weightsQ = df.iloc[0].query_weights
            else:
                weightsQ = torch.ones(Q.shape[0])
            if gpu:
                Q = Q.cuda()
                weightsQ = weightsQ.cuda()
            D = torch.zeros(len(df), factory.args.doc_maxlen, factory.args.dim, device=cuda0)
            df['row_index'] = range(len(df))
            if verbose:
                pt.tqdm.pandas(desc='scorer')
                df.progress_apply(lambda row: _build_interaction(row, D), axis=1)
            else:
                df.apply(lambda row: _build_interaction(row, D), axis=1)
            maxscoreQ = (Q @ D.permute(0, 2, 1)).max(2).values
            scores = (weightsQ * maxscoreQ).sum(1).cpu()
            df["score"] = scores.tolist()

            #             if add_exact_match_contribution:
            idsQ = torch.cat([df.iloc[0].query_toks]).unsqueeze(0)
            idsD = torch.zeros(len(df), factory.args.doc_maxlen, dtype=idsQ.dtype)

            df.apply(lambda row: _build_toks(row, idsD), axis=1)

            # which places in the query are actual tokens, not specials such as MASKs
            token_match = (idsQ != 101) & (idsQ != 102) & (idsQ != 103) & (idsQ != 1) & (idsQ != 2)
            question_match = (idsQ != 2029) & (idsQ != 2129) & (idsQ != 2054) & (idsQ != 2073) & (idsQ != 2339) & (
                        idsQ != 2040) & (idsQ != 2043)

            # perform the interaction
            interaction = (Q @ D.permute(0, 2, 1)).cpu()
            weightsQ = weightsQ.unsqueeze(0).cpu()
            weighted_maxsim = weightsQ * interaction.max(2).values
            # mask out query embeddings that arent tokens
            weighted_maxsim[:, ~token_match[0, :]] = 0
            # get the sum
            denominator = weighted_maxsim.sum(1)

            if add_question:
                interaction = (Q @ D.permute(0, 2, 1)).cpu()

                maxsim, idx = interaction.max(2)
                #here we add all because the non question word tokens get masked out, we cold alternatively use _get_match_matrix_RQ4 from the notebook
                exact_match, semantic_match = _get_match_matrix_remarkable(factory,idfdict, idflist, weighted_maxsim, idx, idsQ, idsD, add_all=True)
                # mask out query embeddings that aren't question tokens
                exact_match[:, question_match[0, :]] = 0
                semantic_match[:, question_match[0, :]] = 0
                weighted_maxsim = weightsQ * maxsim
                weighted_maxsim_exact = weighted_maxsim * exact_match
                weighted_maxsim_semantic = weighted_maxsim * semantic_match
                # get the sum
                numerator_exact = weighted_maxsim_exact.sum(1)
                numerator_semantic = weighted_maxsim_semantic.sum(1)
            else:
                interaction = (Q @ D.permute(0, 2, 1)).cpu()

                maxsim, idx = interaction.max(2)

                exact_match, semantic_match = _get_match_matrix_remarkable(factory, idfdict, idflist, maxsim, idx, idsQ, idsD,
                                                                           add_subword=add_subword,
                                                                           add_stop=add_stop, add_numeric=add_numeric,
                                                                           add_low=add_low, add_med=add_med,
                                                                           add_high=add_high, add_all=add_all,
                                                                           add_stem=add_stem, add_question=add_question)
                # mask out query embeddings that arent tokens
                exact_match[:, ~token_match[0, :]] = 0
                semantic_match[:, ~token_match[0, :]] = 0
                weighted_maxsim = weightsQ * maxsim
                weighted_maxsim_exact = weighted_maxsim * exact_match
                weighted_maxsim_semantic = weighted_maxsim * semantic_match
                # get the sum
                numerator_exact = weighted_maxsim_exact.sum(1)
                numerator_semantic = weighted_maxsim_semantic.sum(1)
            df["exact_numer_exact"] = numerator_exact.tolist()
            df["semantic_numer_exact"] = numerator_semantic.tolist()
            df["exact_denom"] = denominator.tolist()
            df["exact_pct"] = (numerator_exact / denominator).tolist()
            df["semantic_pct"] = (numerator_semantic / denominator).tolist()
        #df = df.drop(columns=['query_toks', 'query_embs'])
        return df

    return pt.apply.by_query(_score_query, add_ranks=True)

def report_smp(cutoff, model, index_root = None, index_name = None, dataset_name='2020', **kwargs):
    """
    Measures the SMP
    """

    factory = make_factory(model, index_root, index_name, **kwargs)

    num_docs, num_all_tokens, idfdict, idflist = make_fnt_info(factory)

    pipe = (factory.query_encoder() >> bm25_terrier_stemmed_text >> factory.text_encoder()
            >> scorer_smp(factory, idfdict, idflist, add_all=True))

    if dataset_name=='2020':
        topics = topics2020
    elif dataset_name=='2019':
        topics = topics2019
    elif dataset_name=='2020-short':
        #here we trunctae the dataset to only the qid's that are in the qrels to save computation time and be consistent
        #with the smp used to compute the S-nDCG
        topics = topics2020[topics2020['qid'].isin(qrels2020['qid'].unique())]
    elif dataset_name=='2019-short':
        # here we trunctae the dataset to only the qid's that are in the qrels to save computation time and be consistent
        # with the smp used to compute the S-nDCG
        topics = topics2019[topics2019['qid'].isin(qrels2019['qid'].unique())]
    else:
        topics=None

    df = pd.concat(pipe.transform_gen(topics, batch_size=1))
    print('df',df)

    smp = df[df['rank']<cutoff].groupby(['qid','query']).mean(numeric_only=True).reset_index().sort_values("semantic_pct", ascending=False).semantic_pct.mean()
    print(smp)
    return smp

def get_smp_df(cutoff, model, index_root = None, index_name = None, dataset_name='2020',add_subword=False, add_stop=False, add_numeric=False,
                                   add_low=False, add_med=False, add_high=False, add_all=True, add_stem=False,
                                   add_question=False, **kwargs):
    """
    Produces the dataframe used for our S-nDCG/L-nDCG calculation
    """

    factory = make_factory(model, index_root, index_name, **kwargs)

    num_docs, num_all_tokens, idfdict, idflist = make_fnt_info(factory)

    pipe = (factory.query_encoder() >>bm25_terrier_stemmed_text >>factory.text_encoder()
            >> scorer_smp(factory, idfdict, idflist, add_all=add_all, add_subword=add_subword, add_stop=add_stop,
                          add_numeric=add_numeric,
                          add_low=add_low, add_med=add_med, add_high=add_high, add_stem=add_stem,
                          add_question=add_question))

    if dataset_name=='2020':
        df = pd.concat(pipe.transform_gen(topics2020, batch_size=1))
    elif dataset_name=='2019':
        df = pd.concat(pipe.transform_gen(topics2019, batch_size=1))
    elif dataset_name=='2020-short':
        #here we trunctae the dataset to only the qid's that are in the qrels to save time
        topics = topics2020[topics2020['qid'].isin(qrels2020['qid'].unique())]
        print(len(topics['qid'].unique()),"topics")
        df = pd.concat(pipe.transform_gen(topics, batch_size=1))
        print(len(topics['qid'].unique()), "topics after")
    elif dataset_name=='2019-short':
        #here we trunctae the dataset to only the qid's that are in the qrels to save time
        topics = topics2019[topics2019['qid'].isin(qrels2019['qid'].unique())]
        print(len(topics['qid'].unique()), "topics")
        df = pd.concat(pipe.transform_gen(topics, batch_size=1))
        print(len(topics['qid'].unique()), "topics after")
    else:
        df=None
    print('df',df)

    return df[df['rank']<cutoff]

def smp_graph(model, save_path='.', cutoff=10, new=False, index_root=None, index_name=None, dataset_name='2020', **kwargs):
    """
    Plots the model's SMP throughout training like for ColBERT in Figure 4.5
    """

    x = [0,1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 150000, 200000]
    if new or (not os.path.isfile(os.path.join(save_path,model+"_smp_graph_0.json"))):
        smp = {}
        steps = ['_0k','_1k', '_2k','_5k','_10k','_20k','_30k','_40k','_50k','_60k','_70k','_80k','_90k','_100k','_150k','_200k']
        for s,n  in zip(steps,x):
                smp[n] = report_smp(cutoff, model+s, index_root, index_name,dataset_name,**kwargs)

        json_path = os.path.join(save_path,model+"_smp_graph_0.json")
        with open(json_path,'w') as f:
            f.write(json.dumps(smp))
        y = [smp[i] for i in x]
    else:
        with open(os.path.join(save_path,model+"_smp_graph_0.json")) as f:
            smp = json.load(f)
        y = [smp[str(i)] for i in x]

    fig, ax = plt.subplots()
    ax.plot(x,y)
    ax.set_title('SMP for '+model+' throughout training')
    plt.savefig(os.path.join(save_path,model+"_smp_graph.png"))


def make_factory(model, index_path=None, index_folder=None,**kwargs):
    if not index_path:
        global index_root
        index_path = index_root

    if not index_folder:
        global index_name
        index_folder = index_name

    if model in CHECKPOINTS:
        checkpoint = CHECKPOINTS[model]
    else:
        checkpoint = model

    factory = ColBERTFactory(checkpoint,index_path,index_folder,faiss_partitions=100,memtype='mem',**kwargs)
    factory.faiss_index_on_gpu = False

    return factory

def make_fnt_info(factory):
    fnt = factory.nn_term(df=True)
    num_docs = fnt.num_docs
    num_all_tokens = len(fnt.emb2tid)
    idfdict = {}
    idflist = []
    for tid in pt.tqdm(range(fnt.inference.query_tokenizer.tok.vocab_size)):
        df = fnt.getDF_by_id(tid)
        idfscore = np.log((1 + num_docs) / (df + 1))
        idfdict[tid] = idfscore
        idflist.append(idfscore)

    return num_docs, num_all_tokens,idfdict, idflist

if __name__=='__main__':
    smp_graph('colbert','evaluation/smp_graphs_2019',cutoff=10, new=False, dataset_name='2019-short')
    smp_graph('ll_none', 'evaluation/smp_graphs_2019', cutoff=10, new=False, linear_layer='none', dim=768, dataset_name='2019-short')
    smp_graph('colbert', 'evaluation/smp_graphs_2020', cutoff=10, new=False, dataset_name='2020-short')
    smp_graph('ll_none', 'evaluation/smp_graphs_2020', cutoff=10, new=False, linear_layer='none', dim=768, dataset_name='2020-short')


