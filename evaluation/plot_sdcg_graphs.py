import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json

"""
Produces the S-nDCG graphs (for all tokens or specific token types) like in Figure A.1
"""


def format_tick_label(x, pos):
    return f"{x / 1000:.0f}k"


def plot_sdcg_token_type(sdcg_file, token_type_file, token_type, save_path):
    with open(sdcg_file) as f:
        gen = json.load(f)
    with open(token_type_file) as f:
        tok = json.load(f)

    fig, ax = plt.subplots(figsize=(6.5, 3), layout='compressed')

    x = [0, 1000, 2000, 5000, 10000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 75000, 80000, 90000, 100000,
         150000, 200000]
    steps = ['0k', '1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k',
             '100k', '150k', '200k']
    ldcg_gen = [gen[s]['nldcg'] for s in steps]
    ndcg = [gen[s]['ndcg'] for s in steps]
    sdcg_tok = [tok[s]['nsdcg'] + ldcg_gen[i] for i, s in enumerate(steps)]
    sdcg = [tok[s]['nsdcg'] for s in steps]

    sdcg_label = 'S-nDCG@10 for ' + token_type + ' tokens'
    if token_type == 'all':
        sdcg_label = 'S-nDCG@10'

    # Set the x-axis tick labels to multiples of e3
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_tick_label))

    ax.plot(x, ldcg_gen, label='L-nDCG@10', color='orange', linewidth=2)
    ax.plot(x, sdcg_tok, label=sdcg_label, color='limegreen', linewidth=2)
    ax.plot(x, ndcg, label='nDCG@10', color='blue', linewidth=2)

    ax.fill_between(x, ldcg_gen, sdcg_tok, alpha=0.5, color='limegreen')

    sndcg_ax = ax.inset_axes([0.5, 0.1, 0.45, 0.3])
    sndcg_ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_tick_label))

    sndcg_ax.plot(x, sdcg, color='limegreen')


    ax.set_xlabel('Training steps')
    ax.set_ylabel('nDCG')

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1))

    fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_sdcg_idf(sdcg_file, low_file, med_file, high_file, save_path):
    with open(sdcg_file) as f:
        gen = json.load(f)
    with open(low_file) as f:
        low = json.load(f)
    with open(med_file) as f:
        med = json.load(f)
    with open(high_file) as f:
        high = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 3), layout='compressed')
    x = [0, 1000, 2000, 5000, 10000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 75000, 80000, 90000, 100000,
         150000, 200000]
    steps = ['0k', '1k', '2k', '5k', '10k', '20k', '25k', '30k', '40k', '50k', '60k', '70k', '75k', '80k', '90k',
             '100k', '150k', '200k']
    ldcg_gen = [gen[s]['nldcg'] for s in steps]
    ndcg = [gen[s]['ndcg'] for s in steps]
    sdcg_low = [low[s]['nsdcg'] + ldcg_gen[i] for i, s in enumerate(steps)]
    sdcg_med = [med[s]['nsdcg'] + sdcg_low[i] for i, s in enumerate(steps)]
    sdcg_high = [high[s]['nsdcg'] + sdcg_med[i] for i, s in enumerate(steps)]

    sdcg_low_small = [low[s]['nsdcg'] for s in steps]
    sdcg_med_small = [med[s]['nsdcg'] for s in steps]
    sdcg_high_small = [high[s]['nsdcg'] for s in steps]

    # Set the x-axis tick labels to multiples of e3
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_tick_label))

    ax.plot(x, sdcg_low, label='S-nDCG@10 for low idf tokens', color='green', linewidth=1)
    ax.plot(x, sdcg_med, label='S-nDCG@10 for med idf tokens', color='orange')
    ax.plot(x, sdcg_high, label='S-nDCG@10 for high idf tokens', color='blue')
    ax.plot(x, ndcg, label='nDCG@10', linestyle='--', color='red')
    ax.plot(x, ldcg_gen, label='L-nDCG@10', color='purple')

    ax.fill_between(x, ldcg_gen, sdcg_low, alpha=0.5, color='limegreen')
    ax.fill_between(x, sdcg_low, sdcg_med, alpha=0.5, color='orange')
    ax.fill_between(x, sdcg_med, sdcg_high, alpha=0.3, color='blue')

    sndcg_ax = ax.inset_axes([0.5, 0.1, 0.45, 0.3])

    sndcg_ax.plot(x, sdcg_high_small, color='blue')
    sndcg_ax.plot(x, sdcg_med_small, color='orange')
    sndcg_ax.plot(x, sdcg_low_small, color='green')


    ax.set_xlabel('Training steps')
    ax.set_ylabel('nDCG')

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1))

    fig.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')


if __name__ == '__main__':
    """#2020
    for token_type in ['all','subword', 'stopword', 'numeric', 'stem', 'question']:
        plot_sdcg_token_type('evaluation/sdcg/colbert_all_results_2020_short.json','evaluation/sdcg/colbert_'+token_type+'_results_2020_short.json',token_type,'evaluation/sdcg_graphs/colbert_'+token_type+'_2020_nsdcg_new.png')

    #2020
    for token_type in ['all','subword', 'stopword', 'numeric', 'stem', 'question']:
        plot_sdcg_token_type('evaluation/sdcg/colbert_all_results_2019_short.json','evaluation/sdcg/colbert_'+token_type+'_results_2019_short.json',token_type,'evaluation/sdcg_graphs/colbert_'+token_type+'_2019_nsdcg_new.png')


    #idf 2020
    plot_sdcg_idf('evaluation/sdcg/colbert_all_results_2020_short.json',
                         'evaluation/sdcg/colbert_low_results_2020_short.json',
                         'evaluation/sdcg/colbert_med_results_2020_short.json',
                         'evaluation/sdcg/colbert_high_results_2020_short.json',
                         'evaluation/sdcg_graphs/colbert_idf_2020_nsdcg_new.png')

    #idf 2019
    plot_sdcg_idf('evaluation/sdcg/colbert_all_results_2019_short.json',
                         'evaluation/sdcg/colbert_low_results_2019_short.json',
                         'evaluation/sdcg/colbert_med_results_2019_short.json',
                         'evaluation/sdcg/colbert_high_results_2019_short.json',
                         'evaluation/sdcg_graphs/colbert_idf_2019_nsdcg_new.png')"""

    # 2020
    for token_type in ['all', 'subword', 'stopword', 'numeric', 'stem', 'question']:
        plot_sdcg_token_type('evaluation/sdcg/ll_none_all_results_2020_short.json',
                             'evaluation/sdcg/ll_none_' + token_type + '_results_2020_short.json', token_type,
                             'evaluation/sdcg_graphs/ll_none_' + token_type + '_2020_nsdcg_new.png')

    # 2020
    for token_type in ['all', 'subword', 'stopword', 'numeric', 'stem', 'question']:
        plot_sdcg_token_type('evaluation/sdcg/ll_none_all_results_2019_short.json',
                             'evaluation/sdcg/ll_none_' + token_type + '_results_2019_short.json', token_type,
                             'evaluation/sdcg_graphs/ll_none_' + token_type + '_2019_nsdcg_new.png')

    # idf 2020
    plot_sdcg_idf('evaluation/sdcg/ll_none_all_results_2020_short.json',
                  'evaluation/sdcg/ll_none_low_results_2020_short.json',
                  'evaluation/sdcg/ll_none_med_results_2020_short.json',
                  'evaluation/sdcg/ll_none_high_results_2020_short.json',
                  'evaluation/sdcg_graphs/ll_none_idf_2020_nsdcg_new.png')

    # idf 2019
    plot_sdcg_idf('evaluation/sdcg/ll_none_all_results_2019_short.json',
                  'evaluation/sdcg/ll_none_low_results_2019_short.json',
                  'evaluation/sdcg/ll_none_med_results_2019_short.json',
                  'evaluation/sdcg/ll_none_high_results_2019_short.json',
                  'evaluation/sdcg_graphs/ll_none_idf_2019_nsdcg_new.png')
