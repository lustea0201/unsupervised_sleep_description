import pandas as pd
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from itertools import groupby
from scipy.stats import mode
from collections import Counter
import networkx as nx

import sys
sys.path.append('../classify_oscillatory')
sys.path.append('..')
sys.path.append('../unsupervised_labeling')

import unsupervised_labeling_helpers as ulh
from helpers import load_ecog
import matplotlib.dates as md
import os

HD = 'G' if os.path.isdir('G://') else 'B' if os.path.isdir('B://') else 'A' if os.path.isdir('A://') else ''
assert HD in ['G', 'A', 'B'], 'No Hard Drive mounted!'

PATH_TO_ECOG='%s://processed'%HD
INT_DATA = '%s://intermediate_data'%HD
BAD_THRESHOLD=0.5

with open('../unsupervised_labeling/selected_periods/plot_electrodes.json', 'r') as f:
    plot_electrodes = json.load(f)


def load_epoch(PATIENT, folder, sleep_start, sleep_end, show=False, padded_ts=False, remove_bad=True):
    is_h5 = (PATIENT[1]=='0')
    plot_electrode = plot_electrodes[PATIENT]

    if is_h5:
        ecog, est_fr, actual_fr = load_ecog(PATH_TO_ECOG, PATIENT)
        idxs = ulh.get_folder_idxs(ecog, actual_fr)
    with open('../unsupervised_labeling/selected_periods/%s.json'%PATIENT, 'r') as f:
        selected = pd.DataFrame.from_dict(json.load(f))

    select_start, select_end = selected.loc[folder]
    bp = pd.read_pickle('%s/bandpowers/%s_%s.pickle'%(INT_DATA, PATIENT, folder)).iloc[select_start:select_end]
    art = pd.read_pickle('%s/artifacts/%s_%s.pickle'%(INT_DATA, PATIENT, folder)).iloc[select_start:select_end]

    if is_h5:
        time_idxs = pd.date_range(start=idxs.loc[folder].start_time
                                 +datetime.timedelta(seconds=float(selected.loc[folder].Start*10)),
                                       periods=len(bp), freq='10s')
        bp = bp.set_index(time_idxs)
        art = art.set_index(time_idxs)
    else:
        bp = bp.tz_localize(None)


    if padded_ts:
        ONE_HOUR=int(60*60/10)
        padded_start = max(0, sleep_start-ONE_HOUR)
        padded_end = min(len(bp), sleep_end+ONE_HOUR)
        ts_before = bp[plot_electrode]['delta'].iloc[padded_start:sleep_start]
        ts_after = bp[plot_electrode]['delta'].iloc[sleep_end:padded_end]

    bp_sleep = bp.iloc[sleep_start:sleep_end]
    art_sleep = art.iloc[sleep_start:sleep_end]

    bands = bp_sleep[0].columns
    art_rep = pd.DataFrame(np.repeat(art_sleep.values, len(bands), axis=1))
    t=bp_sleep.copy()
    t[art_rep.values] = np.nan
    bp_sleep = t.astype(float).interpolate(method='linear', axis=0).fillna(method='ffill', axis=0).fillna(method='bfill', axis=0)
    assert bp_sleep.isna().sum().sum() == 0, 'Have NaNs'


    bad_electrodes = np.where((art_sleep.mean(0) > BAD_THRESHOLD))[0]
    if remove_bad & (bad_electrodes.size>0):
        print('Removing bad electrodes (artifacts) (%s)'%bad_electrodes)
        bp_sleep = bp_sleep.drop(columns=bad_electrodes)
        art_sleep = art_sleep.drop(columns=bad_electrodes)


    with open('../unsupervised_labeling/bad_electrodes/%s.txt'%PATIENT, 'r') as file:
        data = file.read()
    bad_electrodes_manual = json.loads(data)[folder]
    if remove_bad:
        to_remove = list(set(bad_electrodes_manual)-set(bad_electrodes))
        bp_sleep = bp_sleep.drop(columns=to_remove)
        art_sleep = art_sleep.drop(columns=to_remove)

    ts = bp_sleep[plot_electrode]['delta']
    if show:
        plt.figure(figsize=(12,3))
        plt.scatter(ts.index, ts.values, c='k', s=5)
        plt.yscale('log')

    return_vals = [bp_sleep, plot_electrode, ts]
    if padded_ts:
        return_vals = return_vals + [ts_before, ts_after]

    if not remove_bad:
        return_vals = return_vals + [bad_electrodes, bad_electrodes_manual]


    return return_vals

def normalize_bp(bp, normalization='bandpowers_n'):
    if normalization=='bandpowers':
        return bp
    elif normalization=='bandpowers_n':
        bandpowers_n = bp-bp.mean(0)
        bandpowers_n /= bandpowers_n.std(0)
        return bandpowers_n
    else:
        bands=bp[bp.columns[0][0]].columns
        N_bands=len(bands)
        sm = bp.groupby(level=0, axis=1).sum()
        sm = pd.DataFrame(np.repeat(sm.values, N_bands, axis=1), columns = pd.MultiIndex.from_product([sm.columns, bands]), index=bp.index)
        rel = (bp/sm)
        if normalization=='rel':
            return rel
        elif normalization=='rel_n':
            rel_n = rel.copy()-rel.mean(0)
            rel_n /= rel_n.std(0)
            return rel_n
        else:
            raise ValueError('Normalization type %s not implemented. Choose from bandpowers, bandpowers_n, rel or rel_n.'%normalization)



def get_features(bp, normalization='bandpowers_n', reduction='UMAP_10'):
    features = normalize_bp(bp, normalization)
    if reduction is not None:
        method = reduction.split('_')[0]
        Nd = int(reduction.split('_')[1])
        if method=='UMAP':
            features = UMAP(n_components=Nd,  init='spectral').fit_transform(features)
        else:
            raise ValueError('Method %s not implemented.'%method)

    return features


def get_cluster_durations(labels):
    durations = []
    for n, c in groupby(labels):
        num, count = n, sum(1 for i in c)
        durations.append((num, count))

    return pd.DataFrame(durations, columns=['label', 'count'])

def smooth_labels_mode(labels, window):
    return labels.rolling(window=window, center=True).apply(lambda x: mode(x, keepdims=False).mode).astype(int)


def reorder_levels_by_delta(labels, bp):
    ts=bp.xs('delta', level=1, axis=1).mean(1)
    df = pd.DataFrame(ts.groupby(labels).median().sort_values())
    df['rank']=np.arange(len(df))
    mapping=df.reset_index()['index'].to_dict()
    inv_mapping = {v: k for k, v in mapping.items()}
    return labels.apply(lambda x: inv_mapping[x])


def show_ts_clusters(ts, labels, ax=None, ts_before=None, ts_after=None, plot_electrode_location=None, unsupervised_labels=None, manual_labels=None):
    if ax is None:
        f, ax = plt.subplots(figsize=(12, 3))
        
    cm=plt.get_cmap('tab10')
    for val in np.unique(labels):
        mask = (labels==val)
        ax.scatter(ts.index[mask], ts.values[mask], s=10, label=val, color=cm(val))

    ax.legend()
    ax.set_yscale('log')
    if ts_before is not None:
        ax.scatter(ts_before.index, ts_before.values, s=10, c='k', alpha=0.1)
        ax.scatter(ts_after.index, ts_after.values, s=10, c='k', alpha=0.1)

    ylabel = r'$\delta$ power in a selected electrode' if plot_electrode_location is None else r'$\delta$ power in %s %s'%('left' if plot_electrode_location[2]=='L' else 'right', plot_electrode_location[0])
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time of day')
    ax.xaxis.set_major_formatter(md.DateFormatter('%Hh'))

    if unsupervised_labels is not None:
        m, M = ax.get_ylim()
        ax.plot(unsupervised_labels*m, c='k')

    if manual_labels is not None:
        m, M = ax.get_ylim()
        for i in range(len(manual_labels)):
            if manual_labels.Manual[i]:
                ax.text(manual_labels.index[i], m, 'S', c='green')
            else:
                ax.text(manual_labels.index[i], m, 'W', c='red')


def show_cluster_cycles(ts, labels, ax=None):
    if ax is None:
        f, ax = plt.subplots(figsize=(12, 3))
    vals = np.unique(labels)
    
    cm=plt.get_cmap('tab10')   
    for val in vals:
        ax.scatter(ts.index[labels==val], labels[labels==val], color=cm(val))

    ax.set_ylabel('Cluster')
    ax.set_xlabel('Time of day')
    ax.set_yticks(vals, vals)
    ax.xaxis.set_major_formatter(md.DateFormatter('%Hh'))



def get_nodes(labels):
    durations = []
    for n, c in groupby(labels):
        num, count = n, sum(1 for i in c)
        durations.append((num, count))

    return pd.DataFrame(durations, columns=['label', 'count'])

def show_graph(labels, weigh=None, ax=None):
    if ax is None:
        f, ax =plt.subplots()
    durations = get_nodes(labels)
    durations['next']= durations.label.shift(-1)
    edges=list(durations.apply(lambda row: (int(row.label), np.nan if np.isnan(row.next) else int(row.next)), axis=1))

    node_size = durations.groupby('label')['count'].sum()
    G=nx.DiGraph()
    if weigh=='count':
        counted_edges=[(u, v, w) for (u, v), w in Counter(edges).items()]
        G.add_weighted_edges_from(counted_edges)
    else:
        G.add_edges_from(edges)
    
    
    G.remove_node(np.nan)
    cm=plt.get_cmap('tab10')
    colors=[cm(val) for val in G.nodes]
    pos={0: (1,0), 1: (0, 1), 2: (1, 2), 3: (2,1)}
    if weigh=='count':
        weights = np.array(list(map(lambda x: x[2]['weight'], G.edges(data=True))))
        weights=(weights/weights.max()*5)
        nx.draw_networkx(G, arrows=True, node_size=node_size.loc[list(G.nodes)].values, width=weights, arrowsize=list(weights*10), ax=ax, node_color=colors, pos=pos)
    else:
        nx.draw_networkx(G, arrows=True, node_size=node_size.loc[list(G.nodes)].values, ax=ax, node_color=colors, pos=pos)

    ax.set_title('Graph representation of cluster transitions')
