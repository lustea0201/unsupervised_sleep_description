import os 
import sys
import json
import pathlib
import argparse
import numpy as np
np.random.seed(0)
import pandas as pd 
import glob
sys.path.append('..')
import helpers
import matplotlib.pyplot as plt
import unsupervised_labeling_helpers as ulh

HD = 'G' if os.path.isdir('G://') else 'B' if os.path.isdir('B://') else 'A' if os.path.isdir('A://') else ''
assert HD in ['G', 'A', 'B'], 'No Hard Drive mounted!'
PATH_TO_ECOG='%s://processed'%HD


h5_patients=[el.split('.')[0] for el in os.listdir('%s://processed/h5_notch20'%HD)]
edf_patients=['m10000', 'm10003']
ALL_PATIENTS=h5_patients+edf_patients




parser = argparse.ArgumentParser()
parser.add_argument("-r", "--relative", default = 'True', help = "Use relative_BP")
parser.add_argument("-s", "--split", default = 'True', help = "Split recordings longer than 24 hours")
parser.add_argument("-db", "--delta_beta", default = 'True', help = "Consider delta/beta ratio for sleep assignment (alternative is delta power)")
parser.add_argument("-g", "--group", default = 'True', help = "Reassign by group")
args = parser.parse_args()
RELATIVE = eval(args.relative)
SPLIT = eval(args.split)
DB_RATIO = eval(args.delta_beta)
GROUP = eval(args.group)
save_folder='unsupervised_labels_no_tuning_10'


def sort_by_delta(ls, delta_power):
    for i in range(len(ls)):
        rev_mean = (delta_power[i].groupby(ls[i]).mean().sort_values().index[0]==1)
        if rev_mean:
            ls[i] = 1-ls[i]
    return ls 

def get_cluster_labels(folders, gs, show=False, reassign_by_group=False):
    labels=[]
    for folder, g in zip(folders, gs):
        ls = ulh.get_all_clusters([g], [folder], how='dbscan', show=show, eps=0.4, reassign_by_group=reassign_by_group)[0]
        if (np.abs(np.mean(ls)-0.5)>0.3):#if more than 80% or less than 20%sleep, dbscan might have failed. 
            ls = ulh.get_all_clusters([g], [folder], how='kmeans', show=show)[0]
        labels.append(ls)
    return labels

def get_hourly_labels(unsupervised_labels):
    hourly_labels=[]
    for ul, folder in zip(unsupervised_labels, folders):
        smallest_sec = sorted(ul.index.strftime('%S').unique())[0]
        on_hour = '00:%s'%smallest_sec
        hourly_labels.append(ul[ul.index.strftime('%M:%S')==on_hour])
    return hourly_labels


def compare(hourly_labels, folders, verbose=False):
    agr=[]
    comp_all = pd.DataFrame()
    for h_unsupervised, folder in zip(hourly_labels, folders):
        h_unsupervised.name='Unsupervised'
        h_manual = pd.read_csv('unsupervised_labels/%s/%s_manual.csv'%(PATIENT, folder), comment='#', parse_dates=['time'], na_values='?').set_index('time')
        comp = pd.merge(h_unsupervised, h_manual, left_index=True, right_index=True).dropna()
        comp['Agree'] = (comp.Unsupervised == comp.Manual)
        score=100*comp.Agree.mean()
        agr.append(score)
        if verbose:
            print(folder)
            print('%.1f%% agreement'%score)
            print(comp)
            print('----------')
        comp_all = pd.concat([comp_all,comp])
    if verbose:
        print('Overall: %.1f%% agreement'%(100*comp_all.Agree.mean()))
    return agr, 100*comp_all.Agree.mean()

SHOW=False


SPLIT_T=24
SAMPLES_PER_HOUR=60*60/10

rs=[]
for PATIENT in ALL_PATIENTS:
    with open('selected_periods/%s.json'%PATIENT, 'r') as f:
        selected = pd.DataFrame.from_dict(json.load(f))
    is_h5 = (PATIENT[1]=='0')
    if is_h5:
        ecog, est_fr, actual_fr = helpers.load_ecog(PATH_TO_ECOG, PATIENT)
        idxs = ulh.get_folder_idxs(ecog, actual_fr)
        folders, bandpowers, artifacts, time_idxs, plot_electrode = ulh.load_patient_data(PATIENT, idxs, selected=selected, remove_bad_electrodes=True, all_together=False, only_good=True, show_artifacts=False, verbose=False)
    else:
        folders, bandpowers, artifacts, plot_electrode = ulh.load_patient_data(PATIENT, idxs=None, selected=selected,remove_bad_electrodes=True, all_together=False, only_good=True, show_artifacts=False, verbose=False)
        time_idxs = None
    if SPLIT:
        N_split=list((selected.diff(axis=1).End/SAMPLES_PER_HOUR//SPLIT_T).values.astype(int).clip(1))
        folders, bandpowers, artifacts, time_idxs = ulh.split_into_N(folders, bandpowers, artifacts, time_idxs, N=N_split)
    features = ulh.normalize_data(bandpowers, relative=RELATIVE)
    embeddings = ulh.get_all_umap_embeddings(features, show=SHOW, folders=folders, dimension=10)#ulh.get_all_pca_embeddings(features, show=SHOW, folders=folders, dimension=100)################################### 
    labels=get_cluster_labels(folders, embeddings, show=SHOW, reassign_by_group=GROUP)
    ts = [bp.xs('delta', level=1, axis=1).mean(1)/bp.xs('beta', level=1, axis=1).mean(1) for bp in bandpowers] if DB_RATIO else [bp.xs('delta', level=1, axis=1).mean(1) for bp in bandpowers] 
    labels=sort_by_delta(labels, ts)
    if SPLIT:
        folders, bandpowers, artifacts, time_idxs, labels = ulh.recombine_from_N(folders, bandpowers, artifacts, time_idxs, labels)
    if is_h5:
        unsupervised_labels = [pd.Series(labels[i], time_idxs[i]).rename_axis('time') for i in range(len(folders))]
    else:
        unsupervised_labels = [pd.Series(labels[i], bandpowers[i].index).rename_axis('time') for i in range(len(folders))]
    hourly_labels = get_hourly_labels(unsupervised_labels)
    r, t=compare(hourly_labels, folders)
    print(PATIENT, t, r)
    if (t <=30):
        print('LABELS INVERTED!', r)
        for i in range(len(labels)):
            unsupervised_labels[i]=1-unsupervised_labels[i]
        hourly_labels = get_hourly_labels(unsupervised_labels)
        r, t =compare(hourly_labels, folders)
        print('AFTER:', r, t)
    rs.append(r)
    for folder, ul in zip(folders, unsupervised_labels):
        pathlib.Path('%s/%s'%(save_folder, PATIENT)).mkdir(parents=True, exist_ok=True)
        ul.to_json('%s/%s/%s.json'%(save_folder, PATIENT, folder))

print(np.mean(rs))
comp_all = pd.DataFrame()
folder_accuracies=[]
for fn in glob.glob('%s/*/*.json'%save_folder):
    PATIENT = fn.split('\\')[1]
    folder = fn.split('\\')[2].split('.')[0]
    h_unsupervised = pd.read_json('%s/%s/%s.json'%(save_folder, PATIENT, folder), typ='Series')
    h_unsupervised.name='Unsupervised'
    h_manual = pd.read_csv('unsupervised_labels/%s/%s_manual.csv'%(PATIENT, folder), comment='#', parse_dates=['time'], na_values='?').set_index('time').tz_localize(None)
    comp = pd.merge(h_unsupervised, h_manual, left_index=True, right_index=True).dropna()
    comp['Agree'] = (comp.Unsupervised == comp.Manual)
    score=comp.Agree.mean()*100
    folder_accuracies.append({'patient': PATIENT, 'folder': folder, 'accuracy': score})
    if score <50:
        print(PATIENT, folder, score)
    comp_all = pd.concat([comp_all,comp])

folder_accuracies = pd.DataFrame(folder_accuracies)
folder_accuracies.to_csv('folder_accuracies_UMAP_10.csv')
print('FINAL: ', comp_all.Agree.mean()*100, len(comp_all))
plt.hist(folder_accuracies.accuracy)
plt.title(save_folder)
plt.show()

