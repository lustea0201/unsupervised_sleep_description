import os
import glob
import numpy as np
np.random.seed(0)
import pandas as pd
from itertools import groupby
from scipy.stats import mode

WINDOW_S = 10
MIN_RECORDING_ACCURACY = 80
MIN_SLEEP_ACCURACY=80
SMOOTH='40T'

def get_durations_df(l):
    durations = []
    index = 0
    for n, c in groupby(l):
        num, count = n, sum(1 for i in c)
        durations.append({'Label': num, 'Duration': count, 'Start': index})
        index += count
    return pd.DataFrame(durations)

def get_good_sleep(labels, smooth='40T', min_sleep=3*3600, min_wake=3*3600):
    if smooth is not None:
        labels = labels.rolling(window=smooth, center=True).apply(lambda x: mode(x, keepdims=False).mode)
    durations = get_durations_df(labels)
    long_sleep = durations[(durations.Label == 1) & (durations.Duration>=(min_sleep/WINDOW_S))]
    long_wake = durations[(durations.Label == 0) & (durations.Duration>=(min_wake/WINDOW_S))]
    if len(long_wake):
        return long_sleep[['Start', 'Duration']]
    else: 
        print('No continuous wake, fragmented sleep suspected')

def manual_acc(PATIENT, folder, labels, ext='unsupervised_labels'):
    labels.name = 'Unsupervised'
    h_manual = pd.read_csv('%s/%s/%s_manual.csv'%(ext, PATIENT, folder), comment='#', parse_dates=[0]).set_index('time').tz_localize(None)
    comp = pd.merge(labels, h_manual, left_index=True, right_index=True).dropna()
    comp.Manual = comp.Manual.replace('?', np.random.choice([0,1])).astype(int)
    comp['Agree'] = (comp.Unsupervised == comp.Manual)
    return 100*comp.Agree.mean()



def main():
    all_sleep_epochs = []
    ALL_PATIENTS = list(filter(lambda x: not '.' in x, os.listdir('unsupervised_labels')))
    for i, PATIENT in enumerate(ALL_PATIENTS):
        print('%s (%d/%d)'%(PATIENT, i, len(ALL_PATIENTS)))
        for fn in glob.glob('unsupervised_labels/%s/*.json'%PATIENT):
            folder = fn.split('.json')[0][-8:]
            labels = pd.read_json('unsupervised_labels/%s/%s.json'%(PATIENT, folder), typ='Series')
            acc=manual_acc(PATIENT, folder, labels)
            sleep_epochs = get_good_sleep(labels, smooth=SMOOTH)
            if (sleep_epochs is not None) & (acc>=MIN_RECORDING_ACCURACY):
                epoch_accuracies = sleep_epochs.apply(lambda row: manual_acc(PATIENT, folder, labels.iloc[row.Start:row.Start+row.Duration]), axis=1)
                sleep_epochs['folder'] = folder
                sleep_epochs['patient'] = PATIENT
                remaining = sleep_epochs[epoch_accuracies>=MIN_SLEEP_ACCURACY]
                if remaining.empty:
                    sleep_epochs=None
                else:
                    all_sleep_epochs.append(remaining)
            if sleep_epochs is None:
                print('No sleep epochs for %s %s'%(PATIENT, folder))
            else:
                print('%s %s rejected due to low labeling confidence (%.1f%%)'%(PATIENT, folder, acc))
            

            

    all_sleep_epochs = pd.concat(all_sleep_epochs).set_index(['patient', 'folder'], drop=True)
    #all_sleep_epochs.to_csv('all_sleep_epochs.csv')
    print(all_sleep_epochs)




if __name__ == "__main__":
    main()
