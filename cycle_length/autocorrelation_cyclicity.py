import pandas as pd 
import numpy as np
import statsmodels.api as sm
from scipy.signal import find_peaks
import sys
sys.path.append('../N_clusters')
import N_clusters_helpers as Nch

sleep_epochs=pd.read_csv('../unsupervised_labeling/all_sleep_epochs.csv')

def get_cycle_length(signal, w=5*6, min_shift = 6*30, nlags = 6*60*3, prominence=0.05, wlen=int(6*60*1.5)):
    signal2d = (len(signal.shape)==2)
    if signal2d:
        acorrs=np.array([sm.tsa.acf(signal[col], nlags = 6*60*3) for col in signal.columns])
        acorr=acorrs.mean(0)
        delta = acorrs.std(0)*1.96/np.sqrt(bp.shape[1])
    else:
        acorr = sm.tsa.acf(signal, nlags = 6*60*3)
    if w:
        acorr = pd.Series(acorr).rolling(window=w, center=True).mean().values
    y=acorr[min_shift:]
    peaks, _ = find_peaks(y, height=0.05, distance=min_shift, prominence=prominence, wlen=wlen)
    SEC_PER_MIN=60
    WINDOW_S=10
    lag_minutes = np.nan if (len(peaks)==0) else (peaks[0]+min_shift)/(SEC_PER_MIN/WINDOW_S)
    return lag_minutes

def get_all_cycle_lengths(bp):
    ratios = bp.xs('delta', level=1, axis=1)/bp.xs('beta', level=1, axis=1)
    all_ = ratios.iloc[0].copy()
    all_.iloc[:] = np.nan
    all_.name='cycle_length'
    for col in ratios:
        x = ratios[col]
        all_.loc[col] = get_cycle_length(x)
    return all_


for i, (PATIENT, folder, sleep_start, duration) in sleep_epochs.iterrows():
    print('%d/%d'%(i, len(sleep_epochs)))
    sleep_end = sleep_start+duration
    bp, _, _, _, _ = Nch.load_epoch(PATIENT, folder, sleep_start, sleep_end, show=False, padded_ts=True)
    all_ = get_all_cycle_lengths(bp)
    all_.to_csv('epoch_cycle_lengths/%s_%s_%d_%d.csv'%(PATIENT, folder, sleep_start, duration))