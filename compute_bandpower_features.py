import os
import sys
import time
import argparse
import datetime
import numpy as np
import pandas as pd

sys.path.append('../../scripts')
from helpers import load_ecog, bandpower, ranges, ranges_hr

exec_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--patient", default = 'm00032', help = "Patient id")
parser.add_argument("-f", "--folder", default = 'all', help = "Folder")
parser.add_argument("-hr", "--high_resolution", default = 'False', help = "Compute power in many bands instead of alpha, beta, gamma, delta, theta")
parser.add_argument("-g", "--periodogram", default = 'False', help = "Save periodogram instead of PIB")
parser.add_argument("-m", "--method", default = 'welch', help = "welch or multitaper")
args = parser.parse_args()
PATIENT = args.patient
FOLDER = args.folder
HIGH_RES = eval(args.high_resolution)

bands = ranges_hr if HIGH_RES else ranges
RETURN_PSD=eval(args.periodogram)
print('Computing the following bands:')
print('PERIODOGRAM' if RETURN_PSD else bands)
METHOD=args.method

HD = 'G' if os.path.isdir('G://') else 'B' if os.path.isdir('B://') else 'A' if os.path.isdir('A://') else ''
assert HD in ['G', 'A', 'B'], 'No Hard Drive mounted!'
PATH_TO_ECOG = '%s://processed'%HD
WINDOW_S = 10
CONSERVATIVE = False
SAVE_FOLDER = '%s://intermediate_data'%HD




ecog, est_fr, actual_fr = load_ecog(PATH_TO_ECOG, PATIENT)

idxs = np.cumsum([0] + list(ecog.attrs['n_samples']))
idxs = pd.DataFrame({'Start': idxs[:-1], 'End': idxs[1:]}, index=ecog.attrs['files']).astype(int)
sec_per_hr = 60*60
idxs['Hours'] = ((idxs['End']-idxs['Start'])/actual_fr/sec_per_hr).round(1)

if FOLDER != 'all':
    idxs = idxs.loc[[FOLDER]]

print(idxs)
print('Total: %.2f hours (%d electrodes)'%(idxs['Hours'].sum(), ecog['eeg'].shape[1]))


save_list=dir()
save_list.append('save_list')

for i, (folder, row) in enumerate(idxs.iterrows()):
    print('---------- Processing folder %s from %s (%d/%d) ----------'%(folder, PATIENT, i+1, len(idxs)))
    n_electrodes = ecog['eeg'].shape[1]
    folder_start_idx = int(row.Start)

    WINDOW_F = est_fr*WINDOW_S
    n_steps = (int(row.End)-int(row.Start)+1)//WINDOW_F
    samples = []
    for i in range(n_steps):
        window_data = ecog['eeg'][folder_start_idx+i*WINDOW_F:folder_start_idx+(i+1)*WINDOW_F]
        window_data = window_data - window_data.mean(axis=1)[:,np.newaxis]
        samples.append(window_data)

    large_amp = pd.DataFrame(index=range(n_steps), columns = range(n_electrodes))
    small_amp = pd.DataFrame(index=range(n_steps), columns = range(n_electrodes))
    large_slope = pd.DataFrame(index=range(n_steps), columns = range(n_electrodes))
    min_amp, max_amp = 10, 2000
    max_slope = 100
    dt = 1/est_fr*1000

    for i, sample in enumerate(samples):
        large_amps, small_amps = np.zeros((WINDOW_S, n_electrodes)), np.zeros((WINDOW_S, n_electrodes))
        for j in range(WINDOW_S):
            amps = sample[j*est_fr:(j+1)*est_fr].max(axis=0)-sample[j*est_fr:(j+1)*est_fr].min(axis=0)
            large_amps[j] = (amps > max_amp)
            small_amps[j] = (amps < min_amp)

        large_amp.iloc[i] = large_amps.any(axis=0)
        small_amp.iloc[i] = small_amps.any(axis=0)
        slope = (sample[1:,:] - sample[:-1,:]).max(0)/dt
        large_slope.iloc[i] = (slope>max_slope)

    artifacts = (large_amp | small_amp | large_slope)
    artifacts.to_pickle('%s/artifacts/%s_%s.pickle'%(SAVE_FOLDER, PATIENT, folder))

    if RETURN_PSD:
        _, freqs = bandpower(samples[0][:,0], est_fr, bands.values(), method=METHOD, window_sec=WINDOW_S, relative=False, annabelle_params=True, return_psd=RETURN_PSD)
        psd_type = 'periodograms' if (METHOD=='welch') else 'multitapers'
        np.save('%s/%s/%s_%s_freqs.npy'%(SAVE_FOLDER, psd_type, PATIENT, folder), freqs)

    featuress_band = []
    for e in range(n_electrodes):
        print('Electrode %d/%d'%(e+1, n_electrodes), end='\r')

        for i, sample in enumerate(samples):
            powers = bandpower(sample[:,e], est_fr, bands.values(), method=METHOD, window_sec=WINDOW_S, relative=False, annabelle_params=True, return_psd=RETURN_PSD)
            if RETURN_PSD:
                powers, freqs = powers
            if (i == 0):
                all_powers = [np.zeros(len(samples)) for k in range(len(powers))]
            for j in range(len(powers)):
                all_powers[j][i] = powers[j]

        names = np.arange(len(powers)) if RETURN_PSD else bands.keys()
        featuress_band.append(pd.DataFrame({name: array for name, array in zip(names, all_powers)}, index = np.arange(len(samples))))

    features_band = pd.concat(featuress_band, keys=range(n_electrodes), axis=1)
    extension = psd_type if RETURN_PSD else 'bandpowers_HR' if (bands == ranges_hr) else 'bandpowers'
    features_band.to_pickle('%s/%s/%s_%s.pickle'%(SAVE_FOLDER, extension, PATIENT, folder))
    print('Time elapsed: %s'%datetime.timedelta(seconds=(time.time()-exec_start)))
