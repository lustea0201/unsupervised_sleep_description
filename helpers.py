import scipy.io
import mat73
import pickle
import h5py
import numpy as np
import pandas as pd
import glob

from scipy.signal import welch, get_window
from scipy.integrate import simps



def get_manana_labels(path_to_data, patient, behavior):
    mat = scipy.io.loadmat('%s/24hours/features/%s_features'%(path_to_data, patient))
    labels = mat['Behaviors']

    with open('%s/mine/behavior_idx.pkl'%path_to_data, 'rb') as f:
        behavior_idx = pickle.load(f)

    behavior_names = behavior_idx[patient]
    def extract_beh(beh):
        beh_index = behavior_names.index(beh)
        is_beh = labels[beh_index]
        is_beh[is_beh>0] = 1
        return is_beh

    if (type(behavior) == list):
        return [extract_beh(beh) for beh in behavior]
    else:
        return extract_beh(behavior)

def get_annabelle_sleep_labels(path_to_data, patient):
    mat = scipy.io.loadmat('%s/sleep_annotations/annot_ds.mat'%path_to_data)
    data_dict = mat73.loadmat('%s/sleep_annotations/annot_all.mat'%path_to_data)
    subjects = list(map(lambda x: x[0], data_dict['Subjs']))
    sleep_labels = mat['annot_ds'][subjects.index(patient)][0][:,0]
    return sleep_labels

framerates = {
    'm00001': 250,
    'm00004': 256,
    'm00005': 256,
    'm00006': 250,
    'm00017': 256,
    'm00018': 256,
    'm00019': 250,
    'm00021': 256,
    'm00022': 256,
    'm00023': 250,
    'm00024': 250,
    'm00025': 256,
    'm00026': 250,
    'm00027': 256,
    'm00028': 256,
    'm00030': 250,
    'm00032': 250,
    'm00033': 256,
    'm00035': 250,
    'm00037': 250,
    'm00038': 250,
    'm00039': 250,
    'm00043': 250,
    'm00044': 250,
    'm00045': 250,
    'm00047': 250,
    'm00048': 250,
    'm00049': 250,
    'm00052': 250,
    'm00053': 250,
    'm00055': 250,
    'm00056': 250,
    'm00058': 250,
    'm00059': 250,
    'm00060': 250,
    'm00061': 250,
    'm00068': 250,
    'm00071': 250,
    'm00073': 250,
    'm00075': 250,
    'm00079': 250,
    'm00083': 256,
    'm00084': 250,
    'm00095': 250,
    'm00096': 250,
    'm00097': 250,
    'm00100': 250,
    'm00107': 250,
    'm00122': 250,
    'm00124': 256,
    'mSu': 250}


"""
{'m00001': 249.953497,
'm00004': 256.0, 'm00005': 256.0, 'm00006': 249.953497, 'm00017': 256.0, 'm00018': 256.0, 'm00019': 249.953497, 'm00021': 256.0, 'm00022': 256.0, 'm00023': 249.953497, 'm00024': 249.953497, 'm00025': 256.0, 'm00026': 249.953497, 'm00027': 256.0, 'm00028': 256.0, 'm00030': 249.953497, 'm00032': 249.953497, 'm00033': 256.0, 'm00035': 250.0, 'm00037': 250.0, 'm00038': 250.0, 'm00039': 250.0, 'm00043': 250.0, 'm00044': 250.0, 'm00045': 250.0, 'm00047': 250.0, 'm00048': 250.0, 'm00052': 250.0, 'm00053': 250.0, 'm00055': 250.0, 'm00056': 250.0, 'm00058': 250.0, 'm00059': 249.953497, 'm00060': 250.0, 'm00061': 250.0, 'm00068': 250.0, 'm00071': 250.0, 'm00073': 250.0, 'm00075': 250.0, 'm00095': 250.0, 'm00096': 250.0, 'm00097': 250.0, 'm00100': 250.0, 'm00107': 250.0, 'm00122': 250.0, 'm00124': 256.0, 'mSu': 250.0}

"""

def load_ecog(path_to_data, patient, folder=None, verbose=False):
    ecog = h5py.File('%s/h5_notch20/%s.h5'%(path_to_data, patient),'r').get('h5eeg')
    est_fr = framerates[patient]
    actual_fr = ecog['eeg'].attrs['rate'][0]
    if verbose:
        print('Estimated: %d Hz, Actual: %.3f'%(est_fr, actual_fr))
    assert (abs(actual_fr-est_fr)<1), 'Poor estimate for framerate!!! %f vs. %f'%(est_fr, actual_fr)

    return ecog, est_fr, actual_fr

def load_edf(path_to_data, PATIENT):
    from mne.io import read_raw_edf
    fs = glob.glob('%s/%s/test.EDF'%(path_to_data, PATIENT)) # glob.glob('%s/%s/*_FINAL.EDF'%(path_to_data, PATIENT))
    if len(fs) > 1:
        print('Warning, there are multiple files, recode this!!!!!!!!!!!!')
        import sys
        sys.exit(1)

    raw = read_raw_edf(fs[0])

    return raw

def bandpower(data, sf, band, window_sec, method='welch', relative=False, annabelle_params=True, return_psd = False):
    """Compute the average power of the signal x in a specific frequency band.

    Requires MNE-Python >= 0.14.
    Credit: https://raphaelvallat.com/bandpower.html

    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    sf : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'welch' or 'multitaper'
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.

    Return
    ------
    bp : float
      Absolute or relative band power.
    """




    # Compute the modified periodogram (Welch)
    if method == 'welch':
        nperseg = window_sec * sf # else (2 / low) * sf

        if annabelle_params:
            assert window_sec == 10, 'Annabelle defined method for 10s windows'
            mini_w = int(2.2*sf)
            freqs, psd = welch(data, sf, window = get_window('hamming', mini_w), noverlap=mini_w // 2)
        else:
            freqs, psd = welch(data, sf, nperseg=nperseg)



    elif method == 'multitaper':
        from mne.time_frequency import psd_array_multitaper
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose='CRITICAL')

    if return_psd:
        return psd, freqs

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    if type(band) == tuple:
        band = [band]

    bps = []
    total = None
    for b in band:
        low, high = b
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        # Integral approximation of the spectrum using parabola (Simpson's rule)
        bp = simps(psd[idx_band], dx=freq_res)
        if relative:
            if total is None:
                total = simps(psd, dx=freq_res)
            bp /= total
        bps.append(bp)

    if len(bps) == 1:
        return bps[0]
    return bps

def load_locations(path_to_data, patient, chosen_mapping = 'Desikan-Killiany'):
    if not ('://' in path_to_data):
        fn = '%s/24hours/anatomy/%s_all_parcellation.mat'%(path_to_data, patient)
    else:
        fn = '%s/%s/label/all_parcellation.mat'%(path_to_data, patient)
    mat = scipy.io.loadmat(fn)
    mappings = list(map(lambda x: x[0], mat['AtlNames'][0]))
    locations = list(map(lambda x: x[0][0], mat['AtlLabels'][0][mappings.index(chosen_mapping)]))
    names = list(map(lambda x: x[0][0], mat['EleLabels']))
    hemispheres = list(map(lambda x: x[0][0], mat['EleHemi']))
    locations_df = pd.DataFrame({chosen_mapping: locations, 'name': names, 'hemisphere': hemispheres})

    return mat, locations_df



short_name_desikan = {
    'Banks of the Superior Temporal Sulcus': 'bankssts',
    'Caudal Anterior (Frontal)': 'caudalanteriorcingulate',
    'Posterior (Parietal)': 'posteriorcingulate',
    'Rostral Anterior (Frontal)': 'rostralanteriorcingulate'}

lobes = ['Frontal', 'Parietal', 'Temporal', 'Occipital', 'Cingulate']
areas = [
    ['Superior Frontal', 'Rostral Middle Frontal', 'Caudal Middle Frontal',
          'Pars Opercularis', 'Pars Triangularis', 'Pars Orbitalis',
          'Lateral Orbitofrontal', 'Medial Orbitofrontal', 'Precentral',
          'Paracentral', 'Frontal Pole'],
    ['Superior Parietal', 'Inferior Parietal', 'Supramarginal', 'Postcentral', 'Precuneus'],
    ['Superior Temporal', 'Middle Temporal', 'Inferior Temporal',
         'Banks of the Superior Temporal Sulcus',
         'Fusiform', 'Transverse Temporal', 'Entorhinal', 'Temporal Pole', 'Parahippocampal'],
    ['Lateral Occipital', 'Lingual', 'Cuneus', 'Pericalcarine'],
    ['Rostral Anterior (Frontal)', 'Caudal Anterior (Frontal)', 'Posterior (Parietal)', 'Isthmus (Parietal)']
        ]

desik_to_lobe = {'unknown': 'Unknown'}
for lobe, names in zip(lobes, areas):
    for name in names:
        if name in short_name_desikan:
            desik_to_lobe[short_name_desikan[name]] = lobe
        else:

            guessed_name = ''.join(name.lower().split(' '))
            desik_to_lobe[guessed_name] = lobe


def get_lobes(path_to_data, patient):
    mat, locations_df = load_locations(path_to_data, patient)
    locations_df['lobe'] = locations_df['Desikan-Killiany'].apply(lambda x: desik_to_lobe[x])
    return locations_df


def get_levels_binary(conservative):
    bin_levels = {'UNKNOWN': -1, 'OPEN': 0, 'AWAKE': 0, 'DROWSY': -1, 'SLEEP_POSSIBLE': -1, 'SLEEP_CERTAIN': 1}
    if conservative:
        bin_levels['CLOSED'] = -1
        bin_levels['SLEEP_LIKELY'] = -1
    else:
        bin_levels['CLOSED'] = 0
        bin_levels['SLEEP_LIKELY'] = 1
    return bin_levels

def get_folder_coordinates(ecog, folder):
    idxs = np.cumsum([0] + list(ecog.attrs['n_samples']))
    idxs = pd.DataFrame({'Start': idxs[:-1], 'End': idxs[1:]}, index=ecog.attrs['files']).astype(int)
    return idxs.loc[folder]

ranges = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 100)
}

ranges_hr = {
    'b1': (0.5, 2),
    'b2': (2, 4),
    'b3': (4, 8),
    'b4': (8, 14),
    'b5': (14, 20),
    'b6': (20, 30),
    'b7': (30, 40),
    'b8': (40, 50),
    'b9': (50, 60),
    'b10': (60, 70),
    'b11': (70, 80),
    'b12': (80, 90),
    'b13': (90, 100)
}
