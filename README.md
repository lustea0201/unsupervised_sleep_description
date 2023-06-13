# README

## Scripts to run

1. Bandpowers can be computed using compute_bandpower_features.py

2. Unsupervised labeling using unsupervised_labeling/unsupervised_labeling.py

3. select_sleep_epochs.py to extract sleep epochs. 

4. cycle_length/autocorrelation_cyclicity.py, requires all_sleep_epochs.csv and N_clusters/N_cluster_helpers.py

5. sleep_vs_wake_distribution/compute_band_ratios.py to compute ratio so that can make the plots with paper_plots/A3_sleep_vs_wake_VIOLIN.ipynb.

6. paper_plots/cyclicity/demographics_new.ipynb for impact of demographcis on typical cycle length.

7. N_clusters/opt_N_clusters and then paper_plots/N_clusters/N_clusters_comparing_methods.ipynb for plots



## Environment

- Create the environment from env_checkpoints/klab_may19

## Necessary data files

- Bandpowers under A://intermediate_data/bandpowers/*patient*_*folder*.pickle

- Manual annotations for patients under unsupervised_labeling/unsupervised_labels/*patient*/*folder*_manual.csv


