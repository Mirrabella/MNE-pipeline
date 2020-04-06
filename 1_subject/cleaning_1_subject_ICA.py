#!/usr/bin/env python
# coding: utf-8

import mne
import os.path as op
from matplotlib import pyplot as plt
import numpy as np
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs

#load raw data

data_path = '/net/server/data/Archive/speech_learn/meg'
raw_file = op.join(data_path, 'L001/191105/L001_day1_1_passive1_raw_tsss_mc_trans.fif')
raw = mne.io.Raw(raw_file, preload=True)

picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude=[])
raw.filter(1., 50., fir_design='firwin')

#ICA
n_components = 25  # if float, select n_components by explained variance of PCA
method = 'fastica'  # for comparison with EEGLAB try "extended-infomax" here
decim = 3  # we need sufficient statistics, not all time points -> saves time
random_state = 23

ica = ICA(n_components=n_components, method=method, random_state=random_state, allow_ref_meg=False)

reject = dict(mag=9e-12, grad=4e-10) #Alexandra Razorenova conspect
ica.fit(raw, picks=None, decim=decim, reject=reject)
print(ica)

ica_comp_L005 = ica.plot_components();
for id, i in enumerate(ica_comp_L005):
  comp = id+1
  i.savefig('/net/server/data/home/vtretyakova/Desktop/New_experiment/ica_comp_{0}_L001_d1p1.jpeg'.format(comp)) 

#EOG
eog_epochs = create_eog_epochs(raw, reject=reject)  # get single EOG trials
eog_inds, scores = ica.find_bads_eog(eog_epochs)  # find via correlation
print(ica.labels_)
eog_average = create_eog_epochs(raw, reject=reject,
                                picks=picks_meg).average()

score = ica.plot_scores(scores, exclude=eog_inds);  # look at r scores of components
# we can see that only one component is highly correlated and that this
# component got detected by our correlation analysis (red).
score.savefig('/net/server/data/home/vtretyakova/Desktop/New_experiment/ica_score_L001_d1p1.jpeg')

#Note in MNE 20.0 there is no exclude argument in ica.plot_sources
sources = ica.plot_sources(eog_average, exclude=eog_inds);  # look at source time course
sources.savefig('/net/server/data/home/vtretyakova/Desktop/New_experiment/ica_sources_L001_d1p1.jpeg')

properties = ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax': 35.},
                    image_args={'sigma': 1.})

for id, p in enumerate(properties):
  p.savefig('/net/server/data/home/vtretyakova/Desktop/New_experiment/ica_prop_{0}_L001_d1p1.jpeg'.format(id)) 
  
overlay = ica.plot_overlay(eog_average, exclude=eog_inds, show=False);
overlay.savefig('/net/server/data/home/vtretyakova/Desktop/New_experiment/ica_overlay_L001_d1p1.jpeg')

#ECG
ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5, reject=reject)
ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
print(ica.labels_)
ecg_average = create_ecg_epochs(raw, reject=reject,
                                picks=picks_meg).average()

score_ecg = ica.plot_scores(scores, exclude=ecg_inds); 
score_ecg.savefig('/net/server/data/home/vtretyakova/Desktop/New_experiment/ica_score_ecg_L001_d1p1.jpeg')

properties_ecg = ica.plot_properties(ecg_epochs, picks=ecg_inds, psd_args={'fmax': 35.});
for id, p in enumerate(properties_ecg):
  p.savefig('/net/server/data/home/vtretyakova/Desktop/New_experiment/ica_prop_ecg_{0}_L001_d1p1.jpeg'.format(id))
  
overlay_ecg = ica.plot_overlay(ecg_average, exclude=ecg_inds, show=False);
overlay_ecg.savefig('/net/server/data/home/vtretyakova/Desktop/New_experiment/ica_overlay_ecg_L001_d1p1.jpeg')  
  

ica.exclude.extend(eog_inds)
ica.exclude.extend(ecg_inds)
ica.save('/net/server/data/home/vtretyakova/Desktop/New_experiment/L001_d1p1-ica.fif')

#ica = read_ica(op.join('/net/server/data/home/vtretyakova/Desktop/New_experiment', 'L005-ica.fif'))
raw_ica = raw.copy()
ica.apply(raw_ica)
raw_ica.save('/net/server/data/home/vtretyakova/Desktop/New_experiment/raw_ica_L001_d1p1.fif')
