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
raw_file = op.join(data_path, 'L005/191209/L005_day1_1_active1_raw_tsss_mc_trans.fif')
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

#ica_comp_L005 = ica.plot_components();
#ica_comp_L005.savefig('/net/server/data/home/vtretyakova/Desktop/New_experiment/ica_comp_L005.pdf') - list

#EOG
eog_epochs = create_eog_epochs(raw, reject=reject)  # get single EOG trials
eog_inds, scores = ica.find_bads_eog(eog_epochs)  # find via correlation
print(ica.labels_)

#ECG
ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5, reject=reject)
ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
print(ica.labels_)


ica.exclude.extend(eog_inds)
ica.exclude.extend(ecg_inds)
ica.save('/net/server/data/home/vtretyakova/Desktop/New_experiment/L005-ica.fif')

#ica = read_ica(op.join('/net/server/data/home/vtretyakova/Desktop/New_experiment', 'L005-ica.fif'))
raw_ica = raw.copy()
ica.apply(raw_ica)
raw_ica.save('/net/server/data/home/vtretyakova/Desktop/New_experiment/raw_ica_L005_1.fif')
