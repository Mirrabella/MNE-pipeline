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

'''
to select number components , we calculate the rank
'''
rank = mne.compute_rank(raw, rank='info')
rank = int(rank['meg'])

'''
select events for horisontal and vertical eyes movements
'''
eog_events_h = mne.preprocessing.find_eog_events(raw, event_id=998, ch_name='EOG062') #horisontal
eog_events_v = mne.preprocessing.find_eog_events(raw, event_id=997, ch_name='EOG061') #vertical

#ICA
n_components = rank  
method = 'fastica' 
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
#horizontal eyes movements
eog_epochs_h = create_eog_epochs(raw, ch_name='EOG062', event_id=998, reject=reject)  # get single EOG trials
eog_inds_h, scores_h = ica.find_bads_eog(eog_epochs_h, ch_name='EOG062', threshold=5.0)  # find via correlation
score_eog_h = ica.plot_scores(scores_h, exclude=eog_inds_h); 
# we can see that only one component is highly correlated and that this
# component got detected by our correlation analysis (red).
score.savefig('home/..../Desktop/New_experiment/ica_score_h_eog_L001_d1p1.jpeg')

properties_h = ica.plot_properties(eog_epochs_h, picks=eog_inds_h, psd_args={'fmax': 35.}, image_args={'sigma': 1.})
for id, p in enumerate(properties_h):
  p.savefig('home/..../Desktop/New_experiment/ica_prop_v_{0}_L001_d1p1.jpeg'.format(id)) 

eog_average_h = eog_epochs_h.average()
overlay_h = ica.plot_overlay(eog_average_h, exclude=eog_inds_h, show=False);
overlay_h.savefig('/home/....../Desktop/New_experiment/ica_overlay_h_L001_d1p1.jpeg')

#vertical eyes movements
eog_epochs_v = create_eog_epochs(raw, ch_name='EOG061', event_id=997, reject=reject)  # get single EOG trials
eog_inds_v, scores_v = ica.find_bads_eog(eog_epochs_v, ch_name='EOG061', threshold=5.0)  # find via correlation
score_eog_v = ica.plot_scores(scores_v, exclude=eog_inds_v); 
# we can see that only one component is highly correlated and that this
# component got detected by our correlation analysis (red).
score.savefig('home/..../Desktop/New_experiment/ica_score_v_eog_L001_d1p1.jpeg')

properties_v = ica.plot_properties(eog_epochs_v, picks=eog_inds_v, psd_args={'fmax': 35.}, image_args={'sigma': 1.})
for id, p in enumerate(properties_v):
  p.savefig('home/..../Desktop/New_experiment/ica_prop_v_{0}_L001_d1p1.jpeg'.format(id)) 

eog_average_v = eog_epochs_v.average()
overlay_v = ica.plot_overlay(eog_average_v, exclude=eog_inds_v, show=False);
overlay_v.savefig('/home/....../Desktop/New_experiment/ica_overlay_v_L001_d1p1.jpeg')


#Note in MNE 20.0 there is no exclude argument in ica.plot_sources
sources = ica.plot_sources(eog_average_v, exclude=eog_inds-V);  # look at source time course
sources.savefig('/home/........../Desktop/New_experiment/ica_sources-v_L001_d1p1.jpeg')

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
