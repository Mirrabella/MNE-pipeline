#!/usr/bin/env python
# coding: utf-8

import mne
import os.path as op
from matplotlib import pyplot as plt
import numpy as np
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs


def cleaning_raw_ica (subj, raw, n_components, method = 'fastica', decim = 3):
	picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, exclude=[])
	#horisontal
	eog_events_h = mne.preprocessing.find_eog_events(raw, event_id=998, ch_name='EOG062') 

	#vertical
	eog_events_v = mne.preprocessing.find_eog_events(raw, event_id=997, ch_name='EOG061') 
	
	#heat beat
	ecg_events = mne.preprocessing.find_ecg_events(raw, event_id=999, ch_name='ECG063')[0]

	raw.filter(1., 50., fir_design='firwin')

	ica = ICA(n_components=n_components, method=method, allow_ref_meg=False)
	reject = dict(mag=9e-12, grad=4e-10) #Alexandra Razorenova conspect
	ica.fit(raw, picks=None, reject=reject)
	print(ica)
	
	eog_epochs_h = create_eog_epochs(raw, ch_name='EOG062', event_id=998, reject=reject)  
	eog_inds_h, scores_h = ica.find_bads_eog(eog_epochs_h, ch_name='EOG062', threshold=5.0)  
	#eog_average = eog_epochs_h.average()
	#print(ica.labels_)
	properties_h = ica.plot_properties(eog_epochs_h, picks=eog_inds_h, psd_args={'fmax': 35.}, image_args={'sigma': 1.})

	for id, p in enumerate(properties_h):
		p.savefig('/home/vtretyakova/Desktop/New_experiment/active1/ica_prop_horiz_{0}_{1}_d1a1.jpeg'.format(ica.labels_['eog/0/EOG062'][id], subj)) 
	
	eog_epochs_v = create_eog_epochs(raw, ch_name='EOG061', event_id=997, reject=reject)  
	eog_inds_v, scores_v = ica.find_bads_eog(eog_epochs_v, ch_name='EOG061', threshold=5.0)  
	#print(ica.labels_)
	properties_v = ica.plot_properties(eog_epochs_v, picks=eog_inds_v, psd_args={'fmax': 35.}, image_args={'sigma': 1.})
	for id, p in enumerate(properties_v):
		p.savefig('/home/vtretyakova/Desktop/New_experiment/active1/ica_prop_vert_{0}_{1}_d1a1.jpeg'.format(ica.labels_['eog/0/EOG061'][id], subj)) 	
	#ECG
	ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5, reject=reject)
	ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
	print(ica.labels_)
	properties_ecg = ica.plot_properties(ecg_epochs, picks=ecg_inds, psd_args={'fmax': 35.});
	for id, p in enumerate(properties_ecg):
		p.savefig('/home/vtretyakova/Desktop/New_experiment/active1/ica_prop_ecg_{0}_{1}_d1a1.jpeg'.format(ica.labels_['ecg'][id], subj))

	eog_comp = ica.labels_['eog/0/EOG061']
	ecg_comp = ica.labels_['ecg']
	comp_drop = eog_comp + ecg_comp
	raw_ica = raw.copy()
	ica.apply(raw_ica, exclude = comp_drop)
	raw_ica.save('/home/vtretyakova/Desktop/New_experiment/active1/raw_ica_{0}_d1a1.fif'.format(subj), overwrite=True)

	return(ica)
#63

#load raw data

data_path = '/net/server/data/Archive/speech_learn/meg'


subjects = []
for i in range(27,29):
    if i < 10:
        subjects += ['L00' + str(i)]
    else:
        subjects += ['L0' + str(i)]


date1 = ['191105', '191105', '191122', '191206', '191209', '191213', 		'191216', '191220', '200120', '200125', '200125', '200201', 		'200201', '200209', '200209', '200212', '200214', '200214' , 		'200214' ,'200217' ,'200313' , '200313' , '200315', '200315', 		'200318', '200321', '200321', '200321']

'''
Note: integer after day1 in name of raw file can change according to number of subjects (integer afler L).
L001 - L007 - if integer after L is even, then after day1 is 2; if integer after L is odd then after day1 is 1;
L008 - L011 - even 1; odd 2;
L012 - L021 - even 3, odd 4;
LOO22 - even 2;
L023 - L024 - even 4, odd 3;
L025 - L026 - even 2, odd 1;
Lo27 - L028 - even 4, odd 3.
'''

for idx, subj in enumerate(subjects):
	if idx%2 ==0:
		raw_name = '{0}/{1}/{0}_day1_3_active1_raw_tsss_mc_trans.fif'.format(subj, date[idx])
	else:
		raw_name = '{0}/{1}/{0}_day1_4_active1_raw_tsss_mc_trans.fif'.format(subj, date[idx])	
	raw_file = op.join(data_path, raw_name)
	raw = mne.io.Raw(raw_file, preload=True)
	rank = mne.compute_rank(raw, rank='info')
	rank = int(rank['meg'])
	ica = cleaning_raw_ica (subj=subj, raw=raw, n_components = rank, method = 'fastica', decim = 3)
	ica.save('/home/vtretyakova/Desktop/New_experiment/active1/{0}_d1a1-ica.fif'.format(subj))
