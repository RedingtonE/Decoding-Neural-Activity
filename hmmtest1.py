# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:36:43 2020

@author: neurotoolbox
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage.filters
import os
import sys
import warnings
import numpy as np
import pickle
import time
import random
import seaborn as sns
import scipy.stats as stats
import copy
import h5py
from sklearn.model_selection import train_test_split
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nelpy.decoding import k_fold_cross_validation
import nelpy as nel
#import nelpy.analysis
import nelpy.plotting as npl

npl.setup()
from nelpy.hmmutils import PoissonHMM


#Read data in from MATLAB. Needs to be saved with option 'v7.3' to be correctly read in
notebook_path = os.getcwd()
matPath = r'C:\Users\Emmy\Documents\GitHub\Decoding-neural-activity\data\exampledata_c27_2NRE.mat'
f = h5py.File( matPath, 'r')
trandat = f['hmmtestspikesmooth']
timeDat =  f['microscopeTime']
posDat = f['microscopeLocation']
fulldat = np.array(trandat[()])
mTime = timeDat[()]
mPos = posDat[()]

f.close()
#Conver data to formats that nelpy can use

cind = np.arange(0, np.size(fulldat, 0))
slist = [[] for _ in np.arange(0,np.size(fulldat, 0))]
for cell in cind:
    slist[cell] = list(np.array(np.where(fulldat[cell, :] == 1))[0])

bins = np.arange(0, 270, 3)*(300/270)

histcount = np.histogram(mPos[0,slist[1]], bins)

#Create spike trains for all cells, selected subsets of cells, and selected epochs
st = nel.SpikeTrainArray(slist, support=session_bounds, fs=20)
st_pc = nel.SpikeTrainArray(slist, support=session_bounds, fs=20)
st_pc = st._unit_subset(pcells)
npcs = list(set(st._unit_ids)-set(pcells))
st_npcs = st._unit_subset(npcs)
ds = 0.02 # 20 ms bin size for PBEs
ds_run = 0.5
ds_50ms = 0.5
sigma = 0.5 # 300 ms spike smoothing

st_runs = st.bin(ds=ds_50ms)
st_pc_runs = st_pc.bin(ds=ds_50ms)
st_npc_runs = st_npcs.bin(ds=ds_50ms)
st_sub = st[40:, :].bin(ds=ds_50ms)

#Run the HMM 
num_states = 60 # number of states for PBE HMM
hmm = nel.hmmutils.PoissonHMM(n_components=num_states, random_state=0, verbose=False)
hmm.fit(st_runs)
transmat_order = hmm.get_state_order('transmat')
xpos = pos.asarray(at=st_runs.centers).yvals
nbins = 120
nstart = 0
nend = 270
bins = np.linspace(nstart, nend, num=nbins)
hmm.reorder_states(transmat_order)
plt.matshow(hmm.transmat, cmap=plt.cm.Spectral_r, vmin=0, vmax=1)
posterior = hmm.predict_proba(st_runs)

#Generate the raster plot
xpos = pos.asarray(at=st_sub.centers).yvals
nbins = 90
nstart = 0
nend = 270
bins = np.linspace(nstart, nend, num=nbins)
ext = np.digitize(xpos, bins)
ext = ext.astype(float)
extern = hmm.fit_ext(X=st_sub, ext=ext, n_extern=nbins)
new_order = hmm.get_state_order(method='mean')
hmm.reorder_states(new_order)
_, posterior_states = hmm.score_samples(st_sub)
posterior = np.hstack(posterior_states)
fig, ax = plt.subplots(figsize=(st_sub.n_bins/5, 4))    
pixel_width = 0.5
npl.imagesc(x=np.arange(st_sub.n_bins), y=np.arange(270), data=posterior, cmap=plt.cm.gray_r, ax=ax)
npl.plot(xpos, ax=ax)
npl.utils.yticks_interval(270)
npl.utils.no_yticks(ax)
ax.vlines(np.arange(st_sub.lengths.sum())-pixel_width, *ax.get_ylim(), lw=1, linestyle=':', color='0.8')
ax.vlines(np.cumsum(st_sub.lengths)-pixel_width, *ax.get_ylim(), lw=1)
ax.set_xlim(-pixel_width, st_sub.lengths.sum()-pixel_width)

divider = make_axes_locatable(ax)
axRaster = divider.append_axes("top", size=1.5, pad=0)
tc = nel.TuningCurve1D(bst=st_pc_runs, extern=pos, n_extern=nbins, extmin=nstart, extmax=nend, sigma=0.2, min_duration=1)
porder = tc.get_peak_firing_order_ids() 
new_order = tc.get_peak_firing_order_ids() 
tcnpcs = nel.TuningCurve1D(bst=st_npc_runs, extern=pos, n_extern=nbins, extmin=nstart, extmax=nend, sigma=0.2, min_duration=1)
remaining_units = tcnpcs.get_peak_firing_order_ids()
new_order.extend(remaining_units)
st_cut = st[st_sub.support]
st_cut = nel.utils.collapse_time(st_cut)
st_cut.reorder_units_by_ids(new_order, inplace=True)
npl.rasterplot(st_cut, vertstack=True, lh=2.5, lw=2.5, ax = axRaster, color='none')
npl.rasterplot(st_cut[:, porder], vertstack=True, ax = axRaster, lh=2.5, lw=2.5, color='r')
npl.rasterplot(st_cut[:, remaining_units], vertstack=True, ax = axRaster, lh=2.5, lw=2.5, color='b')
axRaster.set_xlim(st_cut.support.time.squeeze())
bin_edges = np.linspace(st_cut.support.time[0,0],st_cut.support.time[0,1], st_sub.n_bins+1)
axRaster.vlines(bin_edges, *axRaster.get_ylim(), lw=1, linestyle=':', color='0.8')
axRaster.vlines(bin_edges[np.cumsum(st_sub.lengths)], *axRaster.get_ylim(), lw=1, color='0.2')
npl.utils.no_xticks(axRaster)
npl.utils.no_xticklabels(axRaster)
npl.utils.no_yticklabels(axRaster)
npl.utils.no_yticks(axRaster)

###Try to fit output of HMM
xpos = pos.asarray(at=st_runs.centers).yvals
nbins = 90
nstart = 0
nend = 270
bins = np.linspace(nstart, nend, num=nbins)
ext = np.digitize(xpos, bins)
ext = ext.astype(float)
extern = hmm.fit_ext(X=st_runs, ext=ext, n_extern=nbins)
new_order = hmm.get_state_order(method='mean')
hmm.reorder_states(new_order)
plt.matshow(hmm.transmat, cmap=plt.cm.Spectral_r, vmin=0, vmax=1)
sigma_tc = 4
lsPF = nel.TuningCurve1D(ratemap=extern, min_duration=0, extmin=nstart, extmax=nend)
lsPF = lsPF.smooth(sigma=sigma_tc)
states_in_track_order = np.array(lsPF.get_peak_firing_order_ids())-1
lsPF.reorder_units(inplace=True)
posterior_pos, bdries, mode_pth, mean_pth = hmm.decode_ext(st_runs, ext_shape=(lsPF.n_bins,))
mean_pth = lsPF.bins[0] + mean_pth*(lsPF.bins[-1] - lsPF.bins[0])
posterior = hmm.predict_proba(st_runs)
tvals, target = pos.asarray(at=st_runs.bin_centers)
target_asa = nel.AnalogSignalArray(ydata=target, timestamps=tvals)
decoded_asa = nel.AnalogSignalArray(ydata=np.atleast_2d(mean_pth), timestamps=tvals)

ax= npl.plot(target_asa, color='0.2', lw=2)
npl.plot(decoded_asa, color=npl.colors.sweet.green, lw=2, ax=ax)

#Plot tuning curves of individual neurons
tc = nel.TuningCurve1D(bst=st_pc_runs, extern=pos, n_extern=nbins, extmin=nstart, extmax=nend, sigma=0.2, min_duration=1)
tc = tc.reorder_units()
npl.set_palette(npl.colors.rainbow)
npl.plot_tuning_curves1D(tc.smooth(sigma=2), normalize=True, pad=0.8)


