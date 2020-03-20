
###########################
# Latent 1D_periodic dataset generator
# Author: Gyuhyeon Sim
###########################


import argparse
import torch

# from generate_timeseries import Periodic_1d
from torch.distributions import uniform
# Create a synthetic dataset
# from __future__ import absolute_import, division
# from __future__ import print_function
import os
import matplotlib
if os.path.exists("/Users/yulia"):
	matplotlib.use('TkAgg')
else:
	matplotlib.use('Agg')

import numpy as np
import numpy.random as npr
from scipy.special import expit as sigmoid
import pickle
import matplotlib.pyplot as plt
import matplotlib.image
import torch
import lib.utils as utils

import pandas as pd

# ======================================================================================

def get_next_val(init, t, tmin, tmax, final = None):
	if final is None:
		return init
	val = init + (final - init) / (tmax - tmin) * t
	return val


def generate_periodic(time_steps, init_freq, init_amplitude, starting_point,
	final_freq = None, final_amplitude = None, phi_offset = 0.):

	tmin = time_steps.min()
	tmax = time_steps.max()

	data = []
	t_prev = time_steps[0]
	phi = phi_offset
	for t in time_steps:
		dt = t - t_prev
		amp = get_next_val(init_amplitude, t, tmin, tmax, final_amplitude)
		freq = get_next_val(init_freq, t, tmin, tmax, final_freq)
		phi = phi + 2 * np.pi * freq * dt # integrate to get phase

		y = amp * np.sin(phi) + starting_point
		t_prev = t
		data.append([t,y])
	return np.array(data)

def assign_value_or_sample(value, sampling_interval = [0.,1.]):
	if value is None:
		int_length = sampling_interval[1] - sampling_interval[0]
		return np.random.random() * int_length + sampling_interval[0]
	else:
		return value

class TimeSeries:
	def __init__(self, device = torch.device("cpu")):
		self.device = device
		self.z0 = None

	def init_visualization(self):
		self.fig = plt.figure(figsize=(10, 4), facecolor='white')
		self.ax = self.fig.add_subplot(111, frameon=False)
		plt.show(block=False)

	def visualize(self, truth):
		self.ax.plot(truth[:,0], truth[:,1])

	def add_noise(self, traj_list, time_steps, noise_weight):
		n_samples = traj_list.size(0)

		# Add noise to all the points except the first point
		n_tp = len(time_steps) - 1
		noise = np.random.sample((n_samples, n_tp))
		noise = torch.Tensor(noise).to(self.device)

		traj_list_w_noise = traj_list.clone()
		# Dimension [:,:,0] is a time dimension -- do not add noise to that
		traj_list_w_noise[:,1:,0] += noise_weight * noise
		return traj_list_w_noise



class Periodic_1d(TimeSeries):
	def __init__(self, device = torch.device("cpu"),
		init_freq = 0.3, init_amplitude = 1.,
		final_amplitude = 10., final_freq = 1.,
		z0 = 0.):
		"""
		If some of the parameters (init_freq, init_amplitude, final_amplitude, final_freq) is not provided, it is randomly sampled.
		For now, all the time series share the time points and the starting point.
		"""
		super(Periodic_1d, self).__init__(device)

		self.init_freq = init_freq
		self.init_amplitude = init_amplitude
		self.final_amplitude = final_amplitude
		self.final_freq = final_freq
		self.z0 = z0

	def sample_traj(self, time_steps, n_samples = 1, noise_weight = 1.,
		cut_out_section = None, sampling_rate=1.0, n_trajectories=1000):
		"""
		Sample periodic functions.
		"""
		print("sampling_rate is "+str(sampling_rate))

		traj_list = []

		cnt=0
		for i in range(n_trajectories):
			init_freq = assign_value_or_sample(self.init_freq, [0.4,0.8])
			if self.final_freq is None:
				final_freq = init_freq
			else:
				final_freq = assign_value_or_sample(self.final_freq, [0.4,0.8])
			init_amplitude = assign_value_or_sample(self.init_amplitude, [0.,1.])
			final_amplitude = assign_value_or_sample(self.final_amplitude, [0.,1.])

			noisy_z0 = self.z0 + np.random.normal(loc=0., scale=0.1)

			traj = generate_periodic(time_steps, init_freq = init_freq,
				init_amplitude = init_amplitude, starting_point = noisy_z0,
				final_amplitude = final_amplitude, final_freq = final_freq)

			# Cut the time dimension
			traj = np.expand_dims(traj[:,:], 0)
			b=cnt*np.ones((n_samples+False, 1))  #False is extrap value
			traj=np.squeeze(traj, axis=0)
			cnt+=1

			#sampling phase
			num_sample=int(n_samples*sampling_rate)
			num_miss=n_samples-num_sample

			index_arange = np.arange(n_samples)
			sampling_idx=sorted(np.random.choice(index_arange, int(num_sample), replace=False))

			traj_mask=np.zeros(n_samples)
			traj_mask[sampling_idx]=1

			#concatenate
			traj=np.c_[b, traj, traj_mask, np.zeros((n_samples+False, 1))]	#no mask included
			traj_list.append(traj)

		return traj_list

	def random_sample(num_traj, num_idx):
		pass

# parameter list
parser = argparse.ArgumentParser("generate 1d periodic data")
parser.add_argument('-n', '--ntraj',  type=int, default=100, help="Size of the dataset")
parser.add_argument('-s', '--sample_tp', type=float, default=1, help="Number of time points to sub-sample."
    "If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")
parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
parser.add_argument('--max-t',  type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")
parser.add_argument('--noise-weight', type=float, default=0.01, help="Noise amplitude for generated traejctories")

args = parser.parse_args()

# sampling time point
n_total_tp = args.timepoints + args.extrap
max_t_extrap = (args.max_t /args.timepoints) * n_total_tp


distribution = uniform.Uniform(torch.Tensor([0.0]),torch.Tensor([max_t_extrap]))
time_steps_extrap =  distribution.sample(torch.Size([n_total_tp-1]))[:,0]
time_steps_extrap = torch.cat((torch.Tensor([0.0]), time_steps_extrap))
time_steps_extrap = torch.sort(time_steps_extrap)[0]

# data sampling
dataset_obj=Periodic_1d(
    init_freq = None, init_amplitude = 1.,
    final_amplitude = 1., final_freq = None,
    z0 = 1.)

dataset = dataset_obj.sample_traj(time_steps_extrap, n_samples = n_total_tp,
    noise_weight = args.noise_weight, sampling_rate=args.sample_tp, n_trajectories=args.ntraj)

dataset = np.concatenate(dataset, axis=0)

col=["ID","Time","Value_1","Mask_1","Cov"]
df = pd.DataFrame(dataset, columns=col)
df.to_csv("1d_periodic_"+str(args.ntraj)+"_"+str(args.sample_tp)+".csv",index=False)
