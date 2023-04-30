import matplotlib.pyplot as plt
import pickle
import json
import csv
from functools import reduce
import os
import numpy as np
from scipy import fftpack
from scipy import interpolate
from multiprocessing import Pool
# from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import kurtosis
# from stldecompose import decompose
# from statsmodels.tsa.x13 import x13_arima_analysis
import pandas as pd
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.graphics.tsaplots import plot_pacf
# import keras
import sys
import itertools
from pca_utils import *

import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)


def plot_main_0():
	ground_truth_list, predicted_list = pickle.load(open(
		'from_notebook/to_transfer/best_seasonal_trend_noise.p','rb'
	))

	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)

	import matplotlib.ticker as ticker
	ax = plt.gca()
	ax.xaxis.set_major_locator(ticker.MultipleLocator(720))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

	plt.plot(
		range( len( ground_truth_list ) ),
		tuple( map( lambda e: e[2] , ground_truth_list ) ),
		label='Ground Truth Noise'
	)

	plt.plot(
		range( len( predicted_list ) ),
		tuple( map( lambda e: e[2] , predicted_list ) ),
		label='Predicted Noise'
	)

	time = 0
	while time < len( predicted_list ):
		plt.plot(
			(time,time),
			(0,1),
			'r-'
		)
		time+=720

	plt.xlabel('Index in The Data Set')
	plt.ylabel('Normalized Value')
	plt.legend()
	plt.show()

def plot_main_1():
	if False:
		ground_truth_list, predicted_list = pickle.load(open(
			'from_notebook/to_transfer/best_site_thp.p','rb'
		))

	if True:
		a_list = pickle.load(open(
			'network_results.p','rb'
		))

		ground_truth_list = list()
		predicted_list = list()

		for i in range(a_list[0].shape[0]):
			ground_truth_list.append(a_list[0][i,0,0])
			predicted_list.append(a_list[1][i,0,0])

	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)

	# import matplotlib.ticker as ticker
	# ax = plt.gca()
	# ax.xaxis.set_major_locator(ticker.MultipleLocator(720))
	# ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

	plt.plot(
		range( len( ground_truth_list ) ),
		ground_truth_list,
		'b+',
		label='Ground Truth Trend',
	)

	plt.plot(
		range( len( predicted_list ) ),
		predicted_list,
		'r+',
		label='Predicted Trend',
	)

	time = 0
	while time < len( predicted_list ):
		plt.plot(
			(time,time),
			(0,1),
			'r-'
		)
		time+=720

	plt.xlabel('Index in The Data Set')
	plt.ylabel('Normalized Value')
	plt.legend()


	plt.show()

def plot_main_2():
	'''
	Plots reads size per time moment vs throughput
	'''
	if False:
		rs_dict = dict()
		for t, rs in json.load(open('first_opt_cern_only_read_value.json', 'rt')):
			if t in rs_dict:
				rs_dict[t] += rs
			else:
				rs_dict[t] = rs
		pickle.dump(
			sorted( rs_dict.items() ),
			open(
				'first_week_rs_per_time.p',
				'wb'
			)
		)
	a_list = pickle.load(open('first_week_rs_per_time.p','rb'))
	min_a, max_a = min(a_list,key=lambda e:e[1])[1],max(a_list,key=lambda e:e[1])[1]
	a_list = tuple(map(lambda e: (e[0],(e[1]-min_a)/(max_a-min_a),),a_list))

	thp_list =\
			tuple(
				filter(
					lambda p: a_list[0][0] <= p[0] <= a_list[-1][0],
					map(
						lambda p: (1000*int(p[0]), float(p[1]),),
						tuple(
							csv.reader(
								open('january_month_throughput.csv','rt')
							)
						)[1:]
					)
			)
		)
	min_a, max_a = min(thp_list,key=lambda e:e[1])[1],max(thp_list,key=lambda e:e[1])[1]
	thp_list = tuple(map(lambda e: (e[0],(e[1]-min_a)/(max_a-min_a),),thp_list))

	plt.plot(
		tuple(map(lambda e: e[0],a_list)), tuple(map(lambda e: e[1],a_list))
	)
	plt.plot(
		tuple(map(lambda e: e[0],thp_list)), tuple(map(lambda e: e[1],thp_list))
	)

	plt.show()

def plot_main_3():
	# min,max=(-34498.30027637715, 75772.12179454978)
	ground_truth_list,predicted_list=pickle.load(open('from_notebook/to_transfer/unitask_nn_trend.p','rb'))
	ground_truth_list = tuple(map(
		lambda e: (75772.12179454978 + 34498.30027637715) * e - 34498.30027637715,
		ground_truth_list
	))
	predicted_list = tuple(map(
		lambda e: (75772.12179454978 + 34498.30027637715) * e - 34498.30027637715,
		predicted_list
	))

	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)

	import matplotlib.ticker as ticker
	ax = plt.gca()
	ax.xaxis.set_major_locator(ticker.MultipleLocator(720))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

	plt.plot(
		range( len( ground_truth_list ) ),
		ground_truth_list,
		# 'b+',
		label='Ground Truth Trend',
	)

	plt.plot(
		range( len( predicted_list ) ),
		predicted_list,
		# 'r+',
		label='Predicted Trend',
	)

	time = 0
	while time < len( predicted_list ):
		plt.plot(
			(time,time),
			(
				min(ground_truth_list + predicted_list),
				max(ground_truth_list + predicted_list),
			),
			'r-'
		)
		time+=720

	plt.xlabel('Index in The Data Set')
	plt.ylabel('MB/s')
	plt.legend()

	plt.show()

def check_noise_main():
	ground_truth_list,predicted_list=pickle.load(open('from_notebook/to_transfer/unitask_nn_noise.p','rb'))
	ground_truth_list = tuple(map(
		lambda e: (75772.12179454978 + 34498.30027637715) * e - 34498.30027637715,
		ground_truth_list
	))

	min_noise, max_noise = min(ground_truth_list), max(ground_truth_list)

	granulation = 100

	step = (max_noise - min_noise) / granulation

	int_list = [ ( min_noise, min_noise + step ) ]
	while int_list[-1][1] - step < max_noise:
		int_list.append((
			int_list[-1][1],
			int_list[-1][1] + step
		))
	int_list.append((
		int_list[-1][1],
		max_noise + 1
	))

	a =\
	reduce(
		lambda acc,x: acc + ( ( x[0] , x[2] , ) , ( x[1] , x[2] , ) , ) ,
		map(
			lambda interval:\
			(
				interval[0],
				interval[1],
				len(
					tuple(filter(
						lambda e: interval[0] <= e < interval[1],
						ground_truth_list
					))
				),
			),
			int_list
		),
		tuple()
	)

	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)

	plt.plot(
		tuple(map(lambda e: e[0],a)),
		tuple(map(lambda e: e[1],a)),
		label='Noise Values Distribution'
	)

	mean_point = sum(ground_truth_list) / len(ground_truth_list)

	plt.plot(
		(mean_point, mean_point),
		( 0 , max(a,key=lambda e: e[1])[1], ),
		label='Noise Mean'
	)

	plt.xlabel('Noise Values')
	plt.ylabel('Number of Occurences in the Noise Set')
	plt.legend()

	plt.show()

def plot_matrices_gaps_main():
	first_moment, last_moment = 1579264801390, 1579875041000

	distance_fn_iterable =\
	sorted(
		filter(
			lambda e: e > last_moment,
			map(
				lambda fn: int(fn.split('_')[0]),
				filter(
					lambda fn: 'distance' in fn,
					os.listdir( '../remote_host/log_folder/' )
				)
			)
		)
	)

	prev_list = [distance_fn_iterable[0],]

	for fn_int in distance_fn_iterable[1:]:
		if fn_int - prev_list[-1] >= 21600000:
			prev_list.append(
				fn_int
			)

	min_a = min(prev_list)

	print(min_a)

	prev_list =\
	tuple(
		map(
			lambda e: (e - min_a) / (3600*1000),
			prev_list
		)
	)

	plt.plot(
		prev_list,
		len(prev_list) * [0],
		'b+'
	)

	t = 0
	while t < prev_list[-1]:

		plt.plot(
			(t, t,),
			(-1,1,),
			'r-'
		)

		print(t)

		t += 168

	plt.show()

	# 168
	# 336
	# 504

	# 1580488142233 -> 1581697742233

def plot_multi_client_data_for_validation():
	X = json.load(open('first_week_normalized_data_set_cern_all_clients.json','rt'))

	plt.plot(
		range(len(X)),
		tuple(map(lambda p: p[1][3],X)),
		'b-'
	)

	plt.show()

def plot_frequency_decomposition():

	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)

	thp_list =\
	tuple(
		map(
			lambda p: (int(p[0]), float(p[1]),),
			tuple(
				csv.reader(
					open('january_month_throughput.csv','rt')
				)
			)[1:]
		)
	)

	print(len(thp_list))

	f = interpolate.interp1d(
		tuple(map(lambda e: e[0], thp_list)),
		tuple(map(lambda e: e[1], thp_list))
	)

	min_t, max_t = min(map(lambda e: e[0], thp_list)), max(map(lambda e: e[0], thp_list))

	t = min_t

	x_list = list()

	while t <= max_t:

		x_list.append( f(t) )

		t+=120

	if False:
		pickle.dump(
			x_list,
			open(
				'two_minutes_spaced_throughput_data.p',
				'wb'
			)
		)

	X = fftpack.fft(x_list)

	freqs = fftpack.fftfreq(len(X)) * (1 / 120)

	plt.plot( freqs[:len(freqs)//2+1] , np.abs(X)[:len(freqs)//2+1] , 'b+')

	plt.xlabel('Frequecy in Hz')
	plt.ylabel('Amplitude of the Frequecy')

	plt.show()

def plot_tsa():
	thp_list = pickle.load(open('two_minutes_spaced_january_throughput_data.p','rb'))

	# result = seasonal_decompose(
	# 	thp_list,
	# 	model='additive',
	# 	freq=4260,
	# )

	result = decompose(thp_list, period=30)

	if False:
		'''
		Nu merge ca vrea quarterly sau monthly data.
		'''

		t = 1578936840000

		thp_dict = dict()

		for val in thp_list:
			thp_dict[pd.Timestamp(t)] = val
			t += 120000

		result = x13_arima_analysis(
			pd.Series(thp_dict,name="Thp"),
			x12path='/home/mircea/Downloads/x13asall_V1.1_B39/x13as'
		)

	preproc_for_plot_func=lambda arr:\
		tuple(\
			filter(\
				lambda e: str(e[1]) != 'nan',\
				enumerate(arr)\
			)\
		)

	a_func=lambda arr, ind: tuple(map(lambda p: p[ind],arr))

	trend_iterable = preproc_for_plot_func(result.trend)

	seasonal_iterable = preproc_for_plot_func(result.seasonal)

	resid_iterable = preproc_for_plot_func(result.resid)

	if False:

		plt.plot(range(len(thp_list)), thp_list, label='Original')

		plt.plot(a_func(trend_iterable,0), a_func(trend_iterable,1), label='Trend')

		plt.plot(a_func(seasonal_iterable,0), a_func(seasonal_iterable,1), label='Seasonal')

		plt.plot(a_func(resid_iterable,0), a_func(resid_iterable,1), label='Residual')

		plt.legend()

		plt.show()

	if True:
		import matplotlib
		font = {'family' : 'normal',
		        'weight' : 'bold',
		        'size'   : 22}
		matplotlib.rc('font', **font)
		import matplotlib.ticker as ticker

		b_func = lambda arr: tuple(\
			map( lambda p: ( 0.001388888888888889 * p[0] , p[1] , ) , arr ) )

		trend_iterable = b_func(trend_iterable)

		seasonal_iterable = b_func(seasonal_iterable)

		resid_iterable = b_func(resid_iterable)

		c_func = lambda arr, f: f(map(lambda e: e[1],arr))
		print('Original: [ ' + str(min(thp_list)) + ' , ' + str(max(thp_list)) + ' ] ')
		print('Trend: [ ' + str(c_func(trend_iterable,min)) + ' , ' + str(c_func(trend_iterable,max)) + ' ] ')
		print('Seasonal: [ ' + str(c_func(seasonal_iterable,min)) + ' , ' + str(c_func(seasonal_iterable,max)) + ' ] ')
		print('Residual: [ ' + str(c_func(resid_iterable,min)) + ' , ' + str(c_func(resid_iterable,max)) + ' ] ')

		if True:
			max_a = max(map(lambda e: 0.001388888888888889 * e, range(len(thp_list))))

		ax=plt.subplot(411)
		ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
		ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.041666666666666664))
		plt.plot(
			tuple(map(lambda e: 0.001388888888888889 * e, range(len(thp_list)))),
			thp_list,
			label='Throughput'
		)
		plt.plot(
			(0,0),
			(min(thp_list),max(thp_list)),
			'r-'
		)
		plt.plot(
			(max_a,max_a),
			(min(thp_list),max(thp_list)),
			'r-'
		)
		plt.plot()
		plt.ylabel('MB/s')
		plt.legend()

		ax=plt.subplot(412)
		ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
		ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.041666666666666664))
		plt.plot(
			a_func(trend_iterable,0),
			a_func(trend_iterable,1),
			label='Trend'
		)
		plt.plot(
			(0,0),
			(min(a_func(trend_iterable,1)),max(a_func(trend_iterable,1))),
			'r-'
		)
		plt.plot(
			(max_a,max_a),
			(min(a_func(trend_iterable,1)),max(a_func(trend_iterable,1))),
			'r-'
		)
		plt.ylabel('MB/s')
		plt.legend()

		ax=plt.subplot(413)
		ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
		ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.041666666666666664))
		plt.plot(
			a_func(seasonal_iterable,0),
			a_func(seasonal_iterable,1),
			label='Seasonal'
		)
		plt.plot(
			(0,0),
			(min(a_func(seasonal_iterable,1)),max(a_func(seasonal_iterable,1))),
			'r-'
		)
		plt.plot(
			(max_a,max_a),
			(min(a_func(seasonal_iterable,1)),max(a_func(seasonal_iterable,1))),
			'r-'
		)
		plt.ylabel('MB/s')
		plt.legend()

		ax=plt.subplot(414)
		ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
		ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.041666666666666664))
		plt.plot(
			a_func(resid_iterable,0),
			a_func(resid_iterable,1),
			label='Residual'
		)
		plt.plot(
			(0,0),
			(min(a_func(resid_iterable,1)),max(a_func(resid_iterable,1))),
			'r-'
		)
		plt.plot(
			(max_a,max_a),
			(min(a_func(resid_iterable,1)),max(a_func(resid_iterable,1))),
			'r-'
		)
		time = 0
		if False:
			while time - 1 < max_a:
				plt.plot(
					(time,time),
					(min(a_func(resid_iterable,1)),max(a_func(resid_iterable,1))),
					'r-'
				)
				time+=1
		plt.xlabel('Time in Days')
		plt.ylabel('MB/s')
		plt.legend()

		plt.show()

def plot_tsa_1():
	from statsmodels.tsa.seasonal import seasonal_decompose
	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 20}
	matplotlib.rc('font', **font)
	import matplotlib.ticker as ticker

	thp_list = pickle.load(open('two_minutes_spaced_january_throughput_data.p','rb'))[:1440]

	preproc_for_plot_func=lambda arr:\
		tuple(\
			map(\
				lambda p: (0.001388888888888889 * p[0], p[1],),\
				filter(\
					lambda e: str(e[1]) != 'nan',\
					enumerate(arr)\
				)\
			)\
		)

	a_func=lambda arr, ind: tuple(map(lambda p: p[ind],arr))

	trend_iterable =\
	preproc_for_plot_func(
		seasonal_decompose(
			thp_list,
			model='additive',
			freq=30
		).trend
	)
	ax=plt.subplot(311)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.041666666666666664))
	plt.plot(
		a_func(trend_iterable,0),
		a_func(trend_iterable,1),
		label='p = 1 hour'
	)
	plt.plot(
		tuple(map(lambda e: 0.001388888888888889 * e, range(len(thp_list)))),
		thp_list,
		label='original'
	)
	plt.ylabel('MB/s')
	plt.legend()

	trend_iterable =\
	preproc_for_plot_func(
		seasonal_decompose(
			thp_list,
			model='additive',
			freq=360
		).trend
	)
	ax=plt.subplot(312)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.041666666666666664))
	plt.plot(
		a_func(trend_iterable,0),
		a_func(trend_iterable,1),
		label='p = 12 hours'
	)
	plt.plot(
		tuple(map(lambda e: 0.001388888888888889 * e, range(len(thp_list)))),
		thp_list,
		label='original'
	)
	plt.ylabel('MB/s')
	plt.legend()

	trend_iterable =\
	preproc_for_plot_func(
		seasonal_decompose(
			thp_list,
			model='additive',
			freq=720
		).trend
	)
	ax=plt.subplot(313)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.041666666666666664))
	plt.plot(
		a_func(trend_iterable,0),
		a_func(trend_iterable,1),
		label='p = 24 hours'
	)
	plt.plot(
		tuple(map(lambda e: 0.001388888888888889 * e, range(len(thp_list)))),
		thp_list,
		label='original'
	)
	plt.xlabel('Time in Days')
	plt.ylabel('MB/s')
	plt.legend()

	plt.show()

def look_per_proc(p):
	k = kurtosis(
		tuple(map(
			lambda p: str(p) != 'nan',
			seasonal_decompose(
				thp_list,
				model='additive',
				freq=p
			).resid
		))
	)
	# k = kurtosis(
	# 	tuple(map(
	# 		lambda p: str(p) != 'nan',
	# 		decompose(thp_list, period=p).resid

	# 	))
	# )
	return (
		p,
		k,
		abs( 0 - k ),
	)

def look_for_best_noise():
	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 20}
	matplotlib.rc('font', **font)

	global thp_list
	if False:
		thp_list = pickle.load(open('two_minutes_spaced_january_throughput_data.p','rb'))

	max_v = 5040
	# max_v = 50

	a = Pool(7).map(
		look_per_proc,
		range( 30 , max_v )
	)

	# print(a)

	print(
		min(
			a,
			key=lambda e: e[-1]
		)
	)

	print(
		max(
			a,
			key=lambda e: e[-1]
		)
	)

	plt.plot(
		range( 60 , 2 * max_v , 2),
		tuple(map(lambda p: p[1],a)),
		'bo'
	)

	# plt.plot(
	# 	range( 30 ,  max_v ),
	# 	tuple(map(lambda p: p[1],a)),
	# 	'bo'
	# )

	plt.xlabel('TSA period in minutes')
	plt.ylabel('Kurtosis')

	plt.show()

def plot_autocorrelation():
	thp_list = pickle.load(open('two_minutes_spaced_january_throughput_data.p','rb'))

	thp_mean = sum(thp_list) / len(thp_list)

	thp_centered_in_zero = tuple(
		map(
			lambda e: e - thp_mean,
			thp_list
		)
	)

	square_sum = sum(
		map(
			lambda e: e * e,
			thp_centered_in_zero
		)
	)

	print(len(thp_list))

	plt.plot(
		range(1, len(thp_list) - 1),
		tuple(
			map(
				lambda k:\
					sum(
						map(
							lambda p: p[0]*p[1],
							zip(
								thp_centered_in_zero[k:],
								thp_centered_in_zero[:len(thp_list)-k]
							)
						)
					) / square_sum,
				range(1, len(thp_list) - 1)
			)
		),
		'ro'
	)

	plot_acf(np.array(thp_list), lags=20158)
	# plot_pacf(np.array(thp_list), lags=50)
	plt.show()

def plot_correlation_between_bins_and_throughput():
	if False:
		data_set_tuple = tuple(
			map(
				lambda line: line[:-1] + [2*line[-1]-1,],
				json.load(open('three_weeks_normalized_data_set.json','rt'))
			)
		)[:4000]

	if True:
		data_set_tuple = tuple(
			map(
				lambda line: line[:-1] + [2*line[-1]-1,],
				json.load(open('three_weeks_normalized_4k_bins_data_set.json','rt'))
			)
		)
		# week_0_first_option_cern_normalized_data_set_FOR_COMPARISON.json
		# three_weeks_normalized_4k_bins_data_set.json
		# first_week_data_set_site_thp.json

	if False:
		plt.plot(
			range(len(data_set_tuple[0])-1),
			tuple(
				map(
					lambda ind:\
						sum(
							map(
								lambda p: p[0] * p[1],
								zip(
									map(lambda e: e[ind], data_set_tuple),
									map(lambda e: e[-1], data_set_tuple),
								)
							)
						),
					range(len(data_set_tuple[0])-1)
				)
			),
			'bo'
		)

	if True:
		print(len(data_set_tuple))
		print(len(data_set_tuple[0]))
		correlations_tuple =\
		list(
			enumerate(
				map(
					lambda ind: float(np.correlate(
							list(map(lambda e: e[ind], data_set_tuple)),
							list(map(lambda e: e[-1], data_set_tuple)),
						)),
					range(len(data_set_tuple[0])-1)
				)
			)
		)
		plt.plot(
			range(len(data_set_tuple[0])-1),
			list(map(lambda e: e[1],correlations_tuple)),
			# 'bo'
			label='Correlation'
		)
		top_k_list = sorted(
			sorted(
				correlations_tuple,
				key=lambda p: p[1],
				reverse=True
			)[:1000]
		)
		plt.plot(
			list(map(lambda e: e[0],top_k_list)),
			list(map(lambda e: e[1],top_k_list)),
			'ro',
			label='Top 1000 Highest Correlated Bins'
		)

		plt.legend()
		plt.xlabel('Bin Series Index')
		plt.ylabel('Correlation with Througput Values')

	plt.show()

def plot_network_results():
	ground_truth_list, predicted_list = pickle.load(open(
		'network_results.p','rb'
	))

	plt.plot(range(len(ground_truth_list)), ground_truth_list, label='Ground Truth Trend')

	plt.plot(range(len(predicted_list)),predicted_list, label='Predicted Trend')

	mape = 0
	div = 0
	for i in range(len(ground_truth_list)):
		d = ground_truth_list[i] - predicted_list[i]

		if d < 0:
			d = -d

		if ground_truth_list[i] != 0:
			mape += d/ground_truth_list[i]
			div += 1

	print('mape: ' +  str(mape/div))
	print('\tdiv: ' + str(div) + '/' + str(len(ground_truth_list)))

	plt.xlabel('Index in The Data Set')
	plt.ylabel('Normalized Value')
	plt.legend()
	plt.show()

def plot_normalization():
	a_list = list(
		map(
			lambda e: e[-1],
			json.load(open('first_week_data_set_site_thp.json','rt'))
		)
	)

	min_thp, max_thp = min(a_list,),max(a_list)

	plt.plot(
		range(len(a_list)),
		list(
			map(
				lambda e: (e - min_thp)/(max_thp-min_thp),
				a_list
			)
		)
	)

	# plt.plot(
	# 	range(len(a_list)),
	# 	a_list
	# )

	plt.show()

def plot_training_process(csv_file):
	if False:
		a_list = list(
			map(
				lambda e: list( map(lambda a: float(a) , e ) ),
				filter(
					lambda e: len(e) == 4 and 'illed' not in e[0],
					csv.reader(
						open(
							csv_file,
							'rt'
						)
					)
				)
			)
		)

	if True:
		csv_file_gen = csv.reader(open(
			'/home/mircea/cern_log_folder/dec_work/pca_csv_folder/'+csv_file,'rt')
		)

		print(csv_file)

		next(csv_file_gen)

		a_list = list(
			map(
				lambda e: list( map(lambda a: float(a) , e[1:] ) ),
				filter(
					lambda e: len(e) == 5,
					csv_file_gen
				)
			)
		)[:100]

	plt.subplot(211)
	plt.plot(
		range(len(a_list)),
		list(
			map(
				lambda e: float(e[0]) if float(e[0]) <= 100 else 100,
				a_list,
			)
		),
		label='Train MAPE'
	)
	plt.plot(
		range(len(a_list)),
		list(
			map(
				lambda e: float(e[2]) if float(e[2]) <= 100 else 100,
				a_list,
			)
		),
		label='Valid MAPE'
	)
	plt.legend()
	plt.ylabel('MAPE')
	plt.subplot(212)
	plt.plot(
		range(len(a_list)),
		list(
			map(
				lambda e: float(e[1]),
				a_list,
			)
		),
		label='Train MAE'
	)
	plt.plot(
		range(len(a_list)),
		list(
			map(
				lambda e: float(e[3]),
				a_list,
			)
		),
		label='Train MAE'
	)
	plt.legend()

	plt.xlabel('Epoch Index')

	plt.ylabel('MAE')

	fig = plt.gcf()
	fig.set_size_inches(11,8)

	# plt.show()
	plt.savefig(
		'/home/mircea/Desktop/plots_6_apr/loss_evolution_'\
		+ csv_file.split('_')[1][:-4]\
		+ '.png'
	)
	plt.clf()

def plot_stats_per_three_weeks():
	stats_dict = pickle.load(open('stats_dict.p','rb'))

	q_count_tag = 0
	min_read_tag = 1
	max_read_tag = 2
	total_read_tag = 3
	avg_read_tag = 4
	min_options_count_tag = 5
	max_options_count_tag = 6
	tm_stats_dict_tag = 7

	if False:
		count_dict = dict()

		for value in stats_dict[tm_stats_dict_tag].values():

			if value[q_count_tag] in count_dict:
				count_dict[value[q_count_tag]] += 1
			else:
				count_dict[value[q_count_tag]] = 1

		a_list = sorted( count_dict.items() )

		plt.plot(
			list(map(lambda e: e[0], a_list)),
			list(map(lambda e: e[1], a_list)),
			label='Queries Frequecy'
		)
		plt.legend()
		plt.xlabel('# queries per time tag')
		plt.ylabel('# time tags')
		plt.show()

def plot_stats_per_time_interval():
	a,b,c,d,e,f,g = pickle.load(open('q_stats_per_time_interval.p','rb'))

	a = list(map(lambda e: e/60000, a))

	plt.plot(a,b,'bo',label='min')
	plt.plot(a,c,'ro',label='max')
	plt.plot(a,d,'go',label='avg')

	plt.legend()
	plt.xlabel('Time window length in minutes')
	plt.ylabel('# of queries')

	plt.show()

def bar_plot_the_option_counts():
	counts_list = [\
		8.017537783595118,
		72.87825082127769,
		3.931079388477518,
		10.646076669174617,
		2.027786609980748,
		0.3851995266218222,
		2.0745325543673503,
		0.027693057824849973,
		0.01184358868028707,
	]

	plt.bar(
		range(1,10),
		counts_list,
		label="Options Percentage"
	)
	plt.legend()
	plt.xlabel('# options')
	plt.ylabel('percentage out of all queries')

	plt.show()

def bar_plot_number_of_trend_value():
	to_plot = (
		(0.05, 435),
		( 0.1, 141),
		(0.15, 210),
		( 0.2, 108),
		(0.25, 333),
		( 0.3, 1949),
		(0.35, 1835),
		( 0.4, 944),
		(0.45, 1759),
		( 0.5, 1602),
		(0.55, 584),
		( 0.6, 467),
		(0.65, 414),
		( 0.7, 1021),
		(0.75, 1198),
		( 0.8, 432),
		(0.85, 332),
		( 0.9, 595),
		(0.95, 1621),
		( 1, 957),
	)

	a = sum(map(lambda e: e[1],to_plot))

	plt.bar(
		list(map(lambda e: e[0],to_plot)),
		list(map(lambda e: 100*e[1]/a,to_plot)),
		width=0.02,
		label="Value Percentages"
	)
	plt.legend()
	plt.ylabel('Value Percentage')
	plt.xlabel('Interval End')

	plt.show()

def plot_network_results_1(dump_file):
	a_dict = dict(pickle.load(open(
		'./pca_results/'+dump_file,'rb'
	)))

	print(dump_file + ': general_performance=' + str(a_dict['general_performance'])\
		+ ' validation_performance=' + str(a_dict['validation_performance'])
	)

	ground_truth_list =\
		a_dict['predictions'][0][0]\
		+a_dict['predictions'][1][0]\
		+a_dict['predictions'][2][0]

	predicted_list =\
		a_dict['predictions'][0][1]\
		+a_dict['predictions'][1][1]\
		+a_dict['predictions'][2][1]

	plt.plot(range(len(ground_truth_list)), ground_truth_list, label='Ground Truth Trend')

	plt.plot(range(len(predicted_list)),predicted_list, label='Predicted Trend')

	plt.plot(
		( len(a_dict['predictions'][0][0]) - 1 , len(a_dict['predictions'][0][0]) - 1 ,),
		( 0 , 1),
		'r-'
	)

	plt.plot(
		(
			len(a_dict['predictions'][0][0]) + len(a_dict['predictions'][1][0]) - 1,
			len(a_dict['predictions'][0][0]) + len(a_dict['predictions'][1][0]) - 1,
		),
		( 0 , 1),
		'r-'
	)

	plt.xlabel('Index in The Data Set')
	plt.ylabel('Normalized Value')
	plt.legend()

	plt.savefig(
		'/home/mircea/Desktop/plots_6_apr/gt_vs_pr_'\
		+ dump_file.split('_')[1][:-2]\
		+ '.png'
	)

	plt.clf()

def print_best_results(fi_ind, la_ind):

	ind_fn_list = list()

	non_existing_list = list()

	existing_list = os.listdir('./pca_results/')

	for ind in range(fi_ind,la_ind):
		if 'result_'+str(ind)+'.p' in existing_list:
			ind_fn_list.append((
				ind,
				dict(pickle.load(open('./pca_results/result_'+str(ind)+'.p','rb')))
			))
		else:
			non_existing_list.append(ind)

	data_list = list(
		map(
			lambda d: (\
				d[0],
				d[1]['general_performance'],
				d[1]['validation_performance'],
			),
			ind_fn_list
		)
	)

	print('nonexisting are: ' + str( non_existing_list ))

	print( 'between ' + str(fi_ind) + ' and ' + str(la_ind) + ' best overall is: ' +\
		str(min(data_list, key=lambda p: p[1]))
	)

	print( 'between ' + str(fi_ind) + ' and ' + str(la_ind) + ' best valid is: ' +\
		str(min(data_list, key=lambda p: p[2]))
	)

	print( 'between ' + str(fi_ind) + ' and ' + str(la_ind) + ' best valid is: ' +\
		str(min(data_list, key=lambda p: p[2]))
	)

	print()

def plot_components():
	component_list = list( map( lambda v: 100 * v,pickle.load(
		open('pipe.p','rb'))))

	n = 40

	# plt.plot(
	# 	range(n),
	# 	component_list[:n],
	# 	'bo',
	# 	label="PC Variation Percentages"
	# )

	accumulated_list = [component_list[0],]
	for el in component_list[1:]:
		accumulated_list.append(
			accumulated_list[-1] + el
		)

	accumulated_list = accumulated_list[:n]

	plt.plot(
		range(1, len(accumulated_list)+1),
		accumulated_list,
		'bo',
		label='PC Cumulative Variation'
	)

	plt.legend()
	plt.xlabel('Component Index')
	plt.ylabel('Percentage')

	plt.show()

def plot_components_from_csv(ind):
	a = csv.reader( open( 'components.csv' , 'rt' ) )
	next(a)
	a = list(a)
	if len( a[-1] ) < 2: a = a[:-1]

	a=a[:100]

	plt.plot(
		range( len(a) ),
		list(
			map(
				lambda e: e[ind],
				a
			)
		)
	)
	plt.show()

def plot_data_set_overlap_evolution():

	plt.plot(
		range(40),
		tuple(
			map(
				lambda e: 17000//(40 - e),
				range(40)
			)
		),
		label='Dataset Size'
	)

	plt.xlabel('Maximum Overlap')
	plt.ylabel('# of examples')

	plt.show()

def plot_network_results_2(dump_file):
	fig = plt.gcf()
	fig.set_size_inches(11,8)

	a_dict = pickle.load(open(
		'./pca_results/'+dump_file,'rb'
	))

	print(
		dump_file\
		+ ': general_performance='\
			+ str(a_dict['whole_data_set_performance'])\
		+ ' validation_performance=' + str(a_dict['best_model_performance'])
	)

	ground_truth_list =\
		a_dict['per_week_predictions'][0][0]\
		+a_dict['per_week_predictions'][1][0]\
		+a_dict['per_week_predictions'][2][0]

	predicted_list =\
		a_dict['per_week_predictions'][0][1]\
		+a_dict['per_week_predictions'][1][1]\
		+a_dict['per_week_predictions'][2][1]

	plt.plot(range(len(ground_truth_list)), ground_truth_list,
		# 'g+',
		label='Ground Truth Trend')

	plt.plot(range(len(predicted_list)),predicted_list,
		# 'b+',
		label='Predicted Trend')

	plt.plot(
		( len(a_dict['per_week_predictions'][0][0]) - 1 , len(a_dict['per_week_predictions'][0][0]) - 1 ,),
		( 0 , 1),
		'r-'
	)

	plt.plot(
		(
			len(a_dict['per_week_predictions'][0][0]) + len(a_dict['per_week_predictions'][1][0]) - 1,
			len(a_dict['per_week_predictions'][0][0]) + len(a_dict['per_week_predictions'][1][0]) - 1,
		),
		( 0 , 1),
		'r-'
	)

	plt.xlabel('Index in The Data Set')
	plt.ylabel('Normalized Value')
	plt.legend()

	plt.savefig(
		'/home/mircea/Desktop/plots_6_apr/gt_vs_pr_'\
		+ dump_file.split('_')[1][:-2]\
		+ '.png'
	)

	plt.clf()

	plt.plot(
		range(
			len(
				a_dict['validation_predictions'][0]
			)
		),
		# 'bo',
		a_dict['validation_predictions'][0],
		label='Ground Truth Trend'
	)
	plt.plot(
		range(
			len(
				a_dict['validation_predictions'][1]
			)
		),
		a_dict['validation_predictions'][1],
		label='Prediction'
	)
	plt.xlabel('Index in The Data Set')
	plt.ylabel('Normalized Value')
	plt.legend()

	plt.savefig(
		'/home/mircea/Desktop/plots_6_apr/valid_gt_vs_pr_'\
		+ dump_file.split('_')[1][:-2]\
		+ '.png'
	)

	plt.clf()

def plot_components_vs_trend():
	fig = plt.gcf()
	fig.set_size_inches(15,8)

	week_0_list, week_1_list, week_2_list =\
		get_normalized_values_per_week( 'complete_data_set_top_80.p' )

	print( len( week_0_list[ 0 ] ) )

	print( week_0_list[ 0 ] )

	for feature_index in range( len( week_0_list[0] ) - 1 ):

		xlabel_name = 'ars' if feature_index == 0 else ( 'pc' + str(feature_index) )
		label_name = xlabel_name + ' vs trend'

		a_list = sorted(
			zip(
				tuple(
					itertools.chain(
						map( lambda e: e[feature_index] , week_0_list ),
						map( lambda e: e[feature_index] , week_1_list ),
						map( lambda e: e[feature_index] , week_2_list ),
					)
				),
				tuple(
					itertools.chain(
						map( lambda e: e[-1] , week_0_list ),
						map( lambda e: e[-1] , week_1_list ),
						map( lambda e: e[-1] , week_2_list ),
					)
				),
			)
		)

		if False:
			plt.plot(
				tuple(
					map(
						lambda e: e[0] ,
						a_list
					)
				),
				tuple(
					map(
						lambda e: e[1] ,
						a_list
					)
				),
				'b+',
				label=label_name
			)
			plt.xlabel(xlabel_name)
			plt.ylabel('Trend')
			plt.legend()
			if feature_index == 0:
				plt.savefig('./pc_vs_trend_plots/dotted_ars_vs_trend.png')
			else:
				plt.savefig(\
					'./pc_vs_trend_plots/dotted_pc_'\
					+ str(feature_index)\
					+'_vs_trend.png'
				)
			plt.clf()

			plt.plot(
				tuple(
					map(
						lambda e: e[0] ,
						a_list
					)
				),
				tuple(
					map(
						lambda e: e[1] ,
						a_list
					)
				),
				label=label_name
			)
			plt.xlabel(xlabel_name)
			plt.ylabel('Trend')
			plt.legend()
			if feature_index == 0:
				plt.savefig('./pc_vs_trend_plots/lined_ars_vs_trend.png')
			else:
				plt.savefig(\
					'./pc_vs_trend_plots/lined_pc_'\
					+ str(feature_index)\
					+'_vs_trend.png'
				)
			plt.clf()

		plt.subplot(211)
		plt.plot(
			tuple(
				map(
					lambda e: e[0] ,
					a_list
				)
			),
			tuple(
				map(
					lambda e: e[1] ,
					a_list
				)
			),
			'b+',
			label=label_name
		)
		plt.ylabel('Trend')
		plt.legend()

		plt.subplot(212)
		plt.plot(
			tuple(
				map(
					lambda e: e[0] ,
					a_list
				)
			),
			tuple(
				map(
					lambda e: e[1] ,
					a_list
				)
			),
		)
		plt.xlabel(xlabel_name)
		plt.ylabel('Trend')

		if feature_index == 0:
			plt.savefig('./pc_vs_trend_plots/both_ars_vs_trend.png')
		else:
			plt.savefig(\
				'./pc_vs_trend_plots/both_pc_'\
				+ str(feature_index)\
				+'_vs_trend.png'
			)
		plt.clf()

def plot_whole_matrices_data_set_1():
	reporting_time_diffs_per_week, _ =\
	pickle.load(
		open(
			'pipe_0.p',
			'rb'
		)
	)

	def plot_week(week_list, offset, label_on_flag=False):
		if label_on_flag:
			plt.plot(
				range(offset, offset + len(week_list)),
				tuple(
					map(
						# lambda e: (e/60000) if e/60000 <= 31 else 31,
						lambda e: e/60000,
						week_list
					)
				),
				'bo',
				label='Time Difference Between Matrices'
			)
		else:
			plt.plot(
				range(offset, offset + len(week_list)),
				tuple(
					map(
						# lambda e: (e/60000) if e/60000 <= 31 else 31,
						lambda e: e/60000,
						week_list
					)
				),
				'bo',
			)
		return offset + len(week_list)

	offset = 0
	offset = plot_week(reporting_time_diffs_per_week[0], offset)
	plt.plot( ( offset , offset ) , ( 0 , 31 ) , 'r-' )
	offset = plot_week(reporting_time_diffs_per_week[1], offset)
	plt.plot( ( offset , offset ) , ( 0 , 31 ) , 'r-' )
	offset = plot_week(reporting_time_diffs_per_week[2], offset, True)

	plt.legend()
	plt.xlabel('Index in The Whole Matrix Data Set')
	plt.ylabel('Difference In Minutes')

	print(\
		str(min(itertools.chain.from_iterable(reporting_time_diffs_per_week))) + ' '\
		+str(\
			sum(itertools.chain.from_iterable(reporting_time_diffs_per_week))/\
			len(list(itertools.chain.from_iterable(reporting_time_diffs_per_week)))\
			) + ' '\
		+str(max(itertools.chain.from_iterable(reporting_time_diffs_per_week)))
	)

	print(len(tuple(filter(lambda d: d==1, itertools.chain.from_iterable(reporting_time_diffs_per_week)))))

	plt.show()

def plot_whole_matrices_data_set_2():
	if False:
		_ , cell_diff_per_week =\
		pickle.load(
			open(
				'pipe_0.p',
				'rb'
			)
		)

	if True:
		matrices_per_week_lists = list()

		for a in pickle.load( open( 'complete_data_set_top_80.p' , 'rb' ) ):
			matrices_per_week_lists.append(
				tuple(
					map(
						lambda p: ( p[0] , p[-1] , ),
						a,
					)
				)
			)

		cell_diff_per_week = list()

		for matrix_week in matrices_per_week_lists:

			time_diff_list = list()

			cells_diff = list()

			for i in range( 1 , len(matrix_week) ):

				time_diff_list.append(
					matrix_week[i][0] - matrix_week[i-1][0]
				)

				cells_diff.append(0)

				for p in zip( matrix_week[i][1] , matrix_week[i-1][1] ):
					if p[0] != p[1]:
						cells_diff[-1] += 1

			print('Finished week !')

			cell_diff_per_week.append( cells_diff )

	print(\
		str(min(itertools.chain.from_iterable(cell_diff_per_week))) + ' '\
		+str(\
			sum(itertools.chain.from_iterable(cell_diff_per_week))/\
			len(list(itertools.chain.from_iterable(cell_diff_per_week)))\
			) + ' '\
		+str(max(itertools.chain.from_iterable(cell_diff_per_week)))
	)

	print( len( tuple( filter( lambda e: e == 0 ,  itertools.chain.from_iterable(cell_diff_per_week)) ) ) )

	print( len( tuple( itertools.chain.from_iterable(cell_diff_per_week))))

	def plot_week(week_list, offset, label_on_flag=False):
		if label_on_flag:
			plt.plot(
				range(offset, offset + len(week_list)),
				tuple(
					map(
						lambda e: e,
						week_list
					)
				),
				'bo',
				label='Cell Count Difference Between Matrices'
			)
		else:
			plt.plot(
				range(offset, offset + len(week_list)),
				tuple(
					map(
						lambda e: e,
						week_list
					)
				),
				'bo',
			)
		return offset + len(week_list)

	offset = 0
	offset = plot_week(cell_diff_per_week[0], offset)
	plt.plot( ( offset , offset ) , ( 0 , 11 ) , 'r-' )
	offset = plot_week(cell_diff_per_week[1], offset)
	plt.plot( ( offset , offset ) , ( 0 , 11 ) , 'r-' )
	offset = plot_week(cell_diff_per_week[2], offset, True)

	plt.legend()
	plt.xlabel('Index in The Whole Matrix Data Set')
	plt.ylabel('Difference In # PC Cells')
	plt.show()

def plot_whole_matrices_diff_histogram():
	diff_dict = dict()

	for week_list in pickle.load( open( 'pipe_0.p', 'rb' ) )[0]:
		# for v in filter( lambda d: d <= 3000 , week_list ):
		for v in week_list:
			if v not in diff_dict: diff_dict[v] = 1
			else: diff_dict[v] += 1

	plt.plot(
		# sorted( diff_dict.keys() ),
		sorted( map(lambda e: e / 60000, diff_dict.keys() )),
		tuple(
			map(
				lambda e: diff_dict[e],
				sorted( diff_dict.keys() )
			)
		),
		'bo',
		label='Time Difference Histogram'
	)

	plt.xlabel( 'Time Difference in Miliseconds' )
	plt.ylabel( '# Occurences In Whole M. Set' )

	plt.show()

def plot_cumsum():
	if False:
		pca_engine = load(open( 'pca_engine.p' , 'rb' ))

		a_cumsum = pca_engine.explained_variance_ratio_.cumsum()[:20]
		current_label = 'Old Cumulative Sum'

	if False:
		a_cumsum = [0.98733121,0.99366513,0.99539214,0.99657829,0.99749954,0.99789256\
	    	,0.9982454,0.99854443,0.99877517,0.99896734,0.99914977,0.99928434\
	    	,0.99939343,0.99949583,0.99959095,0.99966874,0.99972528,0.99977632\
	    	,0.99982505,0.99986134]
		# Asta e cu toate cele 17000 de puncte

	if False:
		a_cumsum = [0.64098809,0.75722604,0.82399221,0.87422334,0.90842523,0.92362604\
			,0.93638813,0.94636114,0.95484503,0.96255456,0.96859185,0.97366096\
			,0.97818377,0.98190494,0.98559887,0.98818473,0.99041058,0.99244519\
			,0.99403916,0.99509542]
		# Asta e cu toate cele 14000 de puncte

		current_label = 'Cumulative Sum'

	if True:
		a_cumsum = [0.90573988,0.9376246,0.95241315,0.96608066,0.97426106,0.97839784\
			,0.98177075,0.98463997,0.98698519,0.98920158,0.99093429,0.99241989\
			,0.99370959,0.99476366,0.99578552,0.99653836,0.99718343,0.99776355\
			,0.9982159,0.9985539]
		# Asta e cu toate cele ~1400 de puncte

		current_label = 'Cumulative Sum'

	plt.plot( range(20) ,  a_cumsum , 'bo' , label=current_label)

	plt.legend()
	plt.xlabel('PC index')
	plt.ylabel('Cumulative Variance')
	plt.show()

def plot_cell_diff():
	# diff_list = pickle.load(
	# 	open( 'pipe_pc_cell_diff_4_may.p' , 'rb' )
	# )
	diff_list = pickle.load(
		open( 'pipe_whole_cell_diff_4_may.p' , 'rb' )
	)

	plt.plot(
		range(len(diff_list)),
		diff_list,
		'bo',
		label='Diff in # Matrix Cells'
	)

	plt.legend()
	plt.xlabel('Index')
	plt.ylabel('# Elements Difference')
	plt.show()

def plot_profile_histogram(image_path,ylabel,index):
	import scipy.stats

	if True:
		dump_name = 'tm_tren_ars_pc_per_week_10_may.p'

	if False:
		dump_name = 'complete_data_set_top_80.p'

	if False:
		data_set_per_week_iterable = get_normalized_values_per_week(dump_name)
		data_set =\
			data_set_per_week_iterable[0]\
			+ data_set_per_week_iterable[1]\
			+ data_set_per_week_iterable[2]

	if True:
		data_set = list()
		for week_list in pickle.load(open('./pca_dumps/' + dump_name, 'rb')):
			data_set +=\
				list(\
					map(\
						lambda e: [e[2],] + e[3] + [e[1],],\
						week_list\
					)\
				)

	# del data_set_per_week_iterable

	# print(len(data_set[0]))

	x = np.array(list(map(lambda e: e[index], data_set)))
	y = np.array(list(map(lambda e: e[-1], data_set)))

	best_to_plot_dict = None

	for bins_no in range(5,50):

		means_result = scipy.stats.binned_statistic(
			x,
			[y, y**2],
			bins=bins_no,
			range=(min(x),max(x),),
			statistic='mean'
		)
		means, means2 = means_result.statistic
		standard_deviations = np.sqrt(means2 - means**2)
		bin_edges = means_result.bin_edges
		bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

		if best_to_plot_dict is None or best_to_plot_dict['average_standard_deviation']\
			> sum(standard_deviations) / len(standard_deviations):

			best_to_plot_dict = {
				'average_standard_deviation' : sum(standard_deviations) / len(standard_deviations),\
				'bins_no' : bins_no,
				'bin_centers':bin_centers,
				'standard_deviations':standard_deviations,
				'means':means,
			}

	plt.subplot(211)

	# plt.plot(
	# 	(-1, 1),
	# 	(0,1),
	# 	'ro'
	# )

	plt.plot(x,y,'r+', label='Scatter Plot')

	plt.legend()

	plt.ylabel('Trend')

	plt.subplot(212)

	# plt.plot(
	# 	(-1, 1),
	# 	(0,1),
	# 	'ro'
	# )

	plt.errorbar(
		x=best_to_plot_dict['bin_centers'],
		y=best_to_plot_dict['means'],
		yerr=best_to_plot_dict['standard_deviations'],
		linestyle='none',
		marker='.',
		label='Prof. Hist.'
	)

	print('For index ' + str(index) + ': bin_number=' + str(best_to_plot_dict['bins_no'])\
		+ ' bin_length=' + str((max(x)-min(x))/best_to_plot_dict['bins_no'])\
		+ ' dynamic_range=' + str((min(x),max(x)))
	)

	plt.legend()

	plt.xlabel(ylabel)
	plt.ylabel('Trend')


	fig = plt.gcf()
	fig.set_size_inches(13,8)

	# plt.show()
	plt.savefig(image_path)
	plt.clf()

def plot_training_process_for_encoder_decoder(pickle_file='result_encoder_decoder_0.p'):

	d = pickle.load(open( './pca_results/' + pickle_file, 'rb' ))

	plt.subplot(211)
	plt.plot(
		range(len(d['mape_evolution'][0])),
		list(
			map(
				lambda e: e if e <= 100 else 100,
				d['mape_evolution'][0],
			)
		),
		label='Train MAPE'
	)
	plt.plot(
		range(len(d['mape_evolution'][1])),
		list(
			map(
				lambda e: e if e <= 100 else 100,
				d['mape_evolution'][1],
			)
		),
		label='Valid MAPE'
	)
	plt.legend()
	plt.ylabel('MAPE')
	plt.subplot(212)
	plt.plot(
		range(len(d['mae_evolution'][0])),
		d['mae_evolution'][0],
		label='Train MAE'
	)
	plt.plot(
		range(len(d['mae_evolution'][1])),
		d['mae_evolution'][1],
		label='Train MAE'
	)
	plt.legend()

	plt.xlabel('Epoch Index')

	plt.ylabel('MAE')

	fig = plt.gcf()
	fig.set_size_inches(11,8)

	# plt.show()
	plt.savefig(
		'/home/mircea/Desktop/plots_6_apr/loss_evolution_'\
		+ 'encoder_decoder.png'
	)
	plt.clf()

def dump_encoder_decoder_plot(index, model_name=None):
	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)

	if False:
		encoder_decoder_dict = pickle.load(open(
			'pca_data_sets/7.p',
			'rb'
		))
		encoder_decoder_dict['non_split_data_set'] = np.array( encoder_decoder_dict['non_split_data_set'] )

		x_set, y_set = generate_random_batch_1(
			len( encoder_decoder_dict['valid_indexes'] ),
			40,
			sorted( encoder_decoder_dict['valid_indexes'] ),
			encoder_decoder_dict['non_split_data_set'],
			only_last_flag=False,
			firs_bs_elements_flag=True,
			one_input_flag=True,
		)

		model = keras.models.load_model(
			'pca_multiple_model_folders/models_' + str(index) + '/' + model_name
		)

	loss_gen = csv.reader( open( './pca_csv_folder/losses_' + str(index) + '.csv' , 'rt' ) )

	next(loss_gen)

	trend_train_list, trend_valid_list, matrices_train_list, matrices_valid_list =\
		list(), list(), list(), list()

	for line_list in loss_gen:
		if len(line_list) > 2:
			trend_train_list.append( float( line_list[4] ) )
			trend_valid_list.append( float( line_list[9] ) )

			matrices_train_list.append( float( line_list[2] ) )
			matrices_valid_list.append( float( line_list[7] ) )

	cap_f = lambda l:\
		list(\
			map(\
				lambda e: e if e <= 100 else 100,\
				l,\
			)\
		)

	plt.subplot(211)
	plt.plot(
		range(len(trend_train_list)),
		cap_f(trend_train_list),
		label='Trend Train MAPE'
	)
	plt.plot(
		range(len(trend_valid_list)),
		cap_f(trend_valid_list),
		label='Trend Valid MAPE'
	)
	plt.legend()
	plt.ylabel('MAPE')

	plt.subplot(212)
	plt.plot(
		range(len(matrices_train_list)),
		cap_f(matrices_train_list),
		label='Matrices Train MAPE'
	)
	plt.plot(
		range(len(matrices_valid_list)),
		cap_f(matrices_valid_list),
		label='Matrices Valid MAPE'
	)
	plt.legend()

	plt.xlabel('Epoch Index')

	plt.ylabel('MAPE')

	fig = plt.gcf()
	fig.set_size_inches(11,8)

	plt.savefig(
		'./pca_plots/' + str(index) + '.png'
	)
	plt.clf()

def plot_fidelity(filepath, index):
	fidelity_list = pickle.load(open(filepath,'rb'))


	plt.plot(
		range(len(fidelity_list)),
		tuple(
			map(
				lambda e: e / index,
				fidelity_list
			)
		)
	)
	plt.plot(
		range(len(fidelity_list)),
		tuple(
			map(
				lambda e: e / index,
				fidelity_list
			)
		), 'ro'
	)
	# plt.plot(
	# 	(499,),(1,)
	# )
	plt.xlabel('Training Sample')
	plt.ylabel('Fidelity')
	plt.show()

def plot_q_agent_results():
	a = csv.reader(open('0.csv','rt'))

	next(a)

	a_list = list()

	for l in a:
		a_list.append( float( l[1][1:-1] ) )

	plt.plot( range(len(a_list)) , a_list )

	plt.show()

def dummy_plot():
	a = pickle.load(open('./agent_dumps/reinforce_agent_fidelity.p','rb'))
	plt.xlabel( 'Step Index' )
	plt.ylabel( 'Fidelity' )
	plt.plot(range(len(a)),a)
	plt.show()

def dump_encoder_decoder_valid_results():
	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)

	encoder_decoder_dict = pickle.load(open(
		'pca_data_sets/7.p',
		'rb'
	))
	encoder_decoder_dict['non_split_data_set'] = np.array( encoder_decoder_dict['non_split_data_set'] )
	encoder_decoder_dict['valid_indexes'] = sorted( encoder_decoder_dict['valid_indexes'] )

	from tensorflow.keras.models import load_model

	model = load_model(
		'pca_multiple_model_folders/models_335/model_0324.hdf5'
	)

	batch_size = 800

	x1_list = np.empty( ( batch_size , 40 , 1 ) )

	x2_list = np.empty( (\
		batch_size,\
		40,\
		len( encoder_decoder_dict['non_split_data_set'][0] ) - 2\
	) )

	print('Len of valid indexes list is:' , len( encoder_decoder_dict['valid_indexes'] ))

	gt_list, p_list = list(), list()

	for i_batch in range( 0 , len( encoder_decoder_dict['valid_indexes'] ) , batch_size ):

		print(i_batch)

		for i_inside_batch , i_random in enumerate(\
			encoder_decoder_dict['valid_indexes'][\
				i_batch\
				: ( ( i_batch + batch_size ) if i_batch + batch_size <= len(encoder_decoder_dict['valid_indexes'])\
					else  len(encoder_decoder_dict['valid_indexes']) )] ):

			# print('\t',i_inside_batch)

			x1_list[ i_inside_batch , : , 0 ] = encoder_decoder_dict['non_split_data_set'][\
				i_random - 40 : i_random , 0 ]

			x2_list[ i_inside_batch , : , : ] = encoder_decoder_dict['non_split_data_set'][\
				i_random - 40 : i_random , 1 : -1 ]

			gt_list.append( encoder_decoder_dict[ 'non_split_data_set' ][ i_random - 1 , -1 ] )

		p_list +=\
		list(\
			map(
				lambda e: e[0],
				model.predict( [ x1_list , x2_list ] )[1]
			)
		)

	plt.plot(
		range(len(gt_list)),
		gt_list,
		label='Ground Truth Trend'
	)
	plt.plot(
		range(len(gt_list)),
		p_list[:len(gt_list)],
		label='Prediction'
	)
	plt.xlabel('Index in The Data Set')
	plt.ylabel('Normalized Value')
	plt.legend()

	fig = plt.gcf()
	fig.set_size_inches(11,8)

	plt.savefig('./test.png')

if __name__ == '__main__':
	# plot_training_process(
	# 	'/home/mircea/cern_log_folder/dec_work/nn_perf/'\
	# 	+ sys.argv[1])
	# bar_plot_number_of_trend_value()
	# for fn in os.listdir('/home/mircea/cern_log_folder/dec_work/pca_csv_folder'):
	# 	if int( fn.split('_')[1].split('.')[0] ) >= 206:
	# 		plot_training_process(fn)
	# plot_training_process('losses_12.csv')
	# plot_training_process('losses_237.csv')

	# for ind , fn in sorted(
	# 		map(
	# 			lambda fn: ( int( fn.split('_')[1].split('.')[0] ) , fn , ),
	# 			os.listdir('/home/mircea/cern_log_folder/dec_work/pca_results')
	# 		)
	# 	):
	# 	if ind >= 106:
	# 		plot_network_results_1(fn)

	# 206 242 283
	# plot_training_process('losses_242.csv')
	# plot_training_process('losses_283.csv')
	# plot_network_results_2( 'result_206.p' )
	# plot_network_results_2( 'result_304.p' )
	# plot_training_process('losses_347.csv')
	# plot_training_process('losses_348.csv')
	# plot_network_results_2( 'result_320.p' )
	# plot_network_results_2( 'result_303.p' )

	# print_best_results(206,226)
	# print_best_results(126,146)
	# print_best_results(146,166)
	# print_best_results(166,186)
	# print_best_results(186,206)

	# plot_components_vs_trend()

	# plot_whole_matrices_data_set_1()

	# plot_whole_matrices_diff_histogram()

	# plot_cumsum()

	# plot_profile_histogram('pc_vs_trend_plots/prof_hist_0.jpg', 'ARS', 0)
	# for p in range(1,12):
	# 	plot_profile_histogram(
	# 		'pc_vs_trend_plots/prof_hist_'+str(p)+'.jpg',
	# 		'PC'+str(p),p)

	# plot_training_process_for_encoder_decoder()

	# dump_encoder_decoder_plot( 338 )

	# plot_fidelity('./agent_dumps/intermediate_fidelity/fidelity_over_samples_98.p',98)

	# plot_q_agent_results()

	dump_encoder_decoder_valid_results()