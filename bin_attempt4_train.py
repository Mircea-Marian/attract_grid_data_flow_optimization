from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
# import keras

import multiprocessing as mp
import numpy as np
import random
import pickle
import os
import csv
from datetime import datetime
from bin_attempt4_models import *
import matplotlib.pyplot as plt
from sys import argv
import json

def train_main_0():
	
	global thp_per_week_iter, rs_per_week_iter, window_size, top_indexes_tuple

	tr_ind_iterable, va_ind_iterable = pickle.load(open(\
		'./bin_attempt4_folders/test_train_indexes/combined_train_test_split.p' , 'rb'))
	tr_ind_iterable, va_ind_iterable = list(tr_ind_iterable),list(va_ind_iterable)

	thp_per_week_iter =\
		tuple(
			map(
				lambda ind:\
					tuple(
						map(
							lambda p: p[1],
							pickle.load(\
								open('./corrected_thp_folder/'+str(ind)+'.p', 'rb')
							)
						)
					),
				(0,1,2,3)
			)
		)
	min_ind_per_week, max_ind_per_week = [ None for _ in range(len(thp_per_week_iter)) ],\
		[ None for _ in range(len(thp_per_week_iter)) ]
	for i_of_week, i_in_week, _ in va_ind_iterable:
		if min_ind_per_week[i_of_week] is None or min_ind_per_week[i_of_week] > i_in_week:
			min_ind_per_week[i_of_week] = i_in_week
		if max_ind_per_week[i_of_week] is None or max_ind_per_week[i_of_week] < i_in_week:
			max_ind_per_week[i_of_week] = i_in_week
	for i_of_week, i_in_week, _ in tr_ind_iterable:
		if min_ind_per_week[i_of_week] > i_in_week:
			min_ind_per_week[i_of_week] = i_in_week
		if max_ind_per_week[i_of_week] < i_in_week:
			max_ind_per_week[i_of_week] = i_in_week
	# print(min_ind_per_week,max_ind_per_week)
	min_ind_per_week = tuple( map( lambda v: v - 40 + 1 , min_ind_per_week ) )
	mit = min( map( lambda w_tuple: min( w_tuple[0][ w_tuple[1] : w_tuple[2] + 1 ] ) , zip( thp_per_week_iter , min_ind_per_week , max_ind_per_week ) ) )
	mat = max( map( lambda w_tuple: max( w_tuple[0][ w_tuple[1] : w_tuple[2] + 1 ] ) , zip( thp_per_week_iter , min_ind_per_week , max_ind_per_week ) ) )
	# pickle.dump(
	# 	( mit , mat ),
	# 	open( 'throughput_min_max.p' , 'wb' )
	# )
	# exit()
	thp_per_week_iter = tuple(
		map(
			lambda w:\
				tuple(
					map(
						lambda p: (p - mit) / (mat - mit),
						w
					)
				),
			thp_per_week_iter
		)
	)
	print(min(map(lambda e: min(e),thp_per_week_iter)))
	print(max(map(lambda e: max(e),thp_per_week_iter)))

	rs_per_week_iter =\
		tuple(
			map(
				lambda ind:\
					tuple(
						map(
							lambda p: p[1],
							pickle.load(\
								open('./bin_attempt4_folders/time_tag_cern_read_size/'+str(ind)+'.p', 'rb')
							)
						)
					),
				(0,1,2,3)
			)
		)

	# top_indexes_tuple = pickle.load( open( 'top_1000_indexes.p' , 'rb' ) )

	window_size = 40
	output_time_window = 10

	# with tf.device('/gpu:1'):

	# 	with tf.Session() as sess:

	# 		K.set_session(sess)
	if True:
		if True:

			model = get_model_17(output_time_window)

			# i = -1
			# j = 0
			# while j < 3:
			# 	if old_model.layers[i].name.startswith('bidirectional') or old_model.layers[i].name.startswith('time_distributed'):
			# 		model.layers[i].set_weights(old_model.layers[i].get_weights())
			# 		j+=1
			# 	i-=1

			model.summary()

			# model.compile(
			# 	optimizer=keras.optimizers.Adam(),
			# 	loss='mse',
			# 	metrics=['mae','mean_absolute_percentage_error']
			# )
			model.compile(
				optimizer=keras.optimizers.Adam(),
				loss='mean_absolute_percentage_error',
				metrics=['mae',]
			)


			model.fit_generator(
				batch_generator_6_for_16(\
					tr_ind_iterable,\
					128,
					output_time_window
				),
				steps_per_epoch=(\
					(len(tr_ind_iterable)//128) if len(tr_ind_iterable)%128 == 0 else (1+len(tr_ind_iterable)//128)\
				),
				epochs=2500,
				validation_data=batch_generator_6_for_16(
					va_ind_iterable,
					128,
					output_time_window
				),
				validation_steps=(\
					(len(va_ind_iterable)//128) if len(va_ind_iterable)%128 == 0 else (1+len(va_ind_iterable)//128)\
				),
				callbacks=[
					keras.callbacks.CSVLogger('./local_train_history.csv'),
					keras.callbacks.ModelCheckpoint(
						"./local_train_models/model_{epoch:04d}.hdf5",
						monitor='val_loss',
						save_best_only=True
					),
					keras.callbacks.EarlyStopping(
						monitor='val_loss',
						patience=700,
					)
				]
			)

def nn_train_per_proc(i):
	'''
	Trains a neural network configuration on a GPU. It works inside a
	pool of processes.
	'''
	gpu_string = proc_q.get()

	print(str(i) + ': will start training on ' + gpu_string)

	with tf.device(gpu_string):

		with tf.Session() as sess:

			K.set_session(sess)

			if arguments_iterable[ i ][ CREATE_MODEL_ARGUMENTS ] == '':
				model =\
					functions_dict[\
						arguments_iterable[ i ][ CREATE_MODEL_FUNCTION ]\
					]()
			else:
				model =\
					functions_dict[\
						arguments_iterable[ i ][ CREATE_MODEL_FUNCTION ]\
					](
						arguments_iterable[ i ][ CREATE_MODEL_ARGUMENTS ]\
					)

			model.compile(
				optimizer=keras.optimizers.Adam(),
				loss='mean_absolute_percentage_error',
				metrics=['mae',]
			)

			functions_dict[\
				arguments_iterable[ i ][ FIT_MODEL_FUNCTION ]\
			](
				model ,
				arguments_iterable[ i ][ FIT_MODEL_ARGUMENTS ],
				i
			)

	print(str(i) + ': Finished training !')

	proc_q.put(gpu_string)

def fit_function_0(model,param_string,i):
	param_dict = dict(\
		zip(\
			map(lambda p: p[1], filter(lambda p: p[0] % 2 == 0, enumerate(param_string.split(':')))),\
			map(lambda p: p[1], filter(lambda p: p[0] % 2 == 1, enumerate(param_string.split(':'))))
		)\
	)

	with open( './bin_attempt4_folders/csv_training_configs/' + str(i) + '.csv' , 'wt' ) as conf_f:
		for k,v in param_dict.items():
			conf_f.write( str(k) + ',' + str(v) + '\n')

	models_path = './bin_attempt4_folders/models/model_' + str(i) + '/'
	if os.path.exists(models_path):
		os.system( 'rm -rf ' + models_path )
	os.system( 'mkdir ' + models_path )

	model.fit_generator(
		batch_generator_2(\
			tr_ind_iterable,\
			int( param_dict[ 'train_bs' ] )
		),
		steps_per_epoch=(\
			(len(tr_ind_iterable)//int( param_dict[ 'train_bs' ] )) if len(tr_ind_iterable)%int(\
				param_dict[ 'train_bs' ] ) == 0 else (1+len(tr_ind_iterable)//int( param_dict[ 'train_bs' ] ))\
		),
		epochs=int( param_dict[ 'epochs' ] ),
		validation_data=batch_generator_2(
			va_ind_iterable,
			int( param_dict[ 'valid_bs' ] )
		),
		validation_steps=(\
			(len(va_ind_iterable)//int( param_dict[ 'valid_bs' ] )) if len(va_ind_iterable)%int(\
				param_dict[ 'valid_bs' ] ) == 0 else (1+len(va_ind_iterable)//int( param_dict[ 'valid_bs' ] ))\
		),
		verbose=0,
		callbacks=[
			keras.callbacks.CSVLogger(
				'./bin_attempt4_folders/histories/' + str(i) + '.csv'
			),
			keras.callbacks.ModelCheckpoint(
				models_path + "model_{epoch:04d}.hdf5",
				monitor='val_loss',
				save_best_only=True
			),
			keras.callbacks.EarlyStopping(
				monitor='val_loss',
				patience=int( param_dict['patience'] ),
			)
		]
	)

def set_const_argument_tags():
	global\
		CREATE_MODEL_FUNCTION,\
		CREATE_MODEL_ARGUMENTS,\
		FIT_MODEL_FUNCTION,\
		FIT_MODEL_ARGUMENTS,\
		functions_dict

	functions_dict = {
		get_model_0.__name__ : get_model_0,
		get_model_1.__name__ : get_model_1,
		get_model_2.__name__ : get_model_2,
		get_model_3.__name__ : get_model_3,
		get_model_4.__name__ : get_model_4,
		get_model_5.__name__ : get_model_5,
		get_model_6.__name__ : get_model_6,
		get_model_7.__name__ : get_model_7,
		fit_function_0.__name__ : fit_function_0,
	}

	CREATE_MODEL_FUNCTION,\
	CREATE_MODEL_ARGUMENTS,\
	FIT_MODEL_FUNCTION,\
	FIT_MODEL_ARGUMENTS\
		=\
		'CREATE_MODEL_FUNCTION',\
		'CREATE_MODEL_ARGUMENTS',\
		'FIT_MODEL_FUNCTION',\
		'FIT_MODEL_ARGUMENTS'

def neural_tework_grid_search():
	global arguments_iterable,proc_q,thp_per_week_iter, rs_per_week_iter, window_size, tr_ind_iterable,va_ind_iterable

	# Backup step
	bfn = 'bin_attempt4_folders/backup/date_time_' +  datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + '/'
	os.system( 'mkdir ' + bfn )
	os.system( 'cp -r bin_attempt4_folders/csv_training_configs ' + bfn )
	os.system( 'cp -r bin_attempt4_folders/histories ' + bfn )
	os.system( 'cp -r bin_attempt4_folders/models ' + bfn )

	set_const_argument_tags()

	tr_ind_iterable, va_ind_iterable = pickle.load(open(\
		'./bin_attempt4_folders/test_train_indexes/combined_train_test_split.p' , 'rb'))
	tr_ind_iterable, va_ind_iterable = list(tr_ind_iterable),list(va_ind_iterable)

	thp_per_week_iter =\
		tuple(
			map(
				lambda ind:\
					tuple(
						map(
							lambda p: p[1],
							pickle.load(\
								open('./corrected_thp_folder/'+str(ind)+'.p', 'rb')
							)
						)
					),
				(0,1,2,3)
			)
		)
	mit = min( map( lambda w: min( w  ) , thp_per_week_iter ) )
	mat = max( map( lambda w: max( w  ) , thp_per_week_iter ) )
	thp_per_week_iter = tuple(
		map(
			lambda w:\
				tuple(
					map(
						lambda p: 2 * (p - mit) / (mat - mit) - 1,
						w
					)
				),
			thp_per_week_iter
		)
	)

	rs_per_week_iter =\
		tuple(
			map(
				lambda ind:\
					tuple(
						map(
							lambda p: p[1],
							pickle.load(\
								open('./bin_attempt4_folders/time_tag_cern_read_size/'+str(ind)+'.p', 'rb')
							)
						)
					),
				(0,1,2,3)
			)
		)

	window_size = 40

	csv_cfg_iterable = tuple(csv.reader( open( 'bin_attempt4_train_configs.csv' , 'rt' ) ) )
	print(csv_cfg_iterable)

	arguments_iterable = tuple(
		map(
			lambda l:\
				dict(\
					zip(\
						map(lambda p: p[1], filter(lambda p: p[0] % 2 == 0, enumerate(l))),\
						map(lambda p: p[1], filter(lambda p: p[0] % 2 == 1, enumerate(l)))
					)\
				),
			csv_cfg_iterable,
		)
	)
	arguments_iterable = {\
		4 : arguments_iterable[0],
		5 : arguments_iterable[1],
		6 : arguments_iterable[2],
	}
	indexes_iterable = (4,5,6)

	proc_q = mp.Queue()
	a = 0
	for gpu_string in ('/gpu:0','/gpu:1','/gpu:2','/gpu:3'):
		proc_q.put( gpu_string )
		a+=1

	print('Will start pool !')

	mp.Pool( a ).map(
		nn_train_per_proc,
		indexes_iterable
	)

def plot_corr():
	thp_per_week_iter =\
		tuple(
			map(
				lambda ind:\
					pickle.load(\
								open('./corrected_thp_folder/'+str(ind)+'.p', 'rb')
							),
				(0,1,2,3)
			)
		)
	mit = min( map( lambda w: min( w , key=lambda p: p[1] ) , thp_per_week_iter ) )[1]
	mat = max( map( lambda w: max( w , key=lambda p: p[1] ) , thp_per_week_iter ) )[1]
	thp_per_week_iter = tuple(
		map(
			lambda w:\
				tuple(
					map(
						lambda p: ( p[ 0 ] , 2 * ( p[ 1 ] - mit ) / ( mat - mit ) - 1 ) , w
					)
				),
			thp_per_week_iter
		)
	)

	rs_per_week_iter =\
		tuple(
			map(
				lambda ind:\
					dict(
						pickle.load(\
							open('./bin_attempt4_folders/time_tag_cern_read_size/'+str(ind)+'.p', 'rb')
						)
					),
				(0,1,2,3)
			)
		)

	# a =\
	# 	(
	# 		(min(thp_per_week_iter[0],key=lambda e:e[0])[0],'min_thp'),\
	# 		(max(thp_per_week_iter[0],key=lambda e:e[0])[0],'max_thp'),\
	# 		(min(rs_per_week_iter[0].keys()),'min_rs'),\
	# 		(max(rs_per_week_iter[0].keys()),'max_rs')\
	# 	)
	# for i in sorted(a): print(i)
	# exit(0)

	# for i , ( tm , rs ) in enumerate(rs_per_week_iter[0][1:]):
	# 	if tm - rs_per_week_iter[0][i][0] != 1000:
	# 		print('Is diff !')
	# 		exit(0)
	count = 0

	score_list = [ 0 for _ in range(4800) ]
	for thp_week, rs_week in zip(thp_per_week_iter,rs_per_week_iter):
		for tm, t in thp_week:
			if tm - 4800000 in rs_week and tm - 1000 in rs_week:
				count += 1
				for j , i_tm in zip( range(4800) , range( tm - 4800000 , tm , 1000 ) ):
					score_list[j] += t * rs_week[ i_tm ]

	top_indexes = tuple(map(lambda pp: pp[0],
		sorted( zip( range(-4800,0) , score_list ) , key=lambda p:p[1] , reverse=True )[:1000]))

	print(top_indexes)

	pickle.dump(
		top_indexes, open( 'top_1000_indexes.p' , 'wb' )
	)

	plt.plot( range(-4800,0) , tuple(map(lambda p: p/count, score_list)) , 'bo')
	plt.show()

def dump_ground_truth_vs_prediction_0():
	thp_per_week_iter =\
		tuple(
			map(
				lambda ind:\
					pickle.load(\
								open('./corrected_thp_folder/'+str(ind)+'.p', 'rb')
							),
				(0,1,2,3)
			)
		)
	# mit = min( map( lambda w: min( w , key=lambda p: p[1] )[1] , thp_per_week_iter ) )
	# mat = max( map( lambda w: max( w , key=lambda p: p[1] )[1] , thp_per_week_iter ) )
	# thp_per_week_iter = tuple(
	# 	map(
	# 		lambda w:\
	# 			tuple(
	# 				map(
	# 					lambda p: ( p[ 0 ] , 2 * ( p[ 1 ] - mit ) / ( mat - mit ) - 1 ) , w
	# 				)
	# 			),
	# 		thp_per_week_iter
	# 	)
	# )

	rs_per_week_iter =\
		tuple(
			map(
				lambda ind:\
					dict(
						pickle.load(\
							open('./bin_attempt4_folders/time_tag_cern_read_size/'+str(ind)+'.p', 'rb')
						)
					),
				(0,1,2,3)
			)
		)

	batch_size = 1024
	window_size = 40
	model = keras.models.load_model( 'local_train_models/model_1573.hdf5' )

	x_arr = np.empty((
		batch_size , window_size , 4800
	))
	ind_arr = -1 * np.ones( ( batch_size , ) )
	# y_arr = np.empty((
	# 	batch_size, window_size , 1
	# ))
	i_arr = 0
	gt_per_week_iter, pred_per_week_iter = list(),list()
	i_of_week = 0
	for thp_week, rs_week in zip(thp_per_week_iter,rs_per_week_iter):
		gt_per_week_iter.append( list() )
		pred_per_week_iter.append( list() )
		for ind, tm, t in map( lambda p: ( p[0] + window_size - 1 , p[1][0] , p[1][1] ) , enumerate(thp_week[window_size-1:]) ):
			if ind % batch_size == 0:
				print( i_of_week , ind , len(thp_week) )

			if thp_week[ind - window_size + 1][0] - 4800000 in rs_week and tm - 1000 in rs_week:
				if i_arr == batch_size:
					y_pred_arr = model.predict( x_arr )

					for i_in_batch , i_in_thp in zip( range( batch_size ) , ind_arr ):
						gt_per_week_iter[-1].append( thp_week[ int(i_in_thp) ][ 1 ] )
						pred_per_week_iter[-1].append( y_pred_arr[ i_in_batch , window_size - 1 , 0 ] )

					i_arr = 0
					ind_arr = -1 * np.ones( ( batch_size , ) )

				for prev_tm , i_in_window in zip(\
						map( lambda a: thp_week[a][0] , range( ind - window_size + 1 , ind + 1 ) ),\
						range( window_size )
					):
					for rs_tm , i_in_rs in zip(
							range( prev_tm - 4800000 , prev_tm , 1000 ),
							range( 4800 )
						):
						x_arr[ i_arr , i_in_window , i_in_rs ] = rs_week[ rs_tm ]
				ind_arr[i_arr] = ind

				i_arr += 1

		i_of_week += 1

		y_pred_arr = model.predict( x_arr )

		for i_in_batch , i_in_thp in zip( range( batch_size ) , ind_arr ):
			if i_in_thp == -1:
				break
			gt_per_week_iter[-1].append( thp_week[ int(i_in_thp) ][ 1 ] )
			pred_per_week_iter[-1].append( y_pred_arr[ i_in_batch , window_size - 1 , 0 ] )

	pickle.dump( (gt_per_week_iter,pred_per_week_iter) ,  open('a.p','wb') )

def plot_ground_truth_vs_prediction_0():
	a = pickle.load(open( 'a.p' , 'rb' ))

	print(
		'MAPE =',
		sum(
			map(
				lambda week_tuple:\
					sum(
						map(
							lambda v_tuple:
								100 * abs( v_tuple[0] - v_tuple[1] ) /  v_tuple[0],
							zip(week_tuple[0],week_tuple[1])
						)
					),
				zip(
					a[0],
					a[1]
				)
			)
		) / sum( map( lambda e: len(e) , a[0] ) )
	)

	plt.plot(range(len(a[0][0])),a[0][0],'b-',label='Ground Truth')
	plt.plot(range(len(a[1][0])),a[1][0],'r-',label='Prediction')
	# plt.plot(range(len(a[0][0])),a[0][0],'bo')
	# plt.plot(range(len(a[1][0])),a[1][0],'ro')
	offset = len(a[0][0])
	for gt, p in zip( a[0][1:] , a[1][1:] ):
		plt.plot(range(offset,offset+len(gt)),gt,'b-')
		plt.plot(range(offset,offset+len(p)),p,'r-')
		# plt.plot(range(offset,offset+len(gt)),gt,'bo')
		# plt.plot(range(offset,offset+len(p)),p,'ro')
		offset += len(p)
	plt.legend()
	plt.show()

def plot_norm_thp():
	tr_ind_iterable, va_ind_iterable = pickle.load(open(\
		'./bin_attempt4_folders/test_train_indexes/combined_train_test_split.p' , 'rb'))
	tr_ind_iterable, va_ind_iterable = list(tr_ind_iterable),list(va_ind_iterable)
	thp_per_week_iter =\
		tuple(
			map(
				lambda ind:\
					tuple(
						map(
							lambda p: p[1],
							pickle.load(\
								open('./corrected_thp_folder/'+str(ind)+'.p', 'rb')
							)
						)
					),
				(0,1,2,3)
			)
		)
	min_ind_per_week, max_ind_per_week = [ None for _ in range(len(thp_per_week_iter)) ],\
		[ None for _ in range(len(thp_per_week_iter)) ]
	for i_of_week, i_in_week, _ in va_ind_iterable:
		if min_ind_per_week[i_of_week] is None or min_ind_per_week[i_of_week] > i_in_week:
			min_ind_per_week[i_of_week] = i_in_week
		if max_ind_per_week[i_of_week] is None or max_ind_per_week[i_of_week] < i_in_week:
			max_ind_per_week[i_of_week] = i_in_week
	for i_of_week, i_in_week, _ in tr_ind_iterable:
		if min_ind_per_week[i_of_week] > i_in_week:
			min_ind_per_week[i_of_week] = i_in_week
		if max_ind_per_week[i_of_week] < i_in_week:
			max_ind_per_week[i_of_week] = i_in_week
	print(min_ind_per_week, max_ind_per_week )
	min_ind_per_week = tuple( map( lambda v: v - 40 + 1 , min_ind_per_week ) )
	mit = min( map( lambda w_tuple: min( w_tuple[0][ w_tuple[1] : w_tuple[2] + 1 ] ) , zip( thp_per_week_iter , min_ind_per_week , max_ind_per_week ) ) )
	mat = max( map( lambda w_tuple: max( w_tuple[0][ w_tuple[1] : w_tuple[2] + 1 ] ) , zip( thp_per_week_iter , min_ind_per_week , max_ind_per_week ) ) )
	thp_per_week_iter = tuple(
		map(
			lambda w:\
				tuple(
					map(
						lambda p: 2 * (p - mit) / (mat - mit) - 1,
						w
					)
				),
			thp_per_week_iter
		)
	)

	plt.plot( 
		range( len(thp_per_week_iter[0]) + len(thp_per_week_iter[1]) + len(thp_per_week_iter[2]) + len(thp_per_week_iter[3]) ),
		thp_per_week_iter[0] + thp_per_week_iter[1] + thp_per_week_iter[2] + thp_per_week_iter[3],
		'bo'
	)
	offset = len(thp_per_week_iter[0])
	plt.plot(
		( offset , offset ),
		(-1,1),
		'r-'
	)
	offset += len(thp_per_week_iter[1])
	plt.plot(
		( offset , offset ),
		(-1,1),
		'r-'
	)
	offset += len(thp_per_week_iter[2])
	plt.plot(
		( offset , offset ),
		(-1,1),
		'r-'
	)
	plt.show()

def plot_feature_vs_throughput(negative_feature_index):
	thp_to_plot, rs_to_plot = list(), list()
	for thp_iterable, rs_dict in zip(
			map(
				lambda ind:\
					pickle.load(\
						open('./corrected_thp_folder/'+str(ind)+'.p', 'rb')
					),
				(0,1,2,3)
			),
			map(
				lambda ind:\
					dict(
						pickle.load(\
							open('./bin_attempt4_folders/time_tag_cern_read_size/'+str(ind)+'.p', 'rb')
						)
					),
				(0,1,2,3)
			)
		):
		i = 0
		while thp_iterable[i][0] - negative_feature_index * 1000 not in rs_dict:
			i+=1
		while i < len(thp_iterable) and thp_iterable[i][0] - negative_feature_index * 1000 in rs_dict:
			thp_to_plot.append( thp_iterable[i][1] )
			rs_to_plot.append( rs_dict[ thp_iterable[i][0] - negative_feature_index * 1000 ] )
			i+=1

	print('Finished data agreggation !')

	mi, ma = min( thp_to_plot ), max(thp_to_plot)
	plt.plot(
		range(len(thp_to_plot)),
		tuple(
			map(
				lambda e: ( e - mi ) / ( ma - mi ),
				thp_to_plot,
			)
		)
	)

	mi, ma = min( rs_to_plot ), max(rs_to_plot)
	plt.plot(
		range(len(rs_to_plot)),
		tuple(
			map(
				lambda e: ( e - mi ) / ( ma - mi ),
				rs_to_plot,
			)
		)
	)

	plt.show()

def svm_train():
	tr_ind_iterable, va_ind_iterable = pickle.load(open(\
		'./bin_attempt4_folders/test_train_indexes/combined_train_test_split.p' , 'rb'))
	tr_ind_iterable, va_ind_iterable = list(tr_ind_iterable),list(va_ind_iterable)

	thp_per_week_iter =\
		tuple(
			map(
				lambda ind:\
					tuple(
						map(
							lambda p: p[1],
							pickle.load(\
								open('./corrected_thp_folder/'+str(ind)+'.p', 'rb')
							)
						)
					),
				(0,1,2,3)
			)
		)

	rs_per_week_iter =\
		tuple(
			map(
				lambda ind:\
					tuple(
						map(
							lambda p: p[1],
							pickle.load(\
								open('./bin_attempt4_folders/time_tag_cern_read_size/'+str(ind)+'.p', 'rb')
							)
						)
					),
				(0,1,2,3)
			)
		)

	window_size = 40

	x_tr, y_tr = np.empty( ( len( tr_ind_iterable ) , 4839 , ) ) , np.empty( ( len( tr_ind_iterable ) , ) )

	for we_i, ( i_week , i_per_week, rs_ind_iter ) in enumerate(tr_ind_iterable):
		if we_i % 1000 == 0:
			print(we_i,len(tr_ind_iterable))
		x_tr[we_i,:4800] = rs_per_week_iter[ i_week ][ rs_ind_iter[-1] - 4799 : rs_ind_iter[-1] + 1 ]
		x_tr[we_i,4800:] = thp_per_week_iter[ i_week ][ i_per_week - window_size + 1 : i_per_week ]
		y_tr[we_i] = thp_per_week_iter[ i_week ][ i_per_week ]

	del tr_ind_iterable, va_ind_iterable, thp_per_week_iter, rs_per_week_iter

	print('Ready to train !')

	from sklearn.svm import SVR

	svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1,verbose=True)

	# svr_rbf.fit( x_tr[:200] , y_tr[:200] )
	svr_rbf.fit( x_tr , y_tr )

	pickle.dump(
		svr_rbf,
		open('my_svr.p','wb')
	)

def plot_ground_truth_vs_prediction_1_for_svr():
	thp_per_week_iter =\
		tuple(
			map(
				lambda ind:\
					pickle.load(\
								open('./corrected_thp_folder/'+str(ind)+'.p', 'rb')
							),
				(0,1,2,3)
			)
		)

	rs_per_week_iter =\
		tuple(
			map(
				lambda ind:\
					dict(
						pickle.load(\
							open('./bin_attempt4_folders/time_tag_cern_read_size/'+str(ind)+'.p', 'rb')
						)
					),
				(0,1,2,3)
			)
		)

	for_pred_input = list()
	gt_list = list()
	ii = -1
	for thp_week, rs_week in zip(thp_per_week_iter,rs_per_week_iter):
		for ind, (tm, thp) in enumerate(thp_week[39:]):
			if ii == 200:
				break
			if tm - 4800000 in rs_week and tm - 1000 in rs_week:
				ii+=1
				if ii % 1000 == 0:
					print( ii )
				gt_list.append( thp )
				for_pred_input.append(
					list(
						map(
							lambda rs_tm: rs_week[ rs_tm ],
							range( tm - 4800000 , tm , 1000 )
						)
					)\
					+ list(
						map(
							lambda t_p: t_p[1],
							thp_week[ind:ind+39]
						)
					)
				)

	rez = pickle.load( open('my_svr.p','rb') ).predict( for_pred_input )

	pickle.dump( (gt_list,rez) ,  open('a_svr.p','wb') )

	plt.plot( range( len(gt_list) )  , gt_list)
	plt.plot( range( len(gt_list) )  , rez )
	plt.show()

def plot_ground_truth_vs_prediction_2_for_15():
	thp_per_week_iter =\
		tuple(
			map(
				lambda ind:\
					pickle.load(\
								open('./corrected_thp_folder/'+str(ind)+'.p', 'rb')
							),
				(0,1,2,3)
			)
		)

	rs_per_week_iter =\
		tuple(
			map(
				lambda ind:\
					dict(
						pickle.load(\
							open('./bin_attempt4_folders/time_tag_cern_read_size/'+str(ind)+'.p', 'rb')
						)
					),
				(0,1,2,3)
			)
		)

	mit,mat = pickle.load(open('throughput_min_max.p','rb'))

	window_size = 40

	for_pred_input = list()
	gt_list = list()
	# ii = -1
	for thp_week, rs_week in zip(thp_per_week_iter,rs_per_week_iter):
		for ind, (tm, thp) in enumerate(thp_week[window_size-1:]):
			# if ii == 200:
				# break
			if thp_week[ind][0] - 120000 in rs_week and tm - 1000 in rs_week:
				for_pred_input.append( list() )
				for j in range( ind , ind + window_size - 1 ):
					for_pred_input[-1].append(
						list(
							map(
								lambda tm_e: rs_week[tm_e],
								range( thp_week[j][0] - 120000 , thp_week[j][0] , 1000 )
							)
						) + [ ( thp_week[j][1] - mit ) / ( mat - mit ) , ]
					)
				gt_list.append( ( thp - mit ) / ( mat - mit ) )

	model = keras.models.load_model( 'local_train_models/model_0417.hdf5' )

	model.summary()

	rez = model.predict( np.array( for_pred_input ) )
	rez = tuple( map( lambda e: rez[e,0] , range(len(gt_list)) ) )

	print( model.evaluate(np.array(\
		for_pred_input ), np.expand_dims( np.array(gt_list) , -1 ), batch_size=128) )

	plt.plot(
		range(len(gt_list)),
		gt_list,
		label='Ground Truth'
	)

	plt.plot(
		range(len(gt_list)),
		rez,
		label='Prediction'
	)

	plt.legend()

	plt.show()

def helper_for_gt_vs_pred_0(thp_per_week_iter,rs_per_week_iter):
	mit,mat = pickle.load(open('throughput_min_max.p','rb'))

	window_size = 40
	output_time_window = 10

	model = keras.models.load_model( 'local_train_models/model_0601.hdf5' )

	encoder_input, decoder_input, gt_list, pred_list = list(),list(),list(),list()

	total_count = 0

	for thp_week, rs_week in zip(thp_per_week_iter,rs_per_week_iter):
		for ind, (tm, thp) in enumerate(thp_week[window_size-1:len(thp_week)-output_time_window]):
			if thp_week[ind][0] - 120000 in rs_week and tm - 1000 in rs_week:
				encoder_input.append( list() )
				for j in range( ind , ind + window_size - 1 ):
					encoder_input[-1].append(
						list(
							map(
								lambda tm_e: rs_week[ tm_e ] if tm_e in rs_week else -1,
								range( thp_week[j][0] - 120000 , thp_week[j][0] , 1000 )
							)
						) + [ ( thp_week[j][1] - mit ) / ( mat - mit ) , ]
					)

				decoder_input.append(
					[ ( thp_week[ ind + window_size - 2 ][1] - mit ) / ( mat - mit ) , ]\
					+ [ 0 for _ in range(output_time_window-1) ]
				)

				gt_list.append( [ ( thp - mit ) / ( mat - mit ) , ] )

				total_count += 1

	print(total_count)

	encoder_input = np.array(encoder_input)
	decoder_input = np.expand_dims( np.array(decoder_input) , -1 )

	for r in model.predict( [ encoder_input , decoder_input ] ):
		pred_list.append( [ r[0,0] , ] )

	for i_output in range(1,output_time_window):
		i_dec = 0
		for thp_week, rs_week in zip(thp_per_week_iter,rs_per_week_iter):
			for ind , ( tm , _ ) in enumerate(thp_week[window_size-1:len(thp_week)-output_time_window]):
				if thp_week[ind][0] - 120000 in rs_week and tm - 1000 in rs_week:
					decoder_input[ i_dec , i_output , 0 ] = pred_list[i_dec][-1]
					gt_list[i_dec].append( 
						( thp_week[ ind + window_size - 1 + i_output ][1] - mit ) / ( mat - mit )
					)
					i_dec += 1
		for i,r in enumerate( model.predict( [ encoder_input , decoder_input ] ) ):
			pred_list[i].append( r[ i_output , 0 ] )

	pickle.dump(
		( gt_list , pred_list ) , open( 'a0123.p' , 'wb' )
	)

def dump_ground_truth_vs_prediction_3_for_16():
	thp_per_week_iter =\
		tuple(
			map(
				lambda ind:\
					pickle.load(\
								open('./corrected_thp_folder/'+str(ind)+'.p', 'rb')
							),
				( 0 , 1, 2, 3 , )
				# ( 4 , )
			)
		)

	rs_per_week_iter =\
		tuple(
			map(
				lambda ind:\
					dict(
						pickle.load(\
						# json.load(\
							open('./bin_attempt4_folders/time_tag_cern_read_size/'+str(ind)+'.p', 'rb')
							# open('./bin_attempt4_folders/time_tag_cern_read_size/'+str(ind)+'.json', 'rt')
						)
					),
				( 0 , 1, 2, 3 , )
				# ( 4 , )
			)
		)
	helper_for_gt_vs_pred_0( thp_per_week_iter , rs_per_week_iter )

def plot_ground_truth_vs_prediction_3_for_16(i,d_name):
	fig = plt.gcf()
	fig.set_size_inches(11,8)
	gt, pred = pickle.load(open(d_name,'rb'))
	gt,pred =\
		tuple(map(lambda e: e[i],gt)),\
		tuple(map(lambda e: e[i],pred))
	plt.plot(
		range(len(gt)),
		gt,
		label='Ground Truth'
	)

	print(len(gt))

	plt.plot(
		range(len(pred)),
		pred,
		label='Prediction'
	)

	plt.xlabel('Index in Data Set',fontweight='bold')
	plt.ylabel('Normalized Value',fontweight='bold')

	# print(
	# 	'MAPE =',
	# 	sum(
	# 		map(
	# 			# lambda p: abs( p[ 0 ] - p[ 1 ] ) / ( 1e-7 if 1e-7 > p[ 0 ] else p[ 0 ] ),
	# 			lambda p: abs( p[ 0 ] - p[ 1 ] ) / p[ 0 ],
	# 			zip( gt , pred )
	# 		)
	# 	) / len( gt )
	# )
	print(
		'MAPE =',
		sum(
			map(
				lambda p: abs( p[ 0 ] - p[ 1 ] ) / p[ 0 ],
				filter(
					lambda pp: pp[0] != 0,
					zip( gt , pred )
				)
			)
		) / len( gt )
	)


	plt.legend()

	plt.show()
	# plt.savefig( '/home/mircea/Desktop/prezentare_grid_guys/new_data_' + str(i) + '.png' )
	# plt.clf()

def plot_MAPE_for_output(d_name):
	gt, pred = pickle.load(open(d_name,'rb'))
	def get_mape_0():
		return tuple(map(
			lambda i:\
				sum(
					map(
						lambda p: abs( p[ 0 ] - p[ 1 ] ) / ( 1e-7 if 1e-7 > p[ 0 ] else p[ 0 ] ),
						# lambda p: abs( p[ 0 ] - p[ 1 ] ) / p[ 0 ],
						# lambda p: ( abs( 1e-7 - p[ 1 ] ) / 1e-7 ) if 1e-7 > p[0] else ( abs( p[ 0 ] - p[ 1 ] ) / p[ 0 ] ),
						# lambda p: ( p[ 0 ] - p[ 1 ] ) * ( p[ 0 ] - p[ 1 ] ),
						zip(
							map( lambda e: e[ i ] , gt ),
							map( lambda e: e[ i ] , pred )
						)
					)
				) / len( gt ),
			range(10)
		))
	def get_mape_1():
		v_list = []
		for i in range(10):
			y_true = np.array( tuple( map( lambda e: e[ i ] , gt ) ) )
			y_true = np.maximum(y_true, 1e-7)
			y_pred = np.array( tuple( map( lambda e: e[ i ] , pred ) ) )
			v_list.append(100. * np.mean(np.abs((y_true - y_pred) / y_true)))
		return v_list

	mape_list = get_mape_1()

	for i,e in enumerate( mape_list ):
		print(i+1,"{:.2f}".format(e))

	# plt.plot(
	# 	range(10),
	# 	mape_list,
	# 	'bo',
	# 	label='Mean Average Percentage Error'
	# )

	# plt.xlabel( 'Output Time Step Index' )
	# plt.ylabel( 'MAPE' )

	# plt.show()

	return mape_list

def normalize_read_sizes():
	mi,ma = pickle.load( open(\
		'bin_attempt4_folders/time_tag_cern_read_size/min_max_read_size.p',\
		'rb'
	) )

	json.dump(
		tuple(
			map(
				lambda p:\
					(
						p[0],
						2 * ( p[1] - mi ) / ( ma - mi ) - 1,
					),
				json.load(
					open(
						'bin_attempt4_folders/time_tag_cern_read_size/4.json',
						'rt'
					)
				)
			)
		),
		open(
			'bin_attempt4_folders/time_tag_cern_read_size/4_1.json',
			'wt'
		)
	)

def check_gaps_in_rs():
	rs_per_week_iter =\
		tuple(
			map(
				lambda ind:\
						# pickle.load(\
						json.load(\
							# open('./bin_attempt4_folders/time_tag_cern_read_size/'+str(ind)+'.p', 'rb')
							open('./bin_attempt4_folders/time_tag_cern_read_size/'+str(ind)+'.json', 'rt')
						),
				( 4 , )
			)
		)[0]
	min_tm, max_tm = rs_per_week_iter[0][0], rs_per_week_iter[-1][0]
	l = len(rs_per_week_iter)
	c = 0
	rs_per_week_iter = dict(rs_per_week_iter)
	for tm in range(min_tm,max_tm,1000):
		if tm not in rs_per_week_iter:
			c+=1
	print(c)
	print(l)

def plot_only_first_and_last_sections(d_name, top_diff_count):
	gt, pred = pickle.load(open(d_name,'rb'))
	gt_0,pred_0 =\
		tuple(map(lambda e: e[0],gt)),\
		tuple(map(lambda e: e[0],pred))
	gt_9,pred_9 =\
		tuple(map(lambda e: e[9],gt)),\
		tuple(map(lambda e: e[9],pred))
	a_list = list()
	for i in range(1,len(gt_9)):
		a_list.append(
			(
				abs( gt_9[i] -  pred_9[i] ),
				i
			)
		)
	a_list.sort(reverse=True)

	plt.plot(
		range(len(gt_0)),
		gt_0,
		label='Ground Truth'
	)

	plt.plot(
		range(len(pred_0)),
		pred_0,
		label='Prediction 0'
	)

	plt.plot(
		(
			a_list[0][1] - 1,
			a_list[0][1],
		),
		(
			pred_9[ a_list[0][1] - 1 ],
			pred_9[ a_list[0][1] ],
		),
		'r-',
		label='Prediction 9'
	)

	for i in range(1,top_diff_count):
		plt.plot(
			(
				a_list[i][1] - 1,
				a_list[i][1],
			),
			(
				pred_9[ a_list[i][1] - 1 ],
				pred_9[ a_list[i][1] ],
			),
			'r-'
		)

	plt.legend()

	plt.show()

def get_performance_per_proc( index ):
	if type(index) == int:
		a = csv.reader( open( './bin_attempt4_folders/histories/' + str(index) + '.csv' , 'rt' ) )
	else:
		a = csv.reader( open( './bin_attempt4_folders/histories/' + index , 'rt' ) )
	next(a)
	return min( map( lambda e: ( float( e[ -2 ] ) , float( e[ -4 ] ) ) , a ) )

def get_performance( index_tuple ):
	for ind, r in zip(index_tuple,mp.Pool(n_proc).map( get_performance_per_proc , index_tuple )):
		print(ind,'tr_perc =',r[1],';','va_perc =',r[0],'overall=',0.8*r[1]+0.2*r[0])

def plot_ground_truth_vs_prediction_3_for_set_27_wrapper(i, ind):
	plot_ground_truth_vs_prediction_3_for_16(
		int(i) , 'bin_attempt4_folders/predictions/'+ind+'_tvds.p'
	)

def plot_MAPE_for_output_for_set_27_wrapper(ind):
	plot_MAPE_for_output(
		'bin_attempt4_folders/predictions/'+ind+'_tvds.p'
	)

def print_model_summary(number):
	keras.models.load_model(
			'./bin_attempt4_folders/models/model_'+number+'/model_0001.hdf5'\
			).summary()


if __name__ == '__main__':
	global n_proc
	n_proc = 20
	# neural_tework_grid_search()
	# train_main_0()
	# plot_corr()

	# global window_size
	# window_size = 40
	# get_model_6().summary()

	# dump_ground_truth_vs_prediction_0()
	# plot_ground_truth_vs_prediction_0()
	# plot_norm_thp()
	# plot_feature_vs_throughput(-1019)

	# svm_train()
	# plot_ground_truth_vs_prediction_1_for_svr()

	# plot_ground_truth_vs_prediction_2()

	# dump_ground_truth_vs_prediction_3_for_16()
	# plot_ground_truth_vs_prediction_3_for_16(int(argv[1]),argv[2])
	# plot_MAPE_for_output(argv[1])

	# normalize_read_sizes()
	# check_gaps_in_rs()

	# for i in range( 10 ):
		# plot_ground_truth_vs_prediction_3_for_16( i , 'a4.p' )

	# plot_only_first_and_last_sections('a0123.p',1000)

	# get_performance( tuple(range(17,30)) + (31,) )
	# plot_ground_truth_vs_prediction_3_for_set_27_wrapper(argv[1] , argv[2])
	# plot_MAPE_for_output_for_set_27_wrapper(argv[1])
	# from bin_attempt4_arguments_0 import get_argument_dict_27_07
	# get_argument_dict_27_07()
	# for i in range(17,42):
	# 	print('\n')
	# 	print(i)
	# 	plot_MAPE_for_output_for_set_27_wrapper(str(i))
	print_model_summary(argv[1])