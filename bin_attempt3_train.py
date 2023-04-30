from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
import functools
import os
import pickle
import numpy as np
import random
import json
from csv import reader
import matplotlib.pyplot as plt
import multiprocessing as mp
import csv
from datetime import datetime

def batch_generator(indexes,batch_size):	
	while True:
		x_arr = np.empty((
			batch_size , window_size , se_len
		))
		y_arr = np.empty((
			batch_size, window_size , 1
		))
		
		for bs_i , ( w , bs_j ) in enumerate(random.sample(indexes, batch_size)):

			# print( w , bs_j - window_size + 1 , bs_j + 1 , len( x_data_set[ w ] ) )

			# for ii,rsi,t in zip(\
			# 			range(window_size),
			# 			x_data_set[ w ][ bs_j - window_size + 1 : bs_j + 1 ],
			# 			y_data_set[ w ][ bs_j - window_size + 1 : bs_j + 1 ],
			# 		):
			# 	x_arr[ bs_i , ii , : ] = rsi
			# 	y_arr[ bs_i , ii , 0 ] = t

			x_arr[bs_i] = x_data_set[ w ][ bs_j - window_size + 1 : bs_j + 1 ]
			y_arr[bs_i,:,0] = y_data_set[ w ][ bs_j - window_size + 1 : bs_j + 1 ]

		# print()

		yield x_arr, y_arr

def get_mode_0():

	# best: 38%

	inp_layer = x = keras.layers.Input(shape=(window_size, se_len,))

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=se_len,
			return_sequences=True,
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=60,
			activation='relu'
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dropout(0.2)(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=30,
			activation='relu'
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dropout(0.1)(x)
	
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=15,
			activation='relu'
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dropout(0.1)(x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=5,
			return_sequences=True,
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)

	# x = keras.layers.Dropout(0.1)(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='sigmoid'
		)
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_mode_1():
	inp_layer = x = keras.layers.Input(shape=(window_size, se_len,))

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=40,
			activation='relu'
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dropout(0.3)(x)
	
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=20,
			activation='relu'
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dropout(0.15)(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=10,
			activation='relu'
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	# x = keras.layers.Dropout(0.1)(x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=5,
			return_sequences=True,
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)

	# x = keras.layers.Dropout(0.1)(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='sigmoid'
		)
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_model_2():
	inp_layer = x = keras.layers.Input(shape=(window_size, se_len,))

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=40,
			activation='relu'
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dropout(0.3)(x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=10,
			return_sequences=True,
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=20,
			activation='relu'
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dropout(0.15)(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=10,
			activation='relu'
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	# x = keras.layers.Dropout(0.1)(x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=5,
			return_sequences=True,
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)

	# x = keras.layers.Dropout(0.1)(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='sigmoid'
		)
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def test_generator(indexes):
	g = batch_generator( indexes , 32 )
	x1,y1 = next(g)
	x2,y2 = next(g)
	print( np.sum( x1 != x2 ) , np.sum( y1 != y2 ) )

def train_main_0():

	global se_len, window_size, x_data_set, y_data_set
	se_len = len( pickle.load( open( './minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p' , 'rb') )[1] )
	x_data_set, y_data_set = list(),list()
	for week_iterable in map( lambda fn: json.load( open( './one_throughput_one_se_rs_iterable_norm/'\
			+ str(fn) + '.json' , 'rt' ) ) , range( len( os.listdir( 'one_throughput_one_se_rs_iterable_norm'\
			) ) ) ):

		x_data_set.append( list() )
		y_data_set.append( list() )

		for _ , t , rsi in week_iterable:
			x_data_set[-1].append( rsi )
			y_data_set[-1].append( t )

	get_general_indexes_iterable_func = lambda p:\
		functools.reduce(\
			lambda acc,x:\
				acc\
					+ tuple(\
						map(\
							lambda e: ( x , e ) ,\
							pickle.load(\
								open(\
									'./split_indexes_folder/'+str(x)+'_one_throughput_one_se_rs.p',\
									'rb'\
								)\
							)[p]\
						)\
					),
			range( len( x_data_set ) ),
			tuple()
		)

	train_general_indexes_iterable = get_general_indexes_iterable_func(0)
	valid_general_indexes_iterable = get_general_indexes_iterable_func(1)
	window_size = 40

	if False:
		test_generator(train_general_indexes_iterable)
		exit(0)

	if False:
		print(\
			tuple(\
				map(
					lambda ind: max(filter(lambda p: p[0] == ind , train_general_indexes_iterable))[1],
					range(4)
				)
			)
		)
		print(\
			tuple(\
				map(
					lambda ind: max(filter(lambda p: p[0] == ind , valid_general_indexes_iterable))[1],
					range(4)
				)
			)
		)
		print(tuple(map(lambda e: len(e), x_data_set)))
		exit(0)

	model = get_mode_1()

	model.summary()

	model.compile(
		optimizer=keras.optimizers.Adam(),
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	print('Will load data !')

	model.fit_generator(
		batch_generator(train_general_indexes_iterable,128),
		steps_per_epoch=160,
		epochs=400,
		validation_data=batch_generator(valid_general_indexes_iterable,128),
		validation_steps=40,
		verbose=0,
		callbacks=[keras.callbacks.CSVLogger('./a.csv'),],
	)

def simple_plot_history():
	csv_generator = reader( open( './a.csv' , 'rt' ) )

	next(csv_generator)

	tr_list, va_list = list(), list()

	for line in csv_generator:
		tr_list.append( float( line[ 1 ] ) )
		if tr_list[-1] > 100: tr_list[-1] = 100
		va_list.append( float( line[ 3 ] ) )
		if va_list[-1] > 100: va_list[-1] = 100

	plt.plot( range( len( tr_list ) ) , tr_list , 'bo' )
	# plt.plot( range( len( tr_list ) ) , tr_list )
	plt.plot( range( len( va_list ) ) , va_list , 'ro' )
	# plt.plot( range( len( va_list ) ) , va_list )
	plt.show()

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

	with open( './bin_attempt3_folders/csv_training_configs/' + str(i) + '.csv' , 'wt' ) as conf_f:
		for k,v in param_dict.items():
			conf_f.write( str(k) + ',' + str(v) + '\n')

	models_path = './bin_attempt3_folders/models/model_' + str(i) + '/'
	if not os.path.exists(models_path):
		os.system( 'rm -rf ' + models_path )
	else:
		os.system( 'mkdir ' + models_path )

	model.fit_generator(
		batch_generator(\
			train_general_indexes_iterable,\
			int( param_dict[ 'train_bs' ] )
		),
		steps_per_epoch=int( param_dict[ 'steps_per_epoch' ] ),
		epochs=int( param_dict[ 'epochs' ] ),
		validation_data=batch_generator(
			valid_general_indexes_iterable,
			int( param_dict[ 'valid_bs' ] )
		),
		validation_steps=int( param_dict[ 'validation_steps' ] ),
		verbose=0,
		callbacks=[
			keras.callbacks.CSVLogger(
				'./bin_attempt3_folders/histories/' + str(i) + '.csv'
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
		get_mode_0.__name__ : get_mode_0,
		get_mode_1.__name__ : get_mode_1,
		get_model_2.__name__ : get_model_2,
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

def get_dataset_tuple():
	x_data_set, y_data_set = list(),list()
	for week_iterable in map( lambda fn: json.load( open( './one_throughput_one_se_rs_iterable_norm/'\
			+ str(fn) + '.json' , 'rt' ) ) , range( len( os.listdir( 'one_throughput_one_se_rs_iterable_norm'\
			) ) ) ):

		x_data_set.append( list() )
		y_data_set.append( list() )

		for _ , t , rsi in week_iterable:
			x_data_set[-1].append( rsi )
			y_data_set[-1].append( t )

	return x_data_set, y_data_set

def neural_tework_grid_search():
	global arguments_iterable,proc_q

	# Backup step
	bfn = 'bin_attempt3_folders/backup/date_time_' +  datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + '/'
	os.system( 'mkdir ' + bfn )
	os.system( 'cp -r bin_attempt3_folders/csv_training_configs ' + bfn )
	os.system( 'cp -r bin_attempt3_folders/histories ' + bfn )
	os.system( 'cp -r bin_attempt3_folders/models ' + bfn )

	set_const_argument_tags()

	global se_len, window_size, x_data_set, y_data_set, train_general_indexes_iterable, valid_general_indexes_iterable
	se_len = len( pickle.load( open( './minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p' , 'rb') )[1] )
	x_data_set,y_data_set = get_dataset_tuple()

	get_general_indexes_iterable_func = lambda p:\
		functools.reduce(\
			lambda acc,x:\
				acc\
					+ tuple(\
						map(\
							lambda e: ( x , e ) ,\
							pickle.load(\
								open(\
									'./split_indexes_folder/'+str(x)+'_one_throughput_one_se_rs.p',\
									'rb'\
								)\
							)[p]\
						)\
					),
			range( len( x_data_set ) ),
			tuple()
		)

	train_general_indexes_iterable = get_general_indexes_iterable_func(0)
	valid_general_indexes_iterable = get_general_indexes_iterable_func(1)
	window_size = 40

	csv_cfg_iterable = tuple(csv.reader( open( 'bin_attempt3_train_configs.csv' , 'rt' ) ) )
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
	indexes_iterable = range( len( arguments_iterable ) )

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

def get_model_summary():
	global se_len, window_size

	window_size = 40
	se_len = len( pickle.load( open( './minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p' , 'rb') )[1] )

	get_model_2().summary()

def dump_ground_truth_vs_prediction_0(model_path):
	se_len = len( pickle.load( open( './minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p' , 'rb') )[1] )
	x_data_set,y_data_set = get_dataset_tuple()
	window_size = 40
	batch_size = 256

	y_data_set =\
		tuple(
			map(
				lambda wi: wi[ window_size - 1 : ],
				y_data_set
			)
		)

	model = keras.models.load_model(model_path)

	x_arr = np.empty(( batch_size , window_size , se_len ))

	prediction_per_week_iterable = list()

	for wi, week_iterable in enumerate( x_data_set ):
		prediction_per_week_iterable.append( list() )
		index_inside_week_iterable = window_size - 1
		bs_i = 0
		while index_inside_week_iterable < len( week_iterable ):
			
			if index_inside_week_iterable % 1000 == 0:
				print(wi,index_inside_week_iterable,len( week_iterable ))

			if batch_size == bs_i:

				prediction_array = model.predict( x_arr )

				for i in range(batch_size):
					prediction_per_week_iterable[-1].append( prediction_array[ i , window_size - 1 , 0 ] )

				bs_i = 0

			x_arr[bs_i] = week_iterable[ index_inside_week_iterable - window_size + 1 : index_inside_week_iterable + 1 ]

			bs_i += 1

			index_inside_week_iterable += 1

		prediction_array = model.predict( x_arr )
		for i in range(batch_size):
			prediction_per_week_iterable[-1].append( prediction_array[ i , window_size - 1 , 0 ] )

		if len(prediction_per_week_iterable[wi]) > len(y_data_set[wi]):
			prediction_per_week_iterable[wi] = prediction_per_week_iterable[wi][:len(y_data_set[wi])]

	pickle.dump(
		( y_data_set , prediction_per_week_iterable ) , open( 'a.p' , 'wb' )
	)
	
def plot_ground_truth_vs_prediction_0():
	a = pickle.load(open( 'a.p' , 'rb' ))

	a_range = range(sum(map(lambda e: len(e), a[0])))

	plt.plot(
		a_range,
		functools.reduce(
			lambda acc,x: acc + x , a[0] , []
		),
		'b-'
	)

	plt.plot(
		a_range,
		functools.reduce(
			lambda acc,x: acc + x , a[1] , []
		),
		'o-'
	)

	plt.show()

if __name__ == '__main__':
	dump_ground_truth_vs_prediction_0('/nfs/public/mipopa/bin_attempt3_folders/models/model_0/model_0489.hdf5')