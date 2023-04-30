import os
import matplotlib.pyplot as plt
from shutil import copyfile
import pickle
import itertools
from sklearn.decomposition import PCA
import numpy as np
from multiprocessing import Pool, Queue
import json
import csv
import random
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf
from collections import namedtuple
import itertools
from pca_utils import *
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf
import sys
import data_processing_utils

WEEK_TIME_MOMENTS = (\
	(1579215600000, 1579875041000),\
	(1580511600000, 1581289199000),\
	(1581289200000, 1581670035995),\
	(1589398710001, 1590609266276),\
)

UNANSWERED_PATH_TUPLE = (\
	'/data/mipopa/unanswered_query_dump_folder_0/',\
	'/optane/mipopa/unanswered_query_dump_folder_1/',\
	'/optane/mipopa/unanswered_query_dump_folder_2/',\
)

PCA_DUMP_FOLDER='./pca_dumps/'

RAW_QUERY_FILES = (\
	'/data/mipopa/apicommands.log',
	'/data/mipopa/apicommands2.log',
	'/data/mipopa/apicommands3.log',
)

MATRICES_FOLDERS = (\
	'./matrices_folder/remote_host_0/log_folder',\
	'./matrices_folder/remote_host_0.5/log_folder',\
	'./matrices_folder/remote_host_1/log_folder',\
	'./matrices_folder/log_folder_4_0',\
	'./matrices_folder/log_folder_4_1',\
)

TREND_FILE_PATHS = (\
	'./trend_folder/week_0.p',\
	'./trend_folder/week_1.p',\
	'./trend_folder/week_2.p',\
	'./trend_folder/week_3.p',\
)

def get_small_encoder_decoder_model_0(arg_dict):
	'''
	Creates and returns a model for the encoder-decoder architecture.

	arg_dict
		Dictionary that contains various tags that alter the
		neural network structure.
	'''
	print( '\n\n\n\nWill construct model for', arg_dict['new_index'] , '!\n\n\n\n' )

	bins_no = 6769

	inp_layer_2 = keras.layers.Input(shape=(40, bins_no,))

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=arg_dict['latent_values_count'],
			activation='relu'
		)
	)(inp_layer_2)
	last_common_x = keras.layers.BatchNormalization()(x)

	y1 = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=bins_no,
			activation='tanh'
		)
	)(last_common_x)

	x = keras.layers.Dropout(arg_dict['dropout_value'])(last_common_x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=1,
			return_sequences=True,
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)

	y2 = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='sigmoid'
		)
	)(x)

	model = keras.models.Model(inputs=inp_layer_2, outputs=[y1,y2,])

	model.compile(
		optimizer=keras.optimizers.Adam(),
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	print( '\n\n\n\nWill exit model creation for', arg_dict['new_index'] , '!\n\n\n\n' )

	return model

def get_small_encoder_decoder_model_1(arg_dict):
	'''
	Creates and returns a model for the encoder-decoder architecture.

	arg_dict
		Dictionary that contains various tags that alter the
		neural network structure.
	'''
	print( '\n\n\n\nWill construct model for', arg_dict['new_index'] , '!\n\n\n\n' )

	inp_layer_1 = keras.layers.Input(shape=(40, 1,))
	inp_layer_2 = keras.layers.Input(shape=(40, arg_dict['bins_no'],))

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=arg_dict['latent_v_count'],
			activation='relu'
		)
	)(inp_layer_2)
	last_common_x = keras.layers.BatchNormalization()(x)

	y1 = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=arg_dict['bins_no'],
			activation='tanh'
		)
	)(last_common_x)

	x = keras.layers.Concatenate()([ inp_layer_1 , last_common_x ])

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=arg_dict['latent_v_count'],
			activation='relu')
	)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dropout(arg_dict['dropout_value'])(x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=arg_dict['latent_v_count'],
			return_sequences=True,
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dropout(arg_dict['dropout_value'])(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=arg_dict['latent_v_count'],
			activation='relu')
	)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dropout(arg_dict['dropout_value'])(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(units=5,
				activation='relu')
	)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dropout(arg_dict['dropout_value'])(x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=1,
			return_sequences=False,
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)

	y2 = keras.layers.Dense(
		units=1,
		activation='sigmoid'
	)(x)

	model = keras.models.Model(inputs=[inp_layer_1, inp_layer_2], outputs=[y1,y2,])

	model.compile(
		optimizer=keras.optimizers.Adam(),
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	print( '\n\n\n\nWill exit model creation for', arg_dict['new_index'] , '!\n\n\n\n' )

	return model

def get_small_encoder_decoder_model_1_with_l2_regularizer(arg_dict):
	'''
	Creates and returns a model for the encoder-decoder architecture.

	arg_dict
		Dictionary that contains various tags that alter the
		neural network structure.
	'''
	print( '\n\n\n\nWill construct model for', arg_dict['new_index'] , '!\n\n\n\n' )

	inp_layer_1 = keras.layers.Input(shape=(40, 1,))
	inp_layer_2 = keras.layers.Input(shape=(40, arg_dict['bins_no'],))

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=arg_dict['latent_v_count'],
			activation='relu',
		)
	)(inp_layer_2)
	last_common_x = keras.layers.BatchNormalization()(x)

	y1 = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=arg_dict['bins_no'],
			activation='tanh'
		)
	)(last_common_x)

	x = keras.layers.Concatenate()([ inp_layer_1 , last_common_x ])

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=10,
			activation='relu',
		    kernel_regularizer=keras.regularizers.l2(1e-4),
			bias_regularizer=keras.regularizers.l2(1e-4),
			activity_regularizer=keras.regularizers.l2(1e-5)
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	# x = keras.layers.Dropout(arg_dict['dropout_value'])(x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=10,
			return_sequences=True,
		    kernel_regularizer=keras.regularizers.l2(1e-4),
			bias_regularizer=keras.regularizers.l2(1e-4),
			activity_regularizer=keras.regularizers.l2(1e-5)
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	# x = keras.layers.Dropout(arg_dict['dropout_value'])(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=10,
			activation='relu',
		    kernel_regularizer=keras.regularizers.l2(1e-4),
			bias_regularizer=keras.regularizers.l2(1e-4),
			activity_regularizer=keras.regularizers.l2(1e-5)
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	# x = keras.layers.Dropout(arg_dict['dropout_value'])(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=5,
			activation='relu',
		    kernel_regularizer=keras.regularizers.l2(1e-4),
			bias_regularizer=keras.regularizers.l2(1e-4),
			activity_regularizer=keras.regularizers.l2(1e-5)
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	# x = keras.layers.Dropout(arg_dict['dropout_value'])(x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=1,
			return_sequences=False,
		    kernel_regularizer=keras.regularizers.l2(1e-4),
			bias_regularizer=keras.regularizers.l2(1e-4),
			activity_regularizer=keras.regularizers.l2(1e-5)
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)

	y2 = keras.layers.Dense(
		units=1,
		activation='sigmoid',
	    kernel_regularizer=keras.regularizers.l2(1e-4),
		bias_regularizer=keras.regularizers.l2(1e-4),
		activity_regularizer=keras.regularizers.l2(1e-5)
	)(x)

	model = keras.models.Model(inputs=[inp_layer_1, inp_layer_2], outputs=[y1,y2,])

	model.compile(
		optimizer=keras.optimizers.Adam(),
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	print( '\n\n\n\nWill exit model creation for', arg_dict['new_index'] , '!\n\n\n\n' )

	return model

def load_last_model(arg_dict):
	'''
	During the training process, models are saved to disk. This function loads the last
	model. The "new_index" tag identifies the series of models.

	arg_dict
		Dictionary
	'''
	models_paths_list = os.listdir( './pca_multiple_model_folders/models_' + str(arg_dict['new_index']) )
	if len(models_paths_list) == 0:
		return get_small_encoder_decoder_model_1(arg_dict)
	return keras.models.load_model(
		'./pca_multiple_model_folders/models_' + str(arg_dict['new_index']) + '/'\
		+ max(map(lambda fn: ( int(fn[6:10]) , fn ) , models_paths_list ))[1]
	)

def encoder_decoder_fit_function(model, model_fit_dict):
	'''
	Used to fit a model. One can observe the "gen_train"
	and "gen_valid". These functions are used as generators,
	because loading the whole data set into memory takes up
	too much space.

	model
		Keras model to be fitted

	model_fit_dict
		Dictionary containing different options for the
		training process
	'''
	def gen_train(batch_size, window_size):
		while True:
			yield generate_random_batch_1(\
				batch_size,
				window_size,
				data_set_dict['enc_dec_set']['train_indexes'],
				data_set_dict['enc_dec_set'],
				only_last_flag=True,
				one_input_flag=False
			)

	def gen_valid(batch_size, window_size):
		while True:
			yield generate_random_batch_1(\
				batch_size,
				window_size,
				data_set_dict['enc_dec_set']['valid_indexes'],
				data_set_dict['enc_dec_set'],
				only_last_flag=True,
				one_input_flag=False,
			)

	print( '\n\n\n\nWill fit for', model_fit_dict['new_index'] , '!\n\n\n\n' )

	model.summary()

	model.fit_generator(
		gen_train(128, 40),
		epochs=model_fit_dict['epochs'],
		steps_per_epoch=100,
		validation_data=gen_valid(128,40),
		validation_steps=20,
		verbose=0,
		callbacks=[
			keras.callbacks.CSVLogger(
				model_fit_dict['csv_log_path']
			),
			keras.callbacks.ModelCheckpoint(
				model_fit_dict['models_dump_path'] + "model_{epoch:04d}.hdf5",
				monitor='val_loss',
				save_best_only=True
			),
			keras.callbacks.EarlyStopping(
				monitor='val_loss',
				patience=model_fit_dict['patience'],
			),
		]
	)

	print(  '\n\n\n\nFinished fit for', model_fit_dict['new_index'] , '!\n\n\n\n'  )

def get_encoder_decoder_parameters_0(new_index, get_log_str_function):
	'''
	Creates list of dictionaries containing different tag values (e.g. latent space dimension,
	dropout values).

	new_index
		hyperparameter configuration index

	get_log_str_function
		function translating the current configuration to a string in order to
		store the meaning of the configuration index
	'''
	pool_arguments_list = list()

	for lat_var_c in (10,15,20,):
		for dr_val in (0.1, 0.3, 0.4):

			with open( 'pca_index_meaning/' + str(new_index) + '.txt' , 'wt' ) as myfile:

				myfile.write(
					get_log_str_function(
						new_index,
						lat_var_c,
						dr_val,
					)
				)

			models_folder_name = 'models_' + str(new_index)

			if models_folder_name not in os.listdir('./pca_multiple_model_folders'):
				os.mkdir(
					'./pca_multiple_model_folders/'\
					+ models_folder_name
				)

			pool_arguments_list.append(
				(
					get_small_encoder_decoder_model_1,
					{
						'one_input_flag' : True,
						'latent_values_count' : lat_var_c,
						'only_last_flag' : False,
						'dropout_value' : dr_val,
						'new_index' : new_index,
					},
					encoder_decoder_fit_function,
					{
						'csv_log_path' :\
							'./pca_csv_folder/losses_'\
							+ str( new_index )
							+'.csv',
						'models_dump_path' :\
							'pca_multiple_model_folders/'\
							+ models_folder_name + '/',
						'epochs' : 300,
						'patience' : 200,
						'new_index' : new_index,
					},
				)
			)

			new_index += 1

	return pool_arguments_list, new_index

def get_encoder_decoder_parameters_1(new_index, get_log_str_function):
	'''
	Creates list of dictionaries containing different tag values (e.g. latent space dimension,
	dropout values).

	new_index
		hyperparameter configuration index

	get_log_str_function
		function translating the current configuration to a string in order to
		store the meaning of the configuration index
	'''
	pool_arguments_list = list()

	for dr_val in (0.1,):
		for lat_var_c in (9,10,11):
			for regularizer_flag in (True, False):

				if not ( dr_val == 0.4 and regularizer_flag == True ):

					with open( 'pca_index_meaning/' + str(new_index) + '.txt' , 'wt' ) as myfile:

						myfile.write(
							get_log_str_function(
								new_index,
								lat_var_c,
								dr_val,
								regularizer_flag
							)
						)

					models_folder_name = 'models_' + str(new_index)

					if models_folder_name not in os.listdir('./pca_multiple_model_folders'):
						os.mkdir(
							'./pca_multiple_model_folders/'\
							+ models_folder_name
						)

					if regularizer_flag:
						pool_arguments_list.append(
							(
								get_small_encoder_decoder_model_1_with_l2_regularizer,
								{
									'latent_v_count' : lat_var_c,
									'dropout_value' : dr_val,
									'new_index' : new_index,
									'bins_no' : 6072,
								},
								encoder_decoder_fit_function,
								{
									'csv_log_path' :\
										'./pca_csv_folder/losses_'\
										+ str( new_index )
										+'.csv',
									'models_dump_path' :\
										'pca_multiple_model_folders/'\
										+ models_folder_name + '/',
									'epochs' : 700,
									'patience' : 200,
									'new_index' : new_index,
								},
							)
						)
					else:
						pool_arguments_list.append(
							(
								load_last_model,
								{
									'latent_v_count' : lat_var_c,
									'dropout_value' : dr_val,
									'new_index' : new_index,
									'bins_no' : 6072,
								},
								encoder_decoder_fit_function,
								{
									'csv_log_path' :\
										'./pca_csv_folder/losses_'\
										+ str( new_index )
										+'.csv',
									'models_dump_path' :\
										'pca_multiple_model_folders/'\
										+ models_folder_name + '/',
									'epochs' : 700,
									'patience' : 200,
									'new_index' : new_index,
								},
							)
						)

					new_index += 1

	return pool_arguments_list, new_index

def get_encoder_decoder_parameters_2(new_index, get_log_str_function):
	'''
	Creates list of dictionaries containing different tag values (e.g. latent space dimension,
	dropout values).

	new_index
		hyperparameter configuration index

	get_log_str_function
		function translating the current configuration to a string in order to
		store the meaning of the configuration index
	'''
	pool_arguments_list = list()

	for dr_val in (0.1, 0.3, 0.4):
		for lat_var_c in (5,7,8):

			with open( 'pca_index_meaning/' + str(new_index) + '.txt' , 'wt' ) as myfile:

				myfile.write(
					get_log_str_function(
						new_index,
						lat_var_c,
						dr_val,
					)
				)

			models_folder_name = 'models_' + str(new_index)

			if models_folder_name not in os.listdir('./pca_multiple_model_folders'):
				os.mkdir(
					'./pca_multiple_model_folders/'\
					+ models_folder_name
				)

			pool_arguments_list.append(
				(
					get_small_encoder_decoder_model_1,
					{
						'latent_v_count' : lat_var_c,
						'dropout_value' : dr_val,
						'new_index' : new_index,
					},
					encoder_decoder_fit_function,
					{
						'csv_log_path' :\
							'./pca_csv_folder/losses_'\
							+ str( new_index )
							+'.csv',
						'models_dump_path' :\
							'pca_multiple_model_folders/'\
							+ models_folder_name + '/',
						'epochs' : 700,
						'patience' : 200,
						'new_index' : new_index,
					},
				)
			)

			new_index += 1

	return pool_arguments_list, new_index

def train_for_generator():
	'''

	DEPRECATED

	Was used to do a grid search in the hyperparameter space. This is part of a
	suboptimal approach. One can not train in parallel multiple models without
	creating new tensorflow Sessions so by calling a new process that
	trains over its assigned configurations, a new Session is created automatically.

	'''
	with tf.device('/gpu:' + sys.argv[1]):
		new_index = 326

		pool_arguments_list = []

		a, new_index = get_encoder_decoder_parameters_0(
			new_index,
			lambda ind, lv_c, dr_v:\
				str(ind)\
				+ ': latent_v_count=' + str(lv_c)\
				+ ' dropout_value=' + str(dr_v)
		)
		pool_arguments_list += a

		print( new_index )

		a_list = [\
			list(),\
			list(),\
			list(),\
			list(),\
		]

		i = 0
		for args in pool_arguments_list:
			a_list[i].append(args)
			i = (i + 1) % 4

		global data_set_dict
		data_set_dict = dict()

		encoder_decoder_dict = pickle.load(open(
			'pca_data_sets/7.p',
			'rb'
		))
		encoder_decoder_dict['non_split_data_set'] = np.array( encoder_decoder_dict['non_split_data_set'] )
		data_set_dict['enc_dec_set'] = encoder_decoder_dict

		for get_model_f, get_model_d, model_fit_f, fit_model_d in a_list[int(sys.argv[1])]:

			model = get_model_f(get_model_d)

			model_fit_f( model , fit_model_d )

def launch_train_per_process(ind):
	'''

	DEPRECATED

	Was used to do a grid search in the hyperparameter space. This is part of a
	suboptimal approach. One can not train in parallel multiple models without
	creating new tensorflow Sessions so by calling a new process that
	trains over its assigned configurations, a new Session is created automatically.

	'''
	gpu_string = proc_q.get()

	try:

		print('\n\n\n\nWill start training for index ' + str(ind) + ' on gpu ' + gpu_string + '\n\n\n\n')

		os.system(
			'python3 encoder_decoder_main.py ' + str(ind) + ' ' + gpu_string
		)

		print('\n\n\n\nFinished training for index ' + str(ind) + ' on gpu ' + gpu_string + '\n\n\n\n')

	except:

		err_string = '\n\n\n\nFailed for index ' + str(ind) + ' on gpu ' + gpu_string + '\n\n\n\n'

		print(err_string)

		with open('econder_decoder_errors.txt','a') as f:
			f.write(err_string)

	proc_q.put(gpu_string)

def train_main_for_generator_grid_search():
	'''

	DEPRECATED

	Was used to do a grid search in the hyperparameter space. This is part of a
	suboptimal approach. One can not train in parallel multiple models without
	creating new tensorflow Sessions so by calling a new process that
	trains over its assigned configurations, a new Session is created automatically.

	'''
	print('Will execute',sys.argv,'!')

	new_index = 335

	pool_arguments_list, new_index =\
		get_encoder_decoder_parameters_1(
			new_index,
			lambda ind, lv_c, dr_v, reg_flag:\
				str(ind)\
				+ ': latent_v_count=' + str(lv_c)\
				+ ' dropout_value=' + str(dr_v)\
				+ ' are_regs_used=' + str(reg_flag)\
				+ ' best_arch_from_pca'
		)

	print( 'Last index is' , new_index )

	exclusion_iterable = tuple( range(335, 340) )

	if sys.argv[1] == '-1':

		file = open('econder_decoder_errors.txt','wt')

		file.close()

		available_gpu_tuple = (\
			'/gpu:0',
			'/gpu:1',
			'/gpu:2',
			'/gpu:3',
		)

		global proc_q
		proc_q = Queue()
		for gpu_string in available_gpu_tuple:
			proc_q.put( gpu_string )

		print('Will start process Pool !')

		Pool(len(available_gpu_tuple)).map(
			launch_train_per_process,
			tuple(
				filter(
					lambda ind: pool_arguments_list[ind][1]['new_index'] not in exclusion_iterable,
					range(len(pool_arguments_list))
				)
			)
		)

	else:

		ind = int( sys.argv[1] )

		if pool_arguments_list[ind][1]['new_index'] not in exclusion_iterable:

			with tf.device(sys.argv[2]):

				global data_set_dict

				data_set_dict = dict()

				encoder_decoder_dict = pickle.load(open(
					'pca_data_sets/7.p',
					'rb'
				))
				encoder_decoder_dict['non_split_data_set'] = np.array( encoder_decoder_dict['non_split_data_set'] )
				data_set_dict['enc_dec_set'] = encoder_decoder_dict

				model = pool_arguments_list[ind][0]( pool_arguments_list[ind][1] )

				pool_arguments_list[ind][2]( model , pool_arguments_list[ind][3] )
		else:
			print('\n\n\n\nSkipped for: ' + str(pool_arguments_list[ind][1]['new_index']) + '\n\n\n\n')

def launch_train_per_process_0(ind):
	'''
	Trains a configuration per GPU. First off, an available GPU string is
	taken out of a shared queue. Then a new session is created for that GPU.
	The training commences. After the training is finished, the GPU string is
	put back into the queue so another process can use it.
	'''
	gpu_string = proc_q.get()


	with tf.device(gpu_string):
		with tf.Session() as sess:

			K.set_session(sess)

			print('\n\n\n\nWill start training for index ' + str(pool_arguments_list[ind][1]['new_index']) + ' on gpu ' + gpu_string + '\n\n\n\n')

			model = pool_arguments_list[ind][0]( pool_arguments_list[ind][1] )

			pool_arguments_list[ind][2]( model , pool_arguments_list[ind][3] )

			print('\n\n\n\nFinished training for index ' + str(pool_arguments_list[ind][1]['new_index']) + ' on gpu ' + gpu_string + '\n\n\n\n')

	proc_q.put(gpu_string)

def train_main_for_generator_grid_search_0():
	'''
	Launches grid search over a set of hyperparameter configurations
	'''
	global pool_arguments_list, proc_q

	new_index = 380

	if True:
		pool_arguments_list, new_index =\
			get_encoder_decoder_parameters_1(
				new_index,
				lambda ind, lv_c, dr_v, reg_flag:\
					str(ind)\
					+ ': latent_v_count=' + str(lv_c)\
					+ ' no dropout' + str(dr_v)\
					+ ' are_regs_used=' + str(reg_flag)\
					+ ' best_arch_from_pca'
			)
	if False:
		pool_arguments_list, new_index =\
			get_encoder_decoder_parameters_2(
				new_index,
				lambda ind, lv_c, dr_v:\
					str(ind)\
					+ ': latent_v_count=' + str(lv_c)\
					+ ' dropout_value=' + str(dr_v)\
					+ ' extended_set'
			)

	# exclusion_iterable = tuple( range(335, 340) )
	# exclusion_iterable = tuple( range(340, 350) )
	exclusion_iterable = tuple()

	global data_set_dict

	data_set_dict = dict()

	encoder_decoder_dict = pickle.load(open(
		'pca_data_sets/8_4th_june.p',
		'rb'
	))
	encoder_decoder_dict['non_split_data_set'] = np.array( encoder_decoder_dict['non_split_data_set'] )
	data_set_dict['enc_dec_set'] = encoder_decoder_dict

	file = open('enconder_decoder_errors.txt','wt')

	file.close()

	available_gpu_tuple = (\
		'/gpu:0',
		'/gpu:1',
		'/gpu:3',
		'/gpu:4',
	)

	proc_q = Queue()
	for gpu_string in available_gpu_tuple:
		proc_q.put( gpu_string )

	print('Will start process Pool !')

	Pool(len(available_gpu_tuple)).map(
		launch_train_per_process_0,
		tuple(
			filter(
				lambda ind: pool_arguments_list[ind][1]['new_index'] not in exclusion_iterable,
				range(len(pool_arguments_list))
			)
		)
	)

def get_biggest_index_model_path(index):
	'''
	Looks into folder containing models for the configuration at "index".

	index
		integer
	'''
	best_tuple = (1,'model_0001.hdf5')

	for model_name in os.listdir('pca_multiple_model_folders/models_' + str(index)):

		model_index = int(model_name[6:10])

		if best_tuple[0] < model_index: best_tuple = ( model_index , model_name, )

	return './pca_multiple_model_folders/models_' + str(index) + '/' + model_name

def get_biggest_index_model_path_by_val_loss(index):
	'''
	DEPRECATED

	Tries to construct the path to dumped model, but the epoch indexes in the
	CSVLogger are different than the ones in the Model Checkpointer. They differ
	by one. (One starts counting at 0 while the other starts counting at 1).
	'''
	g = csv.reader( open( './pca_csv_folder/' + 'losses_' + str( index ) +'.csv' , 'rt' ) )

	val_loss_index = next(g).index('val_loss')

	models_dict = dict()
	for model_name in os.listdir('./pca_multiple_model_folders/models_' + str(index)):
		models_dict[int(model_name[6:10])] = model_name

	print(tuple(models_dict.keys()))

	best_tuple = next(g)
	best_tuple = ( int(best_tuple[0]) , float(best_tuple[val_loss_index]) , )

	for line in g:
		v = float(line[val_loss_index])
		if v < best_tuple[1]:
			best_tuple = (int(line[0]), v,)

	return './pca_multiple_model_folders/models_' + str(index) + '/' + models_dict[best_tuple[0]]

def get_model_path_by_date(index):
	'''
	Returns the model path from a configuration by creation date of the model.
	'''
	a_list = list()

	for fn in os.listdir('./pca_multiple_model_folders/models_' + str(index)):

		a_list.append(
			(
				fn,
				os.stat(
					'./pca_multiple_model_folders/models_' + str(index) + '/' + fn
				).st_mtime
			)
		)

	a_list.sort( key=lambda p: p[1] , reverse=True )

	return './pca_multiple_model_folders/models_' + str(index) + '/' + a_list[0][0]

def dump_encoder_decoder_plot(index):
	'''
	Dumps plots of the loss evolution on training and validation
	data sets.
	'''
	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)

	fig = plt.gcf()
	fig.set_size_inches(11,8)

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
			encoder_decoder_dict,
			only_last_flag=True,
			firs_bs_elements_flag=True,
			one_input_flag=False,
		)
		x_set, y_set = generate_random_batch_1(
			len( encoder_decoder_dict['train_indexes'] )\
				+ len( encoder_decoder_dict['valid_indexes'] ),
			40,
			sorted( encoder_decoder_dict['train_indexes']\
				+ encoder_decoder_dict['valid_indexes'] ),
			encoder_decoder_dict,
			only_last_flag=True,
			firs_bs_elements_flag=True,
			one_input_flag=False,
		)


		best_index_path =\
			get_model_path_by_date(index)
		model = keras.models.load_model(best_index_path)
		
		model.summary()

		print('Best model index at', best_index_path)

		print(index, model.evaluate(x_set,y_set))

		predicted_array = model.predict(x_set)

		predicted_array = predicted_array[1]

		pickle.dump(
			( tuple(y_set[1][:,0]) , tuple(predicted_array[:,0]) ),
			open( 'a.p' , 'wb' )
		)

		# plt.plot(
		# 	range( y_set[1].shape[0] ),
		# 	y_set[1][:,0],
		# 	label='Ground Truth'
		# )

		# plt.plot(
		# 	range( predicted_array.shape[0] ),
		# 	predicted_array[:,0],
		# 	label='Prediction'
		# )

		# plt.legend()

		# plt.xlabel('Index in Data Set')

		# plt.ylabel('Normalized Trend')

		# plt.savefig(
		# 	'./pca_plots/' + str(index) + '_valid_gt_vs_pred.png'
		# )
		# plt.clf()

		exit(0)

	loss_gen = csv.reader( open( './pca_csv_folder/losses_' + str(index) + '.csv' , 'rt' ) )

	first_line_list = next(loss_gen)

	train_trend_index = first_line_list.index('dense_5_loss')
	valid_trend_index = first_line_list.index('val_dense_5_loss')

	train_matrices_index = first_line_list.index('time_distributed_1_loss')
	valid_matrices_index = first_line_list.index('val_time_distributed_1_loss')

	trend_train_list, trend_valid_list, matrices_train_list, matrices_valid_list =\
		list(), list(), list(), list()

	for line_list in loss_gen:
		if len(line_list) > 2:
			trend_train_list.append( float( line_list[train_trend_index] ) )
			trend_valid_list.append( float( line_list[valid_trend_index] ) )

			matrices_train_list.append( float( line_list[train_matrices_index] ) )
			matrices_valid_list.append( float( line_list[valid_matrices_index] ) )

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

	plt.savefig(
		'./pca_plots/' + str(index) + '_losses.png'
	)
	plt.clf()

def one_time_generator_train():
	'''
	Runs one single training configuration.
	'''
	new_index = 361

	data_set_dict = dict()
	encoder_decoder_dict = pickle.load(open(
		'pca_data_sets/8_28_may.p',
		'rb'
	))
	encoder_decoder_dict['non_split_data_set'] = np.array( encoder_decoder_dict['non_split_data_set'] )
	data_set_dict['enc_dec_set'] = encoder_decoder_dict

	def gen_train(batch_size, window_size):
		while True:
			yield generate_random_batch_1(\
				batch_size,
				window_size,
				data_set_dict['enc_dec_set']['train_indexes'],
				data_set_dict['enc_dec_set'],
				only_last_flag=True,
				one_input_flag=True
			)

	def gen_valid(batch_size, window_size):
		while True:
			yield generate_random_batch_1(\
				batch_size,
				window_size,
				data_set_dict['enc_dec_set']['valid_indexes'],
				data_set_dict['enc_dec_set'],
				only_last_flag=True,
				one_input_flag=True,
			)

	if True:
		bins_no = encoder_decoder_dict['non_split_data_set'].shape[-1] - 1

		print('Bins no is', bins_no)

		inp_layer_2 = keras.layers.Input(shape=(40, bins_no,))

		x = keras.layers.TimeDistributed(
			keras.layers.Dense(
				units=10,
				activation='relu',
			)
		)(inp_layer_2)
		last_common_x = keras.layers.BatchNormalization()(x)

		y1 = keras.layers.TimeDistributed(
			keras.layers.Dense(
				units=bins_no,
				activation='tanh'
			)
		)(last_common_x)

		x = keras.layers.TimeDistributed(
			keras.layers.Dense(
				units=10,
				activation='relu',
			    kernel_regularizer=keras.regularizers.l2(1e-4),
				bias_regularizer=keras.regularizers.l2(1e-4),
				activity_regularizer=keras.regularizers.l2(1e-5)
			)
		)(last_common_x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Dropout(0.1)(x)

		x = keras.layers.Bidirectional(
			keras.layers.LSTM(
				units=10,
				return_sequences=True,
			    kernel_regularizer=keras.regularizers.l2(1e-4),
				bias_regularizer=keras.regularizers.l2(1e-4),
				activity_regularizer=keras.regularizers.l2(1e-5)
			)
		)(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Dropout(0.1)(x)

		x = keras.layers.TimeDistributed(
			keras.layers.Dense(
				units=10,
				activation='relu',
			    kernel_regularizer=keras.regularizers.l2(1e-4),
				bias_regularizer=keras.regularizers.l2(1e-4),
				activity_regularizer=keras.regularizers.l2(1e-5)
			)
		)(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Dropout(0.1)(x)

		x = keras.layers.TimeDistributed(
			keras.layers.Dense(
				units=5,
				activation='relu',
			    kernel_regularizer=keras.regularizers.l2(1e-4),
				bias_regularizer=keras.regularizers.l2(1e-4),
				activity_regularizer=keras.regularizers.l2(1e-5)
			)
		)(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Dropout(0.1)(x)

		x = keras.layers.Bidirectional(
			keras.layers.LSTM(
				units=1,
				return_sequences=False,
			    kernel_regularizer=keras.regularizers.l2(1e-4),
				bias_regularizer=keras.regularizers.l2(1e-4),
				activity_regularizer=keras.regularizers.l2(1e-5)
			)
		)(x)
		x = keras.layers.BatchNormalization()(x)

		y2 = keras.layers.Dense(
			units=1,
			activation='sigmoid',
		    kernel_regularizer=keras.regularizers.l2(1e-4),
			bias_regularizer=keras.regularizers.l2(1e-4),
			activity_regularizer=keras.regularizers.l2(1e-5)
		)(x)

		model = keras.models.Model(inputs=inp_layer_2, outputs=[y1,y2,])

		model.compile(
			optimizer=keras.optimizers.Adam(),
			loss='mean_absolute_percentage_error',
			metrics=['mae',]
		)

	model.summary()

	model.fit_generator(
		gen_train(128, 40),
		epochs=700,
		steps_per_epoch=136,
		validation_data=gen_valid(128,40),
		validation_steps=34,
		verbose=2,
		callbacks=[
			keras.callbacks.CSVLogger(
				'pca_csv_folder/losses_' + str(new_index) + '.csv'
			),
			keras.callbacks.ModelCheckpoint(
				'./pca_multiple_model_folders/models_' + str(new_index) + "/model_{epoch:04d}.hdf5",
				monitor='val_loss',
				save_best_only=True
			),
			keras.callbacks.EarlyStopping(
				monitor='val_loss',
				patience=200,
			),
		]
	)

def parse_matrix_per_proc(i):
	pickle.dump(
		(
			matrices_paths_per_week_list[pool_arguments_list[i][0]][pool_arguments_list[i][1]][0],
			matrices_paths_per_week_list[pool_arguments_list[i][0]][pool_arguments_list[i][1]][1],
			data_processing_utils.get_dist_dict_by_path(
				matrices_paths_per_week_list[pool_arguments_list[i][0]][pool_arguments_list[i][1]][2])\
				if matrices_paths_per_week_list[pool_arguments_list[i][0]][pool_arguments_list[i][1]][1] else\
				data_processing_utils.get_dem_dict_by_path(\
				matrices_paths_per_week_list[pool_arguments_list[i][0]][pool_arguments_list[i][1]][2])
		),
		open(
			folder_path\
				+ str(\
					dump_index_list[pool_arguments_list[i][0]]
				)\
				+ '_' + str(pool_arguments_list[i][1]) + '.p',
			'wb'
		)
	)

def aggregate_and_generate_new_data_set(\
		window_size=40,\
		min_se_count=63):
	'''
	
		min_se_count
			It is used, because there are distance matrices
			where a client only comunicated with a couple of SEs so
			those matrices are filtered out.

	'''
	parse_matrices_indexes =\
		process_minimal_indexes =\
		process_whole_matrices_indexes =\
		process_minimal_indexes = create_set_indexes = tuple()

	previous_parsed_folder = 'parsed_0123/'
	previous_minimal_set_fn = 'minimal_sets_for_0123.p'
	previous_whole_matrices = 'whole_for_0123.p'

	next_parsed_folder = 'parsed_0123/'
	next_minimal_set_fn = 'minimal_sets_for_0123.p'
	next_whole_matrices = 'whole_for_0123.p'

	ars_dict = dict()
	if 0 in create_set_indexes\
		or 1 in create_set_indexes\
		or 2 in create_set_indexes:

		a = pickle.load( open(\
			'./read_size_folder/all_queries_read_size.p' , 'rb'
		) )

		for i in filter(lambda ind: ind < 3, create_set_indexes):
			ars_dict[i] = a[i]
	
	if 3 in create_set_indexes:
		ars_dict[3] = sorted(pickle.load( open(\
			'./read_size_folder/rs_per_tm_4.p' , 'rb'
		) ).items())

	if len(parse_matrices_indexes) > 0:
		# Parse matrices filenames
		matrices_fn_list = list(itertools.chain.from_iterable(
			map(
				lambda folder_path:\
					map(\
						lambda fn: (\
							int( fn.split('_')[0] ),\
							'distance' in fn,\
							folder_path + '/' + fn,\
						),\
						os.listdir(folder_path),\
					),
				map(
					lambda a: a[1],
					filter(
						lambda p: p[0] in parse_matrices_indexes,
						enumerate(MATRICES_FOLDERS)
					)
				)
			)
		))

		# Makes sure there are no duplicates.
		new_list = list()
		already_in_list_set = set()
		for a,b,c in matrices_fn_list:
			if (a,b) not in already_in_list_set:
				already_in_list_set.add( (a,b) )
				new_list.append( (a,b,c) )
		matrices_fn_list = new_list
		
		global matrices_paths_per_week_list
		matrices_paths_per_week_list =\
		list(\
			map(
				lambda tm_tup:\
					list(filter(\
						lambda fn_tup:\
							tm_tup[0] - 120000 <= fn_tup[0] < tm_tup[1],\
							matrices_fn_list,\
					)),
				map(
					lambda a: a[1],
					filter(
						lambda p: p[0] in parse_matrices_indexes,
						enumerate(WEEK_TIME_MOMENTS)
					)
				)
			)
		)
		
		for i,l in enumerate(map(lambda e: len(e), matrices_paths_per_week_list)):
			print('For period',i,'there are',l,'matrices')

		global pool_arguments_list, folder_path
		pool_arguments_list = list()
		for i,e in enumerate(matrices_paths_per_week_list):
			for v in range(len(e)):
				pool_arguments_list.append((i,v))
		
		folder_path = './minimal_sets_and_parsed_matrices/' + next_parsed_folder
		if not os.path.isdir(folder_path): os.mkdir(folder_path)
		
		global dump_index_list
		dump_index_list = parse_matrices_indexes

		with Pool(126) as p_pool:
			p_pool.map(\
				parse_matrix_per_proc,
				range(sum(map(lambda e: len(e),\
					matrices_paths_per_week_list)))
			)

		print('Finished parsing !')

		matrices_paths_per_week_list = list()
		for _ in range(len(parse_matrices_indexes)):
			matrices_paths_per_week_list.append(list())
		for fn in os.listdir(folder_path):
			matrices_paths_per_week_list[int(fn.split('_')[0])].append(
				pickle.load(
					open(
						folder_path + fn, 'rb'
					)
				)
			)

	if 0 < len(parse_matrices_indexes) < len(WEEK_TIME_MOMENTS):
		old_folder_path = './minimal_sets_and_parsed_matrices/' + previous_parsed_folder
		new_folder_path = './minimal_sets_and_parsed_matrices/' + next_parsed_folder
		for fn in os.listdir( './minimal_sets_and_parsed_matrices/' + previous_parsed_folder ):
			os.rename(
				old_folder_path + fn,
				new_folder_path + fn,
			)
		os.rmdir( old_folder_path )

	matrices_paths_per_week_list = list()
	for _ in range(len(WEEK_TIME_MOMENTS)):
		matrices_paths_per_week_list.append(list())
	folder_path = './minimal_sets_and_parsed_matrices/' + next_parsed_folder
	a = tuple(os.listdir(folder_path))
	for i,fn in enumerate(a):
		if(i % 5000 == 0):
			print('loading matrix:',i,'/',len(a)-1)
		matrices_paths_per_week_list[int(fn.split('_')[0])].append(
			pickle.load(
				open(
					folder_path + fn, 'rb'
				)
			)
		)

	# add error filtering
	filtered_per_week_list = list()
	for week_list in matrices_paths_per_week_list:
		filtered_per_week_list.append(list())

		for mat_t in week_list:
			if mat_t[1]:
				if len( mat_t[2].keys() ) >= min_se_count:
					se_count_is_valid = True
					for val in mat_t[2].values():
						if len(val.keys()) < min_se_count:
							se_count_is_valid = False
							break
					if se_count_is_valid:
						filtered_per_week_list[-1].append(
							mat_t
						)
			else:
				if len( mat_t[2].keys() ) >= min_se_count:
					filtered_per_week_list[-1].append(
						mat_t
					)
	matrices_paths_per_week_list = filtered_per_week_list

	print('Finished loading matrices !')

	for i, week_list in enumerate( matrices_paths_per_week_list ):
		print('Matrix', i, 'has',\
			len(tuple(filter(lambda fn_t: fn_t[1],week_list))),\
			'distance matrices and',
			len(tuple(filter(lambda fn_t: not fn_t[1],week_list))),\
			'demotion matrices.'
		)

	if len( process_minimal_indexes ) != len(WEEK_TIME_MOMENTS):
		old_minimal_sets = pickle.load(
			open(
				'./minimal_sets_and_parsed_matrices/' + previous_minimal_set_fn,
				'rb'
			)
		)
	
	if len(process_minimal_indexes) > 0:
		minimal_clients_list, minimal_se_list =\
			data_processing_utils.get_clients_and_ses_minimal_list(
				list(
					map(
						lambda week_list:\
							list(map(
								lambda a: (a[0],a[2]),
								filter(
									lambda fn_tup: fn_tup[1],
									week_list
								)
							)),
						map(
							lambda p: p[1],
							filter(
								lambda pp: pp[0] in process_minimal_indexes,
								enumerate(matrices_paths_per_week_list)
							)
						)
					)
				),
				list(
					map(
						lambda week_list:\
							list(map(
								lambda a: (a[0],a[2]),
								filter(
									lambda fn_tup: not fn_tup[1],
									week_list
								)
							)),
						map(
							lambda p: p[1],
							filter(
								lambda pp: pp[0] in process_minimal_indexes,
								enumerate(matrices_paths_per_week_list)
							)
						)
					)
				)
			)


		if len( process_minimal_indexes ) != len(WEEK_TIME_MOMENTS):
			new_cl_list, new_se_list = list(), list()
			for cl in minimal_clients_list:
				if cl in old_minimal_sets[0]:
					new_cl_list.append(cl)
			for se in minimal_se_list:
				if se in old_minimal_sets[1]:
					new_se_list.append(se)

			minimal_clients_list, minimal_se_list =\
				new_cl_list, new_se_list


		pickle.dump(
			(minimal_clients_list, minimal_se_list,),
			open( './minimal_sets_and_parsed_matrices/'\
				+ next_minimal_set_fn , 'wb' )
		)
	else:
		minimal_clients_list, minimal_se_list = old_minimal_sets

	print('# clients:',len(minimal_clients_list))
	print('# ses:',len(minimal_se_list))

	# if 0 < len(process_minimal_indexes) < len(WEEK_TIME_MOMENTS):
	# 	os.remove(previous_minimal_set_fn)

	if len(process_whole_matrices_indexes) != len(WEEK_TIME_MOMENTS):
		whole_matrices_list = pickle.load(
			open(
				'./whole_matrices/' + previous_whole_matrices , 'rb'
			)
		)
	else:
		whole_matrices_list = list()

	if len(process_whole_matrices_indexes) != 0:
		for week_list in matrices_paths_per_week_list:
			whole_matrices_list.append(\
				data_processing_utils.get_complete_distance_matrix_1(
					list(
						map(
							lambda e: (e[0],e[2]),
							filter(
								lambda f: f[1],
								week_list
							)
						)
					),
					list(
						map(
							lambda e: (e[0],e[2]),
							filter(
								lambda f: not f[1],
								week_list
							)
						)
					),	
					minimal_clients_list,
					minimal_se_list	
				)
			)
		pickle.dump(
			whole_matrices_list,
			open(
				'./whole_matrices/' + next_whole_matrices,
				'wb'
			)
		)

	if 0 < len(process_minimal_indexes) < len(WEEK_TIME_MOMENTS):
		os.remove('./whole_matrices/' + previous_whole_matrices)

	print("Finished with the whole matrices !")

	trend_list = list(
		map(
			lambda p: pickle.load(open(p,'rb')),
			TREND_FILE_PATHS
		)
	)
	
	data_set_dict = dict()

	for i in range(len(create_set_indexes)):
		ds = data_processing_utils.create_data_set(
			trend_list[i],
			ars_dict[i],
			whole_matrices_list[i],
		)

		validation_indexes = tuple(random.sample(
			range( window_size - 1 , len(ds) ),
			round( 0.2 * len(ds) )
		))

		data_set_dict[i] =\
			{
				"data_set" : ds,
				"validation_indexes" : validation_indexes,
				"training_indexes" :\
					tuple(
						filter(
							lambda ind: ind not in validation_indexes,
							range( window_size - 1 , len(ds) ),
						)
					)
			}

		pickle.dump(
			data_set_dict[i],
			open(
				'./enc_dec_ready_to_train_non_norm/' + str(i) + '.p' , 'wb'
			)
		)

def norm_data_sets():
	data_set_dict = dict()
	for i in range(len(WEEK_TIME_MOMENTS)):
		data_set_dict[i] = pickle.load(
			open(
				'./enc_dec_ready_to_train_non_norm/' + str(i) + '.p', 'rb'
			)
		)

	min_thp = max_thp = data_set_dict[0]["data_set"][0][1]

	min_rs = max_rs = data_set_dict[0]["data_set"][0][2]

	min_comp = min(data_set_dict[0]["data_set"][0][3])

	max_comp = max(data_set_dict[0]["data_set"][0][3])

	for week_list in data_set_dict.values():
		for t in week_list["data_set"]:
			if t[1] < min_thp:
				min_thp = t[1]
			if t[1] > max_thp:
				max_thp = t[1]

			if t[2] < min_rs:
				min_rs = t[2]
			if t[2] > max_rs:
				max_rs = t[2]

			for cmp_v in t[3]:
				if cmp_v < min_comp:
					min_comp = cmp_v
				if cmp_v > max_comp:
					max_comp = cmp_v

	for k, l in data_set_dict.items():
		pickle.dump(
			{
				"data_set" :\
					tuple(
						map(
							lambda p:\
								[2*(p[2]-min_rs)/(max_rs-min_rs)-1,]\
								+ list(
									map(
										lambda e: 2*(e-min_comp)/(max_comp-min_comp)-1,
										p[3]
									)
								)\
								+ [(p[1]-min_thp)/(max_thp-min_thp),],
							l["data_set"]
						)
					),
				"training_indexes" : l['training_indexes'],
				"validation_indexes": l["validation_indexes"],
			}
			,
			open( './enc_dec_ready_to_train_norm/' + str(k) + '.p' , 'wb' )
		)

def analyze_minimal_set():

	indexes_to_analyze_tuple = (3,)

	matrices_paths_per_week_list = dict()
	for ind in indexes_to_analyze_tuple:
		matrices_paths_per_week_list[ind] = list()
	folder_path = './minimal_sets_and_parsed_matrices/parsed_0123/'
	a = tuple(
			filter(
				lambda fn_t: fn_t[0] in indexes_to_analyze_tuple,
				map(
					lambda fn: (int(fn.split('_')[0]), fn),
					os.listdir(folder_path)
				)
			)
	)
	for i,fn_t in enumerate(a):
		if(i % 5000 == 0):
			print('loading matrix:',i,'/',len(a)-1)
		matrices_paths_per_week_list[fn_t[0]].append(
			pickle.load(
				open(
					folder_path + fn_t[1], 'rb'
				)
			)
		)

	for i, week_list in matrices_paths_per_week_list.items():
		print('Matrix', i, 'has',\
			len(tuple(filter(lambda fn_t: fn_t[1],week_list))),\
			'distance matrices and',
			len(tuple(filter(lambda fn_t: not fn_t[1],week_list))),\
			'demotion matrices.'
		)

	matrices_paths_per_week_list = tuple(
		map(lambda t: t[1],sorted(matrices_paths_per_week_list.items()))
	)

	# add error filtering
	if False:
		filtered_per_week_list = list()
		for week_list in matrices_paths_per_week_list:
			filtered_per_week_list.append(list())

			for mat_t in week_list:
				if mat_t[1]:
					if len( mat_t[2].keys() ) >= 63:
						se_count_is_valid = True
						for val in mat_t[2].values():
							if len(val.keys()) < 63:
								se_count_is_valid = False
								break
						if se_count_is_valid:
							filtered_per_week_list[-1].append(
								mat_t
							)
				else:
					if len( mat_t[2].keys() ) >= 63:
						filtered_per_week_list[-1].append(
							mat_t
						)
		matrices_paths_per_week_list = filtered_per_week_list

	if False:
		process_minimal_indexes = (0,1,2,3,)

		minimal_clients_list, minimal_se_list =\
			data_processing_utils.get_clients_and_ses_minimal_list(
				list(
					map(
						lambda week_list:\
							list(map(
								lambda a: (a[0],a[2]),
								filter(
									lambda fn_tup: fn_tup[1],
									week_list
								)
							)),
						map(
							lambda p: p[1],
							filter(
								lambda pp: pp[0] in process_minimal_indexes,
								enumerate(matrices_paths_per_week_list)
							)
						)
					)
				),
				list(
					map(
						lambda week_list:\
							list(map(
								lambda a: (a[0],a[2]),
								filter(
									lambda fn_tup: not fn_tup[1],
									week_list
								)
							)),
						map(
							lambda p: p[1],
							filter(
								lambda pp: pp[0] in process_minimal_indexes,
								enumerate(matrices_paths_per_week_list)
							)
						)
					)
				)
			)

		print('# clients:',len(minimal_clients_list))
		print('# ses:',len(minimal_se_list))

	if True:
		dem_dict = dict()
		dist_dict = dict()
		for mat_t in matrices_paths_per_week_list[0]:
			if not mat_t[1]:
				a = len(tuple(mat_t[2].items()))
				if a not in dem_dict:
					dem_dict[a] = 1
				else:
					dem_dict[a] += 1
			else:
				for val in mat_t[2].values():
					a = len(tuple(val.items()))
					if a not in dist_dict:
						dist_dict[a] = 1
					else:
						dist_dict[a] += 1

		print('Demotion')
		for k in sorted( dem_dict.keys() ):
			print('\t',k, dem_dict[k])

		print('Distance')
		for k in sorted( dist_dict.keys() ):
			print('\t',k, dist_dict[k])

	if False:
		for mat_t in matrices_paths_per_week_list[0]:
			if not mat_t[1]:
				print(mat_t[2])
				break

		for mat_t in matrices_paths_per_week_list[1]:
			if not mat_t[1]:
				print(mat_t[2])
				break

def single_multiple_ds_train():
	ds_list = tuple(
		map(
			lambda di: {
				"validation_indexes" : di["validation_indexes"],
				"training_indexes" : di["training_indexes"],
				"data_set" : np.array( di[ "data_set" ] ),
			},
			map(
				lambda i:
					pickle.load( open('enc_dec_ready_to_train_norm/' + str(i)\
						+ '.p' , 'rb')),
				range(len(WEEK_TIME_MOMENTS))
			)
		)
	)
	training_pairs_tuple =\
		tuple(
			itertools.chain.from_iterable(
				map(
					lambda d_p: map(\
							lambda ind: (d_p[0],ind),\
							d_p[1]['training_indexes']\
						),
					enumerate(ds_list)
				)
			)
		)
	validation_pairs_tuple =\
		tuple(
			itertools.chain.from_iterable(
				map(
					lambda d_p: map(\
							lambda ind: (d_p[0],ind),\
							d_p[1]['validation_indexes']\
						),
					enumerate(ds_list)
				)
			)
		)

	def extract_batch(batch_size, ind_tuple):
		x1 = np.empty( ( batch_size , 40 , 6072 , ) )
		x2 = np.empty( ( batch_size , 40 , 1 , ) )
		y2 = np.empty( ( batch_size , 1 , ) )
		y1 = np.empty( ( batch_size , 40 , 6072 , ) )

		while True:
			for i, ind_pair in enumerate(random.sample( ind_tuple , batch_size)):
				x1[i,:,:] = y1[i,:,:] = ds_list[ind_pair[0]]['data_set'][ ind_pair[1] - 39 : ind_pair[1] + 1 , 1:-1 ]
				x2[i,:,:] = ds_list[ind_pair[0]]['data_set'][ ind_pair[1] - 39 : ind_pair[1] + 1 , :1 ]
				y2[i,0] = ds_list[ind_pair[0]]['data_set'][ind_pair[1],-1]

			yield ( x2 , x1 ) , ( y1 , y2 )

	model = get_small_encoder_decoder_model_1({
		'new_index' : 0,
		'bins_no': 6072,
		'latent_v_count' : 10,
		'dropout_value' : 0.1,
	})

	model.summary()

	model.fit_generator(
		extract_batch(128,training_pairs_tuple),
		epochs=100000,
		steps_per_epoch=142,
		validation_data=extract_batch(128,validation_pairs_tuple),
		validation_steps=36,
		verbose=0,
		callbacks=[
			keras.callbacks.CSVLogger(
				'./pca_csv_folder/losses_386.csv'
			),
			keras.callbacks.ModelCheckpoint(
				"./pca_multiple_model_folders/models_386/model_{epoch:04d}.hdf5",
				monitor='val_loss',
				save_best_only=True
			),
			keras.callbacks.EarlyStopping(
				monitor='val_loss',
				patience=200,
			),
		]
	)

def plot_enc_dec_hist_and_results(index, history_flag, ds_flag):
	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)

	fig = plt.gcf()
	fig.set_size_inches(11,8)
	if history_flag:
		with open( './pca_csv_folder/losses_' + str(index) + '.csv' , 'rt' ) as f_d:
			csv_r = csv.reader( f_d )
			l = len(next(csv_r))
			train_trend_mape_list, train_recons_mape_list, valid_trend_mape_list, valid_recons_mape_list =\
				list(),list(),list(),list()
			for line in csv_r:
				if len(line) == l:
					train_trend_mape_list.append( float( line[1] ) if float( line[1] ) <= 100 else 100 )
					train_recons_mape_list.append( float( line[4] ) if float( line[4] ) <= 100 else 100 )
					valid_trend_mape_list.append( float( line[6] ) if float( line[6] ) <= 100 else 100 )
					valid_recons_mape_list.append( float( line[9] ) if float( line[9] ) <= 100 else 100 )

			plt.plot( 
				range( len( train_trend_mape_list ) ),
				train_trend_mape_list,
				'r-',
				label='Train Trend Prediction'
			)

			plt.plot( 
				range( len( train_recons_mape_list ) ),
				train_recons_mape_list,
				'g-',
				label='Train Reconstruction'
			)

			plt.plot( 
				range( len( valid_trend_mape_list ) ),
				valid_trend_mape_list,
				'b-',
				label='Valid Trend Prediction'
			)	

			plt.plot( 
				range( len( valid_recons_mape_list ) ),
				valid_recons_mape_list,
				'k-',
				label='Valid Reconstruction'
			)

			plt.xlabel('Epoch Index')
			plt.ylabel("MAPE")
			plt.legend()
			plt.savefig('results.png')

def plot_norm_thp():
	trend_tuple = tuple()
	for i in range(len(WEEK_TIME_MOMENTS)):
		trend_tuple += tuple( map( lambda e: e[-1], pickle.load(
				open( './enc_dec_ready_to_train_norm/' + str(i) + '.p', 'rb' )
				)["data_set"]))

	import matplotlib
	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 22}
	matplotlib.rc('font', **font)

	fig = plt.gcf()
	fig.set_size_inches(11,8)

	plt.plot( 
		range( len( trend_tuple ) ),
		trend_tuple,
	)

	plt.savefig('results.png')

if __name__ == '__main__':
	# train_main_for_generator_grid_search()
	# with tf.device('/gpu:3'):
	# for i in range(335, 340):
	# 	dump_encoder_decoder_plot(i)
	# train_main_for_generator_grid_search_0()
	dump_encoder_decoder_plot(335)
	# test_generator()
	# aggregate_and_generate_new_data_set()
	# analyze_minimal_set()
	# single_multiple_ds_train()
	# plot_enc_dec_hist_and_results(386, True, False)
	# plot_norm_thp()
	# aggregate_and_generate_new_data_set()
