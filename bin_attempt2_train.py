import json
import pickle
import itertools
import numpy as np
import random
import os
import multiprocessing as mp
import functools
from tensorflow import keras
from gc import collect

def batch_generator(indexes_per_week_iterable, batch_size):
	# cf_len, se_len =\
	# 	tuple(
	# 		map(
	# 			lambda e: len(e),
	# 			pickle.load(open('./minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p','rb'))
	# 		)
	# 	)

	# bins_list = tuple( map(\
	# 	lambda e: json.load( open( './norm_thp_bins_folder/' + str(e) + '.json' , 'rt' ) ),
	# 	range(len(indexes_per_week_iterable)) ) )
	# bins_list = tuple( map(\
	# 	lambda e: json.load( open( './minimized_data/' + str(e) + '.json' , 'rt' ) ),
	# 	range(len(indexes_per_week_iterable)) ) )

	for a_list in bins_list:
		for _,_,b_list in a_list: # b_list = list of bins per example
			bin_count = len(b_list)
			break
		break
	
	general_indexes_iterable =\
		functools.reduce(\
			lambda acc,x:\
				acc + tuple(map(lambda e: (x,e), indexes_per_week_iterable[x])),
			range(len(indexes_per_week_iterable)),
			tuple()
		)

	collect()

	# print( 'len(bins_list) = ' , len(bins_list))
	# print( 'len(bins_list[0]) = ' , len(bins_list[0]))
	# print( 'len(bins_list[0][2]) = ' , len(bins_list[0][2]))
	# exit(0)

	print('Generator initialized with bin_count =',bin_count)

	x_arr = np.zeros((
		batch_size , bin_count , 3 * cf_len * se_len
	))
	y_arr = np.empty((
		batch_size, 1
	))
	while True:
		x_arr[:,:,:] = 0
		for bs_i,(w,bs) in enumerate(random.sample(general_indexes_iterable, batch_size)):
			for bc in range(bin_count):
				try:
					for ch,ch_dict in enumerate( bins_list[w][bs][2][bc] ): 
						for cf,se_dict in ch_dict.items():
							for se,v in se_dict.items():
								x_arr[bs_i,bc,ch*cf_len*se_len+cf*se_len+se] = v
				except:
					print( 'Error for:', w , bs , bc )
					exit(0)
			y_arr[bs_i,0] = bins_list[w][bs][1]

		yield x_arr,y_arr

def minimize_into_test_data_proc_func(fnn):
	json.dump(
		json.load( open( './norm_thp_bins_folder/' + fnn , 'rt' ) )[:100],
		open(
			'./minimized_data/' + fnn , 'wt'
		)
	)

def minimize_into_test_data():
	fn_list = os.listdir( './norm_thp_bins_folder/' )
	mp.Pool(len(fn_list)).map(minimize_into_test_data_proc_func,fn_list)

def get_model_0(bin_no, feat_no):
	inp_layer = keras.layers.Input(shape=(bin_no, feat_no,))

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1000,
			activation='relu'
		)
	)(inp_layer)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dropout(0.1)(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=500,
			activation='relu'
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dropout(0.1)(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=250,
			activation='relu'
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dropout(0.1)(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=125,
			activation='relu'
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dropout(0.1)(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=50,
			activation='relu'
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dropout(0.1)(x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=50,
			return_sequences=True,
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=25,
			activation='relu'
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dropout(0.1)(x)

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=1,
			return_sequences=False,
		)
	)(x)
	x = keras.layers.BatchNormalization()(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def train_main_0():
	global cf_len, se_len

	tr_ind_per_week_list, va_ind_per_week_list = list(), list()
	for p in map(\
			lambda fn:\
				pickle.load( open(
					'split_indexes_folder/'+ fn, 'rb'
				)),
			os.listdir( './split_indexes_folder/' )
		):
		tr_ind_per_week_list.append(p[0])
		va_ind_per_week_list.append(p[1])

	cf_len, se_len =\
		tuple(
			map(
				lambda e: len(e),
				pickle.load(open('./minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p','rb'))
			)
		)

	model = get_model_0( 100 , 3*cf_len*se_len )

	model.summary()

	model.compile(
		optimizer=keras.optimizers.Adam(),
		loss='mean_absolute_percentage_error',
		metrics=['mae',]
	)

	print('Will load data !')

	global bins_list

	if True:
		tr_ind_per_week_list = tr_ind_per_week_list[:3]
		va_ind_per_week_list = va_ind_per_week_list[:3]
		collect()

	bins_list = tuple( map(\
		lambda e: json.load( open( './norm_thp_bins_folder/' + str(e) + '.json' , 'rt' ) ),
		range(len(tr_ind_per_week_list)) ) )

	collect()

	print('Loaded data !')

	for a_list in bins_list:
		for _,_,b_list in a_list: # b_list = list of bins per example
			for c_list in b_list: # c_list = list containing choices inside a bin
				for ch_dict in c_list: # ch_dict = dict with key=cf_number and value=dictionary
					cf_tuple = tuple(ch_dict.keys())
					for cf in cf_tuple:
						se_tuple = tuple(ch_dict[cf].keys())
						for se in se_tuple:
							ch_dict[cf][int(se)] = ch_dict[cf][se]
							del ch_dict[cf][se]
						ch_dict[int(cf)] = ch_dict[cf]
						del ch_dict[cf]
		collect()

	print('Transformed into integers !')
	
	model.fit_generator(
		batch_generator(tr_ind_per_week_list, 32),
		steps_per_epoch=100,
		epochs=20000,
		validation_data=batch_generator(va_ind_per_week_list, 32),
		validation_steps=20,
		verbose=0,
		callbacks=[keras.callbacks.CSVLogger('./a.csv'),],
	)

if __name__ == '__main__':
	train_main_0()

	# minimize_into_test_data()

	# a = tuple( map(
	# 	lambda fn:\
	# 		tuple(filter(lambda ind: ind < 100,pickle.load( open(
	# 		'split_indexes_folder/'+ fn[0] + '.p', 'rb'
	# 	))[0])),
	# 	os.listdir( 'split_indexes_folder/' )
	# ) )

	# next(
	# 	batch_generator(
	# 		a,
	# 		3
	# 	)
	# )