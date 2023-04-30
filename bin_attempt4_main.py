# import matplotlib.pyplot as plt
import pickle
from collections import deque
import json
import multiprocessing as mp
import os
import random
import functools
import numpy as np
from tensorflow import keras
# import tensorflow as tf
import tensorflow.keras.backend as K
# import keras
from data_processing_utils import *

WEEK_TIME_MOMENTS = (\
	(1579215600000, 1579875041000),\
	(1580511600000, 1581289199000),\
	(1581289200000, 1581670035995),\
	(1589398710001, 1590609266276),\
	(1603753200000, 1605504964910),\
)

def get_only_cern_tm_rs(tm_rsDict_iterable):
	return\
		tuple(
			map(
				lambda a:\
					(\
						a[0],\
						functools.reduce(
							lambda acc, d:\
								acc + sum(\
									map(\
										lambda p: p[1],\
										filter(\
											lambda pp: int(pp[0]) in indexes_containing_cern_list,
											d.items()
										)
									)\
								),
							a[1][0].values(),
							0
						)
					),
				sorted( tm_rsDict_iterable )
			)
		)

def dump_cern_read_size_per_time_moment(week_ind):
	print('Will calculate rs per tm for:',week_ind)
	pickle.dump(
		get_only_cern_tm_rs(json.load(open('./agreggated_matrices/' + str(week_ind) + '.json','rt'))),
		open(
			'bin_attempt4_folders/time_tag_cern_read_size/' + str(week_ind) + '.p' , 'wb'
		)
	)

def get_min_max(week_ind):
	print('Will get min, max for:',week_ind)
	# a = pickle.load(open('bin_attempt4_folders/time_tag_cern_read_size/' + str(week_ind) + '.p' , 'rb'))
	a = json.load(open('bin_attempt4_folders/time_tag_cern_read_size/' + str(week_ind) + '.json' , 'rt'))
	return\
		min(map(lambda e: e[1], a)),\
		max(map(lambda e: e[1], a))

def norm_week(week_ind,mi,ma):
	print('Will normalize for:',week_ind)
	json.dump(
		tuple(
			map(
				lambda e:\
					(
						e[0],
						2 * ( e[1] - mi ) / ( ma - mi ) - 1,
					),
				json.load(open('bin_attempt4_folders/time_tag_cern_read_size/' + str(week_ind) + '.json' , 'rt'))
			)
		),
		open(
			'bin_attempt4_folders/time_tag_cern_read_size/' + str(week_ind) + '.json' , 'wt'
		)
	)

def train_test_split_proc_func(week_ind, rs_per_thp_count, window_size):
	thp_list = pickle.load(open('./corrected_thp_folder/' + str(week_ind) + '.p','rb'))
	left_thp_lim, right_thp_lim = 0, len(thp_list) - 1

	rs_list = pickle.load(open( 'bin_attempt4_folders/time_tag_cern_read_size/' + str(week_ind) + '.p' , 'rb' ))

	while thp_list[left_thp_lim][0] < rs_list[0][0] + rs_per_thp_count * 1000: left_thp_lim += 1

	left_thp_lim += window_size - 1

	while thp_list[right_thp_lim][0] > rs_list[-1][0]: right_thp_lim -= 1

	same_thp_tm_to_q_dict = dict()
	i_q = 0
	for thp_tm , _ in thp_list[ left_thp_lim - window_size + 1 : right_thp_lim + 1 ]:
		while rs_list[i_q][0] < thp_tm - 1: i_q += 1
		same_thp_tm_to_q_dict[ thp_tm ] = i_q

	valid_indexes_set = set(random.sample( range(left_thp_lim , right_thp_lim + 1) , int( 0.2 * ( right_thp_lim + 1 - left_thp_lim ) ) ))

	t,v =\
		tuple( filter( lambda ind: ind not in valid_indexes_set , range(left_thp_lim , right_thp_lim + 1) ) ),\
		tuple( valid_indexes_set )
	
	print( week_ind , len(t) , len(v) )

	add_info_to_indexes_func = lambda ind_tup:\
		tuple(
			map(
				lambda e:\
					(\
						week_ind,\
						e,\
						tuple(
							map(
								lambda i_thp: same_thp_tm_to_q_dict[ thp_list[ i_thp ][ 0 ] ],
								range( e - window_size + 1 , e + 1 )
							)
						)
					),
				ind_tup
			)
		)

	return add_info_to_indexes_func( t ) , add_info_to_indexes_func( v )

def train_test_split():
	pickle.dump(
		functools.reduce(
			lambda acc, x:\
				(
					acc[0] + x[0],\
					acc[1] + x[1],\
				),
			mp.Pool(4).starmap(
				train_test_split_proc_func,
				( (0,4800,40) , (1,4800,40) , (2,4800,40) , (3,4800,40) )
			),
			( tuple() , tuple() )
		),
		open(
			'./bin_attempt4_folders/test_train_indexes/combined_train_test_split.p' , 'wb'
		)
	)

def fill_possible_zeros(week_ind):
	# rs_list = pickle.load(open('bin_attempt4_folders/time_tag_cern_read_size/' + str(week_ind) + '.p' , 'rb'))
	rs_list = json.load(open('bin_attempt4_folders/time_tag_cern_read_size/' + str(week_ind) + '.json' , 'rt'))
	filled_rs_list = list()
	for i, p in enumerate( rs_list[:-1] ):
		filled_rs_list.append( p )
		if p[0] + 1000 != rs_list[i+1][0]:
			next_t = p[0] + 1000
			while next_t != rs_list[i+1][0]:
				filled_rs_list.append( ( next_t , 0 ) )
				next_t += 1000
	filled_rs_list.append( rs_list[-1] )
	json.dump(
		filled_rs_list,
		open(
			'bin_attempt4_folders/time_tag_cern_read_size/' + str(week_ind) + '.json' , 'wt'
		)
	)

def get_min_max_1(week_ind):
	return\
		functools.reduce(\
			lambda aux, x:\
				(\
					x[ 1 ] if x[ 1 ] < aux[ 0 ] else aux[ 0 ],
					x[ 1 ] if x[ 1 ] > aux[ 1 ] else aux[ 1 ],
				),\
			get_only_cern_tm_rs(json.load(open('./agreggated_matrices/' + str(week_ind) + '.json','rt'))),
			( 9223372036854775807 , -1 )
		)

def generate_read_sizes():
	global indexes_containing_cern_list
	indexes_containing_cern_list = list()
	for i, ( major_se, _ ) in enumerate(pickle.load( open( './minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p' , 'rb') )[1]):
		if 'cern' == major_se:
			indexes_containing_cern_list.append(i)
	indexes_containing_cern_list = set(indexes_containing_cern_list)

	# dump_cern_read_size_per_time_moment(2)
	# exit(0)

	p = mp.Pool(4)

	# p.map( dump_cern_read_size_per_time_moment , ( 0 , 1 , 2 , 3 ) )
	# for week_ind in ( 0 , 1 , 2 , 3 , 4 ):
	# 	os.system('cp ./agreggated_matrices/' + str(week_ind) + '.json bin_attempt4_folders/time_tag_cern_read_size/' + str(week_ind) + '.json' )


	# p.map( fill_possible_zeros , ( 0 , 1 , 2 , 3 , 4 ) )

	mi_ma_list = p.map( get_min_max_1 , ( 0 , 1 , 2 , 3 ) )

	mi,ma =\
		min(map(lambda pp: pp[0], mi_ma_list)),\
		max(map(lambda pp: pp[1], mi_ma_list))

	pickle.dump(
		( mi , ma ),
		open(
			'bin_attempt4_folders/time_tag_cern_read_size/min_max_read_size.p' , 'wb'
		)
	)

	# print(mi)
	# print(ma)

	# p.starmap( norm_week , ( (0,mi,ma) , (1,mi,ma) , (2,mi,ma) , (3,mi,ma) ) )

def rename_in_parsed_4():
	a =\
		tuple(
			filter(
				lambda e: 'parsed_44' in e,
				os.listdir('/nfs/public/mipopa/minimal_sets_and_parsed_matrices/parsed_4')
			)
		)
	for i,fn in enumerate(a):
		if i % 1000 == 0:
			print( i , '/' , len(a) )
		pickle.dump(
			pickle.load(
				open(
					'/nfs/public/mipopa/minimal_sets_and_parsed_matrices/parsed_4/' + fn,
					'rb'
				)
			),
			open(
				'/nfs/public/mipopa/minimal_sets_and_parsed_matrices/parsed_4/4_' + fn.split('_')[2],
				'wb'
			)
		)
		os.remove( '/nfs/public/mipopa/minimal_sets_and_parsed_matrices/parsed_4/' + fn )

def look_for_corrupted_matrices():
	count = 0
	for _ , is_distance , d in map(
			lambda fn: pickle.load( open( './minimal_sets_and_parsed_matrices/parsed_4/' + fn , 'rb' ) ),
			os.listdir( './minimal_sets_and_parsed_matrices/parsed_4/' )
		):
		if len(tuple(d.keys())) < 10:
			count += 1

	print(count)

def dump_matrices_time_moments():
	pickle.dump(
		tuple(
			map(
				lambda fn: pickle.load( open(\
					'./minimal_sets_and_parsed_matrices/parsed_4/' + fn , 'rb' ) )[0],
				os.listdir( './minimal_sets_and_parsed_matrices/parsed_4/' )
			)
		),
		open('a.p','wb')
	)

def plot_differences_in_time_moments_for_raw_matrices_0():
	import matplotlib.pyplot as plt
	a = sorted( pickle.load( open( 'a.p' , 'rb' ) ) )
	d = dict()
	max_diff_int_time = -1
	for diff in map(lambda p: p[1] - p[0] ,zip(a[:-1],a[1:])):
		if diff in d:
			d[ diff ] += 1
		else:
			d[ diff ] = 1
		if diff > max_diff_int_time:
			max_diff_int_time = diff

	print(max_diff_int_time)

	b = sorted( d.items() )
	plt.plot(
		tuple(map(lambda e: e[0],b)),
		tuple(map(lambda e: e[1],b)),
		'bo'
	)
	# plt.plot(
	# 	a,
	# 	[0 for _ in range(len(a))],
	# 	'bo'
	# )
	plt.show()

def plot_differences_in_time_moments_for_raw_matrices_1():
	import matplotlib.pyplot as plt
	a = sorted( pickle.load( open( 'a.p' , 'rb' ) ) )
	diff_iterable = tuple( map(lambda p: p[1] - p[0] ,zip(a[:-1],a[1:])) )

	plt.plot(
		a[1:],
		[0 for _ in range(len(a)-1)],
		'bo'
	)
	plt.plot(
		a[1:],
		diff_iterable,
		'ro'
	)
	plt.show()

def dump_clients_and_se_histogram():
	se_d, cl_d = dict(),dict()
	for _ , is_distance , d in map(
			lambda fn: pickle.load( open(\
				'./minimal_sets_and_parsed_matrices/parsed_4/' + fn , 'rb' ) ),
			os.listdir( './minimal_sets_and_parsed_matrices/parsed_4/' )
		):
		if is_distance:
			a = len( tuple( d.keys() ) )
			if a in cl_d: cl_d[a] += 1
			else: cl_d[a] = 1
			for v in map( lambda nd: len(tuple(nd.keys())) , d.values() ):
				if v in se_d:
					se_d[v] += 1
				else:
					se_d[v] = 1
		else:
			v = len( tuple( d.keys() ) )
			if v in se_d:
				se_d[v] += 1
			else:
				se_d[v] = 1
	pickle.dump((cl_d,se_d),open('a.p','wb'))

def plot_clients_and_se_histogram():
	import matplotlib.pyplot as plt
	cl_d,se_d = pickle.load(open('a.p','rb'))
	cl_list, se_list =\
		sorted(cl_d.items()),\
		sorted(se_d.items())
	miv,mav = min(map(lambda e: e[1],cl_list)), max(map(lambda e: e[1],cl_list))
	print('Clients:',min(map(lambda e: e[0],cl_list)),max(map(lambda e: e[0],cl_list)))
	print('cl_list:',cl_list)
	plt.plot(
		tuple(map(lambda e: e[0],cl_list)),
		tuple(map(lambda e: (e[1]-miv)/(mav-miv),cl_list)),
		'bo'
	)
	miv,mav = min(map(lambda e: e[1],se_list)), max(map(lambda e: e[1],se_list))
	print('Storage Elements:',min(map(lambda e: e[0],se_list)),max(map(lambda e: e[0],se_list)))
	print('se_list:',se_list)
	plt.plot(
		tuple(map(lambda e: e[0],se_list)),
		tuple(map(lambda e: (e[1]-miv)/(mav-miv),se_list)),
		'ro'
	)
	# plt.plot(
	# 	tuple(map(lambda e: e[0],se_list)),
	# 	tuple(map(lambda e: e[1],se_list)),
	# 	'ro'
	# )
	plt.show()

def get_same_number_of_elements_demotion_and_distance_matrices():
	distance_list, demotion_list = list(), list()
	for tm , f , d in map(lambda fn: pickle.load( open('minimal_sets_and_parsed_matrices/parsed_4/' + fn , 'rb' ) ), os.listdir( 'minimal_sets_and_parsed_matrices/parsed_4/' )):
		if f:
			distance_list.append( ( tm , d ) )
		else:
			demotion_list.append( ( tm , d ) )
	min_cl, min_se = pickle.load( open('minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p','rb') )
	distance_list, demotion_list =\
		add_distance_and_demotion_missing_elements(distance_list, demotion_list ,min_cl, min_se)
	for i,e in enumerate(distance_list):
		pickle.dump(
			e,
			open(
				'./minimal_sets_and_parsed_matrices/parsed_4_added_elements/4_di_' + str(i) + '.p',
				'wb'
			)
		)
	for i,e in enumerate(demotion_list):
		pickle.dump(
			e,
			open(
				'./minimal_sets_and_parsed_matrices/parsed_4_added_elements/4_de_' + str(i) + '.p',
				'wb'
			)
		)

def get_whole_matrices():
	distance_list, demotion_list = list(), list()
	for f , p in\
			map(
				lambda fn:\
					(
						'di' in fn,
						pickle.load(
							open(
								'minimal_sets_and_parsed_matrices/parsed_4_added_elements/' + fn,
								'rb'
							)
						)
					),
				os.listdir( 'minimal_sets_and_parsed_matrices/parsed_4_added_elements/' )
			):
		if f:
			distance_list.append( p )
		else:
			demotion_list.append( p )
	distance_list, demotion_list =\
		sorted(distance_list,key=lambda p:p[0]), sorted(demotion_list,key=lambda p:p[0])

	min_cl_iterable, min_se_iterable = pickle.load( open( 'minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p' , 'rb' ) )

	pickle.dump(
		get_complete_distance_matrix_1(
			distance_list , demotion_list , min_cl_iterable , min_se_iterable
		),
		open(
			'./whole_matrices/whole_for_4.p',
			'wb'
		)
	)

def dump_thp_associated_with_read_size():
	g_min_v , g_max_v , g_min_thp , g_max_thp = None , None , None , None
	r_list = []
	for i in ( 0 , 1 , 2 , 3 , 4 ):
		thp_list = pickle.load(open( './corrected_thp_folder/' + str(i) + '.p' , 'rb' ))
		rs_list = json.load(open('bin_attempt4_folders/time_tag_cern_read_size/' + str(i) + '.json' , 'rt'))

		result_list , min_v , max_v , min_thp , max_thp = associate_read_size_to_throughput(
			thp_list , rs_list
		)

		print(i, min_v , max_v , min_thp , max_thp)

		r_list.append( result_list )

		if g_min_v is None or g_min_v > min_v: g_min_v = min_v
		if g_min_thp is None or g_min_thp > min_thp: g_min_thp = min_thp
		if g_max_v is None or g_max_v < max_v: g_max_v = max_v
		if g_max_thp is None or g_max_thp < max_thp: g_max_thp = max_thp

	for i , rr_list in enumerate( r_list ):
		pickle.dump(
			tuple(
				map(
					lambda tup:\
						(
							( tup[0] - g_min_thp ) / ( g_max_thp - g_min_thp ),
							2 * ( tup[1] - g_min_v ) / ( g_max_v - g_min_v ) - 1,
						),
					rr_list
				)
			),
			open(
				'bin_attempt4_folders/thp_rs_normalized/' + str(i) + '.p' , 'wb'
			)
		)

def plot_results_from_dump_thp_associated_with_read_size():
	import matplotlib.pyplot as plt
	y1_list,y2_list = list(),list()
	for l in  map(lambda ii: pickle.load(open('thp_rs_normalized/'+str(ii)+'.p','rb')) , (0,1,2,3,4,) ):
		y1_list += list( map( lambda e: 2 * e[0] - 1 , l ) )
		y2_list += list( map( lambda e: e[1] , l ) )

	plt.plot(range(len(y2_list)),y2_list,label='read size')
	plt.plot(range(len(y1_list)),y1_list,label='throughput')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	generate_read_sizes()
	# train_test_split()
	# process_raw_distance_and_demotion_matrices(\
	# 	4 ,
	# 	'matrices_folder/log_folder_27th_oct/',
	# 	WEEK_TIME_MOMENTS[4][0],
	# 	WEEK_TIME_MOMENTS[4][1],
	# )
	# rename_in_parsed_4()
	# check_if_old_minimal_sets_correspond_with_new_data_and_create_new_minimal_set(
	# 	'./minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p',
	# 	'./minimal_sets_and_parsed_matrices/parsed_4/',
	# 	'./minimal_sets_and_parsed_matrices/minimal_sets_for_01234.p',
	# )
	# get_min_and_max_time_moment( './apicommands_folder/apicommands-nov-2020.log' )
	# look_for_corrupted_matrices()
	# dump_matrices_time_moments()
	# plot_differences_in_time_moments_for_raw_matrices_1()
	# dump_clients_and_se_histogram()
	# plot_clients_and_se_histogram()

	# get_same_number_of_elements_demotion_and_distance_matrices()
	# get_whole_matrices()
	# dump_thp_associated_with_read_size()
	# plot_results_from_dump_thp_associated_with_read_size()