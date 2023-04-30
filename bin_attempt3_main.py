import pickle
from collections import deque
import json
import multiprocessing as mp
import os
import random

def assign_se_readsizes_per_throughput(ind,se_len):

	thp_list = deque(pickle.load(open('./corrected_thp_folder/' + str(ind) + '.p','rb')))
	qrs_list = tuple(sorted(json.load(open('./agreggated_matrices/' + str(ind) + '.json','rt'))))

	print('Just read stuff !')

	while thp_list[0][0] - 120000 < qrs_list[0][0]:
		thp_list.popleft()

	while qrs_list[-1][0] < thp_list[-1][0]:
		thp_list.pop()

	print('Just clipped stuff !')
	rs_len = len(qrs_list)

	qrs_list = iter(qrs_list)

	thp_list =\
		tuple(
			map(
				lambda thp_p:\
					(
						thp_p[0],
						thp_p[1],
						[ 0 for _ in range( se_len ) ]
					),
				thp_list
			)
		)


	thp_i = 0

	for i , ( tm , first_choice_dict ) in enumerate( map( lambda p: ( p[0] , p[1][0] ) , qrs_list ) ):
		
		if i % 10000 == 0:
			print(i,'/',rs_len)
		
		while thp_i < len(thp_list) and thp_list[thp_i][0] <= tm: thp_i += 1

		if thp_i < len(thp_list) and thp_list[thp_i][0] - 120000 <= tm < thp_list[thp_i][0]:

			for di in map(lambda d: d.items(), first_choice_dict.values()):

				for se_ind, rs_v in di:

					thp_list[thp_i][2][int(se_ind)] += rs_v

	json.dump(thp_list,open('./one_throughput_one_se_rs_iterable_raw/'+str(ind)+'.json','wt'))

def norm_thp_per_proc(ind):
	thp_list = json.load(open('./one_throughput_one_se_rs_iterable_raw/'+str(ind)+'.json','rt'))

	tmi, tma, rsmi, rsma = None,None,None,None

	for _,t,rsi in thp_list:

		for rs in rsi:
			if rsmi is None or rs < rsmi: rsmi = rs
			if rsma is None or rs > rsma: rsma = rs

		if tmi is None or t < tmi: tmi = t
		if tma is None or t > tma: tma = t

	print(ind,tmi, tma, rsmi, rsma)

	json.dump(
		tuple(
			map(
				lambda tup:\
					(\
						tup[0],
						( tup[1] - tmi ) / ( tma - tmi ),
						tuple(
							map(
								lambda r: 2 * ( r - rsmi ) / ( rsma - rsmi ) - 1,
								tup[2]
							)
						)
					),
				thp_list
			)
		),
		open(
			'./one_throughput_one_se_rs_iterable_norm/'+str(ind)+'.json' , 'wt'
		)
	)

def dump_per_file_proc_func(file_name):
	len_num = len(json.load(open('./one_throughput_one_se_rs_iterable_norm/'+file_name,'rt')))

	valid_list = set(random.sample( range( time_window - 1 , len_num ) , int( 0.2 * ( len_num - time_window ) ) ))

	pickle.dump(
		(
			tuple( filter( lambda ind: ind not in valid_list , range( time_window - 1 , len_num ) ) ),
			tuple( valid_list ),
		),
		open( './split_indexes_folder/' + file_name.split('.')[0] + '_one_throughput_one_se_rs.p' , 'wb' )
	)

def train_test_split():
	global time_window
	time_window = 40

	path_list = os.listdir('./one_throughput_one_se_rs_iterable_norm/')

	mp.Pool( len( path_list ) ).map( dump_per_file_proc_func , path_list )

def extract_cern_only_rsi_0():
	indexes_containing_cern_list = list()

	for i, ( major_se, _ ) in enumerate(pickle.load( open( './minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p' , 'rb') )[1]):
		if 'cern' == major_se:
			indexes_containing_cern_list.append(i)

	print('# cern:',len(indexes_containing_cern_list))

	min_list, max_list = [ None for _ in range(len(indexes_containing_cern_list)) ],\
		[ None for _ in range(len(indexes_containing_cern_list)) ]

	rs_thp_list_of_lists = [list() for _ in range(len(indexes_containing_cern_list)+1)]

	for week_iterable in map( lambda fn: json.load( open( './one_throughput_one_se_rs_iterable_norm/'\
			+ str(fn) + '.json' , 'rt' ) ) , range( len( os.listdir( 'one_throughput_one_se_rs_iterable_norm'\
			) ) ) ):

		for _ , t , rsi in week_iterable:
			rs_thp_list_of_lists[-1].append(t)
			for i, ind in enumerate(indexes_containing_cern_list):
				rs_thp_list_of_lists[i].append(rsi[ind])
				if min_list[i] is None or rs_thp_list_of_lists[i][-1] < min_list[i]: min_list[i] = rs_thp_list_of_lists[i][-1]
				if max_list[i] is None or rs_thp_list_of_lists[i][-1] > max_list[i]: max_list[i] = rs_thp_list_of_lists[i][-1]

	print(min_list)
	print(max_list)

	for i , ( mi , ma ) in enumerate(zip(min_list,max_list)):
		rs_thp_list_of_lists[i] = tuple(
			map(
				# lambda e: (e - mi) / (ma - mi),
				lambda e: e,
				rs_thp_list_of_lists[i]
			)
		)

	pickle.dump(
		rs_thp_list_of_lists , open( 'a.p' , 'wb' )
	)

def extract_cern_only_rsi_1():
	indexes_containing_cern_list = list()

	for i, ( major_se, _ ) in enumerate(pickle.load( open( './minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p' , 'rb') )[1]):
		if 'cern' == major_se:
			indexes_containing_cern_list.append(i)

	print('# cern:',len(indexes_containing_cern_list))

	rs_thp_list_of_lists = [list() for _ in range(len(indexes_containing_cern_list)+1)]

	for week_iterable in map( lambda fn: json.load( open( './one_throughput_one_se_rs_iterable_norm/'\
			+ str(fn) + '.json' , 'rt' ) ) , range( len( os.listdir( 'one_throughput_one_se_rs_iterable_norm'\
			) ) ) ):

		for _ , t , rsi in week_iterable:
			rs_thp_list_of_lists[-1].append(2*t-1)
			for i, ind in enumerate(indexes_containing_cern_list):
				rs_thp_list_of_lists[i].append(rsi[ind])

	pickle.dump(
		rs_thp_list_of_lists , open( 'a.p' , 'wb' )
	)

if __name__ == '__main__':
	# assign_se_readsizes_per_throughput(2,len( pickle.load( open( './minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p' , 'rb') )[1] ))
	# se_len = len( pickle.load( open( './minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p' , 'rb') )[1] )
	# mp.Pool(3).starmap(\
	# 	assign_se_readsizes_per_throughput,
	# 	( (0,se_len) , (1,se_len) , (3,se_len) )
	# )

	# mp.Pool(4).map(\
	# 	norm_thp_per_proc,
	# 	(0,1,2,3)
	# )

	extract_cern_only_rsi_1()