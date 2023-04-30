import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
import csv
import pickle
import json
import sys
import os
from collections import deque
from gc import collect
import itertools
import psutil
import math
import array
from copy import deepcopy
import sortedcontainers
from functools import reduce
import random

QUERRIES_RAW_FILES = (
	'./apicommands_folder/apicommands.log',
	'./apicommands_folder/apicommands2.log',
	'./apicommands_folder/apicommands3.log',
	'./apicommands_folder/apicommands4.log',
	'./apicommands_folder/apicommands-nov-2020.log'
)

CSV_THROUGHPUT_FILES = (\
	'./throughput_folder/january_month_throughput.csv',\
	'./throughput_folder/february_month.csv',\
	'./throughput_folder/february_month.csv',\
	'./throughput_folder/throughput_may.csv',\
	'./throughput_folder/thp_dec_2020.csv',\
)

def get_per_proc(fn):
	with open( fn , 'rt' ) as f:
		a = csv.reader(f)
		next(a)
		return max( map( lambda line_list: line_list[-1].count( ';' ) , a ) )

def get_max_number_of_options():
	print( tuple(mp.Pool(len(QUERRIES_RAW_FILES)).map( get_per_proc , QUERRIES_RAW_FILES ) ) )

def get_state_dictionaries_per_proc_1(pair):
	whole_i = 0
	a = int(file_variable[pair[0]].split(',')[0])
	while g_whole_matrices_iterable[whole_i+1][0] < a: whole_i+=1

	result_list = [ ( int( file_variable[pair[0]].split(',')[0] ) , [ dict() for _ in range( 9 ) ] , ) , ]

	for time_tag, read_size, cf_name, se_name_tuple_tuple in\
			map(\
				lambda ll:\
					(\
						int(ll[0]),\
						int(ll[2]),\
						ll[3].lower(),\
						tuple(\
							map(\
								lambda e:\
									tuple(
										map(
											lambda a: a.lower(),\
											e.split('::')[1:]
										)\
									),\
								ll[4].split(';')\
							)\
						)\
					),\
				map(
					lambda line: line.split(','),
					# file_variable[pair[0]:pair[1]]
					file_variable
				)
			):
		if time_tag != result_list[-1][0]:
			result_list.append(
				( time_tag , [dict() for _ in range(9)] , )
			)
			while whole_i < len(g_whole_matrices_iterable) and\
				g_whole_matrices_iterable[whole_i+1][0] < time_tag:
				whole_i+=1
		
		if cf_name not in g_cf_dict:
			for dict_cf_name in g_cf_dict.keys():
				if cf_name in dict_cf_name or dict_cf_name in cf_name:
					offset_by_cf = g_cf_dict[dict_cf_name] * g_se_len
					stable_cf_name = dict_cf_name
					break
		else:
			offset_by_cf = g_cf_dict[cf_name] * g_se_len
			stable_cf_name = cf_name

		score_list = list()

		for se_name in se_name_tuple_tuple:

			if se_name in g_se_dict:
				stable_se_name = se_name
			else:
				for k_name in g_se_dict.keys():
					if se_name[0] in k_name[0] or k_name[0] in se_name[0]:
						stable_se_name = k_name
						break

			score_list.append( (\
				g_whole_matrices_iterable[whole_i][1][ offset_by_cf + g_se_dict[ stable_se_name ] ],\
				stable_se_name,\
			) )

		score_list.sort()

		for ind , ( _ , se_name_tuple ) in enumerate(score_list):

			if stable_cf_name in result_list[-1][1][ind]:
				if ( se_name_tuple[0] + '::' + se_name_tuple[1] ) in result_list[-1][1][ind][stable_cf_name]:
					result_list[-1][1][ind][stable_cf_name][se_name_tuple[0] + '::' + se_name_tuple[1]] += read_size
				else:
					result_list[-1][1][ind][stable_cf_name][se_name_tuple[0] + '::' + se_name_tuple[1]] = read_size
			else:
				result_list[-1][1][ind][stable_cf_name] = { se_name_tuple[0] + '::' + se_name_tuple[1] : read_size }

	return result_list
 
def get_state_dictionaries_main_1(file_path_index):
	global file_variable, g_idexes_pairs_tuple, g_dump_folder_name,\
		g_cf_dict, g_se_dict, g_index, g_whole_matrices_iterable,\
		g_se_len

	file_variable = 'PULA MEA'
	print(mp.Pool(10).map(get_state_dictionaries_per_proc, range(10)))
	exit(0)

	file_variable = open(QUERRIES_RAW_FILES[file_path_index],'rt').read()
	file_variable = file_variable.split('\n')[1:]
	# file_variable = pickle.load( open( './apicommands_folder/test_api.p' , 'rb' ) )[1:]
	if len(file_variable[-1]) < 5: file_variable = file_variable[:-1]

	g_dump_folder_name = './answered_querries_dictionaries/'
	g_index = file_path_index

	cf_list, se_list = pickle.load(open('./minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p','rb'))

	g_se_len = len(se_list)

	g_cf_dict, g_se_dict = dict(), dict()

	for i,el in enumerate(cf_list):
		g_cf_dict[el] = i
	for i,el in enumerate(se_list):
		g_se_dict[el] = i

	del cf_list
	del se_list

	print('Loaded minimal sets !')

	g_whole_matrices_iterable = pickle.load(open('./whole_matrices/whole_for_0123.p','rb'))[g_index]

	print('Loaded whole matrices !')

	# Trim quantities
	
	# Trim whole matrices on left side
	i = 0
	a = int( file_variable[0].split(',')[0] )
	while g_whole_matrices_iterable[i+1][0] < a: i+=1
	g_whole_matrices_iterable = g_whole_matrices_iterable[i:]

	print( 'After whole matrices trimming:' )
	print( '\t' + str(g_whole_matrices_iterable[0][0]) + ' <-- g_whole_matrices_iterable[0][0]' )
	print( '\t' + file_variable[0].split(',')[0] + ' <-- file_variable[0][0]' )

	#Trim file variable on the left side
	i = 0
	while g_whole_matrices_iterable[0][0] > int(file_variable[i].split(',')[0]): i+=1
	file_variable = file_variable[i:]

	#Trim file variable on the right
	i = len(file_variable)-1
	while int(file_variable[i].split(',')[0]) > g_whole_matrices_iterable[-1][0]: i-=1
	file_variable = file_variable[:i+1]

	q_num = len(file_variable)
	print( 'File variable: first_time_tag=' + file_variable[0].split(',')[0],\
		'length=' + str(q_num)
	)
	print( 'Whole matrices: first_time_tag=' + str(g_whole_matrices_iterable[0][0]) )

	g_idexes_pairs_tuple = []
	i = 0
	a = q_num // n_proc
	remainder = q_num % n_proc
	while i < q_num:
		if remainder == 0:
			g_idexes_pairs_tuple.append((
				i,
				i + a,
			))
			i+=a
		else:
			g_idexes_pairs_tuple.append((
				i,
				i + a + 1,
			))
			i+=a+1
			remainder-=1

	if g_idexes_pairs_tuple[-1][1] != q_num:
		print('Work indexes are wrong !')

	print( 'Will start pool !' )

	p = mp.Pool(n_proc)

	del file_variable
	del g_whole_matrices_iterable
	del g_cf_dict
	del g_se_dict

	result_list_list = p.map( get_state_dictionaries_per_proc_1 , g_idexes_pairs_tuple )

	del g_idexes_pairs_tuple

	p.close()

	to_dump_list = result_list_list[0]

	for rll in result_list_list[1:]:
		for tm, rl in rll:
			if tm == to_dump_list[-1][0]:
				for old_dict, new_dict in zip( to_dump_list[-1][1] , rl ):
					for cf_name in new_dict.keys():
						if cf_name in old_dict:
							for se_name in new_dict[cf_name].keys():
								if se_name in old_dict[cf_name]:
									old_dict[cf_name][se_name] += new_dict[cf_name][se_name]
								else:
									old_dict[cf_name][se_name] = new_dict[cf_name][se_name]
						else:
							old_dict[cf_name] = new_dict[cf_name]
			else:
				to_dump_list += ( tm , rl )

	json.dump(
		to_dump_list,
		open( './agreggated_matrices/' + str(g_index) + '.json' , 'wt' )
	)

def get_state_dictionaries_per_proc_2(conn, first_q_time_tag):
	global g_whole_matrices_iterable

	result_list = [ ( first_q_time_tag , [ dict() for _ in range( 9 ) ] , ) , ]

	for ind_for_collect, ( time_tag, read_size, cf_name, se_name_tuple_tuple, ) in\
			enumerate(map(\
				lambda ll:\
					(\
						int(ll[0]),\
						int(ll[2]),\
						ll[3].lower(),\
						tuple(\
							map(\
								lambda e:\
									tuple(
										map(
											lambda a: a.lower(),\
											e.split('::')[1:]
										)\
									),\
								ll[4].split(';')\
							)\
						)\
					),\
				map(
					lambda line: line.split(','),
					g_short_f_v
				)
			)):
		if time_tag != result_list[-1][0]:

			result_list.append(
				( time_tag , [dict() for _ in range(9)] , )
			)

			prev_w_m = g_whole_matrices_iterable[0]
			g_whole_matrices_iterable.popleft()
			while g_whole_matrices_iterable and g_whole_matrices_iterable[0][0] < time_tag:
				prev_w_m = g_whole_matrices_iterable[0]
				g_whole_matrices_iterable.popleft()
			g_whole_matrices_iterable.append( prev_w_m )

		if ind_for_collect % 100000 == 0: collect() 

		if cf_name not in g_cf_dict:
			for dict_cf_name in g_cf_dict.keys():
				if cf_name in dict_cf_name or dict_cf_name in cf_name:
					offset_by_cf = g_cf_dict[dict_cf_name] * g_se_len
					stable_cf_name = dict_cf_name
					break
		else:
			offset_by_cf = g_cf_dict[cf_name] * g_se_len
			stable_cf_name = cf_name

		score_list = list()

		for se_name in se_name_tuple_tuple:

			if se_name in g_se_dict:
				stable_se_name = se_name
			else:
				for k_name in g_se_dict.keys():
					if se_name[0] in k_name[0] or k_name[0] in se_name[0]:
						stable_se_name = k_name
						break

			score_list.append( (\
				g_whole_matrices_iterable[0][1][ offset_by_cf + g_se_dict[ stable_se_name ] ],\
				stable_se_name,\
			) )

		score_list.sort()

		for ind , ( _ , se_name_tuple ) in filter( lambda p: p[0] < 9 ,enumerate(score_list) ):

			try:
				if stable_cf_name in result_list[-1][1][ind]:
					if ( se_name_tuple[0] + '::' + se_name_tuple[1] ) in result_list[-1][1][ind][stable_cf_name]:
						result_list[-1][1][ind][stable_cf_name][se_name_tuple[0] + '::' + se_name_tuple[1]] += read_size
					else:
						result_list[-1][1][ind][stable_cf_name][se_name_tuple[0] + '::' + se_name_tuple[1]] = read_size
				else:
					try:
						result_list[-1][1][ind][stable_cf_name] = { se_name_tuple[0] + '::' + se_name_tuple[1] : read_size }
					except:
						print('except 1')
						print(ind)
						print(stable_cf_name)
						print(se_name_tuple)
						print(read_size)
						pickle.dump(
							result_list[-1],
							open('debug.p','wb')
						)
						exit(0)
			except:
				print('except 2')
				print(ind)
				print(stable_cf_name)
				print(se_name_tuple)
				print(read_size)
				pickle.dump(
					result_list[-1],
					open('debug.p','wb')
				)
				exit(0)

	del g_whole_matrices_iterable

	conn.send( result_list )

	conn.close()

def get_state_dictionaries_main_2(file_path_index):
	global g_short_f_v, g_cf_dict, g_se_dict, g_whole_matrices_iterable,\
		g_se_len

	if True:
		file_variable = open(QUERRIES_RAW_FILES[file_path_index],'rt').read().split('\n')[1:]
	if False:
		file_variable = list(map(
			lambda e: ','.join(e),
			pickle.load( open( './apicommands_folder/test_api.p' , 'rb' ) )[1:]
		))
	if len(file_variable[-1]) < 5: file_variable = file_variable[:-1]

	cf_list, se_list = pickle.load(open('./minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p','rb'))

	g_se_len = len(se_list)

	g_cf_dict, g_se_dict = dict(), dict()

	for i,el in enumerate(cf_list):
		g_cf_dict[el] = i
	for i,el in enumerate(se_list):
		g_se_dict[el] = i

	del cf_list
	del se_list

	print('Loaded minimal sets !')

	g_whole_matrices_iterable = pickle.load(open('./whole_matrices/whole_for_0123.p','rb'))[file_path_index]

	print('Loaded whole matrices !')

	# Trim quantities
	
	# Trim whole matrices on left side
	i = 0
	a = int( file_variable[0].split(',')[0] )
	while g_whole_matrices_iterable[i+1][0] < a: i+=1
	g_whole_matrices_iterable = g_whole_matrices_iterable[i:]

	print( 'After whole matrices trimming:' )
	print( '\t' + str(g_whole_matrices_iterable[0][0]) + ' <-- g_whole_matrices_iterable[0][0]' )
	print( '\t' + file_variable[0].split(',')[0] + ' <-- file_variable[0][0]' )

	#Trim file variable on the left side
	i = 0
	while g_whole_matrices_iterable[0][0] > int(file_variable[i].split(',')[0]): i+=1
	file_variable = file_variable[i:]

	#Trim file variable on the right
	i = len(file_variable)-1
	while int(file_variable[i].split(',')[0]) > g_whole_matrices_iterable[-1][0]: i-=1
	file_variable = file_variable[:i+1]

	q_num = len(file_variable)
	print( 'File variable: first_time_tag=' + file_variable[0].split(',')[0],\
		'length=' + str(q_num)
	)
	print( 'Whole matrices: first_time_tag=' + str(g_whole_matrices_iterable[0][0]) )

	print( 'Will start procs one by one !' )

	remainder = q_num % n_proc
	quotient = q_num // n_proc
	proc_list = list()
	parent_pipe_list = list()

	g_whole_matrices_iterable = deque(g_whole_matrices_iterable)

	collect()

	for _ in range(remainder):
		
		g_short_f_v = iter(file_variable[ : quotient + 1 ])
		
		first_q_time_tag = int( file_variable[ 0 ].split( ',' )[ 0 ] )
		
		file_variable = file_variable[ quotient + 1 : ]
		
		prev_w_m = g_whole_matrices_iterable[0]
		g_whole_matrices_iterable.popleft()
		while g_whole_matrices_iterable[0][0] < first_q_time_tag:
			prev_w_m = g_whole_matrices_iterable[0]
			g_whole_matrices_iterable.popleft()
		g_whole_matrices_iterable.appendleft( prev_w_m )
		
		collect()

		parent_conn, child_conn = mp.Pipe()
		
		proc_list.append(
			mp.Process(
				target=get_state_dictionaries_per_proc_2,
				args=(child_conn,first_q_time_tag,),
			)
		)
		
		proc_list[-1].start()
		
		parent_pipe_list.append( parent_conn )

	for _ in range(remainder, n_proc):
		
		g_short_f_v = iter(file_variable[ : quotient ])
		
		first_q_time_tag = int( file_variable[ 0 ].split( ',' )[ 0 ] )
		
		file_variable = file_variable[ quotient : ]
		
		prev_w_m = g_whole_matrices_iterable[0]
		g_whole_matrices_iterable.popleft()
		while g_whole_matrices_iterable[0][0] < first_q_time_tag:
			prev_w_m = g_whole_matrices_iterable[0]
			g_whole_matrices_iterable.popleft()
		g_whole_matrices_iterable.appendleft( prev_w_m )
		
		collect()

		parent_conn, child_conn = mp.Pipe()
		
		proc_list.append(
			mp.Process(
				target=get_state_dictionaries_per_proc_2,
				args=(child_conn,first_q_time_tag,),
			)
		)
		
		proc_list[-1].start()
		
		parent_pipe_list.append( parent_conn )

	del file_variable
	del g_whole_matrices_iterable
	del g_cf_dict
	del g_se_dict
	collect()

	to_dump_list = parent_pipe_list[0].recv()

	for rll in map( lambda conn: conn.recv() , parent_pipe_list[1:] ):
		for tm, rl in rll:
			if tm == to_dump_list[-1][0]:
				for old_dict, new_dict in zip( to_dump_list[-1][1] , rl ):
					for cf_name in new_dict.keys():
						if cf_name in old_dict:
							for se_name in new_dict[cf_name].keys():
								if se_name in old_dict[cf_name]:
									old_dict[cf_name][se_name] += new_dict[cf_name][se_name]
								else:
									old_dict[cf_name][se_name] = new_dict[cf_name][se_name]
						else:
							old_dict[cf_name] = new_dict[cf_name]
			else:
				to_dump_list += ( tm , rl )

	for p in proc_list: p.join()	

	json.dump(
		to_dump_list,
		open( './agreggated_matrices/' + str(file_path_index) + '.json' , 'wt' )
	)

def get_state_dictionaries_per_proc_3(que, offset, work_length, first_q_time_tag, proc_ind, last_q_time_tag):
	global g_whole_matrices_iterable

	fp = open(QUERRIES_RAW_FILES[g_file_path_index],'rt')

	for _ in range( offset ): next(fp)

	collect()
	
	result_list = [ ( first_q_time_tag , [ dict() for _ in range( 9 ) ] , ) , ]

	for _ in range( g_whole_m_offset ): g_whole_matrices_iterable.popleft()

	prev_w_m = g_whole_matrices_iterable[0]
	g_whole_matrices_iterable.popleft()
	while g_whole_matrices_iterable and g_whole_matrices_iterable[0][0] < first_q_time_tag:
		prev_w_m = g_whole_matrices_iterable[0]
		g_whole_matrices_iterable.popleft()
	g_whole_matrices_iterable.appendleft( prev_w_m )

	collect()

	while g_whole_matrices_iterable and g_whole_matrices_iterable[-1][0] > last_q_time_tag:
		g_whole_matrices_iterable.pop()
	
	collect()

	next_partial_dump_index = 0

	results_file_paths_list = list()

	for ind_for_collect, ( time_tag, read_size, cf_name, se_name_tuple_tuple, ) in\
			zip(
				range( work_length ),
				map(\
					lambda ll:\
						(\
							int(ll[0]),\
							int(ll[2]),\
							ll[3].lower(),\
							tuple(\
								map(\
									lambda e:\
										tuple(
											map(
												lambda a: a.lower(),\
												e.split('::')[1:]
											)\
										),\
									ll[4].split(';')\
								)\
							)\
						),\
					map(\
						lambda line: line[:-1].split(','),\
						fp\
					)
				)
			):

		if time_tag != result_list[-1][0]:

			result_list.append(
				( time_tag , [dict() for _ in range(9)] , )
			)

			prev_w_m = g_whole_matrices_iterable[0]
			g_whole_matrices_iterable.popleft()
			while g_whole_matrices_iterable and g_whole_matrices_iterable[0][0] < time_tag:
				prev_w_m = g_whole_matrices_iterable[0]
				g_whole_matrices_iterable.popleft()
			g_whole_matrices_iterable.appendleft( prev_w_m )

		if ind_for_collect % 1000000 == 0 and ind_for_collect > 0:

			results_file_paths_list.append(
				g_dump_folder + str(proc_ind) + '_' + str(next_partial_dump_index) + '.json'
			)
			with open( results_file_paths_list[-1] , 'wt' ) as dump_f:
				json.dump( result_list , dump_f )

			result_list = [ ( time_tag , [ dict() for _ in range( 9 ) ] , ) , ]

			next_partial_dump_index += 1

		if ind_for_collect % 100000 == 0: collect() 

		if ind_for_collect % 500000 == 0:
			print( 'Worker', proc_ind , ':' , ind_for_collect , '/' , work_length )

		if cf_name not in g_cf_dict:
			for dict_cf_name in g_cf_dict.keys():
				if cf_name in dict_cf_name or dict_cf_name in cf_name:
					offset_by_cf = g_cf_dict[dict_cf_name] * g_se_len
					stable_cf_name = dict_cf_name
					break
		else:
			offset_by_cf = g_cf_dict[cf_name] * g_se_len
			stable_cf_name = cf_name

		score_list = list()

		for se_name in se_name_tuple_tuple:

			if se_name in g_se_dict:
				stable_se_name = se_name
			else:
				for k_name in g_se_dict.keys():
					if se_name[0] in k_name[0] or k_name[0] in se_name[0]:
						stable_se_name = k_name
						break


			score_list.append( (\
				g_whole_matrices_iterable[0][1][ offset_by_cf + g_se_dict[ stable_se_name ] ],\
				stable_se_name,\
			) )

		score_list.sort()

		cf_index = g_cf_dict[ stable_cf_name ]

		for ind , ( _ , se_name_tuple ) in filter( lambda p: p[0] < 9 ,enumerate(score_list) ):

			if cf_index in result_list[-1][1][ind]:
				if g_se_dict[se_name_tuple] in result_list[-1][1][ind][cf_index]:
					result_list[-1][1][ind][cf_index][g_se_dict[se_name_tuple]] += read_size
				else:
					result_list[-1][1][ind][cf_index][g_se_dict[se_name_tuple]] = read_size
			else:
				result_list[-1][1][ind][cf_index] = { g_se_dict[se_name_tuple] : read_size }

	del g_whole_matrices_iterable

	results_file_paths_list.append(
		g_dump_folder + str(proc_ind) + '_' + str(next_partial_dump_index) + '.json'
	)
	with open( results_file_paths_list[-1] , 'wt' ) as dump_f:
		json.dump( result_list , dump_f )

	print( 'Worker', proc_ind , ': Finished work !' )

	que.put(results_file_paths_list)

def get_trim_information(conn):
	left_trim_limit = 1
	file_variable = open(QUERRIES_RAW_FILES[g_file_path_index],'rt').read().split('\n')[1:]
	if len(file_variable[-1]) < 5:
		file_variable = file_variable[:-1]

	collect()

	print('Loaded file_variable !')

	g_whole_matrices_iterable = pickle.load(open('./whole_matrices/whole_for_0123.p','rb'))[g_file_path_index]

	print('Loaded whole matrices !')

	print('Before trim:')
	print('\tfile_variable =', len(file_variable))
	print('\tg_whole_matrices_iterable =', len(g_whole_matrices_iterable))

	# Trim quantities
	
	# Trim whole matrices on left side
	i = 0
	a = int( file_variable[0].split(',')[0] )
	while g_whole_matrices_iterable[i+1][0] < a: i+=1
	g_whole_matrices_iterable = g_whole_matrices_iterable[i:]

	whole_matrices_left_trim_limit = i

	print( 'After whole matrices trimming:' )
	print( '\t' + str(g_whole_matrices_iterable[0][0]) + ' <-- g_whole_matrices_iterable[0][0]' )
	print( '\t' + file_variable[0].split(',')[0] + ' <-- file_variable[0][0]' )

	#Trim file variable on the left side
	i = 0
	while g_whole_matrices_iterable[0][0] > int(file_variable[i].split(',')[0]):
		left_trim_limit += 1
		i+=1
	file_variable = file_variable[i:]
	collect()

	#Trim file variable on the right
	i = len(file_variable)-1
	while int(file_variable[i].split(',')[0]) > g_whole_matrices_iterable[-1][0]:
		i-=1
	file_variable = file_variable[:i+1]
	collect()

	q_num = len(file_variable)
	print( 'File variable: first_time_tag=' + file_variable[0].split(',')[0],\
		'length=' + str(q_num)
	)
	print( 'Whole matrices: first_time_tag=' + str(g_whole_matrices_iterable[0][0]) )

	print('After trim:')
	print('\tfile_variable =', len(file_variable))
	print('\tg_whole_matrices_iterable =', len(g_whole_matrices_iterable))

	remainder = q_num % n_proc
	quotient = q_num // n_proc

	work_load_list = []
	first_q_time_tag_list = []
	last_q_time_tag_list = []

	first_index_list, last_index_list = list(), list()

	if remainder == 0:
		first_q_index = 0
		last_q_index = quotient - 1
		work_load_list.append( ( left_trim_limit , quotient ) )
		first_q_time_tag_list.append(
			int( file_variable[ first_q_index ].split( ',' )[ 0 ] )
		)
		last_q_time_tag_list.append(
			int( file_variable[ last_q_index ].split( ',' )[ 0 ] )
		)
		first_index_list.append( first_q_index )
		last_index_list.append( last_q_index )
		first_q_index += quotient
		last_q_index += quotient
		for _ in range( 1 , n_proc ):
			work_load_list.append(
				(
					work_load_list[-1][0] + work_load_list[-1][1],
					quotient
				)
			)
			first_q_time_tag_list.append(
				int( file_variable[ first_q_index ].split( ',' )[ 0 ] )
			)
			last_q_time_tag_list.append(
				int( file_variable[ last_q_index ].split( ',' )[ 0 ] )
			)
			first_index_list.append( first_q_index )
			last_index_list.append( last_q_index )
			first_q_index += quotient
			last_q_index += quotient

	else:
		first_q_index = 0
		last_q_index = quotient
		work_load_list.append( ( left_trim_limit , quotient + 1 ) )
		first_q_time_tag_list.append(
			int( file_variable[ first_q_index ].split( ',' )[ 0 ] )
		)
		last_q_time_tag_list.append(
			int( file_variable[ last_q_index ].split( ',' )[ 0 ] )
		)
		first_index_list.append( first_q_index )
		last_index_list.append( last_q_index )
		first_q_index += quotient+1
		last_q_index += quotient+1
		if remainder > 1:
			for _ in range( 1 , remainder ):
				work_load_list.append(
					(
						work_load_list[-1][0] + work_load_list[-1][1],
						quotient + 1
					)
				)
				first_q_time_tag_list.append(
					int( file_variable[ first_q_index ].split( ',' )[ 0 ] )
				)
				last_q_time_tag_list.append(
					int( file_variable[ last_q_index ].split( ',' )[ 0 ] )
				)
				first_index_list.append( first_q_index )
				last_index_list.append( last_q_index )
				first_q_index += quotient + 1
				last_q_index += quotient + 1

		last_q_index -= 1

		for _ in range(remainder, n_proc):
			work_load_list.append(
				(
					work_load_list[-1][0] + work_load_list[-1][1],
					quotient
				)
			)
			first_q_time_tag_list.append(
				int( file_variable[ first_q_index ].split( ',' )[ 0 ] )
			)
			last_q_time_tag_list.append(
				int( file_variable[ last_q_index ].split( ',' )[ 0 ] )
			)
			first_index_list.append( first_q_index )
			last_index_list.append( last_q_index )
			first_q_index += quotient
			last_q_index += quotient

	for ind, (first_time, last_time, fi_ind, la_ind) in enumerate(zip(
				first_q_time_tag_list,
				last_q_time_tag_list,
				first_index_list,
				last_index_list,
			)):
		print(ind, first_time, last_time, fi_ind, la_ind)

	conn.send( ( whole_matrices_left_trim_limit , work_load_list , first_q_time_tag_list , last_q_time_tag_list ) )

	conn.close()

def get_trim_information_0(conn):
	'''
	incomplete
	'''
	def read_reverse_order(file_name):
	    # Open file for reading in binary mode
	    with open(file_name, 'rb') as read_obj:
	        # Move the cursor to the end of the file
	        read_obj.seek(0, os.SEEK_END)
	        # Get the current position of pointer i.e eof
	        pointer_location = read_obj.tell()
	        # Create a buffer to keep the last read line
	        buffer = bytearray()
	        # Loop till pointer reaches the top of the file
	        while pointer_location >= 0:
	            # Move the file pointer to the location pointed by pointer_location
	            read_obj.seek(pointer_location)
	            # Shift pointer location by -1
	            pointer_location = pointer_location -1
	            # read that byte / character
	            new_byte = read_obj.read(1)
	            # If the read byte is new line character then it means one line is read
	            if new_byte == b'\n':
	                # Fetch the line from buffer and yield it
	                yield buffer.decode()[::-1]
	                # Reinitialize the byte array to save next line
	                buffer = bytearray()
	            else:
	                # If last read character is not eol then add it in buffer
	                buffer.extend(new_byte)
	        # As file is read completely, if there is still data in buffer, then its the first line.
	        if len(buffer) > 0:
	            # Yield the first line too
	            yield buffer.decode()[::-1]

	left_trim_limit = 1
	file_pointer = open(QUERRIES_RAW_FILES[g_file_path_index],'rt')
	next(file_pointer)

	g_whole_matrices_iterable = pickle.load(open('./whole_matrices/whole_for_0123.p','rb'))[g_file_path_index]

	print('Loaded whole matrices !')

	# Trim whole matrices on left side
	i = 0
	a = int( next(file_pointer).split(',')[0] )
	while g_whole_matrices_iterable[i+1][0] < a: i+=1
	g_whole_matrices_iterable = g_whole_matrices_iterable[i:]

	whole_matrices_left_trim_limit = i

	print( 'After whole matrices trimming:' )
	print( '\t' + str(g_whole_matrices_iterable[0][0]) + ' <-- g_whole_matrices_iterable[0][0]' )
	print( '\t' + str(a) + ' <-- file_variable[0][0]' )

	#Trim file variable on the left side
	i = 0
	while g_whole_matrices_iterable[0][0] > a:
		left_trim_limit += 1
		i+=1
		a = int( next(file_pointer).split(',')[0] )
	
	collect()

	#Trim file variable on the right
	i = len(file_variable)-1
	while int(file_variable[i].split(',')[0]) > g_whole_matrices_iterable[-1][0]:
		i-=1
	file_variable = file_variable[:i+1]

	to_elim_from_back_count = 0

	reverse_obj = read_reverse_order( QUERRIES_RAW_FILES[g_file_path_index] )

	last_line = next(reverse_obj)
	if len(last_line) < 5:
		to_elim_from_back_count+=1
	else:
		a = int( last_line.split(',')[0] )

	for line in reverse_obj:
		pass 

	q_num = len(file_variable)
	print( 'File variable: first_time_tag=' + file_variable[0].split(',')[0],\
		'length=' + str(q_num)
	)
	print( 'Whole matrices: first_time_tag=' + str(g_whole_matrices_iterable[0][0]) )

def get_state_dictionaries_main_3(file_path_index, dump_folder):
	global g_short_f_v, g_cf_dict, g_se_dict, g_whole_matrices_iterable,\
		g_se_len, g_whole_m_offset, g_file_path_index, g_dump_folder

	g_dump_folder = dump_folder

	g_file_path_index = file_path_index

	parent_conn, child_conn = mp.Pipe()

	proc = mp.Process(
		target=get_trim_information,
		args=(child_conn,),
	)

	proc.start()

	g_whole_m_offset , work_load_list , first_q_time_tag_list,\
		last_q_time_tag_list = parent_conn.recv()

	proc.join()

	# exit(0)

	del proc
	del parent_conn
	del child_conn

	collect()

	cf_list, se_list = pickle.load(open('./minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p','rb'))

	g_se_len = len(se_list)

	g_cf_dict, g_se_dict = dict(), dict()

	for i,el in enumerate(cf_list):
		g_cf_dict[el] = i
	for i,el in enumerate(se_list):
		g_se_dict[el] = i

	del cf_list
	del se_list

	collect()

	print('Loaded minimal sets !')

	g_whole_matrices_iterable = deque(
		pickle.load(open('./whole_matrices/whole_for_0123.p','rb'))[g_file_path_index]
	)

	proc_list = list()

	conn_que = mp.Queue()

	for ( offset, work_length ) , first_q_time_tag , proc_ind , last_q_time_tag in\
			zip( work_load_list , first_q_time_tag_list , range( n_proc ), last_q_time_tag_list):

		proc_list.append(
			mp.Process(
				target=get_state_dictionaries_per_proc_3,
				args=(conn_que,offset,work_length,first_q_time_tag,proc_ind,last_q_time_tag),
			)
		)
		proc_list[-1].start()
	
	del g_cf_dict
	del g_se_dict
	del g_whole_matrices_iterable

	collect()

	path_list_list = list()
	for _ in range(n_proc):
		path_list_list.append( conn_que.get() )

	for p in proc_list: p.join()

	# to_dump_dict = dict()

	# for path_list in path_list_list:
	# 	for path in path_list:
	# 		for tm, dict_list in json.load(open( path , 'rt' )):
	# 			if tm in to_dump_dict:
	# 				for old_dict, new_dict in zip( to_dump_dict[tm] , dict_list ):
	# 					for cf_ind, se_dict in new_dict.items():
	# 						if cf_ind in old_dict:
	# 							for se_ind, rs in se_dict.items():
	# 								if se_ind in old_dict[cf_ind]:
	# 									old_dict[cf_ind][se_ind] += rs
	# 								else:
	# 									old_dict[cf_ind][se_ind] = rs
	# 						else:
	# 							old_dict[cf_ind] = se_dict
	# 			else:
	# 				to_dump_dict[tm] = dict_list


	# del conn_que
	# del proc_list
	# collect()

	# json.dump(
	# 	sorted(to_dump_dict.items()),
	# 	open( './agreggated_matrices/' + str(file_path_index) + '.json' , 'wt' )
	# )

def seq_proc(week_ind):
	reader_obj = csv.reader(open(QUERRIES_RAW_FILES[week_ind],'rt'))

	next(reader_obj)

	whole_matrices_iterable = deque(pickle.load(open('./whole_matrices/whole_for_0123.p','rb'))[week_ind])

	first_line = next(reader_obj)
	first_time_tag = int(first_line[0])

	while first_time_tag < whole_matrices_iterable[0][0]:
		first_line = next(reader_obj)
		first_time_tag = int(first_line[0])

	prev_w_m = whole_matrices_iterable[0]
	whole_matrices_iterable.popleft()
	while whole_matrices_iterable and whole_matrices_iterable[0][0] < first_time_tag:
		prev_w_m = whole_matrices_iterable[0]
		whole_matrices_iterable.popleft()
	whole_matrices_iterable.append( prev_w_m )

	cf_list, se_list = pickle.load(open('./minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p','rb'))

	g_se_len = len(se_list)

	g_cf_dict, g_se_dict = dict(), dict()

	for i,el in enumerate(cf_list):
		g_cf_dict[el] = i
	for i,el in enumerate(se_list):
		g_se_dict[el] = i

	del cf_list
	del se_list

	collect()

	print('Loaded minimal sets !')

	result_list = [ ( first_time_tag , [ dict() for _ in range( 9 ) ] , ) , ]

	for ind_for_collect, ( time_tag, read_size, cf_name, se_name_tuple_tuple, ) in\
			enumerate(
				map(\
					lambda ll:\
						(\
							int(ll[0]),\
							int(ll[2]),\
							ll[3].lower(),\
							tuple(\
								map(\
									lambda e:\
										tuple(
											map(
												lambda a: a.lower(),\
												e.split('::')[1:]
											)\
										),\
									ll[4].split(';')\
								)\
							)\
						),\
					itertools.chain(
						( first_line , ) , reader_obj
					)
				)
			):

		if time_tag != result_list[-1][0]:

			result_list.append(
				( time_tag , [dict() for _ in range(9)] , )
			)

			prev_w_m = whole_matrices_iterable[0]
			whole_matrices_iterable.popleft()
			while whole_matrices_iterable and whole_matrices_iterable[0][0] < time_tag:
				prev_w_m = whole_matrices_iterable[0]
				whole_matrices_iterable.popleft()
			whole_matrices_iterable.append( prev_w_m )

		if ind_for_collect % 100000 == 0: collect() 

		if ind_for_collect % 500000 == 0:
			print( 'Reached:' , ind_for_collect )

		if cf_name not in g_cf_dict:
			for dict_cf_name in g_cf_dict.keys():
				if cf_name in dict_cf_name or dict_cf_name in cf_name:
					offset_by_cf = g_cf_dict[dict_cf_name] * g_se_len
					stable_cf_name = dict_cf_name
					break
		else:
			offset_by_cf = g_cf_dict[cf_name] * g_se_len
			stable_cf_name = cf_name

		score_list = list()

		for se_name in se_name_tuple_tuple:

			if se_name in g_se_dict:
				stable_se_name = se_name
			else:
				for k_name in g_se_dict.keys():
					if se_name[0] in k_name[0] or k_name[0] in se_name[0]:
						stable_se_name = k_name
						break

			score_list.append( (\
				whole_matrices_iterable[0][1][ offset_by_cf + g_se_dict[ stable_se_name ] ],\
				stable_se_name,\
			) )

		score_list.sort()

		cf_index = g_cf_dict[ stable_cf_name ]

		for ind , ( _ , se_name_tuple ) in filter( lambda p: p[0] < 9 ,enumerate(score_list) ):

			if cf_index in result_list[-1][1][ind]:
				if g_se_dict[se_name_tuple] in result_list[-1][1][ind][cf_index]:
					result_list[-1][1][ind][cf_index][g_se_dict[se_name_tuple]] += read_size
				else:
					result_list[-1][1][ind][cf_index][g_se_dict[se_name_tuple]] = read_size
			else:
				result_list[-1][1][ind][cf_index] = { g_se_dict[se_name_tuple] : read_size }

	json.dump(
		result_list,
		open( './agreggated_matrices/' + str(week_ind) + '.json' , 'wt' )
	)

def correct_and_rectify_thp_file(thp_ind):
	reader_obj = csv.reader( open( CSV_THROUGHPUT_FILES[thp_ind] , 'rt' ) )

	first_line = next(reader_obj)

	# print(first_line)

	alien_db2_index = None
	
	for ind, name in enumerate(first_line):
		if 'aliendb2' in name:
			alien_db2_index = ind
			break

	thp_list = list()
	for line in reader_obj:
		if line[alien_db2_index] != '':
			thp_list.append((
				int(line[0]) * 1000,
				float(line[alien_db2_index]),
			))
			break
	
	for line in reader_obj:
		if line[alien_db2_index] != '':
			time_tag = int(line[0]) * 1000
			if time_tag - thp_list[-1][0] > 120000:
				tt_it = thp_list[-1][0] + 120000
				while tt_it < time_tag:
					thp_list.append((
						tt_it,
						thp_list[-1][1],
					))
					tt_it += 120000
			thp_list.append((
				time_tag,
				float(line[alien_db2_index]),
			))

	is_2_min_spaced_flag = True
	prev_tt = thp_list[0][0]
	for tt, tv in thp_list[1:]:
		if tt - prev_tt != 120000:
			is_2_min_spaced_flag = False
			break
		prev_tt = tt
	if not is_2_min_spaced_flag:
		print('Not 2 minutes spaced !')
		exit(0)
	pickle.dump(
		thp_list,
		open(
			'corrected_thp_folder/' + str(thp_ind) + '.p',
			'wb'
		)
	)

def collect_from_folder_to_single_file(ind):
	result_list = []

	total = len(os.listdir('./proc_aux_folder_' + str(ind)))

	a = 0
	for w_ind, mat_list in\
			map(
				lambda f_tup: (f_tup[2],json.load(open('./proc_aux_folder_' + str(ind) + '/' + f_tup[2],'rt'))),
				sorted(
					map(
						lambda fn_list:\
							(\
								int(fn_list[0]),\
								int(fn_list[1].split('.')[0]),\
								fn_list[0] + '_' + fn_list[1],\
							),
						map(
							lambda fn: fn.split( '_' ),
							os.listdir('./proc_aux_folder_' + str(ind))
						)
					)
				)
			):
		for tm, dict_list in mat_list:
			if len(result_list) == 0 or result_list[-1][0] != tm:
				result_list.append(
					( tm , [ dict() for _ in range(9) ] , )
				)
			for old_dict, new_dict in zip( result_list[-1][1] , dict_list ):
				for cf_ind, se_dict in map(lambda p: (int(p[0]),p[1]) , new_dict.items()):
					if cf_ind in old_dict:
						for se_ind, rs in map(lambda p: (int(p[0]),p[1]) , se_dict.items()):
							if se_ind in old_dict[cf_ind]:
								old_dict[cf_ind][se_ind] += rs
							else:
								old_dict[cf_ind][se_ind] = rs
					else:
						old_dict[cf_ind] = dict(map(lambda p: (int(p[0]),p[1]) , se_dict.items()))				

		collect()
		print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
		print(w_ind, a, total, psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
		a += 1

	json.dump(
		result_list,
		open( './agreggated_matrices/' + str(ind) + '.json' , 'wt' )
	)
			# if len(result_list) == 0 or result_list[-1][0] != tm:
			# 	result_list.append( ( tm , dict_list ) )
			# else:
			# 	for old_dict, new_dict in zip( result_list[-1][1] , dict_list ):
			# 		for cf_ind, se_dict in new_dict.items():
			# 			if cf_ind in old_dict:
			# 				for se_ind, rs in se_dict.items():
			# 					if se_ind in old_dict[cf_ind]:
			# 						old_dict[cf_ind][se_ind] += rs
			# 					else:
			# 						old_dict[cf_ind][se_ind] = rs
			# 			else:
			# 				old_dict[cf_ind] = se_dict

def analyse_non_agregg_results_0(ind):

	num_of_points = file_count = 0

	cf_sum = cf_count = se_sum = se_count = 0

	for w_ind, mat_list in\
			map(
				lambda f_tup: (f_tup[2],json.load(open('./proc_aux_folder_' + str(ind) + '/' + f_tup[2],'rt'))),
				sorted(
					map(
						lambda fn_list:\
							(\
								int(fn_list[0]),\
								int(fn_list[1].split('.')[0]),\
								fn_list[0] + '_' + fn_list[1],\
							),
						map(
							lambda fn: fn.split( '_' ),
							os.listdir('./proc_aux_folder_' + str(ind))
						)
					)
				)
			):

		print( 'Workin on:' , w_ind , 'len =', len(mat_list) )

		num_of_points += len(mat_list)
		file_count += 1

		for _ , dict_list in mat_list:
			cf_sum += len(dict_list[0].keys())
			cf_count += 1
			for se_dict in dict_list[0].values():
				se_sum += len(se_dict.keys())
				se_count += 1

	print('Average points per file:',num_of_points/file_count)
	print('Average cf count:', cf_sum/cf_count)
	print('Average se count:', se_sum/se_count)

	'''
		0:
			Average points per file: 1742.38
			Average cf count: 30.546370891703138
			Average se count: 1.6655648953454134
		3:
			Average points per file: 216241.9071146245
			Average cf count: 1.9824009955180757
			Average se count: 1.1247693185719287
	'''

def analyse_csv_time_order(ind):
	fp = open(QUERRIES_RAW_FILES[ind],'rt')
	next(fp)
	prev_value = int( next(fp).split(',')[0] )
	for ind, line in enumerate( fp ):
		now_value = int( line.split(',')[0] )
		if prev_value > now_value:
			print('Found bad time tag:')
			print('\tnow_value =',now_value)
			print('\tprev_value =',prev_value)
			print('\tind =',ind)
			break
		if ind % 1000000 == 0:
			print(ind)
		prev_value = now_value

def analyse_non_agregg_results_1(ind):
	for w_ind, mat_list in\
			map(
				lambda f_tup: (f_tup[2],json.load(open('./proc_aux_folder_' + str(ind) + '/' + f_tup[2],'rt'))),
				sorted(
					map(
						lambda fn_list:\
							(\
								int(fn_list[0]),\
								int(fn_list[1].split('.')[0]),\
								fn_list[0] + '_' + fn_list[1],\
							),
						map(
							lambda fn: fn.split( '_' ),
							os.listdir('./proc_aux_folder_' + str(ind))
						)
					)
				)
			):

		print( 'Workin on:' , w_ind , 'len =', len(mat_list) )

		tm_set = set()

		for tm, dict_list in mat_list:
			if tm in tm_set:
				print('Found a double !!!')
			else:
				tm_set.add(tm)

def get_state_dictionaries_per_proc_4(conn,proc_ind,offset, work_length):
	# print('Worker',proc_ind,': Started ! Will work for', work_length,'lines.')
	print('Worker',proc_ind,': Started !')
	fp = open(QUERRIES_RAW_FILES[g_file_path_index],'rt')
	for _ in range( offset ): next(fp)
	collect()
	# next_partial_dump_index = 0
	results_dict = dict()
	wm_i = None
	for ind_for_collect, ( time_tag, read_size, cf_name, se_name_tuple_tuple, ) in\
			filter(
				lambda p: g_whole_matrices_iterable[0][0] <= p[1][0]\
					<= g_whole_matrices_iterable[-1][0],
				zip(
					range( work_length ),
					map(\
						lambda ll:\
							(\
								int(ll[0]),\
								int(ll[2]),\
								ll[3].lower(),\
								tuple(\
									map(\
										lambda e:\
											tuple(
												map(
													lambda a: a.lower(),\
													e.split('::')[1:]
												)\
											),\
										ll[4].split(';')\
									)\
								)\
							),\
						map(\
							lambda line: line[:-1].split(','),\
							fp\
						)
					)
				)
			):
		if ind_for_collect % 1000000 == 0 and ind_for_collect > 0:
			# with open( g_dump_folder + str(proc_ind) + '_' + str(next_partial_dump_index) + '.json' , 'wt' ) as dump_f:
			# 	json.dump( results_dict , dump_f )

			# results_dict = dict()

			# next_partial_dump_index += 1

			print('Worker',proc_ind,':',ind_for_collect,'/',work_length,',',wm_i)

		upper_limit = len(g_whole_matrices_iterable)-1
		low = 0
		high = upper_limit
		wm = None
		wm_i = None
		while low + 1 != high:
			if g_whole_matrices_iterable[low][0] <= time_tag < g_whole_matrices_iterable[low+1][0]:
				wm = g_whole_matrices_iterable[low][1]
				wm_i = low
				break
			if ( high == upper_limit and g_whole_matrices_iterable[high][0] == time_tag)\
				or ( g_whole_matrices_iterable[high][0] <= time_tag < g_whole_matrices_iterable[high+1][0] ):
				wm = g_whole_matrices_iterable[high][1]
				wm_i = high
				break
			mid = (low + high) // 2
			if g_whole_matrices_iterable[mid][0] <= time_tag < g_whole_matrices_iterable[mid+1][0]:
				wm = g_whole_matrices_iterable[mid][1]
				wm_i = mid
				break
			if time_tag < g_whole_matrices_iterable[mid][0]:
				high = mid
			else:
				low = mid
		if wm is None:
			if g_whole_matrices_iterable[low][0] <= time_tag < g_whole_matrices_iterable[low+1][0]:
				wm = g_whole_matrices_iterable[low][1]
				wm_i = low

			if ( high == upper_limit and g_whole_matrices_iterable[high][0] == time_tag)\
				or ( g_whole_matrices_iterable[high][0] <= time_tag < g_whole_matrices_iterable[high+1][0] ):
				wm = g_whole_matrices_iterable[high][1]
				wm_i = high

		if cf_name not in g_cf_dict:
			for dict_cf_name in g_cf_dict.keys():
				if cf_name in dict_cf_name or dict_cf_name in cf_name:
					offset_by_cf = g_cf_dict[dict_cf_name] * g_se_len
					stable_cf_name = dict_cf_name
					break
		else:
			offset_by_cf = g_cf_dict[cf_name] * g_se_len
			stable_cf_name = cf_name

		score_list = list()

		for se_name in se_name_tuple_tuple:

			if se_name in g_se_dict:
				stable_se_name = se_name
			else:
				for k_name in g_se_dict.keys():
					if se_name[0] in k_name[0] or k_name[0] in se_name[0]:
						stable_se_name = k_name
						break

			score_list.append( (\
				wm[ offset_by_cf + g_se_dict[ stable_se_name ] ],\
				stable_se_name,\
			) )

		score_list.sort()

		# cf_index = g_cf_dict[ stable_cf_name ]
		# if time_tag not in results_dict:
		# 	results_dict[time_tag] = (dict(),dict(),dict(),dict(),dict(),\
		# 		dict(),dict(),dict(),dict(),)
		# for ind , ( _ , se_name_tuple ) in filter( lambda p: p[0] < 9 , enumerate( score_list ) ):
		# 	if cf_index in results_dict[time_tag][ind]:
		# 		if g_se_dict[se_name_tuple] in results_dict[time_tag][ind][cf_index]:
		# 			results_dict[time_tag][ind][cf_index][g_se_dict[se_name_tuple]] += read_size
		# 		else:
		# 			results_dict[time_tag][ind][cf_index][g_se_dict[se_name_tuple]] = read_size
		# 	else:
		# 		results_dict[time_tag][ind][cf_index] = { g_se_dict[se_name_tuple] : read_size }
	
		if 'cern' in score_list[ 0 ][ 1 ][ 0 ]:
			if time_tag in results_dict:
				results_dict[ time_tag ] += read_size
			else:
				results_dict[ time_tag ] = read_size

	# with open( g_dump_folder + str(proc_ind) + '_' + str(next_partial_dump_index) + '.json' , 'wt' ) as dump_f:
	# 	json.dump( results_dict , dump_f )
	
	print('Worker',proc_ind,': Finished with',len(results_dict), 'dictionary entries !')

	# conn.send( (proc_ind, results_dict) )
	conn.send( results_dict )

def agreggate_dictionary_list_into_single_dict(input_dict_iterable):
	to_dump_dict = sortedcontainers.SortedDict()

	for ind,rd in enumerate(input_dict_iterable):
		print('Started agreggation for ' + str(ind) + '/' + str(n_proc))
		for tm, dict_list in rd.items():
			if tm in to_dump_dict:
				for old_dict, new_dict in zip( to_dump_dict[tm] , dict_list ):
					for cf_name in new_dict.keys():
						if cf_name in old_dict:
							for se_name in new_dict[cf_name].keys():
								if se_name in old_dict[cf_name]:
									old_dict[cf_name][se_name] += new_dict[cf_name][se_name]
								else:
									old_dict[cf_name][se_name] = new_dict[cf_name][se_name]
						else:
							old_dict[cf_name] = new_dict[cf_name]
			else:
				to_dump_dict[tm] = dict_list
		print('Agreggation finished for ' + str(ind) + '/' + str(n_proc))

	return to_dump_dict

def get_state_dictionaries_main_4(file_path_index, dump_folder):
	'''
	Main function for answering queries and adding them to a dictionary list.
	'''

	global g_cf_dict, g_se_dict, g_whole_matrices_iterable,\
		g_se_len, g_dump_folder, g_file_path_index

	g_file_path_index = file_path_index

	g_dump_folder = dump_folder

	cf_list, se_list = pickle.load(open('./minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p','rb'))

	g_se_len = len(se_list)

	g_cf_dict, g_se_dict = dict(), dict()

	for i,el in enumerate(cf_list):
		g_cf_dict[el] = i
	for i,el in enumerate(se_list):
		g_se_dict[el] = i

	del cf_list
	del se_list

	print('Loaded minimal sets !')

	# g_whole_matrices_iterable = tuple(sorted(pickle.load(open('./whole_matrices/whole_for_0123.p','rb'))[file_path_index]))
	g_whole_matrices_iterable = tuple(sorted(pickle.load(open('./whole_matrices/whole_for_4.p','rb'))))

	print('Loaded whole matrices ! There are ' + str(len(g_whole_matrices_iterable))\
		+ ' with an average binary search iteration count of ' + str(math.log2(len(g_whole_matrices_iterable))) +'.'
	)

	q_num = int(os.popen('wc -l ' + QUERRIES_RAW_FILES[file_path_index]).read().split()[0])
	if len(os.popen('tail -1 ' + QUERRIES_RAW_FILES[file_path_index]).read()) < 5:
		q_num -= 1

	print('There are ' + str(q_num) + ' lines to be processed !')

	print( 'Will start procs one by one !' )

	remainder = q_num % n_proc
	quotient = q_num // n_proc
	proc_list = list()
	parent_pipe_list = list()

	collect()

	proc_ind = 0

	offset = 1

	for _ in range(remainder):
				
		collect()

		parent_conn, child_conn = mp.Pipe()
		
		proc_list.append(
			mp.Process(
				target=get_state_dictionaries_per_proc_4,
				args=(child_conn,proc_ind,offset,quotient+1),
			)
		)
		
		proc_list[-1].start()
		
		parent_pipe_list.append( parent_conn )

		proc_ind += 1

		offset += quotient+1

	for _ in range(remainder, n_proc):
		
		collect()

		parent_conn, child_conn = mp.Pipe()
		
		proc_list.append(
			mp.Process(
				target=get_state_dictionaries_per_proc_4,
				args=(child_conn,proc_ind,offset,quotient),
			)
		)
		
		proc_list[-1].start()
		
		parent_pipe_list.append( parent_conn )

		proc_ind += 1

		offset += quotient

	del g_whole_matrices_iterable
	del g_cf_dict
	del g_se_dict
	collect()

	# to_dump_dict = agreggate_dictionary_list_into_single_dict(
	# 	map( lambda conn: conn.recv() , parent_pipe_list )
	# )

	to_dump_dict = dict()
	for d_items in map( lambda conn: conn.recv().items() , parent_pipe_list ):
		for tm , rs in d_items:
			if tm in to_dump_dict:
				to_dump_dict[tm] += rs
			else:
				to_dump_dict[tm] = rs

	for p in proc_list: p.join()	

	print('Will start sort and dump !')

	# json.dump(
	# 	sorted( to_dump_dict.items() ), # <- this shit might not be sorted
	# 	open( './agreggated_matrices/' + str(file_path_index) + '.json' , 'wt' )
	# )
	json.dump(
		sorted( to_dump_dict.items() ), # <- this shit might not be sorted
		open( './bin_attempt4_folders/time_tag_cern_read_size/' + str(file_path_index) + '.json' , 'wt' )
	)

def generate_data_sets(ind,choices_depth,bin_count,inf_time_limit,sup_time_limit):
	'''
	Sequential function to associate query read size matrices into bins per time moment.
	'''

	cf_len, se_len = tuple(map(
		lambda v: len(v),
		pickle.load(open(
			'./minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p','rb'))
	))

	mat_len = cf_len * se_len

	thp_list = deque(pickle.load(open('corrected_thp_folder/' + str(ind) + '.p','rb')))
	qrs_list = json.load(open('./agreggated_matrices/' + str(ind) + '.json','rt'))

	out_file = open('output.txt','wt')
	print('Just read stuff !')
	out_file.write( 'Just read stuff !\n' )
	out_file.flush()

	while thp_list[0][0] - inf_time_limit < qrs_list[0][0]:
		thp_list.popleft()

	while qrs_list[-1][0] < thp_list[-1][0] - sup_time_limit:
		thp_list.pop()

	print('Just clipped stuff !')
	out_file.write( 'Just clipped stuff !\n' )
	out_file.flush()

	finished_list = list()

	work_in_progress_queue = deque()

	bin_ind_queue = deque()

	qrs_len = len(qrs_list)

	qrs_list = iter(qrs_list)

	interval_length = (inf_time_limit - sup_time_limit) / bin_count

	list_model = mat_len * [0]

	for qrs_i,(q_tt, dict_list) in enumerate(qrs_list):
		
		if work_in_progress_queue:
			while work_in_progress_queue and not ( work_in_progress_queue[0][0] - inf_time_limit\
					<= q_tt < work_in_progress_queue[0][0] - sup_time_limit ):
				finished_list.append( work_in_progress_queue[0] )
				work_in_progress_queue.popleft()
				bin_ind_queue.popleft()

		while thp_list[0][0] - inf_time_limit <= q_tt < thp_list[0][0] - sup_time_limit:
			work_in_progress_queue.append(
				(
					thp_list[0][0],
					thp_list[0][1],
					tuple(
						tuple(
							# array.array('L', mat_len * [0]) for _ in range( choices_depth )
							deepcopy(list_model) for _ in range( choices_depth )
						) for _ in range( bin_count )
					)
				)
			)
			bin_ind_queue.append(
				(
					0, # <-- bin index
					thp_list[0][0] - inf_time_limit,
					thp_list[0][0] - inf_time_limit + interval_length,
				)
			)
			thp_list.popleft()

		for i in range( len(work_in_progress_queue) ):
			while not ( bin_ind_queue[i][1] <= q_tt < bin_ind_queue[i][2] ):
				bin_ind_queue[i] = (\
					bin_ind_queue[i][0] + 1,
					bin_ind_queue[i][1] + interval_length,
					bin_ind_queue[i][2] + interval_length,
				)

			for ch_i , d in enumerate(dict_list[:choices_depth]):
				for cf_ind, se_dict in map(lambda p: (int(p[0]),p[1],), d.items()):
					for se_ind, val in map(lambda p: (int(p[0]),p[1],), se_dict.items()):
						work_in_progress_queue[i][2][bin_ind_queue[i][0]][ch_i][cf_ind * se_len + se_ind]\
							+= val
		if qrs_i % 1000 == 0:
			print(qrs_i,'/',qrs_len, len(work_in_progress_queue))
			# print( thp_list[0][0] - inf_time_limit )
			# print( q_tt )
			# print( thp_list[0][0] - sup_time_limit )
			# print()
			out_file.write( str(qrs_i) + ' / ' + str(qrs_len) + '\n' )
			out_file.flush()

	json.dump(
		finished_list,
		open(
			'./raw_thp_bins_folder/' + str(ind) + '.json' , 'wt'
		)
	)
	collect()

def generate_data_sets_proc_func_0(conn,proc_ind):

	print(proc_ind,': Started !')

	finished_list = list()

	work_in_progress_queue = deque()

	bin_ind_queue = deque()

	interval_length = (g_inf_time_limit - g_sup_time_limit) / g_bin_count
	
	for qrs_i,(q_tt, dict_list) in enumerate(p_qrs_list):

		while work_in_progress_queue and not ( work_in_progress_queue[0][0] - g_inf_time_limit\
				<= q_tt < work_in_progress_queue[0][0] - g_sup_time_limit ):
			finished_list.append( work_in_progress_queue[0] )
			work_in_progress_queue.popleft()
			bin_ind_queue.popleft()

		while p_thp_list and p_thp_list[0][0] - g_inf_time_limit <= q_tt < p_thp_list[0][0] - g_sup_time_limit:
			work_in_progress_queue.append(
				(
					p_thp_list[0][0],
					p_thp_list[0][1],
					tuple(
						tuple(
							deepcopy(g_list_model) for _ in range( g_choices_depth )
						) for _ in range( g_bin_count )
					)
				)
			)
			bin_ind_queue.append(
				(
					0, # <-- bin index
					p_thp_list[0][0] - g_inf_time_limit,
					p_thp_list[0][0] - g_inf_time_limit + interval_length,
				)
			)
			p_thp_list.popleft()

		for i in range( len(work_in_progress_queue) ):
			while not ( bin_ind_queue[i][1] <= q_tt < bin_ind_queue[i][2] ):
				bin_ind_queue[i] = (\
					bin_ind_queue[i][0] + 1,
					bin_ind_queue[i][1] + interval_length,
					bin_ind_queue[i][2] + interval_length,
				)

			for ch_i , d in enumerate(dict_list[:g_choices_depth]):
				for cf_ind, se_dict in map(lambda p: (int(p[0]),p[1],), d.items()):
					for se_ind, val in map(lambda p: (int(p[0]),p[1],), se_dict.items()):
						work_in_progress_queue[i][2][bin_ind_queue[i][0]][ch_i][cf_ind * g_se_len + se_ind]\
							+= val

		if qrs_i % 1000 == 0:
			print(proc_ind,':',qrs_i,'/',g_q_len, len(work_in_progress_queue))
			collect()

	json.dump(
		finished_list,
		open(
			'./raw_thp_bins_folder/' + str(g_ind) + '_' + str(proc_ind) + '.json' , 'wt'
		)
	)
	conn.send('./raw_thp_bins_folder/' + str(g_ind) + '_' + str(proc_ind) + '.json')

	# conn.send( finished_list )

def generate_data_sets_in_parallel_0(ind,choices_depth,bin_count,inf_time_limit,sup_time_limit):
	global g_inf_time_limit, g_sup_time_limit, g_bin_count, g_choices_depth, g_list_model,\
		g_se_len, g_q_len, g_ind

	g_inf_time_limit = inf_time_limit
	g_sup_time_limit = sup_time_limit
	g_bin_count = bin_count
	g_choices_depth = choices_depth
	g_ind = ind

	cf_len, g_se_len = tuple(map(
		lambda v: len(v),
		pickle.load(open(
			'./minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p','rb'))
	))
	
	g_list_model = cf_len * g_se_len * [0]

	thp_list = deque(pickle.load(open('corrected_thp_folder/' + str(ind) + '.p','rb')))
	qrs_list = json.load(open('./agreggated_matrices/' + str(ind) + '.json','rt'))

	out_file = open('output.txt','wt')
	print('Just read stuff !')
	out_file.write( 'Just read stuff !\n' )
	out_file.flush()

	while thp_list[0][0] - inf_time_limit < qrs_list[0][0]:
		thp_list.popleft()

	while qrs_list[-1][0] < thp_list[-1][0] - sup_time_limit:
		thp_list.pop()

	print('Just clipped stuff !')
	out_file.write( 'Just clipped stuff !\n' )
	out_file.flush()

	remainder = len(thp_list) % n_proc
	quotient = len(thp_list) // n_proc

	proc_list = list()
	parent_pipe_list = list()
	proc_ind = 0
	global p_thp_list, p_qrs_list
	thp_list = tuple(thp_list)

	for _ in range(remainder):

		i = 0
		while qrs_list[i][0] < thp_list[0][0] - inf_time_limit:
			i += 1
		
		j = i
		while j < len(qrs_list) and qrs_list[j][0] < thp_list[quotient][0] - sup_time_limit:
			j+=1

		print(i,j)

		g_q_len = j

		p_qrs_list = iter( qrs_list[ i : j ] )
		qrs_list = qrs_list[ i : ]

		p_thp_list = deque( thp_list[ : quotient + 1 ] )
		thp_list = thp_list[ : quotient + 1 ]

		parent_conn, child_conn = mp.Pipe()
		
		proc_list.append(
			mp.Process(
				target=generate_data_sets_proc_func_0,
				args=(child_conn,proc_ind),
			)
		)
		
		collect()
		
		proc_list[-1].start()
		
		parent_pipe_list.append( parent_conn )

		proc_ind += 1

	for _ in range(remainder, n_proc):

		i = 0
		while qrs_list[i][0] < thp_list[0][0] - inf_time_limit:
			i += 1
		
		j = i
		while j < len(qrs_list) and qrs_list[j][0] < thp_list[quotient-1][0] - sup_time_limit:
			j+=1

		print(i,j)

		g_q_len = j

		p_qrs_list = iter( qrs_list[ i : j ] )
		qrs_list = qrs_list[ i : ]

		p_thp_list = deque( thp_list[ : quotient ] )
		thp_list = thp_list[ : quotient ]

		parent_conn, child_conn = mp.Pipe()
		
		proc_list.append(
			mp.Process(
				target=generate_data_sets_proc_func_0,
				args=(child_conn,proc_ind),
			)
		)
		
		collect()

		proc_list[-1].start()
		
		parent_pipe_list.append( parent_conn )

		proc_ind += 1

	del thp_list
	del qrs_list
	del p_qrs_list
	del p_thp_list

	# json.dump(
	# 	reduce(
	# 		lambda acc,x: acc + x,
	# 		map(lambda c: c.recv(), parent_pipe_list),
	# 		list()
	# 	),
	# 	open(
	# 		'./raw_thp_bins_folder/' + str(ind) + '.json' , 'wt'
	# 	)
	# )

	json.dump(
		reduce(
			lambda acc,x: acc + x,
			map(
				lambda c: json.load(open(c.recv(),'rt')),
				parent_pipe_list
			),
			list()
		),
		open(
			'./raw_thp_bins_folder/' + str(ind) + '.json' , 'wt'
		)
	)

	for p in proc_list: p.join()

def generate_data_sets_proc_func_1(conn,proc_ind):

	print(proc_ind,': Started !')

	finished_list = list()

	work_in_progress_queue = deque()

	bin_ind_queue = deque()

	interval_length = (g_inf_time_limit - g_sup_time_limit) / g_bin_count
	
	for qrs_i,(q_tt, dict_list) in enumerate(p_qrs_list):

		while work_in_progress_queue and not ( work_in_progress_queue[0][0] - g_inf_time_limit\
				<= q_tt < work_in_progress_queue[0][0] - g_sup_time_limit ):
			finished_list.append( work_in_progress_queue[0] )
			work_in_progress_queue.popleft()
			bin_ind_queue.popleft()

		while p_thp_list and p_thp_list[0][0] - g_inf_time_limit <= q_tt < p_thp_list[0][0] - g_sup_time_limit:
			work_in_progress_queue.append(
				(
					p_thp_list[0][0],
					p_thp_list[0][1],
					tuple(
					 	tuple(
					 		dict() for _ in range( g_choices_depth ) 
					 	) for _ in range(g_bin_count)
					)
				)
			)
			bin_ind_queue.append(
				(
					0, # <-- bin index
					p_thp_list[0][0] - g_inf_time_limit,
					p_thp_list[0][0] - g_inf_time_limit + interval_length,
				)
			)
			p_thp_list.popleft()

		for i, work_tuple in enumerate(work_in_progress_queue):
			while not ( bin_ind_queue[i][1] <= q_tt < bin_ind_queue[i][2] ):
				bin_ind_queue[i] = (\
					bin_ind_queue[i][0] + 1,
					bin_ind_queue[i][1] + interval_length,
					bin_ind_queue[i][2] + interval_length,
				)

			for ch_i , d in enumerate(dict_list[:g_choices_depth]):
				for cf_ind, se_dict in d.items():
					if cf_ind in work_tuple[2][bin_ind_queue[i][0]][ch_i]:
						for se_ind, val in se_dict.items():
							if se_ind in work_tuple[2][bin_ind_queue[i][0]][ch_i][cf_ind]:
								work_tuple[2][bin_ind_queue[i][0]][ch_i][cf_ind][se_ind] += val
							else:
								work_tuple[2][bin_ind_queue[i][0]][ch_i][cf_ind][se_ind] = val
					else:
						work_tuple[2][bin_ind_queue[i][0]][ch_i][cf_ind] = se_dict


		if qrs_i % 1000 == 0:
			print(proc_ind,':',qrs_i,'/',g_q_len, len(work_in_progress_queue))
			collect()

	# json.dump(
	# 	finished_list,
	# 	open(
	# 		'./raw_thp_bins_folder/' + str(g_ind) + '_' + str(proc_ind) + '.json' , 'wt'
	# 	)
	# )
	# conn.send('./raw_thp_bins_folder/' + str(g_ind) + '_' + str(proc_ind) + '.json')

	conn.send( finished_list )

def generate_data_sets_in_parallel_1(ind,choices_depth,bin_count,inf_time_limit,sup_time_limit):
	global g_inf_time_limit, g_sup_time_limit, g_bin_count, g_choices_depth, g_list_model,\
		g_se_len, g_q_len, g_ind

	g_inf_time_limit = inf_time_limit
	g_sup_time_limit = sup_time_limit
	g_bin_count = bin_count
	g_choices_depth = choices_depth
	g_ind = ind

	cf_len, g_se_len = tuple(map(
		lambda v: len(v),
		pickle.load(open(
			'./minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p','rb'))
	))
	
	g_list_model = cf_len * g_se_len * [0]

	thp_list = deque(pickle.load(open('corrected_thp_folder/' + str(ind) + '.p','rb')))
	qrs_list = tuple(json.load(open('./agreggated_matrices/' + str(ind) + '.json','rt')))

	out_file = open('output.txt','wt')
	print('Just read stuff !')
	out_file.write( 'Just read stuff !\n' )
	out_file.flush()

	while thp_list[0][0] - inf_time_limit < qrs_list[0][0]:
		thp_list.popleft()

	while qrs_list[-1][0] < thp_list[-1][0] - sup_time_limit:
		thp_list.pop()

	print('Just clipped stuff !')
	out_file.write( 'Just clipped stuff !\n' )
	out_file.flush()

	remainder = len(thp_list) % n_proc
	quotient = len(thp_list) // n_proc

	proc_list = list()
	parent_pipe_list = list()
	proc_ind = 0
	global p_thp_list, p_qrs_list
	thp_list = tuple(thp_list)

	collect()

	for _ in range(remainder):

		i = 0
		while qrs_list[i][0] < thp_list[0][0] - inf_time_limit:
			i += 1
		
		j = i
		while j < len(qrs_list) and qrs_list[j][0] < thp_list[quotient][0] - sup_time_limit:
			j+=1

		print(i,j)

		g_q_len = j

		p_qrs_list = iter( qrs_list[ i : j ] )
		qrs_list = qrs_list[ i : ]

		p_thp_list = deque( thp_list[ : quotient + 1 ] )
		thp_list = thp_list[ quotient + 1 : ]

		parent_conn, child_conn = mp.Pipe()
		
		proc_list.append(
			mp.Process(
				target=generate_data_sets_proc_func_1,
				args=(child_conn,proc_ind),
			)
		)
		
		collect()
		
		proc_list[-1].start()
		
		parent_pipe_list.append( parent_conn )

		proc_ind += 1

	for _ in range(remainder, n_proc):

		i = 0
		while qrs_list[i][0] < thp_list[0][0] - inf_time_limit:
			i += 1
		
		j = i
		while j < len(qrs_list) and qrs_list[j][0] < thp_list[quotient-1][0] - sup_time_limit:
			j+=1

		print(i,j)

		g_q_len = j

		p_qrs_list = iter( qrs_list[ i : j ] )
		qrs_list = qrs_list[ i : ]

		p_thp_list = deque( thp_list[ : quotient ] )
		thp_list = thp_list[ quotient : ]

		parent_conn, child_conn = mp.Pipe()
		
		proc_list.append(
			mp.Process(
				target=generate_data_sets_proc_func_1,
				args=(child_conn,proc_ind),
			)
		)
		
		collect()

		proc_list[-1].start()
		
		parent_pipe_list.append( parent_conn )

		proc_ind += 1

	del thp_list
	del qrs_list
	del p_qrs_list
	del p_thp_list

	collect()

	json.dump(
		reduce(
			lambda acc,x: acc + x,
			map(lambda c: c.recv(), parent_pipe_list),
			list()
		),
		open(
			'./raw_thp_bins_folder/' + str(ind) + '.json' , 'wt'
		)
	)

	# json.dump(
	# 	reduce(
	# 		lambda acc,x: acc + x,
	# 		map(
	# 			lambda c: json.load(open(c.recv(),'rt')),
	# 			parent_pipe_list
	# 		),
	# 		list()
	# 	),
	# 	open(
	# 		'./raw_thp_bins_folder/' + str(ind) + '.json' , 'wt'
	# 	)
	# )

	for p in proc_list: p.join()

def check_time_moments_compatibility():
	thp_list = pickle.load(open('corrected_thp_folder/' + str(1) + '.p','rb'))
	qrs_list = json.load(open('./agreggated_matrices/' + str(1) + '.json','rt'))
	print(thp_list[0][0], thp_list[-1][0], len(thp_list))
	print(qrs_list[0][0], qrs_list[-1][0], len(qrs_list)) 

def plot_q_time_diffs():
	import matplotlib.pyplot as plt
	qrs_list = json.load(open('1.json','rt'))
	a = list()
	for prev_q, next_q in zip( qrs_list[:-1], qrs_list[1:] ):
		a.append( next_q[0] - prev_q[0] )
	plt.plot(range(len(a)),a)
	plt.show()

def dump_sorted_set_of_time_moments_from_raw_q_file():
	if False:
		csv_reader = csv.reader(open(QUERRIES_RAW_FILES[1],'rt'))
		next(csv_reader)
		from sortedcontainers import SortedSet
		s = SortedSet()
		out_file = open('output.txt','wt')
		for ind,tm in enumerate(map(lambda e: int(e[0]),csv_reader)):
			if ind % 1000000 == 0:
				out_file.write(str(ind) + ' ' + str(len(s)) + '\n')
				out_file.flush()
			s.add(tm)
		json.dump(
			tuple(s),
			open( 'debug.json' , 'wt' )
		)
	if True:
		import matplotlib.pyplot as plt
		b = json.load(open('debug.json','rt'))
		a = []
		for prev_q, next_q in zip( b[:-1], b[1:] ):
			a.append( next_q - prev_q )
		plt.plot(range(len(a)),a)
		plt.show()

def get_max_min_proc_func(file_name):
	print(file_name + ': Started !')
	min_thp, max_thp, min_rs, max_rs = None,None,None,None
	a = json.load(open('./raw_thp_bins_folder/' + file_name,'rt'))
	for _, thp, bin_list in a:
		if min_thp is None or thp < min_thp: min_thp = thp
		if max_thp is None or thp > max_thp: max_thp = thp
		for bb_list in bin_list:
			for ch_dict in bb_list:
				for se_dict in ch_dict.values():
					for rs in se_dict.values():
						if min_rs is None or rs < min_rs: min_rs = rs
						if max_rs is None or rs > max_rs: max_rs = rs

	print(file_name + ': Will try to get lock for value comparison !')

	v_lock.acquire() 
	
	print(file_name + ': Will compare values !')

	if mit.value == -1 or mit.value > min_thp: mit.value = min_thp
	if mat.value == -1 or mat.value < max_thp: mat.value = max_thp
	if mir.value == -1 or mir.value > min_rs: mir.value = min_rs
	if mar.value == -1 or mar.value < max_rs: mar.value = max_rs

	print(file_name + ': Finished comparing and will release lock !')

	v_lock.release()

	print(file_name + ': ' + 'Finished min and max updates !')

	if '0' in file_name:
		print(file_name + ': Waiting to be let to normalize !')
		for _ in range(len(resume_work_lock_dict)):
			signal_resume_work_sem.acquire()
		print(file_name + ': Will release other processes to work !')
		for l in resume_work_lock_dict.values():
			l.release()
	else:
		signal_resume_work_sem.release()
		print(file_name + ': Waiting to be let to normalize !')
		resume_work_lock_dict[file_name].acquire()

	print(file_name + ': Will start to normalize and dump !')
	
	if '0' in file_name:
		print(mit.value, mat.value, mir.value, mar.value)

	for i in range(len(a)):
		a[i][1] = (a[i][1] - mit.value) / (mat.value - mit.value)
		for bb_list in a[i][2]:
			for ch_dict in bb_list:
				for se_dict in ch_dict.values():
					for k in se_dict.keys():
						se_dict[k] = 2 * ( se_dict[k] - mir.value ) / ( mar.value - mir.value ) - 1

	json.dump(
		a,
		open(
			'./norm_thp_bins_folder/' + file_name, 'wt'
		)
	)

def norm_thp_bins():
	global v_lock, resume_work_lock_dict, signal_resume_work_sem,\
		mit, mat, mir, mar

	v_lock = mp.Lock()

	path_list = os.listdir('./raw_thp_bins_folder')

	mit, mat, mir, mar = mp.RawValue('d'),mp.RawValue('d'),mp.RawValue('d'),\
		mp.RawValue('d')
	mit.value, mat.value, mir.value, mar.value = -1 , -1 , -1 , -1


	resume_work_lock_dict = dict()
	for p in filter(lambda p: '0' not in p, path_list):
		resume_work_lock_dict[p] = mp.Lock()
		resume_work_lock_dict[p].acquire()

	print(resume_work_lock_dict)

	signal_resume_work_sem = mp.Semaphore(0)

	print('Will work for:', path_list)

	p_list = list()
	for fn in path_list:
		p_list.append(
			mp.Process(
				target=get_max_min_proc_func,
				args=(fn,)
			)
		)
		p_list[-1].start()
	for p in p_list: p.join()

def dump_per_file_proc_func(file_name):
	len_num = len(json.load(open('./norm_thp_bins_folder/'+file_name,'rt')))

	valid_list = set(random.sample( range( time_window - 1 , len_num ) , int( 0.2 * ( len_num - time_window ) ) ))

	pickle.dump(
		(
			tuple( filter( lambda ind: ind not in valid_list , range( time_window - 1 , len_num ) ) ),
			tuple( valid_list ),
		),
		open( './split_indexes_folder/' + file_name.split('.')[0] + '.p' , 'wb' )
	)

def train_test_split():
	global time_window
	time_window = 10

	path_list = os.listdir('./norm_thp_bins_folder')

	mp.Pool(4).map( dump_per_file_proc_func , path_list )

if __name__ == '__main__':
	global n_proc
	n_proc = 124

	# json.dump(
	# 	sorted(
	# 		agreggate_dictionary_list_into_single_dict(
	# 			map(
	# 				lambda name: json.load(open('./proc_aux_folder_1/'+name,'rt')),
	# 				os.listdir( './proc_aux_folder_1/' )
	# 			)
	# 		).items()
	# 	),
	# 	open( './agreggated_matrices/1.json' , 'wt' )
	# )

	# plot_q_time_diffs()

	# generate_data_sets(0,3,100,3600000,300000)
	# generate_data_sets(1,3,100,3600000,300000)
	# generate_data_sets(2,3,100,3600000,300000)
	# generate_data_sets(3,3,100,3600000,300000)

	# generate_data_sets_in_parallel_1(3,3,100,3600000,300000)
	# generate_data_sets_in_parallel_1(0,3,100,3600000,300000)
	# generate_data_sets_in_parallel_1(1,3,100,3600000,300000)
	# generate_data_sets_in_parallel_1(2,3,100,3600000,300000)

	# norm_thp_bins()
	# train_test_split()

	# get_state_dictionaries_main_4(0,'./proc_aux_folder_1/')
	# get_state_dictionaries_main_4(1,'./proc_aux_folder_1/')
	# get_state_dictionaries_main_4(2,'./proc_aux_folder_1/')
	# get_state_dictionaries_main_4(3,'./proc_aux_folder_1/')
	get_state_dictionaries_main_4(4,'./proc_aux_folder_1/')

	# collect_from_folder_to_single_file(0)
	# get_state_dictionaries_main_3(int(sys.argv[1]),'./proc_aux_folder_'+sys.argv[1]+'/')

	# analyse_csv_time_order(3)

	# analyse_non_agregg_results(0)
	# analyse_non_agregg_results(3)

	# if len(sys.argv) == 1:
	# 	os.system( 'python3 bin_attempt2_main.py 0' )
	# 	os.system( 'python3 bin_attempt2_main.py 1' )
	# 	os.system( 'python3 bin_attempt2_main.py 2' )
	# 	os.system( 'python3 bin_attempt2_main.py 3' )
	# else:
	# 	get_state_dictionaries_main_3(int(sys.argv[1]),'./proc_aux_folder_'+sys.argv[1]+'/')

	# correct_and_rectify_thp_file(4)