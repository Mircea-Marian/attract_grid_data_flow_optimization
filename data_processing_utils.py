from csv import reader
import pickle
import os
from multiprocessing import Pool
from copy import deepcopy

def get_read_size_per_time_moment(api_input_file, pickle_output_file):
	r = reader(open(api_input_file,'rt'))

	next(r)

	tm_d = dict()

	for time_moment, read_size in\
		map(\
			lambda l: ( int(l[0]) , float(l[2]) , ), r
		):

		if time_moment in tm_d: 
			tm_d[time_moment] += read_size
		else:
			tm_d[time_moment] = read_size

	pickle.dump(tm_d,open(pickle_output_file,'wb'))

def get_time_interval_from_matrix_folder(matrix_folder):
	a_list =\
		tuple(
			map(
				lambda fn: int(fn.split('_')[0]),\
				os.listdir(matrix_folder)
			)
		)
	print('Minimum time tag:',min(a_list))
	print('Maximum time tag:',max(a_list))

def compare_time_intervals(matrix_folder, throughput_file, time_moments_dict_file):
	min_max_list = list()
	a_list =\
		tuple(
			map(
				lambda fn: int(fn.split('_')[0]),\
				os.listdir(matrix_folder)
			)
		)

def get_dist_dict_by_path(file_path):
	'''
	This is used to parse a raw distance matrix file into a Python dictionary
	object.
	'''
	d = dict()
	for cl, se, val in\
		map(
			lambda e: ( e[0].lower() , ( e[1][0].lower() , e[1][1].lower() , ) , e[2] , ),
			map(
				lambda e: (e[0], e[1].split('::')[1:], float(e[2]),),
				map(
					lambda r: r.split(';'),
					open(file_path,'r').read().split('\n')[1:-1]
				)
			)
		):
		if cl in d:
			d[cl][se] = val
		else:
			d[cl] = { se : val }
	return d

def get_dem_dict_by_path(file_path):
	'''
	This is used to parse a raw demotion matrix file into a Python dictionary
	object.
	'''
	d = dict()
	for se, val in\
		map(
			lambda e: ( ( e[0][0].lower() , e[0][1].lower() , ) , e[1] , ),
			map(
				lambda e: (e[0].split('::')[1:], float(e[3]),),
				map(
					lambda r: r.split(';'),
					open(file_path,'r').read().split('\n')[1:-1]
				)
			)
		):
		d[se] = val
	return d

def get_clients_and_ses_minimal_list(distance_lists_list, demotion_lists_list):
	clients_sorted_list = sorted( distance_lists_list[0][0][1].keys() )
	se_sorted_list = sorted( demotion_lists_list[0][0][1].keys() )

	for distance_list in distance_lists_list:
		for i , ( _ , cl_dict ) in enumerate(distance_list):
			if i % 1000 == 0:
				print( i , '/', len(distance_list) )
				print( '\t' , clients_sorted_list )
				print( '\t' , se_sorted_list )

			cl_to_remove_list = list()
			for cl in clients_sorted_list:
				if cl not in cl_dict:
					cl_to_remove_list.append(cl)
			if len(cl_to_remove_list) < 30:
				for cl in cl_to_remove_list: clients_sorted_list.remove(cl)

			for se_dict in cl_dict.values():
				se_to_remove_list = list()
				for se in se_sorted_list:
					if se not in se_dict:
						se_to_remove_list.append(se)
				if len(se_to_remove_list) < 30:
					for se in se_to_remove_list: se_sorted_list.remove(se)

	for demotion_list in demotion_lists_list:
		for i , ( _ , se_dict ) in enumerate(demotion_list):
			if i % 1000 == 0:
				print(i,'/',len(demotion_list))
				print( '\t' , clients_sorted_list )
				print( '\t' , se_sorted_list )
				
			se_to_remove_list = list()
			for se in se_sorted_list:
				if se not in se_dict:
					se_to_remove_list.append(se)
			if len(se_to_remove_list) < 30:
				for se in se_to_remove_list: se_sorted_list.remove(se)

	return clients_sorted_list, se_sorted_list

def get_complete_distance_matrix_1(distance_list, demotion_list, clients_sorted_list, se_sorted_list):

	# print( 'clients numbers: ' + str(set( map( lambda p: len(p[1].keys()) , distance_list ) )))

	# print('se numbers 0: '\
	# 	+ str(set( map( lambda p: len(p[1][tuple(p[1].keys())[0]].keys()) , distance_list ) )))

	# print('se numbers 1: ' + str(set( map( lambda p: len(p[1].keys()) , demotion_list ) )))

	dist_i = 0
	dem_i = 0

	while distance_list[dist_i][0] < demotion_list[0][0]: dist_i += 1

	while demotion_list[dem_i][0] < distance_list[0][0]: dem_i += 1

	whole_distance_list = list()

	tm_already_added_set = set()

	aux_ind = 0
	while dist_i < len(distance_list):

		while aux_ind < len(demotion_list) - 1\
			and demotion_list[aux_ind + 1][0] <= distance_list[dist_i][0] : aux_ind += 1

		res_list = list()

		for cl in clients_sorted_list:
			for se in se_sorted_list:
				res_list.append(
					distance_list[dist_i][1][cl][se]\
					+ demotion_list[aux_ind][1][se]
				)

		tm_already_added_set.add( distance_list[dist_i][0] )

		whole_distance_list.append(
			(
				distance_list[dist_i][0],
				res_list,
			)
		)

		dist_i += 1

	aux_ind = 0
	while dem_i < len(demotion_list):

		while aux_ind < len(distance_list) - 1\
			and distance_list[aux_ind + 1][0] <= demotion_list[dem_i][0] : aux_ind += 1

		if demotion_list[dem_i][0] not in tm_already_added_set:

			res_list = list()

			for cl in clients_sorted_list:
				for se in se_sorted_list:
					res_list.append(
						distance_list[aux_ind][1][cl][se]\
						+ demotion_list[dem_i][1][se]
					)

			whole_distance_list.append(
				(
					demotion_list[dem_i][0],
					res_list,
				)
			)

		dem_i += 1

	return sorted(whole_distance_list)

def create_data_set(thp_iterable, read_size_iterable, whole_matrix_iterable,):
	'''
	Matches the trend, average read size, distance and demotion matrices based on time tags.
	'''
	def reduce_throughput_at_front(thp_iterable_0, a_iterable, time_margin=0):
		thp_i = 0

		while thp_iterable_0[thp_i][0] - time_margin < a_iterable[0][0]: thp_i += 1

		return thp_iterable_0[thp_i:]

	thp_iterable = reduce_throughput_at_front(thp_iterable, read_size_iterable, 120000)

	thp_iterable = reduce_throughput_at_front(thp_iterable, whole_matrix_iterable)

	def reduce_throughput_at_back(thp_iterable_0, a_iterable):
		thp_i = len( thp_iterable_0 ) - 1

		while thp_iterable_0[thp_i][0] > a_iterable[-1][0]: thp_i -= 1

		return thp_iterable_0[:thp_i + 1]

	thp_iterable = reduce_throughput_at_back(thp_iterable, read_size_iterable)

	thp_iterable = reduce_throughput_at_back(thp_iterable, whole_matrix_iterable)

	data_set_list = list()

	thp_i = 0

	fi_rs_i = 0
	while read_size_iterable[fi_rs_i][0] < thp_iterable[thp_i][0] - 120000:
		fi_rs_i+=1

	time_window_rs = 0
	la_rs_i = fi_rs_i
	while True:
		time_window_rs += read_size_iterable[la_rs_i][1]
		la_rs_i += 1
		if read_size_iterable[la_rs_i][0] > thp_iterable[thp_i][0]:
			break

	dist_i = 0

	while thp_i < len(thp_iterable):

		while read_size_iterable[fi_rs_i][0] < thp_iterable[thp_i][0] - 120000:
			time_window_rs -= read_size_iterable[fi_rs_i][1]
			fi_rs_i+=1

		if read_size_iterable[la_rs_i][0] <= thp_iterable[thp_i][0]:
			while True:
				time_window_rs += read_size_iterable[la_rs_i][1]
				la_rs_i += 1
				if read_size_iterable[la_rs_i][0] > thp_iterable[thp_i][0]:
					break

		while dist_i < len(whole_matrix_iterable) - 1\
			and whole_matrix_iterable[dist_i+1][0] <= thp_iterable[thp_i][0]:
			dist_i += 1

		data_set_list.append(
			(
				thp_iterable[thp_i][0],
				thp_iterable[thp_i][1],
				time_window_rs / 120,
				whole_matrix_iterable[dist_i][1],
			)
		)

		thp_i += 1

	return data_set_list

def parse_matrix_per_proc(index,time_tag, is_distance_matrix_flag, file_name):
	pickle.dump(
		(
			time_tag,
			is_distance_matrix_flag,
			get_dist_dict_by_path(g_raw_matrices_folder+file_name) if is_distance_matrix_flag else\
				get_dem_dict_by_path(g_raw_matrices_folder+file_name)
		),
		open(
			folder_path + g_dump_index + '_' + str(index) + '.p',
			'wb'
		)
	)

def process_raw_distance_and_demotion_matrices( dump_index , raw_matrices_folder , time_interval_start , time_interval_end ):
		global g_dump_index, folder_path, g_raw_matrices_folder

		if raw_matrices_folder[-1] != '/':
			g_raw_matrices_folder = raw_matrices_folder + '/'
		else:
			g_raw_matrices_folder = raw_matrices_folder
		
		g_dump_index = str(dump_index)
		
		# Parse matrices filenames
		
		# matrices_fn_list =\
		# 	list(
		# 		map(\
		# 			lambda fn: (\
		# 				fn[0],\
		# 				int( fn[1].split('_')[0] ),\
		# 				'distance' in fn[1],\
		# 				fn[1],\
		# 			),\
		# 			enumerate(os.listdir(raw_matrices_folder)),\
		# 		)
		# 	)
		matrices_fn_list =\
			list(
				map(
					lambda tup: ( tup[0], tup[1][0] , tup[1][1] , tup[1][2] ),
					enumerate(
						filter(
							lambda e: time_interval_start <= e[0] <= time_interval_end,
							map(\
								lambda fn: (\
									int( fn.split('_')[0] ),\
									'distance' in fn,\
									fn,\
								),\
								os.listdir(raw_matrices_folder),\
							)
						)
					)
				)
			)
		
		print( 'Will work on' , len( matrices_fn_list ) )

		folder_path = './minimal_sets_and_parsed_matrices/parsed_' + str(dump_index) + '/'
		if not os.path.isdir(folder_path): os.mkdir(folder_path)

		with Pool(126) as p_pool:
			p_pool.starmap(\
				parse_matrix_per_proc,
				matrices_fn_list
			)

		print('Finished parsing !')

def check_if_old_minimal_sets_correspond_with_new_data_and_create_new_minimal_set( old_set_path, new_data_path,\
	output_path):
	old_minimal_sets = pickle.load(
		open(
			old_set_path,
			'rb'
		)
	)

	if new_data_path[-1] != '/': new_data_path += '/'

	print( '# old cl' , len( old_minimal_sets[ 0 ] ) )
	print( '# old se' , len( old_minimal_sets[ 1 ] ) )

	processed_matrices = tuple(
		map(
			lambda fn: pickle.load( open( new_data_path + fn , 'rb' ) ),
			os.listdir( new_data_path )
		)
	)

	print('Will work on',len(processed_matrices),'total matrices')

	minimal_cl_list, minimal_se_list =\
		get_clients_and_ses_minimal_list(
			(
				tuple(
					map(
						lambda a: (a[0],a[2]),
						filter(
							lambda fn_tup: fn_tup[1],
							processed_matrices
						)
					)	
				),\
			),
			(
				tuple(
					map(
						lambda a: (a[0],a[2]),
						filter(
							lambda fn_tup: not fn_tup[1],
							processed_matrices
						)
					)	
				),\
			)
		)

	print(len(minimal_cl_list))
	print(len(minimal_se_list))

	new_cl_list, new_se_list = list(), list()
	for cl in minimal_cl_list:
		if cl in old_minimal_sets[0]:
			new_cl_list.append(cl)
	for se in minimal_se_list:
		if se in old_minimal_sets[1]:
			new_se_list.append(se)

	pickle.dump(
		(new_cl_list, new_se_list),
		open( output_path , 'wb' )
	)

def add_distance_and_demotion_missing_elements(distance_list, demotion_list, clients_sorted_list, se_sorted_list):
	modifiable_distance_dict = dict()
	for cl in clients_sorted_list:
		modifiable_distance_dict[cl] = dict()
		for se in se_sorted_list:
			modifiable_distance_dict[cl][se] = -2
	new_distance_list = list()
	i = 0
	for tm , d in sorted(distance_list):
		if i % 1000 == 0:
			print(i,'/',len(distance_list))
		for cl in clients_sorted_list:
			if cl in d:
				for se in se_sorted_list:
					if se in d[cl]:
						modifiable_distance_dict[cl][se] = d[cl][se]
		new_distance_list.append((
			tm,
			deepcopy( modifiable_distance_dict )
		))
		i += 1
	
	modifiable_demotion_dict = dict( ( se , -2 , ) for se in se_sorted_list )
	new_demotion_list = list()
	i = 0
	for tm , d in sorted(demotion_list):
		if i % 1000 == 0:
			print(i,'/',len(demotion_list))
		for se in se_sorted_list:
			if se in d:
				modifiable_demotion_dict[se] = d[se]
		new_demotion_list.append((
			tm,
			deepcopy( modifiable_demotion_dict )
		))
		i+=1

	return new_distance_list , new_demotion_list

def get_min_and_max_time_moment(file_path):
	file_variable = open(file_path,'rt')
	# len(file_variable) = 318007736

	file_variable.readline()

	min_tm = max_tm = int( file_variable.readline().split(',')[0] )

	count = 0

	while True:
		
		line = file_variable.readline()

		if not line or len(line) < 5:
			break

		count += 1

		n = int( line.split(',')[0] )

		if n > max_tm:
			max_tm = n

		if n < min_tm:
			min_tm = n

		if count % 1000000 == 0:
			print(count)

	print( 'Lower interval limit is' , min_tm )
	print( 'Upper interval limit is' , max_tm )
	print( 'Number of queries is' , count)

def associate_read_size_to_throughput( thp_list , rs_list ):
	i_thp_left = 0
	while thp_list[i_thp_left][0] - 120000 < rs_list[0][0]:
		i_thp_left += 1
	i_thp_right = len( thp_list ) - 1
	while thp_list[i_thp_right][0] > rs_list[-1][0]:
		i_thp_right -= 1

	min_v , max_v = None , None
	min_thp, max_thp = None, None

	i_rs = 0
	result_list = list()
	for thp_tm , thp_v in thp_list[ i_thp_left : i_thp_right + 1 ]:
		while not ( thp_tm - 120000 <= rs_list[ i_rs ][ 0 ] < thp_tm ):
			i_rs += 1

		rs_sum = 0

		while thp_tm - 120000 <= rs_list[ i_rs ][ 0 ] < thp_tm:
			rs_sum += rs_list[ i_rs ][ 1 ]
			i_rs += 1

		if min_v is None or min_v > rs_sum: min_v = rs_sum
		if min_thp is None or min_thp > thp_v: min_thp = thp_v
		if max_v is None or max_v < rs_sum: max_v = rs_sum
		if max_thp is None or max_thp < thp_v: max_thp = thp_v

		result_list.append( ( thp_v , rs_sum ) )

	return result_list , min_v , max_v , min_thp , max_thp