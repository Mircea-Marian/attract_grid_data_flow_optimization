import pickle
import csv
from multiprocessing import Pool, Manager, Process, Lock
from multiprocessing.sharedctypes import RawArray
import os
import shutil
import json
from functools import reduce
import gc
from collections import namedtuple
from copy import deepcopy
import random

def get_unanswered_queries_main(client_name):
	pickle.dump(
		tuple(
			map(
				lambda line: (\
					int(line[0]),
					line[3].lower(),
					tuple(\
						map(
							lambda e: tuple(e.split('::')[1:]),
							line[4].split(';')
						)
					),
					int(line[2]),
				),
				filter(
					lambda line: client_name in line[-1],
					map(
						lambda line: line[:-1] + [line[-1].lower(),],
						csv.reader(open('apicommands.log'),delimiter=',')
					)
				)
			)
		),
		open('unanswered_cern_queries.p', 'wb')
	)

def get_unanswered_queries_main_1(client_name):
	global file_variable

	if False:
		file_variable = '#epoch time,file name,size,accessed from,sorted replica list\n'\
			+ '1579215600000,/alice/cern.ch/user/h/hosokawa/myWorkingDirUPC_MB2_pass1/myOutputDirUPC_MB2_pass1/000246053/022/AnalysisResults.root,6996489,UPB,ALICE::CERN::EOS;ALICE::GSI::SE2\n'\
			+ '1579215600000,/alice/cern.ch/user/h/hosokawa/myWorkingDirUPC_MB2_pass1/myOutputDirUPC_MB2_pass1/000246053/147/AnalysisResults.root,1556821,GSI,ALICE::GSI::SE2;ALICE::NIHAM::EOS\n'\
			+ '1579215600000,/alice/cern.ch/user/h/hosokawa/myWorkingDirUPC_MB2_pass1/myOutputDirUPC_MB2_pass1/000246053/153/AnalysisResults.root,1340549,GSI,ALICE::GSI::SE2;ALICE::CERN::EOS\n'\
			+ '1579215600000,/alice/cern.ch/user/h/hosokawa/myWorkingDirUPC_MB2_pass1/myOutputDirUPC_MB2_pass1/000246053/154/AnalysisResults.root,1507233,GSI,ALICE::CERN::SE2;ALICE::NIHAM::EOS\n'
	if True:
		file_variable = open('apicommands.log','rt').read()
		# len(file_variable) = 43433223135
	if False:
		file_variable = open('test.log','rt').read()

	file_variable = file_variable.split('\n')[1:]
	if len(file_variable[-1]) < 5: file_variable = file_variable[:-1]

	print(len(file_variable))
	for i in\
		map(
			lambda line: line[:-1] + [line[-1].lower(),],
			map(
				lambda line: line.split(','),
				file_variable[:10]
			)
		): print(i)

	print('Will start working !')

	if True:
		print(
			len(
				tuple(
					filter(
						lambda line: client_name in line[-1],
						map(
							lambda line: line[:-1] + [line[-1].lower(),],
							map(
								lambda line: line.split(','),
								file_variable
							)
						)
					)
				)
			)
		)
		exit(0)

	a = tuple(
		map(
			lambda line: (\
				int(line[0]),
				line[3].lower(),
				tuple(\
					map(
						lambda e: tuple(e.split('::')[1:]),
						line[4].split(';')
					)
				),
				int(line[2]),
			),
			filter(
				lambda line: client_name in line[-1],
				map(
					lambda line: line[:-1] + [line[-1].lower(),],
					map(lambda line: line.split(','),file_variable)
				)
			)
		)
	)

	print(len(a))

	pickle.dump(
		a,
		open('unanswered_cern_queries.p', 'wb')
	)

def process_per_proc(i):
	f = open( './unanswered_query_dump_folder/' + str(i) + '.json' , 'wt' )
	json.dump(
		tuple(
			map(
				lambda line: (\
					int(line[0]),
					line[3].lower(),
					tuple(\
						map(
							lambda e: tuple(e.split('::')[1:]),
							line[4].split(';')
						)
					),
					int(line[2]),
				),
				filter(
					lambda line: g_client_name in line[-1],
					map(
						lambda line: line[:-1] + [line[-1].lower(),],
						map(
							lambda line: line.split(','),
							file_variable[\
								g_idexes_pairs_tuple[i][0] : g_idexes_pairs_tuple[i][1]\
							]
						)
					)
				)
			)
		),
		f,
	)
	print('Finished extracting for ' + str(i))
	f.close()
	gc.collect()

def analyse_per_proc(i):
	l =\
	len(
		tuple(
			filter(
				lambda line: g_client_name in line[-1],
				map(
					lambda line: line[:-1] + [line[-1].lower(),],
					map(
						lambda line: line.split(','),
						file_variable[\
							g_idexes_pairs_tuple[i][0] : g_idexes_pairs_tuple[i][1]\
						]
					)
				)
			)
		)
	)
	print('Between ' + str(g_idexes_pairs_tuple[i][0]) + ' and ' + str(g_idexes_pairs_tuple[i][1]) + ' there are ' + str(l))
	return l

def get_unanswered_queries_main_2(client_name):
	global file_variable, g_idexes_pairs_tuple, g_client_name

	g_client_name = client_name

	file_variable = open('apicommands.log','rt').read()
	# len(file_variable) = 318007736

	file_variable = file_variable.split('\n')[1:]
	if len(file_variable[-1]) < 5: file_variable = file_variable[:-1]

	q_num = len(file_variable)

	g_idexes_pairs_tuple = []
	i = 0
	a = q_num // 200
	while i < q_num:
		g_idexes_pairs_tuple.append((
			i,
			i + a,
		))
		i+=a
	if g_idexes_pairs_tuple[-1][1] != q_num:
		g_idexes_pairs_tuple[-1] = (
			g_idexes_pairs_tuple[-1][0],
			q_num
		)

	print( 'Will start pool !' )

	if True: Pool(n_proc).map(process_per_proc, range(len(g_idexes_pairs_tuple)))
	if False:
		print('Total number of q: '\
			+ str(
				reduce(
					lambda acc, x: acc + x,
					Pool(n_proc).map(analyse_per_proc, range(len(g_idexes_pairs_tuple)))
				)
			)
		)

def get_matrices_time_moments_lists(first_moment, last_moment):
	filename_tuple = tuple(os.listdir('./remote_host/log_folder/'))
	distance_list = list(
		map(
			lambda e: int(e.split('_')[0]),
			filter(\
				lambda fn: '_distance' in fn,
				filename_tuple
			)
		)
	)
	demotion_list = list(
		map(
			lambda e: int(e.split('_')[0]),
			filter(\
				lambda fn: '_demotion' in fn,
				filename_tuple
			)
		)
	)
	distance_list.sort()
	demotion_list.sort()

	i = len(distance_list)-1
	while distance_list[i] > first_moment: i-=1
	distance_list = distance_list[i:]

	i = len(demotion_list)-1
	while demotion_list[i] > first_moment: i-=1
	demotion_list = demotion_list[i:]

	i = 0
	while i < len(distance_list) and distance_list[i] < last_moment: i+=1
	distance_list = distance_list[:i+1]

	i = 0
	while i < len(demotion_list) and demotion_list[i] < last_moment: i+=1
	demotion_list = demotion_list[:i+1]

	return distance_list, demotion_list

def get_dist_dict_by_time(time_moment):
	d = dict()
	for cl, se, val in\
		map(
			lambda e: ( e[0].lower() , ( e[1][0].lower() , e[1][1].lower() , ) , e[2] , ),
			map(
				lambda e: (e[0], e[1].split('::')[1:], float(e[2]),),
				map(
					lambda r: r.split(';'),
					open('./remote_host/log_folder/' + str(time_moment) + '_distance','r').read().split('\n')[1:-1]
				)
			)
		):
		if cl in d:
			d[cl][se] = val
		else:
			d[cl] = { se : val }
	return d

def get_dem_dict_by_time(time_moment):
	d = dict()
	for se, val in\
		map(
			lambda e: ( ( e[0][0].lower() , e[0][1].lower() , ) , e[1] , ),
			map(
				lambda e: (e[0].split('::')[1:], float(e[3]),),
				map(
					lambda r: r.split(';'),
					open('./remote_host/log_folder/' + str(time_moment) + '_demotion','r').read().split('\n')[1:-1]
				)
			)
		):
		d[se] = val
	return d

def get_matrices(first_moment, last_moment):

	distance_list, demotion_list = get_matrices_time_moments_lists(first_moment, last_moment)

	dist_res_list = []
	for time_moment in distance_list:
		dist_res_list.append((time_moment,get_dist_dict_by_time(time_moment),))

	dem_res_list = []
	for time_moment in demotion_list:
		dem_res_list.append((time_moment,get_dem_dict_by_time(time_moment),))

	return dist_res_list, dem_res_list

def binary_search_matrixes(matrix_list, time_moment, left_i, right_i):
	if right_i - left_i == 1:
		if matrix_list[left_i][0] <= time_moment < matrix_list[right_i][0]:
			return left_i
		return right_i
	mid_i = (left_i + right_i) // 2
	if time_moment < matrix_list[mid_i][0]:
		return binary_search_matrixes(
			matrix_list,
			time_moment,
			left_i,
			mid_i
		)
	return binary_search_matrixes(
		matrix_list,
		time_moment,
		mid_i,
		right_i
	)

def answer_per_proc(aaaa):

	queries_list = list(
		map(
			lambda q_line:\
				(
					q_line[0],
					q_line[1],
					tuple(map(lambda e: tuple(e),q_line[2])),
					q_line[3],
				),
			filter(
				lambda p: p[0] >= g_first_moment,
				json.load(
					open('./unanswered_query_dump_folder/' + g_file_list[aaaa],'rt')
				)
			)
		)
	)

	dist_i = 0
	dem_i = 0

	for i in range(len(queries_list)):

		ref_dict = g_dist_list[-1][1]
		for k in ref_dict.keys():
			if queries_list[i][1] in k:
				client_name = k
				break

		if queries_list[i][0] >= g_dist_list[-1][0]:
			local_dist_dict = ref_dict[client_name]
		else:
			while not ( g_dist_list[dist_i][0] <= queries_list[i][0] < g_dist_list[dist_i+1][0] ):
				dist_i+=1
			local_dist_dict = g_dist_list[dist_i][1][client_name]
			# local_dist_dict = g_dist_list[\
			# 	binary_search_matrixes(g_dist_list,queries_list[i][0],0,len(g_dist_list)-1)\
			# ][1][client_name]

		if queries_list[i][0] >= g_dem_list[-1][0]:
			local_dem_dict = g_dem_list[-1][1]
		else:
			while not ( g_dem_list[dem_i][0] <= queries_list[i][0] < g_dem_list[dem_i+1][0] ):
				dem_i+=1
			local_dem_dict = g_dem_list[dem_i][1]
			# local_dem_dict = g_dem_list[\
			# 	binary_search_matrixes(g_dem_list,queries_list[i][0],0,len(g_dem_list)-1)\
			# ][1]

		se_list = []
		j = 0
		for se in queries_list[i][2]:
			se_list.append(
				(
					local_dist_dict[se] + local_dem_dict[se],
					j
				)
			)
			j+=1
		se_list.sort(key=lambda p: p[0])

		queries_list[i] = (
			queries_list[i][0],
			queries_list[i][1],
			tuple(map(lambda p: queries_list[i][2][p[1]], se_list)),
			queries_list[i][3],
		)

	json.dump(
		tuple(queries_list),
		open('./answered_query_dump_folder/' + g_file_list[aaaa], 'wt'),
	)

	print('Finished for: ' + g_file_list[aaaa])

	gc.collect()

def get_answered_queries_main_1(first_moment, last_moment):
	global g_dist_list, g_dem_list, g_file_list, g_first_moment

	g_first_moment = first_moment

	g_dist_list, g_dem_list = get_matrices(first_moment, last_moment)

	g_file_list = tuple(\
		map(
			lambda fn: fn,
			os.listdir('./unanswered_query_dump_folder/')
		)
	)

	Pool(n_proc).map(answer_per_proc, range(len(g_file_list)))

def analyse_dist_dem(first_moment, last_moment):
	matrixes_list = os.listdir('remote_host/log_folder')

	dist_list = tuple(
		map(
			lambda fn: int(fn.split('_')[0]),
			filter(
				lambda fn: '_distance' in fn,
				matrixes_list
			)
		)
	)

	dem_list = tuple(
		map(
			lambda fn: int(fn.split('_')[0]),
			filter(
				lambda fn: '_demotion' in fn,
				matrixes_list
			)
		)
	)

	print(str(first_moment) + ' ' + str(last_moment))

	print(str(min(dist_list)) + ' ' + str(max(dist_list)))

	print(str(min(dem_list)) + ' ' + str(max(dem_list)))

def get_first_option_cern():
	json.dump(
		sorted(
			reduce(
				lambda acc,x: acc + x,
				map(
					lambda fn:\
						tuple(
							filter(
								lambda q_line: 'cern' in q_line[2][0][0],
								json.load(
									open(
										'./answered_query_dump_folder/' + fn,
										'rt'
									)
								)
							)
						),
					os.listdir('./answered_query_dump_folder/')
				),
				tuple()
			)
		),
		open(
			'first_option_cern.json',
			'wt'
		)
	)

def get_thp_bin_per_proc(i):
	thp_bins_list = [0 for _ in range(g_bins_no)]

	for q_time, q_read_size in filter(\
		lambda e: g_thp_list[i][0] - g_s <= e[0] < g_thp_list[i][0] - g_f,
		g_queries_list):

		j = g_bins_no - 1

		t = g_thp_list[i][0] - g_f - g_bin_length_in_time

		while j >= 0:

			if q_time >= t:

				thp_bins_list[j] += q_read_size

				break

			t-=g_bin_length_in_time
			j-=1

	g_dict[g_thp_list[i]] = thp_bins_list

def get_five_minute_binned_dataset(
	first_moment,
	millis_interval_start=4000000,
	millis_interval_end=0,
	number_of_bins_per_thp=1000):
	global g_queries_list, g_thp_list, g_dict, g_bin_length_in_time, g_bins_no, g_s, g_f

	g_bin_length_in_time = (millis_interval_start - millis_interval_end) / number_of_bins_per_thp

	g_bins_no = number_of_bins_per_thp

	g_s, g_f = millis_interval_start, millis_interval_end

	g_queries_list = json.load(open('first_option_cern.json', 'rt'))

	print('There are ' + str(len(g_queries_list)) + ' queries.')

	g_thp_list = tuple(
		filter(
			lambda t: t[0] >= first_moment + millis_interval_start,
			pickle.load(open('thp_dump_list.p','rb'))
		)
	)

	print('There are ' + str(len(g_thp_list)) + ' throughput values.')

	g_dict = Manager().dict()

	print('Data is loaded ! Now starting process pool !')

	p = Pool(n_proc)

	p.map(get_thp_bin_per_proc, range(len(g_thp_list)))

	p.close()

	pickle.dump(
		dict( g_dict ),
		open('binned_thp_queries_dict.p','wb')
	)

def get_five_minute_binned_dataset_1(
	first_moment,
	millis_interval_start=4000000,
	millis_interval_end=0,
	number_of_bins_per_thp=1000):

	# queries_list = json.load(open('first_option_cern.json', 'rt'))

	# print('There are ' + str(len(queries_list)) + ' queries.')

	thp_list = sorted(\
		tuple(
			filter(
				lambda t: t[0] >= first_moment + millis_interval_start,
				pickle.load(open('thp_dump_list.p','rb'))
			)
		)
	)

	print('There are ' + str(len(thp_list)) + ' throughput values.')

	Bin_Element = namedtuple('Bin_El',['ind','fi_t','la_t','thp_t','bin_list','thp_v'])

	initial_bin_list = [0 for _ in range(number_of_bins_per_thp)]

	bin_length_in_time = (millis_interval_start - millis_interval_end) / number_of_bins_per_thp

	result_list = []

	queue_list = []

	thp_i = 0

	q_index = 0

	for time_stamp, _, _, read_size in json.load(open('first_option_cern.json', 'rt')):

		if q_index % 100000 == 0:
			print('Reached query index: ' + str(q_index))

			if len(queue_list) >= 10:
				a_str = ''
				for i in random.sample(range(len(queue_list)),10):
					a_str += str(queue_list[i].bin_list[queue_list[i].ind]) + ' '
				print('\t' + a_str)

		while thp_i < len(thp_list) and thp_list[thp_i][0] - millis_interval_start <= time_stamp < thp_list[thp_i][0]:

			bin_i = 0
			bin_t = thp_list[thp_i][0] - millis_interval_start
			while True:
				if bin_i >= g_number_of_bins_per_thp or bin_t <= time_stamp < bin_t + bin_length_in_time:

					queue_list.append(
						Bin_Element(
							bin_i,
							bin_t,
							bin_t + bin_length_in_time,
							thp_list[thp_i][0],
							deepcopy(initial_bin_list),
							thp_list[thp_i][1],
						)
					)

					break

				bin_t += bin_length_in_time

				bin_i += 1

			thp_i += 1

		q_i = 0

		while q_i < len(queue_list):

			if time_stamp < queue_list[q_i].thp_t:

				if q_i - 1 >= 0: queue_list = queue_list[q_i:]

				break

			result_list.append( queue_list[q_i] )

			q_i += 1

		for q_i in range(len(queue_list)):
			if not ( queue_list[q_i].fi_t <= time_stamp < queue_list[q_i].la_t ):
				bin_i = queue_list[q_i].ind
				bin_t = queue_list[q_i].fi_t
				while bin_i < number_of_bins_per_thp:

					if queue_list[q_i].fi_t <= time_stamp < queue_list[q_i].la_t:

						queue_list[q_i] = Bin_Element(
							ind=bin_i,
							fi_t=bin_t,
							la_t=bin_t + bin_length_in_time,
							bin_list=queue_list[q_i].bin_list,
							thp_v=queue_list[q_i].thp_v
						)

						break

					bin_i += 1
					bin_t += bin_length_in_time

		for bin_el in queue_list:
			bin_el.bin_list[bin_el.ind] += read_size

		q_index += 1

	pickle.dump(
		tuple(
			map(
				lambda bin_el: bin_el.bin_list + [bin_el.thp_v,],
				result_list
			)
		),
		open('ten_gigs_bins.p','wb')
	)

def reduce_query_list_size():
	json.dump(
		tuple(
			map(
				lambda p: (p[0],p[-1],),
				json.load(open('first_option_cern.json', 'rt'))
			)
		),
		open('first_opt_cern_only_read_value.json','wt')
	)

def bins_per_proc(ii):
	Bin_Element = namedtuple('Bin_El',['ind','fi_t','la_t','thp_t','bin_list','thp_v', 'thp_i'])

	initial_bin_list = [0 for _ in range(g_number_of_bins_per_thp)]

	queue_list = []

	thp_i = 0
	while thp_i < len(g_thp_list)\
		and not (g_thp_list[thp_i][0] - g_millis_interval_start <=\
			g_query_list[g_slice_list[ii][0]][0] < g_thp_list[thp_i][0]):
		thp_i += 1

	q_index = 0
	for time_stamp, read_size in g_query_list[g_slice_list[ii][0]:g_slice_list[ii][1]]:
	# for time_stamp, read_size in g_query_list[g_slice_list[ii][0]:g_slice_list[ii][0]+10]:
		if q_index % 100000 == 0:
			print(str(os.getpid()) + ': Reached query index: ' + str(q_index) + '/' + str(g_slice_list[ii][1]-g_slice_list[ii][0]))

			# if len(queue_list) >= 10:
			# 	a_str = ''
			# 	for i in random.sample(range(len(queue_list)),10):
			# 		a_str += str(queue_list[i].bin_list[queue_list[i].ind]) + ' '
			# 	print('\t' + a_str)

		while thp_i < len(g_thp_list) and g_thp_list[thp_i][0] - g_millis_interval_start <= time_stamp < g_thp_list[thp_i][0]:

			bin_i = 0
			bin_t = g_thp_list[thp_i][0] - g_millis_interval_start
			while True:
				if bin_i >= g_number_of_bins_per_thp or bin_t <= time_stamp < bin_t + g_bin_length_in_time:

					queue_list.append(
						Bin_Element(
							bin_i,
							bin_t,
							bin_t + g_bin_length_in_time,
							g_thp_list[thp_i][0],
							deepcopy(initial_bin_list),
							g_thp_list[thp_i][1],
							thp_i,
						)
					)

					break

				bin_t += g_bin_length_in_time

				bin_i += 1

			thp_i += 1

		q_i = 0

		while q_i < len(queue_list):

			if time_stamp < queue_list[q_i].thp_t:

				if q_i - 1 >= 0: queue_list = queue_list[q_i:]

				break

			# g_result_list.append((
			# 	queue_list[q_i].thp_i,
			# 	queue_list[q_i].bin_list,
			# ))

			g_lock_list[queue_list[q_i].thp_i].acquire()

			for jj in range(g_number_of_bins_per_thp):
				g_result_list[queue_list[q_i].thp_i][jj] =\
					g_result_list[queue_list[q_i].thp_i][jj] + queue_list[q_i].bin_list[jj]

			g_lock_list[queue_list[q_i].thp_i].release()

			q_i += 1

		for q_i in range(len(queue_list)):
			if not ( queue_list[q_i].fi_t <= time_stamp < queue_list[q_i].la_t ):
				bin_i = queue_list[q_i].ind
				bin_t = queue_list[q_i].fi_t
				while bin_i < g_number_of_bins_per_thp:

					if queue_list[q_i].fi_t <= time_stamp < queue_list[q_i].la_t:

						queue_list[q_i] = Bin_Element(
							ind=bin_i,
							fi_t=bin_t,
							la_t=bin_t + g_bin_length_in_time,
							bin_list=queue_list[q_i].bin_list,
							thp_v=queue_list[q_i].thp_v,
							thp_i=thp_i
						)

						break

					bin_i += 1
					bin_t += g_bin_length_in_time

		for bin_el in queue_list:
			bin_el.bin_list[bin_el.ind] += read_size

		q_index += 1

def get_five_minute_binned_dataset_2(
	first_moment,
	millis_interval_start=4000000,
	millis_interval_end=0,
	number_of_bins_per_thp=1000):

	global g_result_list, g_thp_list, g_query_list, g_slice_list,\
		g_number_of_bins_per_thp, g_bin_length_in_time, g_millis_interval_start,\
		g_lock_list

	# g_result_list = Manager().list()

	g_thp_list = sorted(\
		tuple(
			filter(
				lambda t: t[0] >= first_moment + millis_interval_start,
				pickle.load(open('thp_dump_list.p','rb'))
			)
		)
	)

	print('There are ' + str(len(g_thp_list)) + ' throughput values.')

	g_result_list = [RawArray('d',number_of_bins_per_thp*[0,]) for _ in range(len(g_thp_list))]

	g_lock_list = [Lock() for _ in range(len(g_thp_list))]

	g_query_list = json.load(open('first_opt_cern_only_read_value.json', 'rt'))

	q_count = len(g_query_list)

	print('There are ' + str(q_count) + ' queries values.')

	a_list = [q_count//n_proc for _ in range(n_proc)]
	for i in range(q_count%n_proc):
		a_list[i] += 1

	g_slice_list = [(0, a_list[0],),]
	for i in range(1, n_proc):
		g_slice_list.append((
			g_slice_list[-1][1],
			g_slice_list[-1][1] + a_list[i],
		))

	del a_list

	g_number_of_bins_per_thp = number_of_bins_per_thp

	g_bin_length_in_time = (millis_interval_start - millis_interval_end) / number_of_bins_per_thp

	g_millis_interval_start = millis_interval_start

	p = Pool(n_proc)

	print('Will start pool !')

	# p.starmap(
	# 	bins_per_proc,
	# 	zip(
	# 		range(n_proc),
	# 		[RawArray('d',number_of_bins_per_thp*[0,]) for _ in range(n_proc)]
	# 	)
	# )

	p.map(
		bins_per_proc,
		range(n_proc)
	)

	p.close()

	p.join()

	del g_query_list
	del g_slice_list

	# data_set_dict = dict()
	# for ind, bin_list in g_result_list:
	# 	if ind in data_set_dict:
	# 		data_set_dict[ind] = list(
	# 			map(
	# 				lambda p: p[0] + p[1],
	# 				zip(data_set_dict[ind],bin_list,)
	# 			)
	# 		)

	json.dump(
		tuple(
			map(
				lambda ind: tuple(g_result_list[ind]) + (g_thp_list[ind][1],),
				range(len(g_thp_list))
			)
		),
		open('data_set.json','wt')
	)

def dump_plot_for_matrixes(first_moment, last_moment):
	distance_matrices_list =\
	sorted(
		tuple(
			map(
				lambda e: e / 3600000,
				filter(
					lambda e: first_moment <= e < last_moment,
					map(
						lambda fn: int(fn.split('_')[0]),
						filter(
							lambda e: '_distance' in e,
							os.listdir('log_folder')
						)
					)
				)
			)
		)
	)

	prev = distance_matrices_list[0]

	a_list = [prev]

	for e in distance_matrices_list:
		if e - prev >= 6:
			prev = e
			a_list.append(e)

	pickle.dump(
		a_list,
		open(
			'pipe_1.p',
			'wb'
		)
	)


if __name__ == '__main__':
	first_moment, last_moment = 1579264801390, 1579875041000

	global n_proc

	n_proc = 95

	if False: get_unanswered_queries_main_1('cern')

	if False: get_unanswered_queries_main_2('cern')

	if False: get_answered_queries_main_1(first_moment, last_moment)

	if False: analyse_dist_dem(first_moment, last_moment)

	if False: get_first_option_cern()

	if False: get_five_minute_binned_dataset_2(first_moment,)

	if False: reduce_query_list_size()

	if True:
		dump_plot_for_matrixes(first_moment, last_moment)

	# a = 43433223135 // 200
	# i = 0
	# while i < 43433223135:
	# 	print(str(i) + ' ' + str(i+a))
	# 	i += a
