import json
import pickle
import matplotlib.pyplot as plt
import multiprocessing as mp

def test_data_proc_func_0(file_path):
	cern_rs_list = list()
	thp_list = list()
	minus_one_counts = 0
	min_v_per_choice=[None,None,None]
	max_v_per_choice=[None,None,None]
	for _, thp, b_list in json.load(open(file_path,'rt')):
		thp_list.append( thp )
		for c_list in b_list: # c_list = list containing choices inside a bin
			for ch_i,ch_dict in enumerate(c_list): # ch_dict = dict with key=cf_number and value=dictionary
				if ch_i == 0:
					rs_acc = 0
				for se_d in ch_dict.values():
					for se,v in se_d.items():
						if ch_i == 0 and se == cern_ind:
							rs_acc += v
						if v == -1:
							minus_one_counts += 1
						if min_v_per_choice[ch_i] is None or min_v_per_choice[ch_i] > v:
							min_v_per_choice[ch_i] = v
						if max_v_per_choice[ch_i] is None or max_v_per_choice[ch_i] < v:
							max_v_per_choice[ch_i] = v
				if ch_i == 0:
					cern_rs_list.append( rs_acc )
	return thp_list, cern_rs_list, minus_one_counts, min_v_per_choice, max_v_per_choice

def plot_thp_and_cernRS_and_number_of_zeros():
	_, se_iterable = pickle.load(open('./minimal_sets_and_parsed_matrices/minimal_sets_for_0123.p','rb'))
	global cern_ind
	cern_ind = None
	for ind,se in enumerate(sorted(se_iterable)):
		if 'cern' in se[0]:
			cern_ind = str(ind)
			break
	del se_iterable

	print(cern_ind)

	pickle.dump(
		mp.Pool(4).map(\
			test_data_proc_func_0,
			# (
			# 	'minimized_data/0.json',
			# 	'minimized_data/1.json',
			# 	'minimized_data/2.json',
			# 	'minimized_data/3.json',
			# )
			(
				'norm_thp_bins_folder/0.json',
				'norm_thp_bins_folder/1.json',
				'norm_thp_bins_folder/2.json',
				'norm_thp_bins_folder/3.json',
			)
		),
		open('dump.p','wb')
	)

if __name__ == '__main__':
	plot_thp_and_cernRS_and_number_of_zeros()