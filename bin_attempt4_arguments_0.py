import bin_attempt4_grid_search
import pickle
import bin_attempt4_models
import bin_attempt4_batch_generators
import os
import numpy as np
import json

CREATE_MODEL_FUNCTION,\
CREATE_MODEL_ARGUMENTS,\
FIT_MODEL_FUNCTION,\
FIT_MODEL_ARGUMENTS,\
GENERATOR_FUNCTION,\
TRAIN_GENERATOR,\
TRAIN_ARGUMENTS,\
VALID_GENERATOR,\
VALID_ARGUMENTS,\
	=\
	'CREATE_MODEL_FUNCTION',\
	'CREATE_MODEL_ARGUMENTS',\
	'FIT_MODEL_FUNCTION',\
	'FIT_MODEL_ARGUMENTS',\
	'GENERATOR_FUNCTION',\
	'TRAIN_GENERATOR',\
	'TRAIN_ARGUMENTS',\
	'VALID_GENERATOR',\
	'VALID_ARGUMENTS'

def get_data_and_indexes(keep_time_moments=False, indexes_list=(0,1,2,3)):
	tr_ind_iterable, va_ind_iterable = pickle.load(open(\
		'./bin_attempt4_folders/test_train_indexes/combined_train_test_split.p' , 'rb'))
	tr_ind_iterable, va_ind_iterable = list(tr_ind_iterable),list(va_ind_iterable)

	f = lambda dir_name:\
		tuple(
			map(
				lambda ind: (str(ind)+'.p') if str(ind)+'.p' in os.listdir(dir_name)\
					else (str(ind)+'.json'),
				indexes_list
			)
		)
	g = lambda s:\
		pickle.load(open( s , 'rb' )) if s[-1] == 'p' else\
		json.load(open( s , 'rt' ))

	thp_per_week_iter =\
		tuple(
			map(
				lambda ind: g('./corrected_thp_folder/'+ind),
				f( './corrected_thp_folder/' )
			)
		) if keep_time_moments else\
			tuple(
				map(
					lambda ind:\
						tuple(
							map(
								lambda p: p[1],
								g('./corrected_thp_folder/'+ind)
							)
						),
					f( './corrected_thp_folder/' )
				)
			)
	mit, mat = pickle.load(open('bin_attempt4_folders/throughput_min_max.p','rb'))
	thp_per_week_iter =\
		tuple(
			map(
				lambda w:\
					tuple(
						map(
							lambda p: ( p[0] , (p[1] - mit) / (mat - mit) ),
							w
						)
					),
				thp_per_week_iter
			)
		) if keep_time_moments else\
			tuple(
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

	rs_per_week_iter =\
		tuple(
			map(
				lambda ind: g('./bin_attempt4_folders/time_tag_cern_read_size/'+ind),
				f( './bin_attempt4_folders/time_tag_cern_read_size/' )
			)
		) if keep_time_moments else\
			tuple(
				map(
					lambda ind:\
						tuple(
							map(
								lambda p: p[1],
								g('./bin_attempt4_folders/time_tag_cern_read_size/'+ind)
							)
						),
					f( './bin_attempt4_folders/time_tag_cern_read_size/' )
				)
			)

	return tr_ind_iterable, va_ind_iterable, thp_per_week_iter, rs_per_week_iter

def create_model_func_wrapper_27_07(args):
	return bin_attempt4_models.get_model_17(args['input_window_size'],args['output_window_size'])

def generator_func_wrapper_27_07(args):
	yield from\
		bin_attempt4_batch_generators.batch_generator_6_for_16(
			args[ 'indexes' ],
			args[ 'batch_size' ],
			args[ 'thp_per_week_iter' ],
			args[ 'rs_per_week_iter' ],
			args[ 'output_time_window' ],
			args[ 'window_size' ] + 1,
		)

def get_arg_dict_for_train_or_valid_27_07(indexes_iterable, ai, aj, thp_per_week_iter, rs_per_week_iter):
	return\
		{
			'indexes' :\
				list(
					filter(
						lambda t: ai <= t[ 1 ] < len( thp_per_week_iter[ t[ 0 ] ] ) - aj + 1 ,
						indexes_iterable
					)
				),
			'batch_size' : 128 ,
			'thp_per_week_iter' : thp_per_week_iter ,
			'rs_per_week_iter' : rs_per_week_iter ,
			'output_time_window' : aj ,
			'window_size' : ai ,
		}

def get_argument_dict_27_07():
	tr_ind_iterable, va_ind_iterable, thp_per_week_iter, rs_per_week_iter = get_data_and_indexes()
	
	i = 10
	j = 10

	argument_dict = dict()
	for ii in range(17,17+25):

		print(ii,i,j)

		argument_dict[ii] = dict()

		argument_dict[ii][ CREATE_MODEL_FUNCTION ] = create_model_func_wrapper_27_07
		argument_dict[ii][ CREATE_MODEL_ARGUMENTS ] =\
			{
				'input_window_size' : i , 'output_window_size' : j
			}
		argument_dict[ii][ FIT_MODEL_FUNCTION ] = bin_attempt4_grid_search.fit_function

		argument_dict[ii][ FIT_MODEL_ARGUMENTS ] =\
			{
				TRAIN_GENERATOR : generator_func_wrapper_27_07 ,
				TRAIN_ARGUMENTS : get_arg_dict_for_train_or_valid_27_07( tr_ind_iterable , i , j ,\
					thp_per_week_iter , rs_per_week_iter ) ,
				VALID_GENERATOR : generator_func_wrapper_27_07 ,
				VALID_ARGUMENTS : get_arg_dict_for_train_or_valid_27_07( va_ind_iterable , i , j ,\
					thp_per_week_iter , rs_per_week_iter ) ,
				'log_path' : './bin_attempt4_folders/histories/' + str(ii) + '.csv',
				'model_checkpoint_path' : './bin_attempt4_folders/models/model_' + str(ii) + '/'
			}

		if not os.path.isdir(argument_dict[ii][ FIT_MODEL_ARGUMENTS ][ 'model_checkpoint_path' ]):
			os.mkdir( argument_dict[ii][ FIT_MODEL_ARGUMENTS ][ 'model_checkpoint_path' ] )

		if j == 50:
			i += 10
			j = 10
		else:
			j += 10
	
	return argument_dict

def set_27_07():
	argument_dict = get_argument_dict_27_07()

	bin_attempt4_grid_search.main( argument_dict )

def show_performance(argument_dict,dump_end_string):
	_, _, thp_per_week_iter, rs_per_week_iter = get_data_and_indexes(True,(4,))
	rs_per_week_iter = tuple( map( lambda e: dict(e) , rs_per_week_iter ) )

	print('Loaded input data !')

	from tensorflow import keras
	
	for k , v_dict in argument_dict.items():

		window_size = v_dict[ CREATE_MODEL_ARGUMENTS ][ 'input_window_size' ] + 1
		output_time_window = v_dict[ CREATE_MODEL_ARGUMENTS ][ 'output_window_size' ]
		print('Started for', k , window_size , output_time_window )

		model = keras.models.load_model(
			'./bin_attempt4_folders/models/model_' + str(k) + '/'\
				+ max(
					map(
						lambda fn:\
							(\
								int( fn[ 6 : 10 ] ),
								fn,
							),
						os.listdir( './bin_attempt4_folders/models/model_' + str(k) + '/' )
					)
				)[1]
		)

		encoder_input, decoder_input, gt_list, pred_list = list(),list(),list(),list()

		total_count = 0

		for thp_week, rs_week in zip(thp_per_week_iter,rs_per_week_iter):
			debug_thp_ind = window_size-1
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
							) + [ thp_week[j][1] , ]
						)
						print('j=',j)

					decoder_input.append(
						[ thp_week[ ind + window_size - 2 ][ 1 ] , ]\
						+ [ 0 for _ in range(output_time_window-1) ]
					)

					gt_list.append( [ thp , ] )



					total_count += 1
				
					print('ind =',ind)
					print('debug_thp_ind =',debug_thp_ind)
					print('window_size =',window_size)

					return

				debug_thp_ind += 1
		
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
						gt_list[i_dec].append( thp_week[ ind + window_size - 1 + i_output ][1] )
						i_dec += 1
			for i,r in enumerate( model.predict( [ encoder_input , decoder_input ] ) ):
				pred_list[i].append( r[ i_output , 0 ] )
		
		pickle.dump(
			( gt_list , pred_list ) ,
			open( './bin_attempt4_folders/predictions/' + str(k) + '_'+ dump_end_string + '.p' , 'wb')
		)

if __name__ == '__main__':
	# set_27_07()
	show_performance( get_argument_dict_27_07() , 'cds' )
	pass