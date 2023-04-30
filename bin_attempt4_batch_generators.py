import random
import numpy as np

window_size = 40

def batch_generator_2(indexes,batch_size,thp_per_week_iter,rs_per_week_iter):	
	while True:
		random.shuffle( indexes )  # #STEPS = #EXAMPLE / BATCH SIZE
		
		x_arr = np.empty((
			batch_size , window_size , 4800
		))
		y_arr = np.empty((
			batch_size, window_size , 1
		))
		i_arr = 0

		for we_i, ( i_week , i_per_week, rs_ind_iter ) in enumerate(indexes):

			if i_arr == batch_size:

				yield x_arr, y_arr

				i_arr = 0

				x_arr = np.empty((
					batch_size , window_size , 4800
				))
				y_arr = np.empty((
					batch_size, window_size , 1
				))

			for i_window , i_rs in zip( range( window_size ) , rs_ind_iter ):
				x_arr[ i_arr , i_window , : ] =\
					rs_per_week_iter[ i_week ][ i_rs - 4799 : i_rs + 1 ]
			y_arr[ i_arr , : , 0 ] = thp_per_week_iter[ i_week ][ i_per_week - window_size + 1 : i_per_week + 1 ]

			i_arr += 1

		yield x_arr[:i_arr], y_arr[:i_arr] 

def batch_generator_3(indexes,batch_size,thp_per_week_iter,rs_per_week_iter):	
	while True:
		random.shuffle( indexes )  # #STEPS = #EXAMPLE / BATCH SIZE
		
		x_arr = np.empty((
			batch_size , window_size , 4800
		))
		y_arr = np.empty((
			batch_size, 1
		))
		i_arr = 0

		for we_i, ( i_week , i_per_week, rs_ind_iter ) in enumerate(indexes):

			if i_arr == batch_size:

				yield x_arr, y_arr

				i_arr = 0

				x_arr = np.empty((
					batch_size , window_size , 4800
				))
				y_arr = np.empty((
					batch_size, 1
				))

			for i_window , i_rs in zip( range( window_size ) , rs_ind_iter ):
				x_arr[ i_arr , i_window , : ] =\
					rs_per_week_iter[ i_week ][ i_rs - 4799 : i_rs + 1 ]
			y_arr[ i_arr , 0 ] = thp_per_week_iter[ i_week ][ i_per_week ]

			i_arr += 1

		yield x_arr[:i_arr], y_arr[:i_arr] 

def batch_generator_4(indexes,batch_size,thp_per_week_iter,rs_per_week_iter,top_indexes_tuple):	
	while True:
		random.shuffle( indexes )  # #STEPS = #EXAMPLE / BATCH SIZE
		
		x_arr = np.empty((
			batch_size , window_size , 1000
		))
		y_arr = np.empty((
			batch_size, window_size , 1
		))
		i_arr = 0

		for we_i, ( i_week , i_per_week, rs_ind_iter ) in enumerate(indexes):

			if i_arr == batch_size:

				yield x_arr, y_arr

				i_arr = 0

				x_arr = np.empty((
					batch_size , window_size , 1000
				))
				y_arr = np.empty((
					batch_size, window_size , 1
				))

			for i_window , i_rs in zip( range( window_size ) , rs_ind_iter ):
				for j_rs, i_top_rs in zip( range(1000) , top_indexes_tuple ):
					x_arr[ i_arr , i_window , j_rs ] =\
						rs_per_week_iter[ i_week ][ i_rs + i_top_rs + 1 ]
			y_arr[ i_arr , : , 0 ] = thp_per_week_iter[ i_week ][ i_per_week - window_size + 1 : i_per_week + 1 ]

			i_arr += 1

		yield x_arr[:i_arr], y_arr[:i_arr] 

def batch_generator_5_for_15(indexes,batch_size,thp_per_week_iter,rs_per_week_iter,):
	while True:
		random.shuffle( indexes )  # #STEPS = #EXAMPLE / BATCH SIZE
		
		x_arr = np.empty((
			batch_size , window_size - 1 , 121
		))
		y_arr = np.empty((
			batch_size, 1
		))
		i_arr = 0

		for we_i, ( i_week , i_per_week, rs_ind_iter ) in enumerate(indexes):

			if i_arr == batch_size:

				yield x_arr, y_arr

				i_arr = 0

				x_arr = np.empty((
					batch_size , window_size - 1 , 121
				))
				y_arr = np.empty((
					batch_size, 1
				))

			for i_window , i_rs , i_reverse in zip( range( window_size - 1 ) , rs_ind_iter[:-1] , range( window_size - 1 , 0 , -1 ) ):
				x_arr[ i_arr , i_window , : 120 ] =\
					rs_per_week_iter[ i_week ][ i_rs - 119 : i_rs + 1 ]
				x_arr[ i_arr , i_window , 120 ] = thp_per_week_iter[ i_week ][ i_per_week - i_reverse ]
			
			y_arr[ i_arr , 0 ] = thp_per_week_iter[ i_week ][ i_per_week ]

			i_arr += 1

		yield x_arr[:i_arr], y_arr[:i_arr] 

def batch_generator_6_for_16(\
		indexes,\
		batch_size,\
		thp_per_week_iter,\
		rs_per_week_iter,\
		output_time_window,\
		window_size\
		):
	while True:
		random.shuffle( indexes )  # #STEPS = #EXAMPLE / BATCH SIZE
		
		x1_arr = np.empty((
			batch_size , window_size - 1 , 121
		))
		x2_arr = np.empty((
			batch_size , output_time_window , 1
		))
		y_arr = np.empty((
			batch_size, output_time_window , 1
		))
		i_arr = 0

		for i_week , i_per_week, rs_ind_iter in indexes:

			# if i_per_week + output_time_window <= len( thp_per_week_iter[ i_week ] ):
			if True:

				if i_arr == batch_size:

					yield [ x1_arr , x2_arr ] , y_arr

					i_arr = 0

					x1_arr = np.empty((
						batch_size , window_size - 1 , 121
					))
					x2_arr = np.empty((
						batch_size , output_time_window , 1
					))
					y_arr = np.empty((
						batch_size, output_time_window , 1
					))

				for i_window , i_rs , i_reverse in zip( range( window_size - 1 ) , rs_ind_iter[:-1] , range( window_size - 1 , 0 , -1 ) ):
					x1_arr[ i_arr , i_window , : 120 ] =\
						rs_per_week_iter[ i_week ][ i_rs - 119 : i_rs + 1 ]
					x1_arr[ i_arr , i_window , 120 ] = thp_per_week_iter[ i_week ][ i_per_week - i_reverse ]
				
				x2_arr[ i_arr , : , 0 ] = thp_per_week_iter[ i_week ][ i_per_week - 1 : i_per_week + output_time_window - 1 ]
				y_arr[ i_arr , : , 0 ] = thp_per_week_iter[ i_week ][ i_per_week : i_per_week + output_time_window ]

				i_arr += 1

		yield [ x1_arr[:i_arr] , x2_arr[:i_arr] ] , y_arr[:i_arr]