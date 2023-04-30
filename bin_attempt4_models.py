# import keras
from tensorflow import keras

window_size = 40

def get_td_bn_dr(prev_x, unit_count, dr=-1):
	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=unit_count,
			activation='relu'
		)
	)(prev_x)
	x = keras.layers.BatchNormalization()(x)
	if dr == -1: return x
	return keras.layers.Dropout(dr)(x)

def get_lstm_bn_dr(prev_x, unit_count, dr=-1, return_eq_last=True):
	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=unit_count,
			return_sequences=return_eq_last,
		)
	)(prev_x)
	x = keras.layers.BatchNormalization()(x)
	if dr == -1: return x
	return keras.layers.Dropout(dr)(x)

def get_model_0():
	inp_layer = x = keras.layers.Input(shape=(window_size, 4800,))

	x = get_td_bn_dr( x , 40 )
	x = get_lstm_bn_dr( x , 40 )
	x = get_td_bn_dr( x , 30 )
	x = get_td_bn_dr( x , 15 )
	x = get_td_bn_dr( x , 7 )
	x = get_lstm_bn_dr( x , 7 )

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='tanh'
		)
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_model_1():
	inp_layer = x = keras.layers.Input(shape=(window_size, 4800,))

	x = get_td_bn_dr( x , 64 )
	x = get_lstm_bn_dr( x , 32 )
	x = get_td_bn_dr( x , 32 )
	x = get_td_bn_dr( x , 16 )
	x = get_td_bn_dr( x , 8 )
	x = get_td_bn_dr( x , 4 )
	x = get_td_bn_dr( x , 2 )
	x = get_lstm_bn_dr( x , 2 )

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='tanh'
		)
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_model_2():
	inp_layer = x = keras.layers.Input(shape=(window_size, 4800,))

	x = get_td_bn_dr( x , 256 )
	x = get_lstm_bn_dr( x , 128 )
	x = get_td_bn_dr( x , 64 )
	x = get_td_bn_dr( x , 16 )
	x = get_td_bn_dr( x , 4 )
	x = get_lstm_bn_dr( x , 4 )

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='tanh'
		)
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_model_3():
	inp_layer = x = keras.layers.Input(shape=(window_size, 4800,))

	x = get_td_bn_dr( x , 600 , 0.45 )
	x = get_td_bn_dr( x , 40 )
	x = get_lstm_bn_dr( x , 40 )
	x = get_td_bn_dr( x , 30 )
	x = get_td_bn_dr( x , 15 )
	x = get_td_bn_dr( x , 2 )
	x = get_lstm_bn_dr( x , 2 )

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='tanh'
		)
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_model_4():
	inp_layer = x = keras.layers.Input(shape=(window_size, 4800,))

	x = get_td_bn_dr( x , 8 )
	x = get_lstm_bn_dr( x , 4 )
	x = get_td_bn_dr( x , 4 )
	x = get_lstm_bn_dr( x , 2 )
	x = get_td_bn_dr( x , 2 )
	x = get_lstm_bn_dr( x , 2 )

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='tanh'
		)
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_model_5():
	inp_layer = x = keras.layers.Input(shape=(window_size, 4800,))

	x = get_td_bn_dr( x , 32 )
	x = get_lstm_bn_dr( x , 16 )
	x = get_td_bn_dr( x , 16 )
	x = get_td_bn_dr( x , 8 )
	x = get_td_bn_dr( x , 4 )
	x = get_td_bn_dr( x , 2 )
	x = get_lstm_bn_dr( x , 2 )

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='tanh'
		)
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_model_6():
	inp_layer = x = keras.layers.Input(shape=(window_size, 4800,))

	x = get_td_bn_dr( x , 16 )
	x = get_lstm_bn_dr( x , 8 )
	x = get_td_bn_dr( x , 8 )
	x = get_td_bn_dr( x , 4 )
	x = get_td_bn_dr( x , 2 )
	x = get_lstm_bn_dr( x , 2 )

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='tanh'
		)
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_model_7():
	inp_layer = x = keras.layers.Input(shape=(window_size, 4800,))

	x = get_td_bn_dr( x , 8 )
	x = get_lstm_bn_dr( x , 4 )
	x = get_td_bn_dr( x , 4 )
	x = get_td_bn_dr( x , 2 )
	x = get_lstm_bn_dr( x , 2 )

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='tanh'
		)
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_model_8():
	inp_layer = x = keras.layers.Input(shape=(window_size, 4800,))

	x = get_td_bn_dr( x , 4 )
	x = get_lstm_bn_dr( x , 2 )
	x = get_td_bn_dr( x , 2 )
	x = get_lstm_bn_dr( x , 2 )

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='tanh'
		)
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_model_9():
	inp_layer = x = keras.layers.Input(shape=(window_size, 4800,))

	x = get_td_bn_dr( x , 16 , 0.4 )
	x = get_lstm_bn_dr( x , 8 )
	x = get_td_bn_dr( x , 8 )
	x = get_td_bn_dr( x , 4 )
	x = get_td_bn_dr( x , 2 )
	x = get_lstm_bn_dr( x , 2 )

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='tanh'
		)
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_model_10():
	inp_layer = x = keras.layers.Input(shape=(window_size, 4800,))

	x = get_td_bn_dr( x , 4 ) # time distributed - batch norm - dr
	x = get_lstm_bn_dr( x , 2 )
	x = get_td_bn_dr( x , 2 )
	x = get_lstm_bn_dr( x , 1 )

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='tanh'
		)
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_model_11():
	inp_layer = x = keras.layers.Input(shape=(window_size, 4800,))

	x = get_td_bn_dr( x , 8 , 0.4 )
	x = get_lstm_bn_dr( x , 4 , 0.4 )
	x = get_td_bn_dr( x , 4 , 0.4 )
	x = get_td_bn_dr( x , 2 )
	x = get_lstm_bn_dr( x , 1 )

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='tanh'
		)
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_model_12():
	inp_layer = x = keras.layers.Input(shape=(window_size, 4800,))

	x = get_td_bn_dr( x , 8 )
	x = get_lstm_bn_dr( x , 4 )
	x = get_td_bn_dr( x , 4 )
	x = get_td_bn_dr( x , 2 )
	x = get_lstm_bn_dr( x , 1 , return_eq_last=False )

	x = keras.layers.Dense(
		units=1,
		activation='tanh'
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_model_13():
	inp_layer = x = keras.layers.Input(shape=(window_size, 1000,))

	x = get_td_bn_dr( x , 4 )
	x = get_lstm_bn_dr( x , 2 )
	x = get_td_bn_dr( x , 2 )
	x = get_lstm_bn_dr( x , 1 )

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='tanh'
		)
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_model_14():
	inp_layer = x = keras.layers.Input(shape=(window_size, 4800,))

	x = get_td_bn_dr( x , 4 ) # time distributed - batch norm - dr
	x = get_lstm_bn_dr( x , 2 )
	x = get_td_bn_dr( x , 2 )
	x = get_lstm_bn_dr( x , 1 )

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='relu'
		)
	)(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_model_15():
	inp_layer = x = keras.layers.Input(shape=(window_size - 1 , 121,))

	# x = get_td_bn_dr( x , 60 , 0.2 )
	# x = get_lstm_bn_dr( x , 30 )
	x = get_td_bn_dr( x , 25 )
	x = get_td_bn_dr( x , 15 )
	x = get_lstm_bn_dr( x , 5 , return_eq_last=False)

	x = keras.layers.Dense(units=1,activation='relu')(x)

	return keras.models.Model(inputs=inp_layer, outputs=x)

def get_model_16(output_time_window):
	encoder_input = keras.layers.Input(shape=(window_size - 1 , 121,))

	x = get_td_bn_dr( encoder_input , 25 )
	x = get_td_bn_dr( x , 15 )

	_ , forward_state_h, forward_state_c, backward_state_h, backward_state_c = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=5,
			return_state=True
		)
	)(x)


	decoder_input = keras.layers.Input(shape=(output_time_window, 1))

	x = keras.layers.Bidirectional(
		keras.layers.LSTM(
			units=5,
			return_sequences=True
		)
	)(decoder_input, initial_state=(forward_state_h, forward_state_c, backward_state_h, backward_state_c),)

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='relu'
		)
	)(x)

	return keras.models.Model([encoder_input, decoder_input], x)

def get_model_17(window_size,output_time_window):
	encoder_input = keras.layers.Input(shape=(window_size, 121,))

	x = get_td_bn_dr( encoder_input , 25 )
	x = get_td_bn_dr( x , 15 )

	_ , forward_state_h, forward_state_c =\
		keras.layers.LSTM(
			units=5,
			return_state=True
		)(x)


	decoder_input = keras.layers.Input(shape=(output_time_window, 1))

	x = keras.layers.LSTM(
			units=5,
			return_sequences=True,
	)(decoder_input,initial_state=(forward_state_h, forward_state_c))

	x = keras.layers.TimeDistributed(
		keras.layers.Dense(
			units=1,
			activation='relu'
		)
	)(x)

	return keras.models.Model([encoder_input, decoder_input], x)