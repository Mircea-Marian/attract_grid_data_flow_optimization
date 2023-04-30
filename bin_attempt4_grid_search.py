from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
import multiprocessing as mp

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
	'VALID_ARGUMENTS',\

def nn_train_per_proc(i):
	'''
	Trains a neural network configuration on a GPU. It works inside a
	pool of processes.
	'''
	gpu_string = proc_q.get()

	print(str(i) + ': will start training on ' + gpu_string)

	with tf.device(gpu_string):
	# if True:

		with tf.Session() as sess:
		# if True:

			K.set_session(sess)

			model = argument_dict[ i ][ CREATE_MODEL_FUNCTION ](
				argument_dict[ i ][ CREATE_MODEL_ARGUMENTS ]
			)

			model.compile(
				optimizer=keras.optimizers.Adam(),
				loss='mean_absolute_percentage_error',
				metrics=['mae',]
			)

			argument_dict[ i ][ FIT_MODEL_FUNCTION ](
				model,
				argument_dict[ i ][ FIT_MODEL_ARGUMENTS ]
			)

	print(str(i) + ': Finished training !')

	proc_q.put(gpu_string)

def fit_function(model, args):

	# gen_obj = args[ TRAIN_GENERATOR ]( args[ TRAIN_ARGUMENTS ] )

	# a = next(gen_obj)
	# print( type( a ) )
	# print( len(a) )
	# print( type( a[0] ) )
	# print( type( a[1] ) )
	# print( type( a[0][0] ) )
	# print( type( a[0][1] ) )
	# print( a[0][0].shape )
	# print( a[0][1].shape )
	# print( a[1].shape )
	# b = next(gen_obj)
	# print( (a[0][0] == b[0][0]).all() )
	# print( (a[0][1] == b[0][1]).all() )
	# print( (a[1] == b[1]).all() )

	# model.summary()

	# exit(0)

	model.fit_generator(
		args[ TRAIN_GENERATOR ]( args[ TRAIN_ARGUMENTS ] ),
		steps_per_epoch=(len(args[ TRAIN_ARGUMENTS ][ 'indexes' ])//128)\
			if len(args[ TRAIN_ARGUMENTS ][ 'indexes' ])%128 == 0 else (1+len(args[ TRAIN_ARGUMENTS\
			][ 'indexes' ])//128),
		epochs=2500,
		validation_data=args[ VALID_GENERATOR ]( args[ VALID_ARGUMENTS ] ),
		validation_steps=(len(args[ VALID_ARGUMENTS ][ 'indexes' ])//128)\
			if len(args[ VALID_ARGUMENTS ][ 'indexes' ])%128 == 0 else (1+len(args[ VALID_ARGUMENTS\
			][ 'indexes' ])//128),
		callbacks=[
			keras.callbacks.CSVLogger( args[ 'log_path' ] ),
			keras.callbacks.ModelCheckpoint(
				args[ 'model_checkpoint_path' ] + "model_{epoch:04d}.hdf5",
				monitor='val_loss',
				save_best_only=True
			),
			keras.callbacks.EarlyStopping(
				monitor='val_loss',
				patience=700,
			)
		]
	)

def main(arg_d):
	global proc_q, argument_dict

	argument_dict = arg_d

	proc_q = mp.Queue()
	a = 0
	for gpu_string in ('/gpu:0','/gpu:1','/gpu:2','/gpu:3'):
		proc_q.put( gpu_string )
		a+=1

	print('Will start pool !')

	mp.Pool( a ).map(
		nn_train_per_proc,
		argument_dict.keys()
	)