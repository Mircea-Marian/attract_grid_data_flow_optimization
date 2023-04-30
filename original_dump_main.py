from urllib.request import urlopen

import time

from datetime import datetime

def log_main_0():

	current_distance, current_demotion =\
		urlopen('http://alimonitor.cern.ch/export/seDistance.jsp').read().decode('UTF-8'),\
		urlopen('http://alimonitor.cern.ch/export/seList.jsp').read().decode('UTF-8')


	file = open('./log_folder/' + str(int(round(time.time() * 1000))) + '_distance','wt')

	file.write(current_distance)

	file.close()

	file = open('./log_folder/' + str(int(round(time.time() * 1000))) + '_demotion','wt')

	file.write(current_demotion)

	file.close()

	while True:

		new_distance, new_demotion =\
			urlopen('http://alimonitor.cern.ch/export/seDistance.jsp').read().decode('UTF-8'),\
			urlopen('http://alimonitor.cern.ch/export/seList.jsp').read().decode('UTF-8')

		if new_distance != current_distance:
			file = open('./log_folder/' + str(int(round(time.time() * 1000))) + '_distance','wt')

			file.write(new_distance)

			file.close()

			current_distance = new_distance

		if new_demotion != current_demotion:
			file = open('./log_folder/' + str(int(round(time.time() * 1000))) + '_demotion','wt')

			file.write(new_demotion)

			file.close()

			current_demotion = new_demotion

		time.sleep(2)

def log_main_1(log_folder_path):
	current_distance, current_demotion =\
		urlopen('http://alimonitor.cern.ch/export/seDistance.jsp').read().decode('UTF-8'),\
		urlopen('http://alimonitor.cern.ch/export/seList.jsp').read().decode('UTF-8')


	file = open(log_folder_path + str(int(round(time.time() * 1000))) + '_distance','wt')

	file.write(current_distance)

	file.close()

	file = open(log_folder_path + str(int(round(time.time() * 1000))) + '_demotion','wt')

	file.write(current_demotion)

	file.close()

	errors_file = './logging_errors.txt'

	while True:

		try:

			with urlopen('http://alimonitor.cern.ch/export/seDistance.jsp') as new_distance:

				new_distance = new_distance.read().decode('UTF-8')

				if new_distance != current_distance:
					file = open(log_folder_path + str(int(round(time.time() * 1000))) + '_distance','wt')

					file.write(new_distance)

					file.close()

					current_distance = new_distance

		except:

			with open(errors_file, "a") as myfile:
				myfile.write(\
					'Distance update failed at '\
					+ datetime.now().strftime("%d/%m/%Y %H:%M:%S")\
					+ '\n'\
				)

		try:

			with urlopen('http://alimonitor.cern.ch/export/seList.jsp') as new_demotion:

				new_demotion = new_demotion.read().decode('UTF-8')

				if new_demotion != current_demotion:
					file = open(log_folder_path + str(int(round(time.time() * 1000))) + '_demotion','wt')

					file.write(new_demotion)

					file.close()

					current_demotion = new_demotion

		except:

			with open(errors_file, "a") as myfile:
				myfile.write(\
					'Demotion update failed at '\
					+ datetime.now().strftime("%d/%m/%Y %H:%M:%S")\
					+ '\n'\
				)

		time.sleep(2)

if __name__ == '__main__':
	log_main_1('./log_folder_27th_oct/')
