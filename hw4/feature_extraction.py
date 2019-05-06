import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
from statistics import mean
from statistics import stdev


def separate_files():
	cc_wavs = []
	cl_wavs = []
	gg_wavs = []
	jg_wavs = []
	mf_wavs = []
	mk_wavs = []
	mm_wavs = []

	for file in os.listdir('data/'):
		if 'cc_001' in file:
			cc_wavs.append(file)
		elif 'cl_001' in file:
			cl_wavs.append(file)
		elif 'gg_001' in file:
			gg_wavs.append(file)
		elif 'jg_001' in file:
			jg_wavs.append(file)
		elif 'mf_001' in file:
			mf_wavs.append(file)
		elif 'mk_001' in file:
			mk_wavs.append(file)
		elif 'mm_001' in file:
			mm_wavs.append(file)

	return cc_wavs, cl_wavs, gg_wavs, jg_wavs, mf_wavs, mk_wavs, mm_wavs


def build_one_speaker_features(wavs, name):
	# -------------------------------------------
	# Set up csv file to write
	# -------------------------------------------
	print('--------------------------------------')
	print("Building " + name + " csv file for features...\n")
	csvinput = open('extracted_data/speaker_features/' + name + ".csv", 'w')
	writer = csv.writer(csvinput, delimiter=',', lineterminator='\n')

	headers = ['filename', 'min_pitch', 'max_pitch', 'mean_pitch', 'min_intensity', 'max_intensity', 'mean_intensity']
	writer.writerow([h for h in headers])

	# -------------------------------------------
	# Start using parselmouth for data extraction
	# -------------------------------------------
	for file in wavs:
		sound = parselmouth.Sound('data/' + file).extract_left_channel()
		pitch = sound.to_pitch()
		intensity = sound.to_intensity()

		writer.writerow([
			file,
			parselmouth.praat.call(pitch, "Get quantile", 0.0, 0.0, 0.01, "Hertz"),
			parselmouth.praat.call(pitch, "Get quantile", 0.0, 0.0, 0.99, "Hertz"),
			parselmouth.praat.call(pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz"),
			parselmouth.praat.call(intensity, "Get quantile", 0.0, 0.0, 0.01),
			parselmouth.praat.call(intensity, "Get quantile", 0.0, 0.0, 0.99),
			parselmouth.praat.call(intensity, "Get quantile", 0.0, 0.0, 0.5)
		])

	csvinput.close()

	# -------------------------------------------
	# Get z-scores
	# -------------------------------------------
	print("Calculating z-scores...")
	df = pd.read_csv('extracted_data/speaker_features/' + name + '.csv')
	cols = list(df.columns)
	cols.remove('filename')
	for col in cols:
		col_zscore = col + '_zscore'
		df[col_zscore] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

	df.to_csv('extracted_data/speaker_features/' + name + '.csv', index=False)
	print("Calculated z-scores")

	print("\nFinished building csv file")
	return df


def get_opensmile_z_scores(name):
	# -------------------------------------------
	# Get z-scores
	# -------------------------------------------
	print("Calculating z-scores...")
	df = pd.read_csv('extracted_data/opensmile/' + name + '.csv', header=None)
	cols = len(df.columns)
	df[len(df.columns)] = ''
	for col in range(1, cols - 1):
		df[len(df.columns)] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

	df.to_csv('extracted_data/opensmile/' + name + '.csv', index=False)
	print("Calculated z-scores")

	print("\nFinished building csv file")
	return df


def build_features():
	cc, cl, gg, jg, mf, mk, mm = separate_files()

	cc_df = build_one_speaker_features(cc, "cc_001")
	cl_df = build_one_speaker_features(cl, "cl_001")
	gg_df = build_one_speaker_features(gg, "gg_001")
	jg_df = build_one_speaker_features(jg, "jg_001")
	mf_df = build_one_speaker_features(mf, "mf_001")
	mk_df = build_one_speaker_features(mk, "mk_001")
	mm_df = build_one_speaker_features(mm, "mm_001")

	result = cc_df.append(cl_df)
	result = result.append(gg_df)
	result = result.append(jg_df)
	result = result.append(mf_df)
	result = result.append(mk_df)
	result = result.append(mm_df)

	result.to_csv('all_features.csv', index=False)
	print("Created all_features.csv that holds all data together")


def get_plot(feature, normalized=True):
	if feature == 'min_pitch':
		x = 1
	elif feature == 'max_pitch':
		x = 2
	elif feature == 'mean_pitch':
		x = 3
	elif feature == 'min_intensity':
		x = 4
	elif feature == 'max_intensity':
		x = 5
	elif feature == 'mean_intensity':
		x = 6
	else:
		print("Must put valid feature. Valid features are the following:")
		print("'min_pitch', 'max_pitch', 'mean_pitch', 'min_intensity', 'max_intensity', mean_intensity'")
		return False

	if normalized:
		x += 6
		feature += '_normalized'

	csv_file = open('all_features.csv', 'r')
	reader = csv.reader(csv_file, delimiter=',')
	next(reader)  # Skip headers

	# All 15 emotions
	anxiety = []
	boredom = []
	cold_anger = []
	contempt = []
	despair = []
	disgust = []
	elation = []
	happy = []
	hot_anger = []
	interest = []
	neutral = []
	panic = []
	pride = []
	sadness = []
	shame = []

	# total = []  # Has all values

	for row in reader:
		name = row[0]
		val = float(row[x])
		# total.append(val)
		if '_anxiety_' in name:
			anxiety.append(val)
		elif '_boredom_' in name:
			boredom.append(val)
		elif '_cold-anger_' in name:
			cold_anger.append(val)
		elif '_contempt_' in name:
			contempt.append(val)
		elif '_despair_' in name:
			despair.append(val)
		elif '_disgust_' in name:
			disgust.append(val)
		elif '_elation_' in name:
			elation.append(val)
		elif '_happy_' in name:
			happy.append(val)
		elif '_hot-anger_' in name:
			hot_anger.append(val)
		elif '_interest_' in name:
			interest.append(val)
		elif '_neutral_' in name:
			neutral.append(val)
		elif '_panic_' in name:
			panic.append(val)
		elif '_pride_' in name:
			pride.append(val)
		elif '_sadness_' in name:
			sadness.append(val)
		elif '_shame_' in name:
			shame.append(val)

	x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

	means = [mean(anxiety), mean(boredom), mean(cold_anger), mean(contempt), mean(despair),
			mean(disgust), mean(elation), mean(happy), mean(hot_anger), mean(interest),
			mean(neutral), mean(panic), mean(pride), mean(sadness), mean(shame)]
	stddevs = [stdev(anxiety), stdev(boredom), stdev(cold_anger), stdev(contempt), stdev(despair),
			stdev(disgust), stdev(elation), stdev(happy), stdev(hot_anger), stdev(interest),
			stdev(neutral), stdev(panic), stdev(pride), stdev(sadness), stdev(shame)]

	plt.errorbar(x, np.asarray(means), np.asarray(stddevs), linestyle='None', marker='^')
	plt.title(feature)
	plt.xlabel('emotions')
	plt.xticks(x, ('anx', 'bor', 'c-ang', 'cont', 'desp', 'disg', 'ela', 'hap',
						'h-ang', 'int', 'neut', 'pan', 'pri', 'sad', 'shame'))
	plt.show()


def get_all_plots():
	get_plot('min_pitch', normalized=False)
	get_plot('min_pitch', normalized=True)

	get_plot('max_pitch', normalized=False)
	get_plot('max_pitch', normalized=True)

	get_plot('mean_pitch', normalized=False)
	get_plot('mean_pitch', normalized=True)

	get_plot('min_intensity', normalized=False)
	get_plot('min_intensity', normalized=True)

	get_plot('max_intensity', normalized=False)
	get_plot('max_intensity', normalized=True)

	get_plot('mean_intensity', normalized=False)
	get_plot('mean_intensity', normalized=True)


def perform_commands(wav_files):
	open_smile_raw = './extracted_data/opensmile/' + wav_files[0][:6] + '_openSmile_raw.csv'
	open_smile_tmp = './extracted_data/opensmile/' + wav_files[0][:6] + '_openSmile_tmp.csv'
	open_smile = './extracted_data/opensmile/' + wav_files[0][:6] + '_openSmile.csv'

	# -------------------------------------------
	# Execute the OpenSmile command to get arff file as csv
	# -------------------------------------------
	command = 'SMILExtract -C C:/Windows/opensmile-2.3.0/config/IS09_emotion.conf -I ./data/'
	for file in wav_files:
		os.system(command + file + ' -O ' + open_smile_raw)

	# -------------------------------------------
	# Remove the first 391 rows that do not have anything
	# -------------------------------------------
	n = 391
	with open(open_smile_raw) as f, open(open_smile_tmp, 'w') as out:
		for x in range(n):
			next(f)
		for line in f:
			out.write(line)

	# -------------------------------------------
	# Rename 'unknown' as the correct file name
	# -------------------------------------------
	csv_out = open(open_smile, 'w')
	writer = csv.writer(csv_out, delimiter=',', lineterminator='\n')
	csv_in = open(open_smile_tmp, 'r')
	reader = csv.reader(csv_in, delimiter=',')

	count = 0
	for row in reader:
		row[0] = wav_files[count]
		writer.writerow(row)
		count += 1

	# -------------------------------------------
	# Clean up
	# -------------------------------------------
	csv_out.close()
	csv_in.close()
	os.remove(open_smile_raw)
	os.remove(open_smile_tmp)


def get_emotion_list():
	emotions = []
	for file in os.listdir('data/'):
		emotions.append(file.split('_')[2])
	return emotions


def all_except_csv(a, b, c, d, e, f):
	# -------------------------------------------
	# MUST HAVE SAME ORDER AS all_except_emotions WHEN CALLED
	# -------------------------------------------
	result = a.append(b)
	result = result.append(c)
	result = result.append(d)
	result = result.append(e)
	result = result.append(f)

	return result


def all_except_emotions(a, b, c, d, e, f):
	# -------------------------------------------
	# MUST HAVE SAME ORDER AS all_except_csv WHEN CALLED
	# -------------------------------------------
	result = []
	for file in a:
		result.append(file.split('_')[2])
	for file in b:
		result.append(file.split('_')[2])
	for file in c:
		result.append(file.split('_')[2])
	for file in d:
		result.append(file.split('_')[2])
	for file in e:
		result.append(file.split('_')[2])
	for file in f:
		result.append(file.split('_')[2])
	return result


def remaining_emotions(rem):
	result = []
	for file in rem:
		result.append(file.split('_')[2])
	return result


def get_opensmile_csv():
	print('Getting opensmile csv files...')
	cc, cl, gg, jg, mf, mk, mm = separate_files()

	# -------------------------------------------
	# Get initial opensmile csvs for speakers
	# -------------------------------------------
	perform_commands(cc)
	perform_commands(cl)
	perform_commands(gg)
	perform_commands(jg)
	perform_commands(mf)
	perform_commands(mk)
	perform_commands(mm)

	# -------------------------------------------
	# Obtain dataframe of csv with z-score
	# -------------------------------------------
	cc_df = get_opensmile_z_scores('cc_001_openSmile')
	cl_df = get_opensmile_z_scores('cl_001_openSmile')
	gg_df = get_opensmile_z_scores('gg_001_openSmile')
	jg_df = get_opensmile_z_scores('jg_001_openSmile')
	mf_df = get_opensmile_z_scores('mf_001_openSmile')
	mk_df = get_opensmile_z_scores('mk_001_openSmile')
	mm_df = get_opensmile_z_scores('mm_001_openSmile')

	# -------------------------------------------
	# Create total result with all speakers
	# -------------------------------------------
	total_result = cc_df.append(cl_df)
	total_result = total_result.append(gg_df)
	total_result = total_result.append(jg_df)
	total_result = total_result.append(mf_df)
	total_result = total_result.append(mk_df)
	total_result = total_result.append(mm_df)

	emotions = get_emotion_list()
	total_result.insert(1, 'emotion_key', emotions)

	total_result.to_csv('all_opensmile_features.csv', index=False)
	print("Created all_opensmile_features.csv that holds all data together")

	# -------------------------------------------
	# Create all_but results with every speaker except one
	# -------------------------------------------
	# All Except CC
	all_except = all_except_csv(cl_df, gg_df, jg_df, mf_df, mk_df, mm_df)
	em = all_except_emotions(cl, gg, jg, mf, mk, mm)
	all_except.insert(1, 'emotion_key', em)
	all_except.to_csv('extracted_data/all_except_x/all_except_cc.csv', index=False)

	# All Except CL
	all_except = all_except_csv(cc_df, gg_df, jg_df, mf_df, mk_df, mm_df)
	em = all_except_emotions(cc, gg, jg, mf, mk, mm)
	all_except.insert(1, 'emotion_key', em)
	all_except.to_csv('extracted_data/all_except_x/all_except_cl.csv', index=False)

	# All Except GG
	all_except = all_except_csv(cc_df, cl_df, jg_df, mf_df, mk_df, mm_df)
	em = all_except_emotions(cc, cl, jg, mf, mk, mm)
	all_except.insert(1, 'emotion_key', em)
	all_except.to_csv('extracted_data/all_except_x/all_except_gg.csv', index=False)

	# All Except JG
	all_except = all_except_csv(cc_df, cl_df, gg_df, mf_df, mk_df, mm_df)
	em = all_except_emotions(cc, cl, gg, mf, mk, mm)
	all_except.insert(1, 'emotion_key', em)
	all_except.to_csv('extracted_data/all_except_x/all_except_jg.csv', index=False)

	# All Except MF
	all_except = all_except_csv(cc_df, cl_df, gg_df, jg_df, mk_df, mm_df)
	em = all_except_emotions(cc, cl, gg, jg, mk, mm)
	all_except.insert(1, 'emotion_key', em)
	all_except.to_csv('extracted_data/all_except_x/all_except_mf.csv', index=False)

	# All Except MK
	all_except = all_except_csv(cc_df, cl_df, gg_df, jg_df, mf_df, mm_df)
	em = all_except_emotions(cc, cl, gg, jg, mf, mm)
	all_except.insert(1, 'emotion_key', em)
	all_except.to_csv('extracted_data/all_except_x/all_except_mk.csv', index=False)

	# All Except MM
	all_except = all_except_csv(cc_df, cl_df, gg_df, jg_df, mf_df, mk_df)
	em = all_except_emotions(cc, cl, gg, jg, mf, mk)
	all_except.insert(1, 'emotion_key', em)
	all_except.to_csv('extracted_data/all_except_x/all_except_mm.csv', index=False)

	# -------------------------------------------
	# Get remaining csvs
	# -------------------------------------------
	# CC
	em = remaining_emotions(cc)
	cc_df.insert(1, 'emotion_key', em)
	cc_df.to_csv('extracted_data/remaining/cc_001.csv', index=False)

	# CL
	em = remaining_emotions(cl)
	cl_df.insert(1, 'emotion_key', em)
	cl_df.to_csv('extracted_data/remaining/cl_001.csv', index=False)

	# GG
	em = remaining_emotions(gg)
	gg_df.insert(1, 'emotion_key', em)
	gg_df.to_csv('extracted_data/remaining/gg_001.csv', index=False)

	# JG
	em = remaining_emotions(jg)
	jg_df.insert(1, 'emotion_key', em)
	jg_df.to_csv('extracted_data/remaining/jg_001.csv', index=False)

	# MF
	em = remaining_emotions(mf)
	mf_df.insert(1, 'emotion_key', em)
	mf_df.to_csv('extracted_data/remaining/mf_001.csv', index=False)

	# MK
	em = remaining_emotions(mk)
	mk_df.insert(1, 'emotion_key', em)
	mk_df.to_csv('extracted_data/remaining/mk_001.csv', index=False)

	# MM
	em = remaining_emotions(mm)
	mm_df.insert(1, 'emotion_key', em)
	mm_df.to_csv('extracted_data/remaining/mm_001.csv', index=False)


if __name__ == '__main__':
	# build_features()
	# get_all_plots()
	get_opensmile_csv()
