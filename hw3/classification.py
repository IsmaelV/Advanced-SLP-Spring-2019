from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


def get_prediction(x_train, y_final, x_test):
	model = OneVsRestClassifier(RandomForestClassifier())
	model.fit(x_train, y_final)

	return model.predict(x_test)


def test_liwc(train, test):

	y_train = train['act_tag']
	x_train = train.iloc[:, 12:100]

	y_test = test['act_tag']
	x_test = test.iloc[:, 12:100]

	le = LabelEncoder().fit(y_train)
	y_final = le.transform(y_train)
	y_true = le.transform(y_test)

	y_pred = get_prediction(x_train, y_final, x_test)

	evaluate_specifics(le.inverse_transform(y_pred), y_test)
	print("LIWC Total Accuracy:", accuracy_score(y_true, y_pred, normalize=True))


def test_ugram(train, train_ugram, test, test_ugram):
	y_train = train['act_tag']
	y_test = test['act_tag']

	le = LabelEncoder().fit(y_train)
	y_final = le.transform(y_train)
	y_true = le.transform(y_test)

	y_pred = get_prediction(train_ugram, y_final, test_ugram)

	evaluate_specifics(le.inverse_transform(y_pred), y_test)
	print("Unigram Accuracy:", accuracy_score(y_true, y_pred, normalize=True))


def test_misc_features(train, test):
	y_train = train['act_tag']
	x_train = train.iloc[:, 102]
	x_train = x_train.values.reshape(-1, 1)

	y_test = test['act_tag']
	x_test = test.iloc[:, 102]
	x_test = x_test.values.reshape(-1, 1)

	le = LabelEncoder().fit(y_train)
	y_final = le.transform(y_train)
	y_true = le.transform(y_test)

	y_pred = get_prediction(x_train, y_final, x_test)

	evaluate_specifics(le.inverse_transform(y_pred), y_test)
	print("Miscellaneous Features Accuracy:", accuracy_score(y_true, y_pred, normalize=True))


def test_last_pos(train, test):
	y_train = train['act_tag']
	x_train = train.iloc[:, 103]

	y_test = test['act_tag']
	x_test = test.iloc[:, 103]

	le = LabelEncoder().fit(y_train)
	y_final = le.transform(y_train)
	y_true = le.transform(y_test)

	le2 = LabelEncoder().fit(x_train)
	x_train_final = le2.transform(x_train).reshape(-1, 1)
	x_test_final = le2.transform(x_test).reshape(-1, 1)

	y_pred = get_prediction(x_train_final, y_final, x_test_final)

	evaluate_specifics(le.inverse_transform(y_pred), y_test)
	print("Last POS Accuracy:", accuracy_score(y_true, y_pred, normalize=True))


def test_my_features(train, train_ugram, test, test_ugram):
	y_train = train['act_tag']
	y_test = test['act_tag']

	# Y setup
	le = LabelEncoder().fit(y_train)
	y_final = le.transform(y_train)
	y_true = le.transform(y_test)

	# Miscellaneous setup
	x_test_misc = test.iloc[:, 102].values.reshape(-1, 1)
	x_train_misc = train.iloc[:, 102].values.reshape(-1, 1)

	# POS setup
	x_test_pos = test.iloc[:, 103]
	x_train_pos = train.iloc[:, 103]
	le2 = LabelEncoder().fit(x_train_pos)
	x_train_pos_final = le2.transform(x_train_pos).reshape(-1, 1)
	x_test_pos_final = le2.transform(x_test_pos).reshape(-1, 1)

	x_train_tmp = np.concatenate((x_train_misc, x_train_pos_final), axis=1)
	x_test_tmp = np.concatenate((x_test_misc, x_test_pos_final), axis=1)

	x_train = np.concatenate((x_train_tmp, train_ugram), axis=1)
	x_test = np.concatenate((x_test_tmp, test_ugram), axis=1)

	y_pred = get_prediction(x_train, y_final, x_test)

	evaluate_specifics(le.inverse_transform(y_pred), y_test)
	print("All My Features Accuracy:", accuracy_score(y_true, y_pred, normalize=True))


def test_all_features(train, train_ugram, test, test_ugram):
	y_train = train['act_tag']
	y_test = test['act_tag']

	# Y setup
	le = LabelEncoder().fit(y_train)
	y_final = le.transform(y_train)
	y_true = le.transform(y_test)

	# Miscellaneous setup
	x_test_misc = test.iloc[:, 102].values.reshape(-1, 1)
	x_train_misc = train.iloc[:, 102].values.reshape(-1, 1)

	# POS setup
	x_test_pos = test.iloc[:, 103]
	x_train_pos = train.iloc[:, 103]
	le2 = LabelEncoder().fit(x_train_pos)
	x_train_pos_final = le2.transform(x_train_pos).reshape(-1, 1)
	x_test_pos_final = le2.transform(x_test_pos).reshape(-1, 1)

	x_train = np.concatenate((x_train_misc, x_train_pos_final), axis=1)
	x_test = np.concatenate((x_test_misc, x_test_pos_final), axis=1)

	# Ugram setup
	x_train = np.concatenate((x_train, train_ugram), axis=1)
	x_test = np.concatenate((x_test, test_ugram), axis=1)

	# LIWC setup
	x_train = np.concatenate((x_train, train.iloc[:, 12:100].values), axis=1)
	x_test = np.concatenate((x_test, test.iloc[:, 12:100].values), axis=1)

	y_pred = get_prediction(x_train, y_final, x_test)

	evaluate_specifics(le.inverse_transform(y_pred), y_test)
	print("All My Features Accuracy:", accuracy_score(y_true, y_pred, normalize=True))


def evaluate_specifics(y_decoded, y_test):

	total = {'sd': 0, 'b': 0, 'sv': 0, 'aa': 0, 'ba': 0, 'qy': 0, 'x': 0, 'ny': 0, 'fc': 0, 'qw': 0}
	correct = {'sd': 0, 'b': 0, 'sv': 0, 'aa': 0, 'ba': 0, 'qy': 0, 'x': 0, 'ny': 0, 'fc': 0, 'qw': 0}

	for i in range(0, len(y_decoded)):
		correct_ans = y_test[i]
		if y_decoded[i] == correct_ans:
			correct[correct_ans] = correct[correct_ans] + 1
		total[correct_ans] = total[correct_ans] + 1

	for key in correct:
		print(key, "has accuracy of", correct[key] / total[key])


def extract_ugrams(file_to_open):
	loaded_unigrams = np.load(file_to_open)       # Load unigrams for train
	ugrams_1 = loaded_unigrams['ugrams_1']
	# ugrams_2 = loaded_unigrams['ugrams_2']
	# ugrams_3 = loaded_unigrams['ugrams_3']
	# tot_size = len(ugrams_1) + len(ugrams_2) + len(ugrams_3)
	tot_size = len(ugrams_1)
	all_ugrams = np.empty((tot_size, 16108))  # Create empty DataFrame

	for idx, u in enumerate(ugrams_1):
		all_ugrams[idx] = u.flatten()

	# for idx, u in enumerate(ugrams_2):
	# 	to_add = idx + len(ugrams_1)
	# 	all_ugrams[to_add] = u.flatten()
	#
	# for idx, u in enumerate(ugrams_3):
	# 	to_add = idx + len(ugrams_1) + len(ugrams_2)
	# 	all_ugrams[to_add] = u.flatten()

	# ugrams_1, ugrams_2, ugrams_3 = None, None, None
	ugrams_1 = None
	loaded_unigrams.close()

	return all_ugrams


if __name__ == "__main__":
	trainData = pd.read_csv('data/final_train_merged.csv')
	trainData = trainData[0:50000]
	print("Extracting train unigrams ...")
	train_ugrams = extract_ugrams('data/train_npz_ugrams_compressed.npz')
	print("Finished extracting train unigrams\n")

	testData = pd.read_csv('data/final_test_merged.csv')
	print("Extracting test unigrams ...")
	test_ugrams = extract_ugrams('data/test_npz_ugrams_compressed.npz')
	print("Finished extracting test unigrams\n")

	# test_liwc(trainData, testData)
	# test_ugram(trainData, train_ugrams, testData, test_ugrams)
	# test_misc_features(trainData, testData)
	# test_last_pos(trainData, testData)
	# test_my_features(trainData, train_ugrams, testData, test_ugrams)
	test_all_features(trainData, train_ugrams, testData, test_ugrams)
