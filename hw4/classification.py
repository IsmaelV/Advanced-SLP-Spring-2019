from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd


def get_prediction(x_train, y_final, x_test):
	model = OneVsRestClassifier(RandomForestClassifier())
	model.fit(x_train, y_final)

	return model.predict(x_test)


def test_accuracy(train, test, name):

	y_train = train['emotion_key']
	x_train = train.iloc[:, 388:]

	y_test = test['emotion_key']
	x_test = test.iloc[:, 388:]

	le = LabelEncoder().fit(y_train)
	y_final = le.transform(y_train)
	y_true = le.transform(y_test)

	y_pred = get_prediction(x_train, y_final, x_test)

	# evaluate_specifics(le.inverse_transform(y_pred), y_test)
	print(name + " Total Accuracy:", accuracy_score(y_true, y_pred, normalize=True))
	return classification_report(y_true, y_pred, target_names=target_names)


def test_cc():
	# -----------------------
	# Testing CC
	# -----------------------
	trainData = pd.read_csv('extracted_data/all_except_x/all_except_cc.csv')
	testData = pd.read_csv('extracted_data/remaining/cc_001.csv')

	trainData = trainData.drop('568', 1)  # Remove NaN from this column
	trainData = trainData.drop('571', 1)  # Remove NaN from this column
	testData = testData.drop('568', 1)  # Remove NaN from this column
	testData = testData.drop('571', 1)  # Remove NaN from this column

	print(test_accuracy(trainData, testData, 'CC'))


def test_cl():
	# -----------------------
	# Testing CL
	# -----------------------
	trainData = pd.read_csv('extracted_data/all_except_x/all_except_cl.csv')
	testData = pd.read_csv('extracted_data/remaining/cl_001.csv')

	trainData = trainData.drop('568', 1)  # Remove NaN from this column
	trainData = trainData.drop('571', 1)  # Remove NaN from this column
	testData = testData.drop('568', 1)  # Remove NaN from this column
	testData = testData.drop('571', 1)  # Remove NaN from this column

	print(test_accuracy(trainData, testData, 'CL'))


def test_gg():
	# -----------------------
	# Testing GG
	# -----------------------
	trainData = pd.read_csv('extracted_data/all_except_x/all_except_gg.csv')
	testData = pd.read_csv('extracted_data/remaining/gg_001.csv')

	trainData = trainData.drop('568', 1)  # Remove NaN from this column
	trainData = trainData.drop('571', 1)  # Remove NaN from this column
	testData = testData.drop('568', 1)  # Remove NaN from this column
	testData = testData.drop('571', 1)  # Remove NaN from this column

	print(test_accuracy(trainData, testData, 'GG'))


def test_jg():
	# -----------------------
	# Testing JG
	# -----------------------
	trainData = pd.read_csv('extracted_data/all_except_x/all_except_jg.csv')
	testData = pd.read_csv('extracted_data/remaining/jg_001.csv')

	trainData = trainData.drop('568', 1)  # Remove NaN from this column
	trainData = trainData.drop('571', 1)  # Remove NaN from this column
	testData = testData.drop('568', 1)  # Remove NaN from this column
	testData = testData.drop('571', 1)  # Remove NaN from this column

	print(test_accuracy(trainData, testData, 'JG'))


def test_mf():
	# -----------------------
	# Testing MF
	# -----------------------
	trainData = pd.read_csv('extracted_data/all_except_x/all_except_mf.csv')
	testData = pd.read_csv('extracted_data/remaining/mf_001.csv')

	trainData = trainData.drop('568', 1)  # Remove NaN from this column
	trainData = trainData.drop('571', 1)  # Remove NaN from this column
	testData = testData.drop('568', 1)  # Remove NaN from this column
	testData = testData.drop('571', 1)  # Remove NaN from this column

	print(test_accuracy(trainData, testData, 'MF'))


def test_mk():
	# -----------------------
	# Testing MK
	# -----------------------
	trainData = pd.read_csv('extracted_data/all_except_x/all_except_mk.csv')
	testData = pd.read_csv('extracted_data/remaining/mk_001.csv')

	trainData = trainData.drop('568', 1)  # Remove NaN from this column
	trainData = trainData.drop('571', 1)  # Remove NaN from this column
	testData = testData.drop('568', 1)  # Remove NaN from this column
	testData = testData.drop('571', 1)  # Remove NaN from this column

	print(test_accuracy(trainData, testData, 'MK'))


def test_mm():
	# -----------------------
	# Testing MM
	# -----------------------
	trainData = pd.read_csv('extracted_data/all_except_x/all_except_mm.csv')
	testData = pd.read_csv('extracted_data/remaining/mm_001.csv')

	trainData = trainData.drop('568', 1)  # Remove NaN from this column
	trainData = trainData.drop('571', 1)  # Remove NaN from this column
	testData = testData.drop('568', 1)  # Remove NaN from this column
	testData = testData.drop('571', 1)  # Remove NaN from this column

	print(test_accuracy(trainData, testData, 'MM'))


def test_all():
	test_cc()
	test_cl()
	test_gg()
	test_jg()
	test_mf()
	test_mk()
	test_mm()


if __name__ == '__main__':
	target_names = ['anxiety', 'boredom', 'cold-anger', 'contempt', 'despair', 'disgust', 'elation', 'happy',
																	'hot-anger', 'interest', 'neutral', 'panic',
																	'pride', 'sadness', 'shame']
	test_all()
