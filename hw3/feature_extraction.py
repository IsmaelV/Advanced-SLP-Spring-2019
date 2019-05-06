from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv
import re
import sys

global new_unigram_vectorizer
# global new_bigram_vectorizer
global dim_uni
# global dim_bi


def new_ngrams(text):
	try:
		return new_unigram_vectorizer.transform([text]).toarray()
	except ValueError:
		return np.zeros((1, dim_uni), dtype=int)


def remove_arrowheads(arrowed_text):
	amt_matches = re.findall('<.*?>', arrowed_text)
	result = re.sub('<.*?>', '', arrowed_text)
	return result.strip(), len(amt_matches)


def remove_curlies(curlied_text):
	# Remove all {} instances and replace with wanted word
	result = curlied_text
	m = re.findall(r"\{(.*?)\}", result)
	for i in range(0, len(m)):
		replacement = m[i][2:-1]
		result = re.sub(r"\{(.*?)\}", str(replacement), result, 1)

	return result


def remove_double_parentheses(p_text):
	result = p_text.replace('((', '')
	result = result.replace('))', '')
	return result.strip()


def remove_hashtags(h_text):
	result = h_text.replace('#', '')
	return result.strip()


def remove_EoU(total_text):
	result = total_text.replace('/', '')
	return result.strip()


def remove_comments(total_text):
	return total_text.split("*")[0].strip()


def remove_init_dashes(text):
	dashes = '-- '
	if text.find(dashes) == 0:
		return text[3:]
	else:
		return text


def remove_multi_space(text):
	result = text.replace('  ', ' ')
	result = result.replace('   ', ' ')
	result = result.replace('    ', ' ')
	return result.strip()


def remove_brackets(bracketed_text):
	amt_of_open_brackets = bracketed_text.count('[')
	amt_of_closed_brackets = bracketed_text.count(']')
	if amt_of_open_brackets == 0 and amt_of_closed_brackets == 0:
		return bracketed_text

	bracket_stack = []
	total_times = []
	to_replace = []
	read_in = False
	tmp_txt = ''
	tmp_open = None

	if amt_of_open_brackets == amt_of_closed_brackets:
		for c in range(0, len(bracketed_text)):
			if len(bracket_stack) == 0:
				read_in = False
				tmp_txt = ''
			else:
				read_in = True

			if bracketed_text[c] == '[' and tmp_open is None:
				bracket_stack.append('[')
				tmp_open = c
			elif bracketed_text[c] == '[' and tmp_open is not None:
				bracket_stack.append('[')
			elif bracketed_text[c] == ']':
				try:
					bracket_stack.pop()
					if len(bracket_stack) == 0:
						total_times.append(tmp_txt.strip())
						to_replace.append((tmp_open, c))
						tmp_open = None
				except IndexError:
					# print("Mismatched brackets")
					break
			elif read_in:
				tmp_txt += bracketed_text[c]

		for i in range(0, len(total_times)):
			total_times[i] = total_times[i].split('+')[-1]

		tmp = ''
		offset = 0
		for j in range(0, len(to_replace)):
			start = to_replace[j][0] - offset
			end = to_replace[j][1] - offset

			first = bracketed_text[0:start]
			replacement = total_times[j]
			last = bracketed_text[end+1:]

			tmp = first + replacement + last
			offset += abs(len(tmp) - len(bracketed_text))
			bracketed_text = first + replacement + last

	return bracketed_text


def clean_text(text_to_clean):
	under = text_to_clean.lower()               # Make all lowercase to normalize data

	result, occurrences = remove_arrowheads(under)           # Remove features such as laughter or taking_breathe
	result = remove_curlies(result)
	result = remove_double_parentheses(result)
	result = remove_hashtags(result)
	result = remove_brackets(result)
	result = remove_EoU(result)
	result = remove_comments(result)
	result = remove_init_dashes(result)
	result = remove_multi_space(result)

	return result, occurrences


def get_data(text):
	cleaned_text, arrow_occurrences = clean_text(text)
	not_to_delete = any(c.isalpha() for c in cleaned_text)

	if not_to_delete:
		return cleaned_text, arrow_occurrences, False
	else:
		return cleaned_text, arrow_occurrences, True


def get_pos(pos_text):
	tokens = pos_text.split(' ')
	last_pos = None
	for t in reversed(tokens):
		try:
			p = t.split('/')[1]
		except IndexError:
			continue
		if not any(c.isalpha() for c in p):
			continue
		else:
			last_pos = p
			break

	return last_pos


def remove_rows(train=False):
	# -----------------------------------------
	# Remove unnecessary rows
	# -----------------------------------------
	print("Starting to remove unnecessary rows...")
	if train:
		inp = open('data/train_merged.csv', 'r')
		output = open('data/edited_train_merged.csv', 'w')
	else:
		inp = open('data/test_merged.csv', 'r')
		output = open('data/edited_test_merged.csv', 'w')

	top_ten_da = ['sd', 'b', 'sv', 'aa', 'ba', 'qy', 'x', 'ny', 'fc', 'qw']

	read = csv.reader(inp, delimiter=',')
	write = csv.writer(output, delimiter=',', lineterminator='\n')

	write.writerow(next(read))
	for row in read:
		if row[4] in top_ten_da:
			write.writerow(row)
	inp.close()
	output.close()

	print("Finished removing unnecessary rows")


def remove_most_rows(train=False):
	# -----------------------------------------
	# Remove unnecessary rows
	# -----------------------------------------
	print("Starting to remove unnecessary rows...")
	if train:
		inp = open('data/train_merged.csv', 'r')
		output = open('data/small_train_merged.csv', 'w')
	else:
		inp = open('data/test_merged.csv', 'r')
		output = open('data/small_test_merged.csv', 'w')

	top_ten_da = ['sd', 'b', 'sv', 'aa', 'ba', 'qy', 'x', 'ny', 'fc', 'qw']

	read = csv.reader(inp, delimiter=',')
	write = csv.writer(output, delimiter=',', lineterminator='\n')

	write.writerow(next(read))
	count = 1
	for row in read:
		if count <= 1000:
			break
		if row[4] in top_ten_da:
			write.writerow(row)
		count += 1
	inp.close()
	output.close()

	print("Finished removing unnecessary rows")


def create_final_csv(train=False):
	# -----------------------------------------
	# Create cleaned data set
	# -----------------------------------------
	print("Starting to clean data...")
	if train:
		csvinput = open('data/edited_train_merged.csv', 'r')
		csvoutput = open('data/final_train_merged.csv', 'w')
	else:
		csvinput = open('data/edited_test_merged.csv', 'r')
		csvoutput = open('data/final_test_merged.csv', 'w')

	reader = csv.reader(csvinput, delimiter=',')
	writer = csv.writer(csvoutput, lineterminator='\n', delimiter=',')

	row = next(reader)
	row.append("Cleaned")
	row.append("Misc_Feature_Occurrences")
	row.append("POS_tags")
	writer.writerow(row)

	count = 1
	for row in reader:
		cleaned, arrowed_occurrences, delete_status = get_data(row[8])
		pos = get_pos(row[9])
		if pos is None:
			pos = 'none'

		row.append(cleaned)
		row.append(arrowed_occurrences)
		row.append(pos)
		writer.writerow(row)

		sys.stdout.write("\rFinished cleaning row %i" % count)
		sys.stdout.flush()
		count += 1

	print("\nFinished reading file")


def extract_ugrams(file_to_open):
	csvinput = open(file_to_open, 'r')
	reader = csv.reader(csvinput, delimiter=',')

	next(reader)
	count = 1
	ugrams_1 = []
	# ugrams_2 = []
	# ugrams_3 = []
	for row in reader:
		cleaned, ar = clean_text(row[8])
		ugram = new_ngrams(cleaned)

		if count <= 50000:
			ugrams_1.append(ugram)
		elif count <= 100000:
			break
			# ugrams_2.append(ugram)
		# elif count <= 150000:
		# 	ugrams_3.append(ugram)

		sys.stdout.write("\rFinished reading row %i" % count)
		sys.stdout.flush()
		count += 1
	print("\nFinished reading file")

	print("Saving ugrams with np.savez_compressed...")
	if 'train' in file_to_open:
		# np.savez_compressed('data/train_npz_ugrams_compressed.npz', ugrams_1=ugrams_1, ugrams_2=ugrams_2, ugrams_3=ugrams_3)
		np.savez_compressed('data/train_npz_ugrams_compressed.npz', ugrams_1=ugrams_1)
	else:
		# np.savez_compressed('data/test_npz_ugrams_compressed.npz', ugrams_1=ugrams_1, ugrams_2=ugrams_2, ugrams_3=ugrams_3)
		np.savez_compressed('data/test_npz_ugrams_compressed.npz', ugrams_1=ugrams_1)
	print("Finished saving ugrams with np.savez_compressed")


# def extract_bgrams(file_to_open):
# 	csvinput = open(file_to_open, 'r')
# 	reader = csv.reader(csvinput, delimiter=',')
#
# 	next(reader)
# 	count = 1
# 	bgrams_1 = []
# 	bgrams_2 = []
# 	bgrams_3 = []
# 	for row in reader:
# 		cleaned, ar = clean_text(row[8])
# 		ugram, bgram = new_ngrams(cleaned)
#
# 		if count <= 50000:
# 			bgrams_1.append(bgram)
# 		elif count <= 100000:
# 			bgrams_2.append(bgram)
# 		elif count <= 150000:
# 			bgrams_3.append(bgram)
#
# 		sys.stdout.write("\rFinished extracting bigram on row %i" % count)
# 		sys.stdout.flush()
# 		count += 1
# 	print("\nFinished reading file")
#
# 	print("Saving bgrams with np.savez_compressed...")
# 	if 'train' in file_to_open:
# 		np.savez_compressed('data/train_npz_bgrams_compressed.npz', bgrams_1=bgrams_1, bgrams_2=bgrams_2, bgrams_3=bgrams_3)
# 	else:
# 		np.savez_compressed('data/test_npz_bgrams_compressed.npz', bgrams_1=bgrams_1, bgrams_2=bgrams_2, bgrams_3=bgrams_3)


def get_corpus(train=True):
	# -----------------------------------------
	# Create corpus
	# -----------------------------------------
	print("Starting to create corpus...")
	if train:
		csvinput = open('data/edited_train_merged.csv', 'r')
		reader = csv.reader(csvinput, delimiter=',')
	else:
		csvinput = open('data/edited_test_merged.csv', 'r')
		reader = csv.reader(csvinput, delimiter=',')

	row = next(reader)

	c = []
	for row in reader:
		cleaned, arrowed_occurrences = clean_text(row[8])
		row.append(cleaned)
		c.append(cleaned)

	csvinput.close()
	print("Created corpus\n")
	return c


if __name__ == "__main__":
	# corpus = get_corpus(train=True)

	# -----------------------------------------
	# Create final csv files
	# -----------------------------------------
	print("Creating final csv ...")
	create_final_csv(train=False)
	print("Created final csv")

	# -----------------------------------------
	# Create n-gram vectorizers
	# -----------------------------------------
	# print("Creating unigram_vectorizer...")
	# new_unigram_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
	# new_unigram_vectorizer.fit(corpus)
	# print("Created unigram_vectorizer\n")

	# print("Creating bigram_vectorizer...")
	# new_bigram_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
	# new_bigram_vectorizer.fit(corpus)
	# print("Created bigram_vectorizer\n")

	# -----------------------------------------
	# Extract n-grams
	# -----------------------------------------
	# dim_uni = len(new_unigram_vectorizer.vocabulary_)  # Getting dimensions for checking later
	# dim_bi = len(new_bigram_vectorizer.vocabulary_)  # Getting dimensions for checking later

	train_file = 'data/edited_train_merged.csv'
	test_file = 'data/edited_test_merged.csv'
	# extract_ugrams(train_file)
	# extract_ugrams(test_file)
	# print("Finished extracting n-grams")
