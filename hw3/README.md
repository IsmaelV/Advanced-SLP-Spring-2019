# Dialogue Act Recognition - Advanced Spoken Language Processing - HW 3
## Author: Ismael Villegas Molina - iv2181

### feature_extraction.py
This file is simple. It cleans the csv data and creates a new csv file with all the original data along with the cleaned
data, unigrams, bigrams, and counter for <<>> features. I am attaching the updated csv files onto the assignments, so 
it's not necessary to run this program, but if you do decide to run it, all you have to do is click run and it will 
extract all the necessary information. This also creates a npz file with all the necessary n-grams (also attached). This
is because without this compressed version of the n-grams, the files would be much too large. WARNING: The program takes
a LONG time to run when extracting the n-grams because of the sheer amount of data. Proceed with caution!

### classification.py
This file just needs to be run and it will calculate all the different models. This file also takes a long time to 
execute because of the unpacking of np arrays gotten from `feature_extraction`, so if you would want to run it again, it
will take a while. If you want a faster version (with much less information) you could run the program but with the 
ngram evaluation commented out (in the main function). This will give you the evaluation of misc_features and LIWC, but 
not on any other portion.
