# Emotion Recognition - Advanced Spoken Language Processing - HW 4
## Author: Ismael Villegas Molina - iv2181

### feature_extraction.py
This is the file that will generate all the files needed for classification as well as plots to visualize the emotional 
features. Must have the wav files in directory called `.data/` in the same directory. You get these with the following:
* Getting Plots
	* First run `build_features()` in order to read the data files from `./data/` and create a master csv file with all 
	the features. The normalized features are also labeled at the last columns. Normalization was done with individual 
	speaker z-scores.
	* Run `get_all_plots()` which will generate a plot for mean, min, and max pitch as well as mean, min, and max 
	intensity. It will first generate the plot for the regular values, then it will generate the normalized plot. You 
	can then save these files with the save function that is provided by matplotlib.
* Getting classification files
	* You first need to make sure that you have the latest version of [openSmile](https://www.audeering.com/opensmile/).
	Without this, you will not get the features needed for classification.
	* After getting openSmile, all you need to run is `get_opensmile_csv()` which will generate all the necessary files 
	for classification. These files are posted in two different places:
		* `./extracted_data/all_except_x/` will hold all the csv files of every speaker except speaker 'x'. This will be 
		necessary for the all-vs.-one technique used for classification as the training data.
		* `./extracted_data/remaining/` will hold all the csv files of individual speakers by themselves which is used 
		as test data for all-vs.-one technique.

### classification.py
This is the file that will use the data generated from `feature_extraction.py` and try to recognize emotions with a 
one-vs-Rest Classifier. You can run `test_all()` to get all reports, but you can run individual tests for each speaker 
as test data. The "best" classifier is usually changing but the classifiers are hovering around 17% accuracy most times.