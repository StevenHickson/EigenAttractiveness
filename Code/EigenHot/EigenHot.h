#pragma once

#include "resource.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <boost/filesystem.hpp>
#include "Eigenfaces.h"

using namespace cv;
using namespace std;
using namespace boost::filesystem;

class categorizer {
private:
	map<string, Mat> objects, positive_data, negative_data; //maps from category names to data
	//map<string, vector<Mat>> hists;
	multimap<string, Mat> train_set; //training images, mapped by category name
	vector<Mat> images;
	vector<int> labels;
	map<string, CvSVM> svm; //trained SVMs, mapped by category name
	vector<string> category_names; //names of the categories found in TRAIN_FOLDER
	int categories; //number of categories
	Mat vocab; //vocabulary
	string train_folder, test_folder, vocab_folder;

	EigenfacesOpen *model;
	HistInfo histInfo;

	void make_train_set(); //function to build the training set multimap
	void make_pos_neg(); //function to extract BOW features from training images and organize them into positive and negative samples 
	string remove_extension(string); //function to remove extension from file name, used for organizing templates into categories
	void predict(FILE *fp, Mat &input, Mat &mean, Mat &W, vector<Mat> &projections);
public:
	categorizer(string direc); //constructor
	void save_vocab(); //function to save the classifiers
	void load_vocab(); //function to load the classifiers
	void train_classifiers(); //function to train the one-vs-all SVM classifiers for all categories
	void categorize(VideoCapture); //function to perform real-time object categorization on camera frames
	void categorize(); //function to perform real-time object categorization on saved frames
	inline void predictSVM(Mat &input, int &predictedLabel);
};

#define NUM_K 100