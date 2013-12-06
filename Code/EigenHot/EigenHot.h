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

Mat VisualizeHoG(Mat& origImg, vector<float>& descriptorValues);
int AlignImage(Mat &in, Mat &out);
int FindFace(Mat &in, Mat &out);

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
	Ptr<HOGDescriptor> hogDescriptor;

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
	void categorize(Mat &in);
	inline void predictSVM(Mat &input, int &predictedLabel);
	void trainskin(vector<int> &skin);
	void trainhog(vector<vector<float>> &descriptors);
	void trainhog2(vector<vector<float>> &descriptors);
	int testhogknn(Mat &img, vector<vector<float>> &descriptors, int num_neighbors);
};

#define NUM_K 100
#define TRAIN_DB "C:/Users/Steve/Documents/Data/EigenHot/people.train"
#define FACE_XML "C:/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
#define EYE_XML "C:/opencv/data/haarcascades/haarcascade_eye.xml"