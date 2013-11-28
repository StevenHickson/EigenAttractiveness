#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <boost/filesystem.hpp>
#include <set>
#include <limits>
#include <opencv2/core/internal.hpp>
#include <opencv2/contrib/contrib.hpp>
#include "DecisionTree.h"
//#include "precomp.hpp"

using namespace cv;

class EigenfacesOpen : public FaceRecognizer
{
private:

public:
	int _num_unique_labels;
    int _num_components;
    double _threshold;
    std::vector<Mat> _projections;
    Mat _labels;
    Mat _eigenvectors;
    Mat _eigenvalues;
    Mat _mean;


    using FaceRecognizer::save;
    using FaceRecognizer::load;

    // Initializes an empty Eigenfaces model.
    EigenfacesOpen(int num_unique_labels, int num_components = 0, double threshold = DBL_MAX) :
        _num_unique_labels(num_unique_labels),
		_num_components(num_components),
        _threshold(threshold) {}

    // Initializes and computes an Eigenfaces model with images in src and
    // corresponding labels in labels. num_components will be kept for
    // classification.
    EigenfacesOpen(InputArrayOfArrays src, InputArray labels,
            int num_components = 0, double threshold = DBL_MAX) :
        _num_components(num_components),
        _threshold(threshold) {
        train(src, labels);
    }

    // Computes an Eigenfaces model with images in src and corresponding labels
    // in labels.
    void train(InputArrayOfArrays src, InputArray labels);

    // See FaceRecognizer::load.
    void load(const FileStorage& fs);
	void loadBinary(string filename);
    // See FaceRecognizer::save.
    void save(FileStorage& fs) const;
	void saveBinary(string filename) const;

    //AlgorithmInfo* info() const;

	//my new stuff
	void CalcWeights(InputArray _src, Mat &weights) const;
	inline void CalculateMean(int *label_size, Mat &mean_weights);
	inline void CalculateStd(int *label_size, Mat &mean_weights, Mat &std_weights);
	std::vector<Mat>* getEigenValues() {
		return &_projections;
	}

	// Predicts the label and confidence for a given sample.
    int predict(InputArray src) const;
    void predict(InputArray _src, int &label, double &dist) const;
	void predictKNN(Mat &input, int &predictedLabel, int num_neighbors);
	void predictMeanofDist(Mat &input, EigenDecisionTree &prediction);
	void predictDistofMean(Mat &input, EigenDecisionTree &prediction);
};

//Ptr<FaceRecognizer> createEigenFaceOpenRecognizer(int num_unique_labels, int num_components, double threshold)
//{
//    return new EigenfacesOpen(num_unique_labels, num_components, threshold);
//}
//
//Ptr<FaceRecognizer> createEigenFaceOpenRecognizer(int num_unique_labels)
//{
//    return new EigenfacesOpen(num_unique_labels, 0, DBL_MAX);
//}