// EigenHot.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "EigenHot.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// The one and only application object

CWinApp theApp;

using namespace std;

int _tmain(int argc, TCHAR* argv[], TCHAR* envp[])
{
	int nRetCode = 0;

	HMODULE hModule = ::GetModuleHandle(NULL);

	if (hModule != NULL)
	{
		// initialize MFC and print and error on failure
		if (!AfxWinInit(hModule, NULL, ::GetCommandLine(), 0))
		{
			// TODO: change error code to suit your needs
			_tprintf(_T("Fatal Error: MFC initialization failed\n"));
			nRetCode = 1;
		}
		else
		{
			// TODO: code your application's behavior here.
			try {
				// Number of clusters for building BOW vocabulary from SURF features
				categorizer c(argv[1]);
				if(atoi(argv[2]) == 0) {
					c.train_classifiers();
					c.save_vocab();
				} else {
					cout << "loading vocab" << endl;
					c.load_vocab();
					cout << "vocab loaded" << endl;
				}

				c.categorize();
			} catch(cv::Exception &e) {
				printf("Error: %s\n", e.what());
			}
			cin.get();
		}
	}
	else
	{
		// TODO: change error code to suit your needs
		_tprintf(_T("Fatal Error: GetModuleHandle failed\n"));
		nRetCode = 1;
	}

	return nRetCode;
}

inline static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	// Create and return normalized image:
	Mat dst;
	switch(src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

void PrintHist(HistInfo &histInfo, vector<Mat> &hist) {
	for(int j = 0; j < hist.size(); j++) {
		printf("%d: ",j);
		for(int i = 0; i < histInfo.bins; i++) {
			printf("%f, ", hist[j].at<float>(i));
		}
		printf("\n");
	}
}

void DisplayHist(HistInfo &histInfo, vector<Mat> &hist) {
	//for displaying the histogram
	double maxVal=0;
	for(int i = 0; i < hist.size(); i++) {
		double tmp;
		minMaxLoc(hist[i], 0, &tmp, 0, 0);
		if(tmp > maxVal)
			maxVal = tmp;
	}
	for(int i = 0; i < hist.size(); i++) {
		Mat histImg = Mat::zeros( 500,  histInfo.bins*10, CV_8UC3);
		for(int h = 0; h < histInfo.bins; h++) {
			float binVal = hist[i].at<double>(0, h);
			int val = cvRound(binVal*500/maxVal);
			if(val != 0)
			rectangle( histImg, Point(h*10, 0),
				Point( (h+1)*10 - 1, val),
				Scalar::all(128),
				CV_FILLED );
		}
		char buff[50];
		sprintf(buff,"Hist %d",i);
		namedWindow(buff, 1);
		imshow(buff, histImg);
	}
	cvWaitKey();
}

inline string categorizer::remove_extension(string full) {
	int last_idx = full.find_last_of(".");
	string name = full.substr(0, last_idx);
	return name;
}

inline int get_label(string full) {
	int last_idx = full.find_last_of("_");
	string before = full.substr(last_idx - 1, last_idx);
	return atoi(before.c_str());
}

categorizer::categorizer(string direc) {
	//set up folders
	test_folder = direc + "test_images\\";
	train_folder = direc + "train_images\\";
	vocab_folder = direc;

	cout << "Initialized" << endl;

	// Organize training images by category
	make_train_set();
	// Initialize pointers to all the eigen stuff
	model = new EigenfacesOpen(category_names.size());
	//model = createFisherFaceRecognizer();
}

void categorizer::make_train_set() {
	string category;
	// Boost::filesystem recursive directory iterator to go through all contents of TRAIN_FOLDER
	for(recursive_directory_iterator i(train_folder), end_iter; i != end_iter; i++) {
		// Level 0 means a folder, since there are only folders in TRAIN_FOLDER at the zeroth level
		if(i.level() == 0) {
			// Get category name from name of the folder
			category = (i->path()).filename().string();
			category_names.push_back(category);
		}
		// Level 1 means a training image, map that by the current category
		else {
			// File name with path
			string filename = string(train_folder) + category + string("/") + (i->path()).filename().string();
			Mat img = imread(filename,0);
			pair<string, Mat> p(category, img);
			train_set.insert(p);
			images.push_back(img);
			labels.push_back(atoi(category.c_str()));
		}
	}
	// Number of categories
	categories = category_names.size();
	cout << "Discovered " << categories << " categories of objects" << endl;
}

void categorizer::make_pos_neg() {
	vector<Mat> *eigenvalues = model->getEigenValues();
	vector<Mat>::const_iterator pE = eigenvalues->begin();
	// Iterate through the whole training set of images
	for(multimap<string, Mat>::iterator i = train_set.begin(); i != train_set.end(); i++) {
		// Category name is the first element of each entry in train_set
		string category = i->first;
		// Training image is the second elemnt
		Mat im = i->second;
		//cvtColor(im,im_g,CV_BGR2GRAY);

		// Mats to hold the positive and negative training data for current category
		Mat tmp = pE->reshape(1,1);
		++pE;
		tmp.convertTo(tmp,CV_32F);
		for(int cat_index = 0; cat_index < categories; cat_index++) {
			string check_category = category_names[cat_index];
			// Add BOW feature as positive sample for current category ...
			if(check_category.compare(category) == 0)
				positive_data[check_category].push_back(tmp);
			//... and negative sample for all other categories
			else
				negative_data[check_category].push_back(tmp);
		}
	}

	// Debug message
	for(int i = 0; i < categories; i++) {
		string category = category_names[i];
		cout << "Category " << category << ": " << positive_data[category].rows << " Positives, " << negative_data[category].rows << " Negatives" << endl;
	}
}

void categorizer::save_vocab() {
	/*FileStorage fs(vocab_folder + "vocab.xml", FileStorage::WRITE);
	model->save(fs);
	fs.release();*/
	model->saveBinary(vocab_folder + "vocab.bin");
}

void categorizer::load_vocab() {
	/*FileStorage fs(vocab_folder + "vocab.xml", FileStorage::READ);
	model->load(fs);
	fs.release();*/
	model->loadBinary(vocab_folder + "vocab.bin");

	//load the classifiers
	/*for(int i = 0; i < categories; i++) {
	string category = category_names[i];
	string svm_filename = string(vocab_folder) + category + string("SVM.xml");
	svm[category].load(svm_filename.c_str());
	}*/
}

void categorizer::train_classifiers() {

	model->train(images, labels);
	make_pos_neg();
	for(int i = 0; i < categories; i++) {
		string category = category_names[i];

		// Postive training data has labels 1
		Mat train_data = positive_data[category], train_labels = Mat::ones(train_data.rows, 1, CV_32S);
		// Negative training data has labels 0
		train_data.push_back(negative_data[category]);
		Mat m = Mat::zeros(negative_data[category].rows, 1, CV_32S);
		train_labels.push_back(m);

		// Train SVM!
		//svm[category].train(train_data, train_labels);

		//// Save SVM to file for possible reuse
		//string svm_filename = string(vocab_folder) + category + string("SVM.xml");
		//svm[category].save(svm_filename.c_str());

		//cout << "Trained and saved SVM for category " << category << endl;
	}
}

void categorizer::categorize(VideoCapture cap) {
}

inline void DisplayStuff(Mat mean, Mat eigenvalues, Mat W, int height, string folder) {
	imshow("mean", norm_0_255(mean.reshape(1, height)));
	imwrite(folder + "mean.png", norm_0_255(mean.reshape(1, height)));
	// Display or save the Eigenfaces:
	for (int i = 0; i < min(20, W.cols); i++) {
		string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
		cout << msg << endl;
		// get eigenvector #i
		Mat ev = W.col(i).clone();
		// Reshape to original size & normalize to [0...255] for imshow.
		Mat grayscale = norm_0_255(ev.reshape(1, height));
		// Show the image & apply a Jet colormap for better sensing.
		Mat cgrayscale;
		applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
		// Display 

		/*char img_file[400];
		sprintf(img_file, "%seigen_face_%d.png", folder.c_str(), i);
		imwrite(img_file,cgrayscale);*/
		imshow(format("eigenface_%d", i), cgrayscale);
	}
}

inline void SaveStats(Mat &mean_weights, Mat &std_weights, string folder) {
	/*FileStorage fs(folder + "stats.xml", FileStorage::WRITE);
	fs << "mean" << mean_weights;
	fs << "std" << std_weights;
	fs.release();*/
	FILE *fp;
	fopen_s(&fp, string(folder + "mean.txt").c_str(), "w");
	if(fp == NULL)
		throw std::exception("Couldn't open mean.txt");
	double *w = (double*)mean_weights.data;
	int x, y;
	for(x = 0; x < mean_weights.rows; x++) {
		for(y = 0; y < mean_weights.cols; y++) {
			fprintf_s(fp,"%e,",*w);
			++w;
		}
		fprintf_s(fp,"\n");
	}
	fclose(fp);
	fopen_s(&fp, string(folder + "std.txt").c_str(), "w");
	if(fp == NULL)
		throw std::exception("Couldn't open std.txt");
	w = (double*)std_weights.data;
	for(x = 0; x < std_weights.rows; x++) {
		for(y = 0; y < std_weights.cols; y++) {
			fprintf_s(fp,"%e,",*w);
			++w;
		}
		fprintf_s(fp,"\n");
	}
	fclose(fp);
}

void categorizer::predict(FILE *fp, Mat &input, Mat &mean, Mat &W, vector<Mat> &projections) {
	// project into PCA subspace
	Mat q = subspaceProject(W, mean, input.reshape(1,1));
	//double minDist = DBL_MAX;
	//double minClass = -1;
	for(size_t sampleIdx = 0; sampleIdx < projections.size(); sampleIdx++) {
		double dist = norm(projections[sampleIdx], q, NORM_L2);
		fprintf_s(fp,"%e,",dist);
	}
	fprintf_s(fp,"\n");
}

inline void categorizer::predictSVM(Mat &input, int &predictedLabel) {
	float best_score = 777;
	for(int i = 0; i < categories; i++) {
		string category = category_names[i];
		Mat eigenvalues;
		model->CalcWeights(input,eigenvalues);
		Mat tmp = eigenvalues.reshape(1,1);
		tmp.convertTo(tmp,CV_32F);
		float prediction = svm[category].predict(tmp, true);
		//cout << category << " " << prediction << " ";
		if(prediction < best_score) {
			best_score = prediction;
			predictedLabel = atoi(category.c_str());
		}
	}
}

void GetMinAndMax(vector<Mat> &projections, double &min, double &max, double &mean) {
	mean = 0;
	min = max = projections[0].at<double>(0);
	vector<Mat>::const_iterator p = projections.begin();
	int size = projections.size();
	while(p != projections.end()) {
		double *data = (double *)p->data, *end = data + size;
		while(data != end) {
			if(*data < min)
				min = *data;
			else if(*data > max)
				max = *data;
			mean += *data;
			++data;
		}
		++p;
	}
	mean /= (size * size);
}

void categorizer::categorize() {
	int height = images[0].rows;
	int num_labels = category_names.size();
	Mat confusion = Mat::zeros(num_labels, num_labels, CV_32S);

	/*vector<Mat> hist;
	model->GetHistograms(histInfo,hist);
	PrintHist(histInfo,hist);
	DisplayHist(histInfo,hist);*/

	/*CalculateMean(projections,model_labels,label_size,mean_weights);
	CalculateStd(projections,model_labels,label_size,mean_weights,std_weights);
	SaveStats(mean_weights, std_weights, vocab_folder);*/

	//DisplayStuff(model->_mean, model->_eigenvalues, model->_eigenvectors, height, vocab_folder);

	//imshow("frame", frame);
	//waitKey();

	/*FILE *fp;
	fopen_s(&fp, string(vocab_folder + "weights.txt").c_str(), "w");
	if(fp == NULL)
	throw std::exception("Couldn't open weights.txt");*/
	int count = 0;
	int old_label = 0;
	for(directory_iterator i(test_folder), end_iter; i != end_iter; i++) {
		Mat frame;
		// Prepend full path to the file name so we can imread() it
		string filename = string(test_folder) + i->path().filename().string();
		//cout << "Opening file: " << filename << endl;
		frame = imread(filename,0);
		EigenDecisionTree decision, decision2, decision3;
		int predictedLabel = -1;
		double distance = 0.0;
		//model->predict(frame, predictedLabel, distance);
		//model->predictKNN(frame,predictedLabel,5);
		/*if(predictedLabel < 3)
			predictedLabel = 0;
		else if(predictedLabel >= 3)
			predictedLabel = 1;*/
		/*if(predictedLabel == 2)
			predictedLabel = 1;
		else if(predictedLabel == 1) {
			predictedLabel = 3;
		}*/
		model->predictKNN(frame,decision,4);
		model->predictDistofMean(frame,decision2);
		model->predictMeanofDist(frame,decision3);
		//decision *= decision2;
		//decision *= decision3;
		int l1 = decision.GetBottomVote();
		int l2 = decision2.GetBottomVote();
		int l3 = decision3.GetBottomVote();
		if(l2 == l3)
			predictedLabel = l2;
		else
			predictedLabel = l1;
		//predictedLabel = decision.GetBottomVote();
		//predictSVM(frame,predictedLabel);
		/*model->predictHist(frame,histInfo,hist,decision);
		predictedLabel = decision.GetBottomVote();*/

		//cout << "Predicted as: " << predictedLabel << " with confidence: " << confidence << endl;
		int label = get_label(filename);
		confusion.at<int>(label,predictedLabel)++;
	}
	//fclose(fp);
	//imshow("confusion", confusion);
	for(int y = 0; y < confusion.rows; y++) {
		for(int x = 0; x < confusion.cols; x++) {
			cout << confusion.at<int>(y,x) << ", ";
		}
		cout << endl;
	}
	//waitKey();
}

 /* void categorizer::categorize() {
	int height = images[0].rows;
	int num_labels = category_names.size();
	vector<Mat> confusionVector;
	confusionVector.resize(NUM_K);
	int k;
	for(k = 0; k < NUM_K; k++)
		confusionVector[k] = Mat::zeros(num_labels, num_labels, CV_32S);

	int count = 0;
	int old_label = 0;
	double max = 0;
	int label = 0;
	for(directory_iterator i(test_folder), end_iter; i != end_iter; i++) {
		Mat frame;
		// Prepend full path to the file name so we can imread() it
		string filename = string(test_folder) + i->path().filename().string();
		//cout << "Opening file: " << filename << endl;
		frame = imread(filename,0);
		EigenDecisionTree decision, decision2;
		int predictedLabel = -1;
		double distance = 0.0;
		int label = get_label(filename);
		for(k = 1; k < NUM_K; k++) {
			model->predictKNN(frame,decision,k);
			predictedLabel = decision.GetBottomVote();
			confusionVector[k].at<int>(label,predictedLabel)++;
		}
	}
	for(k = 1; k < NUM_K; k++) {
		double stuff = 0;
		for(int y = 0; y < confusionVector[k].rows; y++) {
			stuff += confusionVector[k].at<int>(y,y);
		}
		if(stuff > max) {
			max = stuff;
			label = k;
		}
		cout << stuff << endl;
	}
	//fclose(fp);
	//imshow("confusion", confusion);
	cout << label << endl;
	//waitKey();
} */