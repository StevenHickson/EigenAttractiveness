
#include "stdafx.h"
#include "Eigenfaces.h"

using namespace cv;

// Reads a sequence from a FileNode::SEQ with type _Tp into a result vector.
template<typename _Tp>
inline void readFileNodeList(const FileNode& fn, std::vector<_Tp>& result) {
	if (fn.type() == FileNode::SEQ) {
		for (FileNodeIterator it = fn.begin(); it != fn.end();) {
			_Tp item;
			it >> item;
			result.push_back(item);
		}
	}
}

// Writes the a list of given items to a cv::FileStorage.
template<typename _Tp>
inline void writeFileNodeList(FileStorage& fs, const String& name,
							  const std::vector<_Tp>& items) {
								  // typedefs
								  typedef typename std::vector<_Tp>::const_iterator constVecIterator;
								  // write the elements in item to fs
								  fs << name << "[";
								  for (constVecIterator it = items.begin(); it != items.end(); ++it) {
									  fs << *it;
								  }
								  fs << "]";
}

static Mat asRowMatrix(InputArrayOfArrays src, int rtype, double alpha=1, double beta=0) {
	// make sure the input data is a vector of matrices or vector of vector
	if(src.kind() != _InputArray::STD_VECTOR_MAT && src.kind() != _InputArray::STD_VECTOR_VECTOR) {
		String error_message = "The data is expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< std::vector<...> >).";
		throw std::exception(error_message.c_str());
	}
	// number of samples
	size_t n = src.total();
	// return empty matrix if no matrices given
	if(n == 0)
		return Mat();
	// dimensionality of (reshaped) samples
	size_t d = src.getMat(0).total();
	// create data matrix
	Mat data((int)n, (int)d, rtype);
	// now copy data
	for(unsigned int i = 0; i < n; i++) {
		// make sure data can be reshaped, throw exception if not!
		if(src.getMat(i).total() != d) {
			String error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src.getMat(i).total());
			throw std::exception(error_message.c_str());
		}
		// get a hold of the current row
		Mat xi = data.row(i);
		// make reshape happy by cloning for non-continuous matrices
		if(src.getMat(i).isContinuous()) {
			src.getMat(i).reshape(1, 1).convertTo(xi, rtype, alpha, beta);
		} else {
			src.getMat(i).clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
		}
	}
	return data;
}


// Removes duplicate elements in a given vector.
template<typename _Tp>
inline std::vector<_Tp> remove_dups(const std::vector<_Tp>& src) {
	typedef typename std::set<_Tp>::const_iterator constSetIterator;
	typedef typename std::vector<_Tp>::const_iterator constVecIterator;
	std::set<_Tp> set_elems;
	for (constVecIterator it = src.begin(); it != src.end(); ++it)
		set_elems.insert(*it);
	std::vector<_Tp> elems;
	for (constSetIterator it = set_elems.begin(); it != set_elems.end(); ++it)
		elems.push_back(*it);
	return elems;
}

void EigenfacesOpen::train(InputArrayOfArrays _src, InputArray _local_labels) {
	if(_src.total() == 0) {
		String error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
		throw std::exception(error_message.c_str());
	} else if(_local_labels.getMat().type() != CV_32SC1) {
		String error_message = format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _local_labels.type());
		throw std::exception(error_message.c_str());
	}
	// make sure data has correct size
	if(_src.total() > 1) {
		for(int i = 1; i < static_cast<int>(_src.total()); i++) {
			if(_src.getMat(i-1).total() != _src.getMat(i).total()) {
				String error_message = format("In the Eigenfaces method all input samples (training images) must be of equal size! Expected %d pixels, but was %d pixels.", _src.getMat(i-1).total(), _src.getMat(i).total());
				throw std::exception(error_message.c_str());
			}
		}
	}
	// get labels
	Mat labels = _local_labels.getMat();
	// observations in row
	Mat data = asRowMatrix(_src, CV_64FC1);

	// number of samples
	int n = data.rows;
	// assert there are as much samples as labels
	if(static_cast<int>(labels.total()) != n) {
		String error_message = format("The number of samples (src) must equal the number of labels (labels)! len(src)=%d, len(labels)=%d.", n, labels.total());
		throw std::exception(error_message.c_str());
	}
	// clear existing model data
	_labels.release();
	_projections.clear();
	// clip number of components to be valid
	if((_num_components <= 0) || (_num_components > n))
		_num_components = n;

	// perform the PCA
	PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, _num_components);
	// copy the PCA results
	_mean = pca.mean.reshape(1,1); // store the mean vector
	_eigenvalues = pca.eigenvalues.clone(); // eigenvalues by row
	transpose(pca.eigenvectors, _eigenvectors); // eigenvectors by column
	// store labels for prediction
	_labels = labels.clone();
	// save projections
	for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
		Mat p = subspaceProject(_eigenvectors, _mean, data.row(sampleIdx));
		_projections.push_back(p);
	}
}

void EigenfacesOpen::CalcWeights(InputArray _src, Mat &weights) const {
	// get data
	Mat src = _src.getMat();
	// make sure the user is passing correct data
	if(_projections.empty()) {
		// throw error if no data (or simply return -1?)
		String error_message = "This Eigenfaces model is not computed yet. Did you call Eigenfaces::train?";
		throw std::exception(error_message.c_str());
	} else if(_eigenvectors.rows != static_cast<int>(src.total())) {
		// check data alignment just for clearer exception messages
		String error_message = format("Wrong input image size. Reason: Training and Test images must be of equal size! Expected an image with %d elements, but got %d.", _eigenvectors.rows, src.total());
		throw std::exception(error_message.c_str());
	}
	// project into PCA subspace
	weights = subspaceProject(_eigenvectors, _mean, src.reshape(1,1));
}

int EigenfacesOpen::predict(InputArray _src) const {
	int label;
	double dummy;
	predict(_src, label, dummy);
	return label;
}

void EigenfacesOpen::load(const FileStorage& fs) {
	//read matrices
	fs["num_components"] >> _num_components;
	fs["mean"] >> _mean;
	fs["eigenvalues"] >> _eigenvalues;
	fs["eigenvectors"] >> _eigenvectors;
	// read sequences
	readFileNodeList(fs["projections"], _projections);
	fs["labels"] >> _labels;
}

void EigenfacesOpen::loadBinary(string filename) {
	FILE *fp;
	fp = fopen(filename.c_str(),"rb");
	if(fp == NULL)
		throw std::exception("Couldn't open file for loading Eigen data");
	int num_eigen;
	fread(&_num_components,sizeof(int),1,fp);
	fread(&num_eigen,sizeof(int),1,fp);
	//hopefully this is enough
	//char *buffer = (char*) malloc(sizeof(CV_64F) * _num_components * num_eigen);
	_mean.create(1,num_eigen,CV_64F);
	_eigenvalues.create(_num_components,1,CV_64F);
	_eigenvectors.create(num_eigen,_num_components,CV_64F);
	_labels.create(1,_num_components,CV_32S);
	_projections.resize(_num_components);
	fread(_mean.data,sizeof(double),_mean.rows * _mean.cols,fp);
	fread(_eigenvalues.data,sizeof(double),_eigenvalues.rows * _eigenvalues.cols,fp);
	fread(_eigenvectors.data,sizeof(double),_eigenvectors.rows * _eigenvectors.cols,fp);
	vector<Mat>::iterator p =_projections.begin();
	while(p != _projections.end()) {
		p->create(1,_num_components,CV_64F);
		fread(p->data,sizeof(double),p->rows * p->cols,fp);
		++p;
	}
	fread(_labels.data,sizeof(int),_labels.rows * _labels.cols,fp);
	fclose(fp);
}

void EigenfacesOpen::save(FileStorage& fs) const {
	// write matrices
	fs << "num_components" << _num_components;
	fs << "mean" << _mean;
	fs << "eigenvalues" << _eigenvalues;
	fs << "eigenvectors" << _eigenvectors;
	// write sequences
	writeFileNodeList(fs, "projections", _projections);
	fs << "labels" << _labels;
}

void EigenfacesOpen::saveBinary(string filename) const {
	FILE *fp;
	fp = fopen(filename.c_str(),"wb");
	if(fp == NULL)
		throw std::exception("Couldn't open file for saving Eigen data");
	int num_eigen = _mean.cols;
	fwrite(&_num_components,sizeof(int),1,fp);
	fwrite(&num_eigen,sizeof(int),1,fp);
	fwrite(_mean.data,sizeof(double),_mean.rows * _mean.cols,fp);
	fwrite(_eigenvalues.data,sizeof(double),_eigenvalues.rows * _eigenvalues.cols,fp);
	fwrite(_eigenvectors.data,sizeof(double),_eigenvectors.rows * _eigenvectors.cols,fp);
	vector<Mat>::const_iterator p =_projections.begin();
	while(p != _projections.end()) {
		fwrite(p->data,sizeof(double),p->rows * p->cols,fp);
		++p;
	}
	fwrite(_labels.data,sizeof(int),_labels.rows * _labels.cols,fp);
	fclose(fp);
}

inline void EigenfacesOpen::CalculateMean(int *label_size, Mat &mean_weights) {
	double *w;
	vector<Mat>::iterator p = _projections.begin();
	int *c = (int*)_labels.data;
	while(p != _projections.end()) {
		w = (double*)p->data;
		int d = 0;
		label_size[*c]++;
		while(w != (double*)p->dataend) {
			mean_weights.at<double>(*c,d) += *w;
			++w; ++d;
		}
		++p; ++c;
	}
	w = (double*)mean_weights.data;
	int x, y, count;
	for(x = 0; x < mean_weights.rows; x++) {
		count = label_size[x];
		for(y = 0; y < mean_weights.cols; y++) {
			*w /= count;
			++w;
		}
	}
}

inline void EigenfacesOpen::CalculateStd(int *label_size, Mat &mean_weights, Mat &std_weights) {
	double *w;
	vector<Mat>::iterator p = _projections.begin();
	int *c = (int*)_labels.data;
	while(p != _projections.end()) {
		w = (double*)p->data;
		int d = 0;
		while(w != (double*)p->dataend) {
			double val = *w - mean_weights.at<double>(*c,d);
			std_weights.at<double>(*c,d) += val * val;
			++w; ++d;
		}
		++p; ++c;
	}
	w = (double*)std_weights.data;
	int x, y, count;
	for(x = 0; x < std_weights.rows; x++) {
		count = label_size[x];
		for(y = 0; y < std_weights.cols; y++) {
			*w = sqrt( *w / count);
			++w;
		}
	}
}

inline void CheckErrors(InputArray &src, vector<Mat> _projections, Mat _eigenvectors) {
	// make sure the user is passing correct data
	if(_projections.empty()) {
		// throw error if no data (or simply return -1?)
		String error_message = "This Eigenfaces model is not computed yet. Did you call Eigenfaces::train?";
		throw std::exception(error_message.c_str());
	} else if(_eigenvectors.rows != static_cast<int>(src.total())) {
		// check data alignment just for clearer exception messages
		String error_message = format("Wrong input image size. Reason: Training and Test images must be of equal size! Expected an image with %d elements, but got %d.", _eigenvectors.rows, src.total());
		throw std::exception(error_message.c_str());
	}
}

inline void CalcHist(const Mat &in, HistInfo &histInfo, Mat *hist) {
	Mat hist_base;
	hist_base = Mat::zeros(1,histInfo.bins,CV_32F);
	float *data = (float *)in.data, *end = data + in.cols;
	float mul = histInfo.bins / (histInfo.ranges[1] - histInfo.ranges[0]);
	while(data != end) {
		int loc = int((*data - histInfo.ranges[0]) * mul);
		if(loc < 0)
			loc = 0;
		if(loc >= histInfo.bins)
			loc = histInfo.bins - 1;
		hist_base.at<float>(0,loc)++;
		++data;
	}
	normalize( hist_base, *hist, 0, 1, NORM_MINMAX, -1, Mat() );
}

void EigenfacesOpen::GetHistograms(HistInfo &histInfo, vector<Mat> &hist) {
	vector<Mat>::const_iterator p = _projections.begin();
	hist.resize(_num_unique_labels);
	int *pL = (int *)_labels.data;
	int last_label = 0;
	for(int i = 0; i < _num_unique_labels; i++)
		hist[i] = Mat::zeros(1,histInfo.bins,CV_32F);
	while(p != _projections.end()) {
		Mat tmp;
		CalcHist(*p,histInfo,&tmp);
		hist[*pL] += tmp;
		++p; ++pL;
	}
	for(int i = 0; i < _num_unique_labels; i++)
		normalize( hist[i], hist[i], 0, 1, NORM_MINMAX, -1, Mat() );
}

inline double EigenNorm(const Mat &src1, const Mat &src2) {
	Mat in1 = Mat(1,20,src1.type());
	Mat in2 = Mat(1,20,src2.type());
	const int corr[20] = {4,8,9,10,11,19,37,48,51,63,66,77,78,86,142,152,189,292,778,1388};
	for(int i = 0; i < 20; i++) {
		in1.at<double>(0,i) = src1.at<double>(0,corr[i]);
		in2.at<double>(0,i) = src2.at<double>(0,corr[i]);
	}
	return norm(in1, in2, NORM_L2);
}

void EigenfacesOpen::predict(InputArray _src, int &minClass, double &minDist) const {
	// get data
	Mat src = _src.getMat();
	CheckErrors(src,_projections,_eigenvectors);
	// project into PCA subspace
	Mat q = subspaceProject(_eigenvectors, _mean, src.reshape(1,1));
	minDist = DBL_MAX;
	minClass = -1;
	for(size_t sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
		double dist = norm(_projections[sampleIdx], q, NORM_L2);
		if((dist < minDist) && (dist < _threshold)) {
			minDist = dist;
			minClass = _labels.at<int>((int)sampleIdx);
		}
	}
}

void EigenfacesOpen::predictKNN(Mat &input, int &predictedLabel, int num_neighbors) {
	CheckErrors(input,_projections,_eigenvectors);
	// project into PCA subspace
	vector<Vote> dist;
	dist.resize(_projections.size());
	Mat q = subspaceProject(_eigenvectors, _mean, input.reshape(1,1));
	vector<Vote>::iterator pD = dist.begin();
	vector<Mat>::const_iterator pP = _projections.begin();
	int *pL = (int *)_labels.data;
	for(size_t sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
		pD->dist = norm(*pP, q, NORM_L2);
		pD->label = *pL;
		++pD; ++pP; ++pL;
	}
	//sort the results
	std::sort(dist.begin(),dist.end());
	//count up the votes
	vector<int> votes;
	votes.resize(_num_unique_labels);
	int i;
	for(i = 0; i < num_neighbors; i++) {
		votes[dist[i].label]++;
	}
	int max = votes[0];
	predictedLabel = 0;
	for(i = 0; i < _num_unique_labels; i++) {
		if(votes[i] > max) {
			max = votes[i];
			predictedLabel = i;
		}
	}
}

void EigenfacesOpen::predictKNN(Mat &input, EigenDecisionTree &prediction, int num_neighbors) {
	CheckErrors(input,_projections,_eigenvectors);
	// project into PCA subspace
	vector<Vote> dist;
	dist.resize(_projections.size());
	Mat q = subspaceProject(_eigenvectors, _mean, input.reshape(1,1));
	vector<Vote>::iterator pD = dist.begin();
	vector<Mat>::const_iterator pP = _projections.begin();
	int *pL = (int *)_labels.data;
	_count_labels.resize(_num_unique_labels);
	int i;
	for(int k = 0; k < _num_unique_labels; k++)
		_count_labels[k] = 0;
	for(size_t sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
		pD->dist = norm(*pP, q, NORM_L2);
		pD->label = *pL;
		_count_labels[*pL]++;
		++pD; ++pP; ++pL;
	}
	//sort the results
	std::sort(dist.begin(),dist.end());
	//count up the votes
	vector<int> votes;
	votes.resize(_num_unique_labels);
	for(i = 0; i < num_neighbors; i++) {
		votes[dist[i].label]++;
	}
	//int max = votes[0];
	for(i = 0; i < _num_unique_labels; i++) {
		prediction.bottom[i].label = i;
		if(votes[i] == 0)
			prediction.bottom[i].dist = 2;
		else
			prediction.bottom[i].dist = 1.0f / votes[i];
		//prediction.bottom[i].dist *= double(_count_labels[i]) / double(_projections.size());
	}
	prediction.FillTiersBasedOnBottom();
}

void EigenfacesOpen::predictHist(Mat &input, HistInfo &histInfo, vector<Mat> &histComp, EigenDecisionTree &decision) {
	Mat q = subspaceProject(_eigenvectors, _mean, input.reshape(1,1));
	Mat hist = Mat::zeros(1,histInfo.bins,CV_32F);
	CalcHist(q,histInfo,&hist);
	int type = hist.type();
	for(int i = 0; i < histComp.size(); i++) {
		int type2 = histComp[i].type();
		decision.bottom[i].dist = compareHist(hist,histComp[i],CV_COMP_BHATTACHARYYA );
		decision.bottom[i].label = i;
	}
	decision.FillTiersBasedOnBottom();
}

void EigenfacesOpen::predictMeanofDist(Mat &input, EigenDecisionTree &prediction) {
	CheckErrors(input,_projections,_eigenvectors);
	// project into PCA subspace
	vector<Vote> dist;
	dist.resize(_num_unique_labels);
	Mat q = subspaceProject(_eigenvectors, _mean, input.reshape(1,1));
	vector<Vote>::iterator pD = dist.begin();
	vector<Mat>::const_iterator pP = _projections.begin();
	int *pL = (int *)_labels.data;
	int last_label = 0, count = 0;
	pD->dist = 0;
	pD->label = 0;
	while(pP != _projections.end()) {
		if(*pL != last_label) {
			pD->dist /= count;
			last_label = *pL;
			pD++;

			pD->dist = 0;
			pD->label = last_label;
			count = 0;
		}
		pD->dist += EigenNorm(*pP, q);
		++pP; ++pL; ++count;
	}
	pD->dist /= count;
	last_label = 4;
	for(int i = 0; i < NUM_CLASSES; i++) {
		prediction.bottom[i] = dist[i];
	}
	prediction.FillTiersBasedOnBottom();
}

void EigenfacesOpen::predictDistofMean(Mat &input, EigenDecisionTree &prediction) {
	CheckErrors(input,_projections,_eigenvectors);
	// project into PCA subspace
	vector<Vote> dist;
	dist.resize(_num_unique_labels);
	Mat q = subspaceProject(_eigenvectors, _mean, input.reshape(1,1));
	Mat mean = Mat::zeros(q.rows,q.cols,q.type());
	vector<Vote>::iterator pD = dist.begin();
	vector<Mat>::const_iterator pP = _projections.begin();
	int *pL = (int *)_labels.data;
	int last_label = 0, count = 0;
	pD->label = 0;
	while(pP != _projections.end()) {
		if(*pL != last_label) {
			mean /= count;
			pD->dist = EigenNorm(mean, q);
			last_label = *pL;
			pD++;

			pD->label = last_label;
			count = 0;
			mean = Mat::zeros(q.rows,q.cols,q.type());
		}
		mean += *pP;
		++pP; ++pL; ++count;
	}
	mean /= count;
	pD->dist = norm(mean, q, NORM_L2);
	last_label = 4;
	for(int i = 0; i < NUM_CLASSES; i++) {
		prediction.bottom[i] = dist[i];
	}
	prediction.FillTiersBasedOnBottom();
}