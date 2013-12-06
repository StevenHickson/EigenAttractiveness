#include "stdafx.h"
#include "Eigenfaces.h"
#include <iostream>
#include <math.h>

const float Pif = 3.141592653589793238f;
//local #defines
#define B_MEAN 99.1442f
#define R_MEAN 138.0654f	

//this is actually the inverse of the covariance matrix
#define COV_11 0.00376353477484179f //552.6847
#define COV_12 -0.00330727927436376f //326.5669
#define COV_21 -0.00330727927436376f //326.5669
#define COV_22 0.00559726810338469f //371.6184

#define COV_DET 9.87418612943281e+04

//detects skin color, (you should erode and dilate manually as you see fit)
void SkinDetection2(Mat &img, Mat &skin, float thresh) {
	int type = img.type();
	Mat chrom = Mat::zeros(img.rows, img.cols, CV_32F);
	float *pO = (float *)chrom.data, *end = (float *)chrom.dataend;
	uchar *pI = img.data;
	float cr,cb,tmp, max = 0;
	int r,g,b;
	while(pO != end) {
		//convert from rgb to ycbcr and subtract mean, ignore y, taken from wikipedia page on ycbcr
		//Vec3b color = img.at<Vec3b>(y,x);
		//b = color[0]; g = color[1]; r = color[2];
		b = *pI++; g = *pI++; r = *pI++;
		cb = -0.16874f * r - 0.33126f * g + 0.50000f * b  + 128.0f - B_MEAN;
		cr =  0.50000f * r - 0.41869f * g - 0.08131f * b  + 128.0f - R_MEAN;
		//matrix multiplication [cr , cb] * inv(cov) * [cr ; cb]
		tmp = cr * (COV_11 * cr + COV_21 * cb) + cb * (COV_12 * cr + COV_22 * cb);
		tmp = float(expf(-0.5f * tmp) / (2.0f*Pif*pow(COV_DET,5.0e-1)));
		*pO++ = tmp;
		if(tmp > max)
			max = tmp;
		//chrom.at<float>(y,x) = tmp;
		/*if(b != 0 || g != 0 || r != 0)
		std::cout << int(b) << ", " << int(g) << ", " << int(r) << ", " << tmp << std::endl;*/
	}
	//pass through low-pass filter 1/9 * ones(3);
	blur(chrom,chrom,Size(3,3));
	//normalize by maximum
	pO = (float *)chrom.data;
	for(int y = 0; y < img.rows; y++) {
		for(int x = 0; x < img.cols; x++) {
			*pO /= max;
			++pO;
		}
	}
	threshold(chrom,skin,thresh,1,CV_THRESH_BINARY);
}

void SkinDetection3(Mat &img, Mat &skin, float thresh) {
	if (img.empty())
		throw std::exception("Empty image in skin detection");
	blur( img, img, Size(3,3) );
	Mat ycbcr;
	cvtColor(img, ycbcr, CV_BGR2YCrCb);
	inRange(ycbcr, Scalar(30, 133, 77), Scalar(255, 173, 127), skin);
}

void SkinDetection(Mat &img, Mat &skin, float thresh) {
	if (img.empty())
		throw std::exception("Empty image in skin detection");
	blur( img, img, Size(3,3) );
	Mat hsv;
	cvtColor(img, hsv, CV_BGR2HSV);
	//inRange(hsv, Scalar(0, 40, 60), Scalar(20, 150, 255), bw);
	inRange(hsv, Scalar(0, 48, 80), Scalar(20, 255, 255), skin);
}

int GetFaceSize(Mat &img) {
	Mat skin, tmp;
	SkinDetection2(img,skin,0.4f);
	//imshow("frame", img);
	//imshow("skin",skin);
	//waitKey();
	Mat kernel = Mat::ones(3,3,CV_32S);

	erode(skin,tmp,kernel);
	dilate(tmp,skin,kernel);

	imshow("frame",img);
	imshow("skin",skin);
	waitKey();

	int count = 0;
	int type = skin.type();
	float* p = (float *)skin.data, *end = (float *)skin.dataend;
	while(p != end) {
		if(*p > 0)
			count++;
		++p;
	}
	return count;
}

