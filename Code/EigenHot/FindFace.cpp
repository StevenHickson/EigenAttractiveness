
#include "stdafx.h"
#include "EigenHot.h"


int FindFace(Mat &in, Mat &out) {
	CascadeClassifier face_cascade = CascadeClassifier(FACE_XML);
	vector<Rect> faces;
	face_cascade.detectMultiScale(in, faces);
	if(faces.size() == 0)
		return 0;
	out = in(faces[0]);
	return faces.size();
}