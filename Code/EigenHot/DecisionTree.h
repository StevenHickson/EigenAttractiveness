#pragma once

#define NUM_CLASSES 4

class Vote {
public:
	int label;
	double dist;

	bool operator<(Vote &other) {
		return (dist < other.dist);
	}

	Vote(int _label, double _dist) : label(_label), dist(_dist) {	}
	Vote() : label(-1), dist(-1) { }
};

class EigenDecisionTree {
public:
	bool propogated;
	Vote votes[NUM_CLASSES + 5];
	Vote *bottom, *middle, *top;
	EigenDecisionTree() : propogated(false) {
		bottom = votes;
		middle = votes + NUM_CLASSES;
		top = middle + 3;
	}

	void operator*=(EigenDecisionTree &in) {
		for(int i = 0; i < NUM_CLASSES + 5; i++) {
			if(in.votes[i].dist != -1 && votes[i].dist != -1)
				votes[i].dist *= in.votes[i].dist;
		}
	}
	void operator+=(EigenDecisionTree &in) {
		for(int i = 0; i < NUM_CLASSES + 5; i++) {
			if(in.votes[i].dist >= 0 && votes[i].dist >= 0)
				votes[i].dist += in.votes[i].dist;
		}
	}

	void SetBottomTier(Vote v0, Vote v1, Vote v2, Vote v3) {
		bottom[0] = v0; bottom[1] = v1; bottom[2] = v2; bottom[3] = v3; 
	}
	void SetMiddleTier(Vote vl, Vote vm, Vote vh) {
		propogated = true;
		middle[0] = vl; middle[1] = vm; middle[2] = vh;
	}
	void SetTopTier(Vote vb, Vote vt) {
		propogated = true;
		top[0] = vb; top[1] = vt;
	}
	void Normalize() {
		double sum = 0;
		for(int i = 0; i < NUM_CLASSES; i++) {
			sum += votes[i].dist;
		}
		for(int i = 0; i < NUM_CLASSES; i++) {
			votes[i].dist /= sum;
		}
	}
	void FillTiersBasedOnBottom() {
		Normalize();
		propogated = true;
		middle[0].label = 0;
		middle[1].label = 1;
		middle[2].label = 2;
		top[0].label = 0;
		top[1].label = 1;
		middle[0].dist = (bottom[0].dist + bottom[1].dist) / 2;
		middle[1].dist = (bottom[1].dist + bottom[2].dist) / 2;
		middle[2].dist = (bottom[2].dist + bottom[3].dist) / 2;
		top[0].dist = (bottom[0].dist + bottom[1].dist + bottom[2].dist) / 3;
		top[1].dist = (bottom[1].dist + bottom[2].dist + bottom[3].dist) / 3;
	}

	//DEPRECATED!!
	int GetBestVote() {
		double best[NUM_CLASSES];
		best[0] = bottom[0].dist * middle[0].dist * top[0].dist;
		best[1] = bottom[1].dist * middle[0].dist * top[0].dist;
		best[2] = bottom[2].dist * middle[1].dist * top[0].dist;
		best[3] = bottom[3].dist * middle[2].dist * top[1].dist;
		double min = best[0];
		int out = 0;
		for(int i = 1; i < NUM_CLASSES; i++) {
			if(best[i] < min) {
				min = best[i];
				out = i;
			}
		}
		return out;
	}

	int GetBottomVote() {
		double min = bottom[0].dist;
		int out = bottom[0].label;
		for(int i = 1; i < NUM_CLASSES; i++) {
			if(bottom[i].dist < min) {
				min = bottom[i].dist;
				out = bottom[i].label;
			}
		}
		return out;
	}
	int GetMiddleVote() {
		double min = middle[0].dist;
		int out = middle[0].label;
		for(int i = 1; i < 3; i++) {
			if(middle[i].dist < min) {
				min = middle[i].dist;
				out = middle[i].label;
			}
		}
		return out;
	}
	int GetTopVote() {
		if(top[0].dist < top[1].dist)
			return top[0].label;
		return top[1].label;
	}
};