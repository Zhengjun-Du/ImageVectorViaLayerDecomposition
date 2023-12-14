#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include "Region.h"
#include "Tree.h"
#include "Utility.h"
#include "Graph.h"
#include <autodiff/forward/dual.hpp>

using namespace autodiff;

class Object {
public:
	int obj_id, layer_id;
	Vec4i bbox_coord;
	MatrixXd param;				// object's linear gradient params 
	set<int> covered_rids;		// a object may consist of several regions
	vector<int> covered_pids;

	Object(int id = 0) { obj_id = id; }
	Object(int oid, int lid, set<int> rids_) { obj_id = oid; layer_id = lid;  covered_rids = rids_; }


	bool IsOverlapWith(Object& obj) {
		set<int> union_rids;
		for (int rid : covered_rids) union_rids.insert(rid);
		for (int rid : obj.covered_rids) union_rids.insert(rid);
		return union_rids.size() < (covered_rids.size() + obj.covered_rids.size());
	}
};

struct ObjectParams {
	vector<dual> vars;
	vector<double> gradients;

	ObjectParams() {}
	ObjectParams(vector<MatrixXd>& mats) {
		Initialize(mats);
	}
	void Initialize(int mat_n) {
		vars.resize(mat_n * 9);
		gradients.resize(mat_n * 9);
	}
	void Initialize(vector<MatrixXd>& mats) {
		int mat_n = mats.size();
		vars.resize(mat_n * 9);
		gradients.resize(mat_n * 9);

		for (int i = 0; i < mat_n; i++) {
			int k = 9 * i;
			MatrixXd mat = mats[i];
			double theta = mat(0, 0) != 0 ? atan(mat(1, 0) / mat(0, 0)) : 3.141592653 / 2;
			double sin_t = sin(theta);
			double cos_t = cos(theta);

			vars[k + 0] = theta;
			vars[k + 1] = abs(cos_t) > 1e-6 ? mat(0, 0) / cos_t : mat(1, 0) / sin_t;
			vars[k + 2] = abs(cos_t) > 1e-6 ? mat(0, 1) / cos_t : mat(1, 1) / sin_t;
			vars[k + 3] = abs(cos_t) > 1e-6 ? mat(0, 2) / cos_t : mat(1, 2) / sin_t;
			vars[k + 4] = abs(cos_t) > 1e-6 ? mat(0, 3) / cos_t : mat(1, 3) / sin_t;
			vars[k + 5] = mat(2, 0);
			vars[k + 6] = mat(2, 1);
			vars[k + 7] = mat(2, 2);
			vars[k + 8] = mat(2, 3);
		}
	}
	vector<MatrixXd> Convert2Mats() {
		int mat_n = vars.size() / 9;
		vector<MatrixXd> mats(mat_n);

		for (int i = 0; i < mat_n; i++) {
			int k = i * 9;
			MatrixXd mat = MatrixXd::Ones(3, 4);
			mat(0, 0) = cos((double)vars[k]) * (double)vars[k + 1];
			mat(0, 1) = cos((double)vars[k]) * (double)vars[k + 2];
			mat(0, 2) = cos((double)vars[k]) * (double)vars[k + 3];
			mat(0, 3) = cos((double)vars[k]) * (double)vars[k + 4];
			mat(1, 0) = sin((double)vars[k]) * (double)vars[k + 1];
			mat(1, 1) = sin((double)vars[k]) * (double)vars[k + 2];
			mat(1, 2) = sin((double)vars[k]) * (double)vars[k + 3];
			mat(1, 3) = sin((double)vars[k]) * (double)vars[k + 4];
			mat(2, 0) = (double)vars[k + 5];
			mat(2, 1) = (double)vars[k + 6];
			mat(2, 2) = (double)vars[k + 7];
			mat(2, 3) = (double)vars[k + 8];
			mats[i] = mat;
		}
		return mats;
	}
};

struct PixPassedObjects {
	int pix_id;
	Vec2d coord;
	vector<int> covered_objects;

	PixPassedObjects(int pix_id_ = 0, Vec2d coord_ = Vec2d(0, 0)) {
		pix_id = pix_id_;
		coord = coord_;
	}
};
