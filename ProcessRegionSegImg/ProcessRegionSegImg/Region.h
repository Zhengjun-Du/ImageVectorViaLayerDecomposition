#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace Eigen;

class Region {
public:
	int			rid;
	set<int>	pids;
	Vec3i		reg_color;
	Vec4i		bbox;

public:
	Region() {};
	Region(int rid, set<int> pids) {
		this->rid = rid;
		this->pids = pids;
	}

	vector<int> SamplePixels(int n) {
		vector<int> sample_pids;
		n = min(n, (int)pids.size());
		double step = pids.size() * 1.0 / n;

		vector<int> pid_vec(pids.begin(), pids.end());
		for (double i = 0; i < pid_vec.size(); i += step)
			sample_pids.push_back(pid_vec[(int)i]);
		return sample_pids;
	}

	void GetRegionBbox(int img_w, int img_h) {
		int min_r = img_h + 1, min_c = img_w + 1, max_r = -1, max_c = -1;
		for (int pid : pids) {
			int r = pid / img_w;
			int c = pid % img_w;
			if (r < min_r) min_r = r;
			if (c < min_c) min_c = c;
			if (max_r < r) max_r = r;
			if (max_c < c) max_c = c;
		}
		bbox = Vec4i(min_r, min_c, max_r, max_c);
	}
};