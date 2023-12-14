#pragma once

#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
using namespace cv;
using namespace std;
using namespace Eigen;

struct ImageObj {
	vector<Vec3d> colors;
	int h, w, c;

	ImageObj() { }
	ImageObj(vector<Vec3d>& colors_, int h_, int w_, int c_) {
		colors = colors_;
		h = h_, w = w_, c = c_;
	}

	ImageObj(string path) {
		cv::Mat img = cv::imread(path);
		h = img.rows;
		w = img.cols;
		c = img.channels();
		colors.clear();
		colors.resize(h * w);

		//read the original image pixels
#pragma omp parallel for
		for (int row = 0; row < h; row++) {
			uchar* data = img.ptr<uchar>(row);
			for (int col = 0; col < w; col++) {
				uchar B = data[c * col + 0];
				uchar G = data[c * col + 1];
				uchar R = data[c * col + 2];
				Vec3d color(R / 255.0, G / 255.0, B / 255.0);
				int k = row * w + col;
				colors[k] = color;
			}
		}
	}

	double operator -(Mat& img1) {
		double diff = 0;
		for (int row = 0; row < h; row++) {
			uchar* data = img1.ptr<uchar>(row);
			for (int col = 0; col < w; col++) {
				uchar B = data[c * col + 0];
				uchar G = data[c * col + 1];
				uchar R = data[c * col + 2];
				Vec3d color(R / 255.0, G / 255.0, B / 255.0);
				int k = row * w + col;
				diff += norm(color - this->colors[k], NORM_L1);
			}
		}
		return diff;
	}

	//include those outside the boundary
	//Vec2i(a,b): a = 1:inside, a = 0:boundary, b is the pid
	vector<Vec2i> GetAllNeighbors(int pos) {
		int r = pos / w, c = pos % w;
		int r_[8] = { -1,0,1,0,-1,1,-1,1 };
		int c_[8] = { 0,1,0,-1,-1,1,1,-1 };
		vector<Vec2i> nb;
		for (int i = 0; i < 8; i++) {
			int new_r = r + r_[i];
			int new_c = c + c_[i];
			if (0 <= new_r && new_r < h && 0 <= new_c && new_c < w)
				nb.push_back(Vec2i(1, new_r * w + new_c));
			else
				nb.push_back(Vec2i(0, new_r * w + new_c));
		}
		return nb;
	}
};



struct vec2icmp {
	bool operator()(Vec2i v1, Vec2i v2) const {
		if (v1[0] != v2[0])  return v1[0] < v2[0];
		else  return v1[1] < v2[1];
	}
};

struct vec3icmp {
	bool operator()(Vec3i v1, Vec3i v2) const {
		if (v1[0] != v2[0])  return v1[0] < v2[0];
		else if (v1[1] != v2[1]) return v1[1] < v2[1];
		else return v1[2] < v2[2];
	}
};

inline bool edgecmp(Vec2i v1, Vec2i v2) {
	if (v1[0] != v2[0]) return v1[0] < v2[0];
	else return v1[1] < v2[1];
}

struct vec4icmp {
	bool operator()(Vec4i v1, Vec4i v2) const {
		int a = v1[0] + v1[1] + v1[2] + v1[3];
		int b = v2[0] + v2[1] + v2[2] + v2[3];
		return a < b;
	}
};

inline Mat GetChessboard(int h = 128, int w = 128) {
	int grid_len = 16;
	Mat img(h, w, CV_8UC3, Scalar(255, 255, 255));
	int k = 0;
	for (int r = 0; r < h; r += grid_len) {
		k = r / grid_len % 2;
		for (int c = 0; c < w; c += grid_len) {
			if (k++ % 2 == 1) continue;
			for (int r_ = r; r_ < min(r + grid_len, h); r_++)
				for (int c_ = c; c_ < min(c + grid_len, w); c_++) {
					img.at<cv::Vec3b>(r_, c_)[0] = 210;
					img.at<cv::Vec3b>(r_, c_)[1] = 210;
					img.at<cv::Vec3b>(r_, c_)[2] = 210;
				}
		}
	}
	return img;
}