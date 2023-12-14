#pragma once
#include <string>
#include <vector> 
#include <opencv2/opencv.hpp>
#include <set>
#include <map>
#include <omp.h>
#include <Eigen/Core>

using namespace std;
using namespace cv;
using namespace Eigen;

class Image {
public:
	int h, w, c;
	vector<cv::Vec3d> m_rgb;
private:
	Mat img;

public:
	Image() {};
	Image(string path, string blur_path="") {
		img = cv::imread(path);
		h = img.rows, w = img.cols, c = img.channels();

		cout << "0. Read imge..." << endl;
		m_rgb.resize(h * w);

		for (int row = 0; row < h; row++) {
			uchar* data = img.ptr<uchar>(row);
			for (int col = 0; col < w; col++) {
				uchar B = data[c * col + 0];
				uchar G = data[c * col + 1];
				uchar R = data[c * col + 2];
				m_rgb[row * w + col] = cv::Vec3d(R / 255.0, G / 255.0, B / 255.0);
			}
		}
	};

	vector<int> GetNeighborsOf(int r, int c) {
		int r_[8] = { -1,0,1,0,-1,1,-1,1 };
		int c_[8] = { 0,1,0,-1,-1,1,1,-1 };
		vector<int> nb;
		for (int i = 0; i < 8; i++) {
			int new_r = r + r_[i];
			int new_c = c + c_[i];
			if (0 <= new_r && new_r < h && 0 <= new_c && new_c < w)
				nb.push_back(new_r * w + new_c);
		}
		return nb;
	}

	vector<int> GetNeighborsOf(int pos) {
		return GetNeighborsOf(pos / w, pos % w);
	}

	//just for checking a pixel in mask is foreground or not 
	bool IsFgPixelAt(int id) {
		return (abs(m_rgb[id][0] + m_rgb[id][1] + m_rgb[id][2]) > 0.0001);
	}
};

struct vec2icmp {
	bool operator()(Vec2i v1, Vec2i v2) const {
		if (v1[0] != v2[0]) return v1[0] < v2[0];
		else return v1[1] < v2[1];
	}
};

inline bool IsEqualOfXjunctions(Vec4i xj1, Vec4i xj2) {
	set<int> s1; s1.insert(xj1[0]);s1.insert(xj1[1]);s1.insert(xj1[2]);s1.insert(xj1[3]);
	set<int> s2; s2.insert(xj2[0]);s2.insert(xj2[1]);s2.insert(xj2[2]);s2.insert(xj2[3]);
	set<int>::iterator it1 = s1.begin(), it2 = s2.begin();
	for (; it1 != s1.end(); it1++, it2++) {
		if (*it1 != *it2)
			return false;
	}
	return true;
}

inline vector<Vec2i> GenerateSquareOutermostCoords(int s) {
	int  k = 0;
	int cnt = 2 * s + 2 * (s - 2);
	vector<Vec2i> offset(cnt);

	for (int i = 0; i < s; i++) {
		offset[k][0] = -s / 2 + i;
		offset[k][1] = -s / 2;
		k++;
	}
	for (int i = 0; i < s - 2; i++) {
		offset[k][0] = s / 2;
		offset[k][1] = -s / 2 + 1 + i;
		k++;
	}
	for (int i = s - 1; i >= 0; i--) {
		offset[k][0] = -s / 2 + i;
		offset[k][1] = s / 2;
		k++;
	}
	for (int i = s - 2; i > 0; i--) {
		offset[k][0] = -s / 2;
		offset[k][1] = -s / 2 + i;
		k++;
	}
	return offset;
}