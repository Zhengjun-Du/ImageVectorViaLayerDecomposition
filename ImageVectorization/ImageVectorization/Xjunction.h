#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class Xjunction {
public:
	vector<Vec4i> m_xjunctions;
	vector<vector<Vec4i>> m_possible_configs;

public:
	Xjunction() {}
	Xjunction(vector<Vec4i> xjunctions) {
		m_xjunctions = xjunctions;
		GetAllPosibleConfigs();
	}

	vector<array<int, 4>> Conver2Arrrint4() {
		vector<array<int, 4>> ans;
		for (Vec4i xj : m_xjunctions)
			ans.push_back({ xj[0],xj[1],xj[2],xj[3] });
		return ans;
	}

	// 0 | 1
	// ¡ª¡ª¡ª
	// 3 | 2
	void GetAllPosibleConfigs() {
		m_possible_configs.clear();
		for (Vec4i xrids : m_xjunctions) {
			vector<Vec4i> single_xjunction_configs(4);

			// 0->1, 3->2
			Vec4i config1(xrids[0], xrids[1], xrids[3], xrids[2]);
			if (xrids[3] < xrids[0])
				config1 = Vec4i(xrids[3], xrids[2], xrids[0], xrids[1]);
			single_xjunction_configs[0] = config1;

			// 1->0, 2->3
			Vec4i config2(xrids[1], xrids[0], xrids[2], xrids[3]);
			if (xrids[2] < xrids[1])
				config2 = Vec4i(xrids[2], xrids[3], xrids[1], xrids[0]);
			single_xjunction_configs[1] = config2;

			// 0->3, 1->2
			Vec4i config3(xrids[0], xrids[3], xrids[1], xrids[2]);
			if (xrids[1] < xrids[0])
				config3 = Vec4i(xrids[1], xrids[2], xrids[0], xrids[3]);
			single_xjunction_configs[2] = config3;

			// 3->0, 2->1
			Vec4i config4(xrids[3], xrids[0], xrids[2], xrids[1]);
			if (xrids[2] < xrids[3])
				config4 = Vec4i(xrids[2], xrids[1], xrids[3], xrids[0]);
			single_xjunction_configs[3] = config4;

			m_possible_configs.push_back(single_xjunction_configs);
		}
	}

	bool ContainsRegions(int r1_id, int r2_id) {
		for (vector<Vec4i> xjs : m_possible_configs) { //2 10 3 11
			for (Vec4i xj : xjs) {
				if ((r1_id == xj[0] && r2_id == xj[1]) || (r1_id == xj[1] && r2_id == xj[0]) ||
					(r1_id == xj[2] && r2_id == xj[3]) || (r1_id == xj[3] && r2_id == xj[2]))
					return true;
			}
		}
		return false;
	}

	bool ContainsRegion(int rid) {
		for (vector<Vec4i> xjs : m_possible_configs) { 
			for (Vec4i xj : xjs)
				if (rid == xj[0] || rid == xj[1] || rid == xj[2] || rid == xj[3])
					return true;
		}
		return false;
	}
};