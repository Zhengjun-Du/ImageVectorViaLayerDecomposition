#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "Utility.h"
#include "Xjunction.h"

using namespace std;
using namespace Eigen;
using namespace cv;

class Region {
public:
	int					m_region_id;
	set<int>			m_adj_regions;
	Vec3i				m_region_color;
	vector<int>			m_region_pids;
	vector<MatrixXd>	m_layer_params;
	MatrixXd			m_recon_pix_colors;
	Vec4i				m_bbox;
	vector<int>			m_shared_pixcnt_with_reg;
	double				m_perimeter;
	bool				m_is_at_bottom;

public:
	Region() { m_is_at_bottom = false; }
	Region(int region_id, set<int> adjacent_regions, vector<int> region_pix_index) {
		m_region_id = region_id;
		m_adj_regions = adjacent_regions;
		m_region_pids = region_pix_index;
		m_is_at_bottom = false;
	}

	bool Enclose(Region& R1) {
		return	m_bbox[0] <= R1.m_bbox[0] && m_bbox[1] <= R1.m_bbox[1] &&
			m_bbox[2] >= R1.m_bbox[2] && m_bbox[3] >= R1.m_bbox[3];
	}

	bool CouldSupport(Region& R2) {
		if (Enclose(R2)) return true;
		if (R2.Enclose(*this)) return false;

		int r1_area = m_region_pids.size();
		int r2_area = R2.m_region_pids.size();
		return (r1_area >= r2_area * 2.0 / 3);
	}
};

//===================================================================================

class RegionInfo {
public:
	vector<Region>	regions;
	vector<int>		possible_bottom_rids;
	Xjunction		xjunction;
	vector<int>		pix_region_ids;

	RegionInfo(string region_img_path, string region_param_path) {
		GetAllRegionInfoFrom(region_img_path, region_param_path);
	}

	int GetInitialEdgeCnt() {
		int en = regions.size() - 1;
		for (int i = 1; i < regions.size(); i++) {
			Region R = regions[i];
			en += R.m_adj_regions.size();
		}
		return en;
	}

	void GetAllRegionInfoFrom(string region_img_path, string region_param_path) {
		ifstream ifs(region_param_path);
		int region_num = 0; ifs >> region_num;
		regions.resize(region_num + 1);

		//1. define the canvas region==============================================
		regions[0].m_region_id = 0;
		for (int i = 1; i < regions.size(); i++)
			regions[0].m_adj_regions.insert(i);

		MatrixXd region_param(3, 3);
		region_param << 0, 0, 0, 0, 0, 0, 1, 1, 1;
		regions[0].m_layer_params.push_back(region_param);

		for (int i = 0; i < regions.size(); i++)
			regions[i].m_shared_pixcnt_with_reg.resize(regions.size(), 0);

		//2. read other regions' parameters and mask colors========================
		map<Vec3i, int, vec3icmp> mp;
		for (int i = 1; i <= region_num; i++) {
			int R, G, B; ifs >> R >> G >> B;
			Vec3i region_color(R, G, B);
			regions[i].m_region_id = i;
			regions[i].m_region_color = region_color;
			mp[region_color] = i;

			//region bounding box coordinates
			Vec4i bbox;
			ifs >> bbox[0] >> bbox[1] >> bbox[2] >> bbox[3];
			regions[i].m_bbox = bbox;
		}

		//3. record each region's pix ids================================================
		ImageObj region_img(region_img_path);
		pix_region_ids.resize(region_img.h * region_img.w, 0);
		for (int i = 0; i < region_img.colors.size(); i++) {
			Vec3i color = region_img.colors[i] * 255;
			int row = i / region_img.w, col = i % region_img.w;
			if (mp.find(color) != mp.end()) {
				int region_id = mp[color];
				regions[region_id].m_region_pids.push_back(i);
				pix_region_ids[i] = region_id;
			}
		}

		//4. read possible bottom regions==========================================
		int bot_rid_cnt = 0; ifs >> bot_rid_cnt;
		possible_bottom_rids.resize(bot_rid_cnt);
		for (int i = 0; i < bot_rid_cnt; i++)
			ifs >> possible_bottom_rids[i];

		if (bot_rid_cnt == 1) {
			int bot_rid = possible_bottom_rids[0];
			regions[bot_rid].m_is_at_bottom = true;
		}

		//5. read region adjacent info=============================================
		int cnt = 0; ifs >> cnt;
		int rid = 0, adj_cnt, adj_rid;
		for (int i = 0; i < cnt; i++) {
			ifs >> rid >> adj_cnt;
			for (int j = 0; j < adj_cnt; j++) {
				ifs >> adj_rid;
				regions[rid].m_adj_regions.insert(adj_rid);
			}
		}

		//6. read xjunction info===================================================
		int xj_cnt = 0; ifs >> xj_cnt;
		int xj1, xj2, xj3, xj4;
		vector<Vec4i> xjunction_vec;
		for (int i = 0; i < xj_cnt; i++) {
			ifs >> xj1 >> xj2 >> xj3 >> xj4;
			xjunction_vec.push_back(Vec4i(xj1, xj2, xj3, xj4));
		}
		xjunction = Xjunction(xjunction_vec);

		//7. record adjacent region boundary pixcnt
		for (int rid = 1; rid < regions.size(); rid++) {
			double perimeter = 0;
			double max_share_boundary = -1;
			for (int j : regions[rid].m_region_pids) {
				vector<Vec2i> neighbor_pids = region_img.GetAllNeighbors(j);
				for (Vec2i location_pid : neighbor_pids) {
					bool inside = location_pid[0];
					int pid = location_pid[1];
					if (inside) {
						int nb_rid = pix_region_ids[pid];
						if (rid != nb_rid) {
							regions[rid].m_shared_pixcnt_with_reg[nb_rid]++;
							perimeter++;
						}
					}
					else {
						regions[rid].m_shared_pixcnt_with_reg[0]++;
						perimeter++;
					}
				}
			}
			regions[rid].m_perimeter = perimeter;
		}
	}
};