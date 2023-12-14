#pragma once
#include <vector>
#include "Utility.h"
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <fstream>
#include "Region.h"

using namespace std;
using namespace Eigen;
using namespace cv;

class RegionInfo {
private:
	Image m_ori_img, m_reg_img, m_mask_img;
	vector<int> m_pix_regid;
	vector<Region> m_regions;
	vector<set<int>> m_adj_regions;
	set<int> m_bound_pids;
	vector<Vec4i> m_xjunction_vec;

public:
	RegionInfo() {}
	RegionInfo(string img_path, string reg_path, string mask_path) {
		srand(600);
		m_ori_img = Image(img_path);
		m_reg_img = Image(reg_path);
		m_mask_img = Image(mask_path);
		GetAllRegions();
	}

	//BFS, assign each pixel a number to indicate the region id
	int BFS(int start, int id) {
		if (m_pix_regid[start] != 0) return 0;
		int h = m_reg_img.h;
		int w = m_reg_img.w;
		int cnt = 1;
		m_pix_regid[start] = id;
		int r, c, pid, new_r, new_c, new_pid;
		Vec3d cur_color = m_reg_img.m_rgb[start];

		queue<int> Q; Q.push(start);
		while (!Q.empty()) {
			pid = Q.front(); Q.pop();
			vector<int> adj_pids = m_reg_img.GetNeighborsOf(pid);
			for (int adj_pid : adj_pids) {
				Vec3d adj_color = m_reg_img.m_rgb[adj_pid];
				double diff = norm(cur_color - adj_color);
				if (m_pix_regid[adj_pid] == 0 && diff < 0.05 && m_mask_img.IsFgPixelAt(adj_pid)) {
					Q.push(adj_pid);
					m_pix_regid[adj_pid] = id;
					cnt++;
				}
			}
		}
		return cnt;
	}

	void GetAllRegions() {
		int pix_cnt = m_reg_img.h * m_reg_img.w;
		m_pix_regid.resize(pix_cnt, 0);

		int rid = 1; //region 0 is virtual, so number from 1
		for (int i = 0; i < m_ori_img.h * m_ori_img.w; i++)
			if (m_pix_regid[i] == 0 && m_mask_img.IsFgPixelAt(i)) {
				int cnt = BFS(i, rid++);
				cout << "region " << rid-1 << " pix_cnt: " << cnt << ", coord: " << i / m_reg_img.w << ", " << i % m_reg_img.w << endl;
			}

		m_regions.resize(rid);
		for (int i = 0; i < m_ori_img.h * m_ori_img.w; i++)
			if(m_mask_img.IsFgPixelAt(i))
				m_regions[m_pix_regid[i]].pids.insert(i);

		//assign each region a color, for debug
		for (int i = 1; i < m_regions.size(); i++) {
			m_regions[i].rid = i;
			m_regions[i].reg_color = Vec3i(rand() % 256, rand() % 256, rand() % 256);
			m_regions[i].GetRegionBbox(m_ori_img.w, m_ori_img.h);
		}

		//process small noise regions
		ProcessSmallNoiseRegions();
	}

	void ProcessSmallNoiseRegions() {
		vector<int> to_remove_rids;
		for (int i = 1; i < m_regions.size(); i++) {
			if (m_regions[i].pids.size() > 20) continue;

			int closest_rid = 0;
			double min_diff = 1e8;

			for (int pid : m_regions[i].pids) {
				vector<int> neighbor_pids = m_ori_img.GetNeighborsOf(pid);
				for (int nb_pid : neighbor_pids) {
					int nb_pid_rid = m_pix_regid[nb_pid];
					if (nb_pid_rid != i && m_regions[nb_pid_rid].pids.size() > 20) {
						double diff = norm(m_ori_img.m_rgb[pid] - m_ori_img.m_rgb[nb_pid]);
						if (diff < min_diff) {
							min_diff = diff;
							closest_rid = nb_pid_rid;
						}
					}
				}
			}
			m_regions[closest_rid].pids.insert(m_regions[i].pids.begin(), m_regions[i].pids.end());
			to_remove_rids.push_back(i);
		}
		for (int i = to_remove_rids.size() - 1; i >= 0; i--)
			m_regions.erase(m_regions.begin() + to_remove_rids[i]);

		for (int i = 1; i < m_regions.size(); i++) {
			m_regions[i].rid = i;
			for (int pid : m_regions[i].pids)
				m_pix_regid[pid] = i;
		}
	}

	void GetAdjacencyInfo(string reg_info_path) {
		m_adj_regions.resize(m_regions.size());
		map<Vec2i, set<int>, vec2icmp>  region_boundary_pixcnt;
		int h = m_reg_img.h;
		int w = m_reg_img.w;
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				int pid = i * w + j;
				int rid = m_pix_regid[pid];
				if (rid == 0) continue; //0: ignore background pixels

				vector<int> adj_pids = m_reg_img.GetNeighborsOf(i, j);
				for (int adj_pid : adj_pids) {
					int adj_rid = m_pix_regid[adj_pid];
					if (adj_rid != rid) {
						m_adj_regions[rid].insert(adj_rid);
						m_adj_regions[adj_rid].insert(rid);
						m_bound_pids.insert(pid);
						int min_rid = min(rid, adj_rid);
						int max_rid = max(rid, adj_rid);
						region_boundary_pixcnt[Vec2i(min_rid, max_rid)].insert(pid);
					}
				}

				//recored the outmost regions that on the boundary
				if(adj_pids.size() < 8) //pixel has less than 8 neighbors, on the image rectangle boundary
					m_adj_regions[rid].insert(0);
			}
		}

		//some region pair have only share few pixels, they are not really neighbors.
		for (auto it = region_boundary_pixcnt.begin(); it != region_boundary_pixcnt.end(); it++) {
			Vec2i reg_pair = it->first;
			int bound_pix_cnt = it->second.size();
			if (bound_pix_cnt < 5) {
				int rid1 = reg_pair[0];
				int rid2 = reg_pair[1];
				if (rid1 != -1 && rid2 != -1) {
					m_adj_regions[rid1].erase(rid2);
					m_adj_regions[rid2].erase(rid1);
				}
			}
		}
	}

	//detect the X-junctions in the segmentation image, but it may nor stable.
	void GetXjunctionInfo(string reg_info_path) {
		m_xjunction_vec.clear();
		vector<Vec2i> coords = GenerateSquareOutermostCoords(7);
		int h = m_reg_img.h;
		int w = m_reg_img.w;
		for (int pid : m_bound_pids) {
			int r = pid / w;
			int c = pid % w;
			vector<int> region_ids;
			for (int j = 0; j < coords.size(); j++) {
				int new_r = r + coords[j][0];
				int new_c = c + coords[j][1];
				if (0 <= new_r && new_r < m_ori_img.h && 0 <= new_c && new_c < m_ori_img.w) {
					int rid = m_pix_regid[new_r * w + new_c];
					region_ids.push_back(rid);
				}
			}
			vector<int> uniq_rids;
			for (auto it = region_ids.begin(); it < region_ids.end(); it++)
				if (find(region_ids.begin(), it, *it) == it)
					uniq_rids.push_back(*it);

			if (uniq_rids.size() == 4) {
				Vec4i x(uniq_rids[0], uniq_rids[1], uniq_rids[2], uniq_rids[3]);
				bool different_with_existing_xj = true;
				for (int i = 0; i < m_xjunction_vec.size(); i++) {
					if (IsEqualOfXjunctions(x, m_xjunction_vec[i])) {
						different_with_existing_xj = false;
						break;
					}
				}
				if (different_with_existing_xj)
					m_xjunction_vec.push_back(x);
			}
		}
	}

	void OutputRegionInfo_s1(string reg_path, string reg_info_path, string region_ind_path) {
		//1. output region image (region.png)=========================================================
		Mat regions(m_reg_img.h, m_reg_img.w, CV_8UC3, Scalar(255, 255, 255));
		int start = 1;

		for (int i = start; i < m_regions.size(); i++) {
			Region reg = m_regions[i];
			for (int pid : reg.pids) {
				int r = pid / m_reg_img.w;
				int c = pid % m_reg_img.w;
				regions.at<cv::Vec3b>(r, c)[2] = reg.reg_color[0];
				regions.at<cv::Vec3b>(r, c)[1] = reg.reg_color[1];
				regions.at<cv::Vec3b>(r, c)[0] = reg.reg_color[2];
			}
		}
		imwrite(reg_path, regions);
		waitKey();

		//2. output region color index (region_index.png)=============================================
		int reg_cnt = m_regions.size() - 1;
		int grid_len = 100;
		int W = 8;
		int H = (reg_cnt + 7) / 8;
		Mat region_ind(H * grid_len, W * grid_len, CV_8UC3, Scalar(255, 255, 255));
		int k = 1;

		for (int i = 0; i < H; i++) {
			for (int j = 0; j < W; j++) {
				for (int r = i * grid_len; r < (i + 1) * grid_len; r++) {
					for (int c = j * grid_len; c < (j + 1) * grid_len; c++) {
						region_ind.at<cv::Vec3b>(r, c)[2] = m_regions[k].reg_color[0];
						region_ind.at<cv::Vec3b>(r, c)[1] = m_regions[k].reg_color[1];
						region_ind.at<cv::Vec3b>(r, c)[0] = m_regions[k].reg_color[2];
					}
				}

				int center_x = (i + 0.5) * grid_len;
				int center_y = (j + 0.5) * grid_len;
				int num = k;
				putText(region_ind, to_string(num), Point(center_y - 9, center_x + 9), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0));
				if (++k >= m_regions.size()) {
					break;
				}
			}
		}
		imwrite(region_ind_path, region_ind);
		waitKey();


		//3. output region color and bounding box=====================================================
		ofstream of(reg_info_path);
		of << reg_cnt << endl;
		for (int i = start; i < m_regions.size(); i++) {
			Region& reg = m_regions[i];
			of << reg.reg_color[0] << " " << reg.reg_color[1] << " " << reg.reg_color[2] << endl;
			of << reg.bbox[0] << " " << reg.bbox[1] << " " << reg.bbox[2] << " " << reg.bbox[3] << endl;
			of << endl;
		}
		of << endl; of.close();
	}

	//out most region should be connect to 0
	void OutputRegionInfo_s2(string reg_info_path) {
		
		//1 outputs outermost regions' ids, which may be the bottom layers
		ofstream of(reg_info_path, ios::app);
		vector<int> m_outmost_regions;
		for (int i = 1; i < m_adj_regions.size(); i++) {
			if (m_adj_regions[i].find(0) != m_adj_regions[i].end()) {
				m_outmost_regions.push_back(i);
				m_adj_regions[i].erase(0);
			}
		}
		
		of << m_outmost_regions.size() << endl;
		for (int i = 0; i < m_outmost_regions.size(); i++)
			of << m_outmost_regions[i] << " ";
		of << endl << endl;

		//2 output regions' all adjacent regions
		of << m_adj_regions.size() - 1 << endl;
		for (int i = 1; i < m_adj_regions.size(); i++) {
			of << i << " " << m_adj_regions[i].size() << endl;
			for (int j : m_adj_regions[i])
				of << j << "  ";
			of << endl;
		}

		//3 output xjunction info, x-junction detection may not correct, if so£¬ it can be specified by user
		of << endl << m_xjunction_vec.size() << endl;
		for (Vec4i xj : m_xjunction_vec)
			of << xj[0] << " " << xj[1] << " " << xj[2] << " " << xj[3] << endl;
		of.close();
	}
};