#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include "Region.h"
#include "Utility.h"
#include "Graph.h"
#include "LayerParameterOptimization.h"
#include "Object.h"

using namespace std;
using namespace cv;

class LayerVectorizing {
private:
	ImageObj* m_input_img;
	vector<Region> m_regions;
	vector<vector<Object>> m_layer_objects; //a layer may contain several objects
	vector<Mat> m_layer_imgs;				//store each layer's objects
	Mat m_reconstructed_img;

public:
	double m_total_loss = 1e8;
	double m_layer_cnt = 0;
	double m_recon_gamut_loss = 0;
	double m_ave_region_layer_cnt = 0;
	double m_big_over_small_cnt = 0;

	double m_wr = 20.0;
	double m_wg = 10.0;
	double m_wc = 0.02;

	// for eva
	double m_data_loss = 0;
	double m_gamut_loss = 0;
	double m_bigoversmall_loss = 0;

public:
	LayerVectorizing() {}

	LayerVectorizing(vector<Region> regions, ImageObj* input_img, vector<vector<Object>> layer_objs) {
		m_regions = regions;
		m_input_img = input_img;
		m_layer_objects = layer_objs;
	}

	bool operator < (LayerVectorizing& Ld) {
		return m_total_loss < Ld.m_total_loss;
	}

	Vec2i FindObjContainsRegion(int rid, int layer_id = -1) {
		int stat_layer_id = layer_id, end_layer_id = layer_id + 1;
		if (layer_id == -1)
			stat_layer_id = 1, end_layer_id = m_layer_objects.size();

		for (int i = stat_layer_id; i < end_layer_id; i++) {
			for (int j = 0; j < m_layer_objects[i].size(); j++) {
				Object& obj = m_layer_objects[i][j];
				if (obj.covered_rids.find(rid) != obj.covered_rids.end())
					return Vec2i(i, j);
			}
		}
		return Vec2i(-1, -1);
	}

	vector<int> SamplePixelsInAllRegions() {
		vector<int> sample_pids;

		//region 0 is the canvas, no need to sample
		for (int i = 1; i < m_regions.size(); i++) {
			int k = m_regions[i].m_region_pids.size();
			int sample_n = 30;
			double step = k * 1.0 / sample_n;
			for (double j = 0; j < k; j += step) {
				int pid = m_regions[i].m_region_pids[int(j)];
				sample_pids.push_back(pid);
			}
		}
		return sample_pids;
	}

	vector<PixPassedObjects> GetPixelPassedObjectsFromBottom2Top(vector<int>& pixels) {
		vector<PixPassedObjects> pix_passed_objs(pixels.size());
		for (int i = 0; i < pixels.size(); i++) {
			int pid = pixels[i];
			double x = pid / m_input_img->w * 1.0 / (m_input_img->h - 1);
			double y = pid % m_input_img->w * 1.0 / (m_input_img->w - 1);
			PixPassedObjects ppo(pid, Vec2d(x, y));

			for (int j = 0; j < m_layer_objects.size(); j++) {
				for (Object& obj : m_layer_objects[j]) {
					if (find(obj.covered_pids.begin(), obj.covered_pids.end(), pid) != obj.covered_pids.end()) {
						ppo.covered_objects.push_back(obj.obj_id);
						break;
					}
				}
			}
			pix_passed_objs[i] = ppo;
		}
		return pix_passed_objs;
	}

	void CalculateLayerObjectParamsWithGlobalOptimization() {
		map<int, int> obj_layer_map;
		int k = 0; //reassign object id from bottom to top layer

		for (int i = 1; i < m_layer_objects.size(); i++) {
			for (int j = 0; j < m_layer_objects[i].size(); j++) {
				Object& obj = m_layer_objects[i][j];
				obj.obj_id = k++;

				obj_layer_map[obj.obj_id] = i;

				for (auto it = obj.covered_rids.begin(); it != obj.covered_rids.end(); it++) {
					Region& R = m_regions[*it];
					obj.covered_pids.insert(obj.covered_pids.end(), R.m_region_pids.begin(), R.m_region_pids.end());
				}
			}
		}

		vector<int> sample_pids = SamplePixelsInAllRegions();
		vector<PixPassedObjects> pix_passed_objs = GetPixelPassedObjectsFromBottom2Top(sample_pids);

		LayerParameterOptimization LPO(m_input_img->colors, pix_passed_objs, m_layer_objects.size(), obj_layer_map, m_wr, m_wg);
		ObjectParams obj_params = LPO.CalculateLayerObjectParameters();
		m_recon_gamut_loss = LPO.m_recon_gamut_loss;

		vector<MatrixXd> result_params = obj_params.Convert2Mats();
		for (int i = 1; i < m_layer_objects.size(); i++) {
			for (int j = 0; j < m_layer_objects[i].size(); j++) {
				Object& obj = m_layer_objects[i][j];
				obj.param = result_params[obj.obj_id];
			}
		}
	}

	void CalculateTotalLoss(string error_path = "", int id = 0) {
		//1. average region covering layers
		double region_cover_cnt = 0;
		for (int i = 1; i < m_regions.size(); i++) {
			for (int j = 0; j < m_layer_objects.size(); j++) {
				Vec2i pos = FindObjContainsRegion(i, j);
				if (pos[0] != -1) region_cover_cnt++;
			}
		}
		m_ave_region_layer_cnt = region_cover_cnt / (m_regions.size() - 1);

		//2. bigger layer over smaller layer
		vector<Object> all_objects;
		for (int i = 0; i < m_layer_objects.size(); i++) {
			for (int j = 0; j < m_layer_objects[i].size(); j++)
				all_objects.push_back(m_layer_objects[i][j]);
		}

		m_big_over_small_cnt = 0;
		for (int i = 0; i < all_objects.size() - 1; i++) {
			Object& obj_1 = all_objects[i];
			for (int j = i + 1; j < all_objects.size(); j++) {
				Object& obj_2 = all_objects[j];
				if (obj_1.covered_pids.size() < obj_2.covered_pids.size())
					m_big_over_small_cnt += obj_1.IsOverlapWith(obj_2);
			}
		}

		// total loss
		m_total_loss = m_recon_gamut_loss + m_wc * m_big_over_small_cnt;
	}

	//============================================================================================================================
	void GenerateResultingLayers() {
		cv::Mat chessboard = GetChessboard(m_input_img->h, m_input_img->w);
		m_layer_imgs.clear();
		m_layer_imgs.resize(m_layer_objects.size());

		//cout << "\nLinear gradient parameters======================================" << endl;
		for (int i = 1; i < m_layer_objects.size(); i++) {
			cv::Mat layer_img = chessboard.clone();
			vector<Object>& objs = m_layer_objects[i];

			//cout << "Layer " << i << "==========================" << endl;
			for (int j = 0; j < objs.size(); j++) {
				MatrixXd layer_param = objs[j].param;
				//cout << "Object " << j + 1<< "'s param:" << endl;
				//cout << layer_param << endl;

				for (auto it = objs[j].covered_rids.begin(); it != objs[j].covered_rids.end(); it++) {
					vector<int> pids = m_regions[*it].m_region_pids;
					//#pragma omp parallel for
					for (int k = 0; k < pids.size(); k++) {
						int pid = pids[k];
						int r = pid / m_input_img->w;
						int c = pid % m_input_img->w;
						double x = pid / m_input_img->w * 1.0 / (m_input_img->h - 1);
						double y = pid % m_input_img->w * 1.0 / (m_input_img->w - 1);
						MatrixXd pix_pos_mat(1, 3); pix_pos_mat << x, y, 1;

						double pix_alpha = (pix_pos_mat * layer_param.col(3))(0, 0);
						MatrixXd top_color = pix_pos_mat * layer_param.leftCols(3);
						MatrixXd bot_color = MatrixXd::Ones(1, 3);
						Vec3b cv_bot_color = chessboard.at<Vec3b>(r, c);
						bot_color << cv_bot_color[0] / 255.0, cv_bot_color[1] / 255.0, cv_bot_color[2] / 255.0;
						MatrixXd result_color = pix_alpha * top_color + (1 - pix_alpha) * bot_color;


						layer_img.at<cv::Vec3b>(r, c)[2] = clamp(result_color(0, 0), 0.0, 1.0) * 255;
						layer_img.at<cv::Vec3b>(r, c)[1] = clamp(result_color(0, 1), 0.0, 1.0) * 255;
						layer_img.at<cv::Vec3b>(r, c)[0] = clamp(result_color(0, 2), 0.0, 1.0) * 255;
					}
				}
			}
			m_layer_imgs[i] = layer_img;
		}
	}

	cv::Mat ReconstructImageWithLayers() {
		cv::Mat recon_img(m_input_img->h, m_input_img->w, CV_8UC3, Scalar(255, 255, 255));
		for (int i = 1; i < m_layer_objects.size(); i++) {
			for (int j = 0; j < m_layer_objects[i].size(); j++) {
				Object obj = m_layer_objects[i][j];
				vector<int> obj_pids = obj.covered_pids;
				MatrixXd obj_param = obj.param;
				//#pragma omp parallel for
				for (int k = 0; k < obj.covered_pids.size(); k++) {
					int pid = obj.covered_pids[k];
					int r = pid / m_input_img->w;
					int c = pid % m_input_img->w;
					double x = pid / m_input_img->w * 1.0 / (m_input_img->h - 1);
					double y = pid % m_input_img->w * 1.0 / (m_input_img->w - 1);
					MatrixXd pix_pos_mat(1, 3); pix_pos_mat << x, y, 1;

					MatrixXd bottom_color = MatrixXd::Ones(1, 3);
					bottom_color << recon_img.at<cv::Vec3b>(r, c)[2] / 255.0,
						recon_img.at<cv::Vec3b>(r, c)[1] / 255.0,
						recon_img.at<cv::Vec3b>(r, c)[0] / 255.0;
					double  pix_alpha = (pix_pos_mat * obj_param.col(3))(0, 0);
					MatrixXd pix_color = pix_pos_mat * obj_param.leftCols(3);
					MatrixXd blend_color = pix_alpha * pix_color + (1 - pix_alpha) * bottom_color;

					recon_img.at<cv::Vec3b>(r, c)[2] = clamp(blend_color(0, 0), 0.0, 1.0) * 255;
					recon_img.at<cv::Vec3b>(r, c)[1] = clamp(blend_color(0, 1), 0.0, 1.0) * 255;
					recon_img.at<cv::Vec3b>(r, c)[0] = clamp(blend_color(0, 2), 0.0, 1.0) * 255;
				}
			}
		}
		return recon_img;
	}

	void SaveReconstructedImageAndLayers(string layer_path) {
		int pos = layer_path.rfind('/');
		string parent_dir = layer_path.substr(0, pos);
		string grand_dir = parent_dir.substr(0, parent_dir.rfind('/'));
		if (_access(grand_dir.c_str(), 0) == -1)  int ret = _mkdir(grand_dir.c_str());
		if (_access(parent_dir.c_str(), 0) == -1) int ret = _mkdir(parent_dir.c_str());
		string str_id = layer_path.substr(pos + 1, layer_path.size());

		m_reconstructed_img = ReconstructImageWithLayers();

		Mat result;
		m_layer_imgs[0] = m_reconstructed_img;
		hconcat(m_layer_imgs, result);
		cv::imwrite(layer_path + ".png", result);
	}

	void OutputLayerMask(string layer_mask_path) {
		int pos = layer_mask_path.rfind('/');
		string parent_dir = layer_mask_path.substr(0, pos);
		if (_access(parent_dir.c_str(), 0) == -1) int ret = _mkdir(parent_dir.c_str());

		for (int i = 1; i < m_layer_objects.size(); i++) {
			vector<Object>& objs = m_layer_objects[i];
			for (int j = 0; j < objs.size(); j++) {
				cv::Mat layer_mask(m_input_img->h, m_input_img->w, CV_8UC3, Scalar(0, 0, 0));

				for (auto it = objs[j].covered_rids.begin(); it != objs[j].covered_rids.end(); it++) {
					vector<int> pids = m_regions[*it].m_region_pids;
					//#pragma omp parallel for
					for (int k = 0; k < pids.size(); k++) {
						int pid = pids[k];
						int r = pid / m_input_img->w;
						int c = pid % m_input_img->w;
						layer_mask.at<cv::Vec3b>(r, c)[2] = 255;
						layer_mask.at<cv::Vec3b>(r, c)[1] = 255;
						layer_mask.at<cv::Vec3b>(r, c)[0] = 255;
					}
				}
				string path = layer_mask_path + "/mask_" + to_string(i) + "_" + to_string(j + 1) + ".png";
				imwrite(path, layer_mask);
			}
		}
	}

	void OutputJsonForPresentation(string json_path) {
		int pos = json_path.rfind('/');
		string parent_dir = json_path.substr(0, pos);
		int pos1 = parent_dir.rfind('/');
		string parent_parent_dir = parent_dir.substr(0, pos1);
		if (_access(parent_parent_dir.c_str(), 0) == -1) int ret = _mkdir(parent_parent_dir.c_str());
		if (_access(parent_dir.c_str(), 0) == -1) int ret = _mkdir(parent_dir.c_str());

		ofstream of(json_path);
		of << "{" << endl;
		of << "\t\"width\": " << m_input_img->w << "," << endl;
		of << "\t\"height\": " << m_input_img->h << "," << endl;
		of << "\t\"linearGradients\": [" << endl;

		for (int i = 1; i < m_layer_objects.size(); i++) {
			of << "\t\t{" << endl;
			for (int j = 0; j < m_layer_objects[i].size(); j++) {
				Object& obj = m_layer_objects[i][j];
				MatrixXd mat = obj.param.transpose();

				of << "\t\t\"#" << to_string(i) + "-" + to_string(j + 1) + "\": [" << endl;
				of << "\t\t\t\t\t[" << mat(0, 0) << "," << mat(0, 1) << "," << mat(0, 2) << "]," << endl;
				of << "\t\t\t\t\t[" << mat(1, 0) << "," << mat(1, 1) << "," << mat(1, 2) << "]," << endl;
				of << "\t\t\t\t\t[" << mat(2, 0) << "," << mat(2, 1) << "," << mat(2, 2) << "]," << endl;
				of << "\t\t\t\t\t[" << mat(3, 0) << "," << mat(3, 1) << "," << mat(3, 2) << "]" << endl;

				if (j != m_layer_objects[i].size() - 1)
					of << "\t\t\t\t]," << endl;
				else
					of << "\t\t\t\t]" << endl;
			}
			if (i != m_layer_objects.size() - 1)
				of << "\t\t}," << endl;
			else
				of << "\t\t}" << endl;
		}

		of << "\t]" << endl;
		of << "}" << endl;
	}
};