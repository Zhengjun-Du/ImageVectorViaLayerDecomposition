#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class Tree {
public:
	vector<Vec4i> m_meeted_xjconstrains;	//Vec4i:bot1 top1 bot2 top2
	vector<vector<int>> m_sons_of;			//store node's son nodes
	int m_vn, m_id;

private:
	vector<Vec2i> m_edges;
	map<int, int> m_son_parent_map;

public:
	Tree() :m_id(0) { }
	Tree(int vn, vector<Vec2i> tree_edges) :m_edges(tree_edges), m_vn(vn), m_id(0) {
		m_sons_of.resize(m_vn);
		for (int i = 0; i < tree_edges.size(); i++) {
			int u = tree_edges[i][0];
			int v = tree_edges[i][1];
			m_sons_of[u].push_back(v);
		}

		for (int i = 0; i < m_edges.size(); i++)
			m_son_parent_map[m_edges[i][1]] = m_edges[i][0];
	}

	bool IsExist2Edges(int& v1, int& v2, int& v3, int& v4) {
		bool exist = false;
		for (int j = 0; j < m_edges.size(); j++) {
			if (m_edges[j][0] == v1 && m_edges[j][1] == v2) {
				exist = true;
				break;
			}
		}
		if (!exist) return false;
		for (int j = 0; j < m_edges.size(); j++)
			if (m_edges[j][0] == v3 && m_edges[j][1] == v4)
				return true;
		return false;
	}

	//check if there are 2 edges with the same direction in the x-junction
	bool SatisfySingleXjunctionConstrain(vector<Vec4i> xj_constrain, Vec4i& ans) {
		for (int i = 0; i < xj_constrain.size(); i++) {
			int bot1 = xj_constrain[i][0], top1 = xj_constrain[i][1];
			int bot2 = xj_constrain[i][2], top2 = xj_constrain[i][3];
			if (IsExist2Edges(bot1, top1, bot2, top2)) {
				ans = xj_constrain[i];
				return true;
			}
		}
		return false;
	}

	bool SatisfyAllXjunctionConstrains(vector<vector<Vec4i>> all_xjunctions) {
		if (all_xjunctions.size() == 0) return true;
		vector<vector<Vec4i>> unmeeted_xjunctions;
		Vec4i ans;

		//get the meeted x-junctions and unmeeted x-junctions initially
		for (int i = 0; i < all_xjunctions.size(); i++) {
			bool satisfy = SatisfySingleXjunctionConstrain(all_xjunctions[i], ans);
			if (satisfy) m_meeted_xjconstrains.push_back(ans);
			else unmeeted_xjunctions.push_back(all_xjunctions[i]);
		}

		//for all unmeeted xjunctions
		for (vector<Vec4i> u_xjs : unmeeted_xjunctions) {
			//for all 4 possible configuration in xjunction_constrains
			for (Vec4i xj1 : u_xjs) {
				//if xj1 is consistent with meeted xj2
				bool consistent_with_meeted_xj = false;
				for (Vec4i xj2 : m_meeted_xjconstrains) {

					//get the different region ids between xj1 and xj2
					vector<int> diff_region_ids;
					for (int q = 0; q < 4; q++) {
						if (xj1[q] != xj2[q])
							diff_region_ids.push_back(q);
					}

					//if only one region different, and with same depth
					if (diff_region_ids.size() == 1) {
						int v1_id = xj1[diff_region_ids[0]];
						int v2_id = xj2[diff_region_ids[0]];
						if (GetDepthOf(v1_id) == GetDepthOf(v2_id)) {
							consistent_with_meeted_xj = true;
							m_meeted_xjconstrains.push_back(xj1);
							break;
						}
					}
				}
				if (consistent_with_meeted_xj)
					break;
			}
		}
		return m_meeted_xjconstrains.size() == all_xjunctions.size();
	}

	//Draw the spanning tree for debug=============================
	void Draw(string path = "", int rank = 0, double error = 0) {
		int depth = GetTreeDepth() + 1;
		vector<vector<int>> layer_nodes(depth);
		layer_nodes[0].push_back(0);
		for (int i = 1; i < m_vn; i++) {
			int depth = GetDepthOf(i);
			layer_nodes[depth].push_back(i);
		}

		int h = 400, w = 400;
		Mat img = Mat(h, w, CV_8UC3, Scalar(255, 255, 255));
		int h_space = 400 / 8;
		int w_space = 300 / 4;

		int max_node_in_a_layer = 0;
		vector<Vec2i> node_poses(m_vn);
		for (int i = 0; i < layer_nodes.size(); i++) {
			if (max_node_in_a_layer < layer_nodes[i].size())
				max_node_in_a_layer = layer_nodes[i].size();

			for (int j = 0; j < layer_nodes[i].size(); j++) {
				int vid = layer_nodes[i][j];
				int x_coord = (i + 1) * w_space;
				int y_coord = (j + 1) * h_space;
				node_poses[vid] = Vec2i(x_coord, y_coord);
			}
		}

		for (auto it = m_son_parent_map.begin(); it != m_son_parent_map.end(); it++) {
			int v1 = it->second; int v1x = node_poses[v1][0], v1y = node_poses[v1][1];
			int v2 = it->first;  int v2x = node_poses[v2][0], v2y = node_poses[v2][1];

			line(img, Point(v1x, v1y), Point(v2x, v2y), Scalar(0, 0, 0), 1);
		}

		for (int i = 0; i < node_poses.size(); i++) {
			int x = node_poses[i][0];
			int y = node_poses[i][1];
			circle(img, Point(x, y), 20, Scalar(255, 255, 255), -1);
			circle(img, Point(x, y), 20, Scalar(0, 0, 0), 1);
			putText(img, to_string(i), Point(x - 9, y + 9), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0));
		}

		//int x = node_poses[0][0];
		//int y = (max_node_in_a_layer + 1) * h_space + 50;
		//putText(img, "rank " + to_string(rank) + ", err=" + to_string(error), Point(x - 26, y), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0));

 		imwrite(path, img);
	}

	//for debug==================================================================
	void OutputEdges(int id) {
		cout << endl << "Tree " << id << " : The tree edges are:=================================" << endl;
		vector<Vec2i> tree_edges = GetEdgeList();
		for (int j = 0; j < tree_edges.size(); j++)
			cout << tree_edges[j][0] << ", " << tree_edges[j][1] << endl;
	}

	int GetDepthOf(int v) {
		if (v == 0) return 0;
		int parent = m_son_parent_map[v];
		int depth = 1;
		while (parent != 0) {
			parent = m_son_parent_map[parent];
			depth++;
		}
		return depth;
	}

	int GetTreeDepth() {
		int tree_depth = 1;
		for (int v = 0; v < m_sons_of.size(); v++) {
			int parent = m_son_parent_map[v];
			int depth = 1;
			while (parent != 0) {
				parent = m_son_parent_map[parent];
				depth++;
			}
			tree_depth = max(tree_depth, depth);
		}
		return tree_depth;
	}

	bool IsContainEdges(vector<Vec2i> edges) {
		for (int i = 0; i < edges.size(); i++) {
			int v1 = edges[i][0];
			int v2 = edges[i][1];
			if (v1 != m_son_parent_map[v2])
				return false;
		}
		return true;
	}
	vector<Vec2i>& GetEdgeList() {
		return m_edges; 
	}
};