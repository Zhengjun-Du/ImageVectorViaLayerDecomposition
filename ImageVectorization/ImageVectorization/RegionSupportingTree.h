#pragma once

#include <vector>
#include <queue>
#include <map>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "io.h"
#include <direct.h>
#include "Region.h"
#include "Graph.h"
#include "Tree.h"
#include "Xjunction.h"
#include "Utility.h"

using namespace std;
using namespace cv;

class RegionSupportingTree {
private:
	ImageObj		m_ori_img;						//input image
	vector<Vec2i>	m_adj_region_graph_edges;		//adj region graph's edges
	Xjunction		m_xjunction;					//x-junctions in the region image
	vector<int>		m_possible_bottom_rids;
	vector<int>		m_pix_rid;						//record pix's region

public:
	vector<Region>	m_regions;						//regions in the input image
	vector<Tree>	m_valid_region_support_trees;	//region order tree

public:
	RegionSupportingTree() {
		srand(time(0));
	}

	RegionSupportingTree(ImageObj& ori_img, vector<Region>& regs, Xjunction& xj, vector<int>& possible_bot_rids) {
		m_ori_img = ori_img;
		m_regions = regs;
		m_xjunction = xj;
		m_possible_bottom_rids = possible_bot_rids;
		srand(time(0));
	}

	int GetSimplifiedEdgeSize() {
		return m_adj_region_graph_edges.size();
	}


	void RemoveEdges(vector<Vec2i>& to_remove_edges) {
		for (int i = m_adj_region_graph_edges.size() - 1; i >= 0; i--) {
			Vec2i e((int)m_adj_region_graph_edges[i][0], (int)m_adj_region_graph_edges[i][1]);
			for (Vec2i e1 : to_remove_edges) {
				if (e == e1) m_adj_region_graph_edges.erase(m_adj_region_graph_edges.begin() + i);
			}
		}
	}

	void RemoveWeakConnectionEdge(vector<vector<int>> whoSupportRegion) {
		vector<Vec2i> to_remove_edges;
		for (int i = 1; i < m_regions.size(); i++) {

			if (whoSupportRegion[i].size() < 2) continue;
			int max_share_boundary_length = -1;
			vector<int> rids = whoSupportRegion[i];
			for (int rid : rids) {
				//if (rid == 0) continue;
				max_share_boundary_length = max(max_share_boundary_length, m_regions[i].m_shared_pixcnt_with_reg[rid]);
			}

			for (int rid : rids) {
				//if (rid == 0) continue;
				if (m_regions[i].m_shared_pixcnt_with_reg[rid] < max_share_boundary_length * 0.4)
					if (!m_xjunction.ContainsRegions(rid, i))
						to_remove_edges.push_back(Vec2i(rid, i));
			}
		}
		RemoveEdges(to_remove_edges);
	}

	//if a region Ra in an X-junction X is supported by another region Rb in X and does not support others in X,
	//if some other regions not in X support Ra, remove these connections.
	void RemoveEdgeWithXjunctionConstrain() {
		vector<vector<int>> whoSupportRegion(m_regions.size());
		vector<vector<int>> SupportRegionsOf(m_regions.size());

		for (Vec2i e : m_adj_region_graph_edges) {
			SupportRegionsOf[e[0]].push_back(e[1]);
			whoSupportRegion[e[1]].push_back(e[0]);
		}

		vector<Vec2i> to_remove_edges;
		for (int ri_id = 1; ri_id < SupportRegionsOf.size(); ri_id++) {
			if (!SupportRegionsOf[ri_id].empty()) continue; //this region 
			if (!m_xjunction.ContainsRegion(ri_id)) continue;

			for (int j = 0; j < whoSupportRegion[ri_id].size(); j++) {
				int rj_id = whoSupportRegion[ri_id][j];
				if (!m_xjunction.ContainsRegions(rj_id, ri_id))
					to_remove_edges.push_back(Vec2i(rj_id, ri_id));
			}
		}
		RemoveEdges(to_remove_edges);
	}

	void BuildAdjacentRegionGraph() {
		m_adj_region_graph_edges.clear();
		vector<vector<int>> whoSupportRegion(m_regions.size());

		//surrounding rule: a surrounded region cannot be supported by r0, nor support the region that surrounds it
		for (int rid : m_possible_bottom_rids) {
			m_adj_region_graph_edges.push_back(Vec2i(0, rid));
			whoSupportRegion[rid].push_back(0);
		}

		for (int i = 1; i < m_regions.size(); i++) {
			Region& R = m_regions[i];
			for (auto it = R.m_adj_regions.begin(); it != R.m_adj_regions.end(); it++) {
				Region& adjR = m_regions[*it];

				//size rule: remove very smaller region supports larger region
				bool support = R.CouldSupport(adjR);
				if (support) {
					whoSupportRegion[adjR.m_region_id].push_back(R.m_region_id);
					m_adj_region_graph_edges.push_back(Vec2i(R.m_region_id, adjR.m_region_id));
				}
			}
		}
		//adjacency connection strength rule, remove weak connection
		RemoveWeakConnectionEdge(whoSupportRegion);

		//remove some invalid connection with X-junction
		RemoveEdgeWithXjunctionConstrain();

		HandleIsloatedNodes();
		
		int eid = 0;
		sort(m_adj_region_graph_edges.begin(), m_adj_region_graph_edges.end(), edgecmp);
		cout << "the simplified adjacent graph has " << m_adj_region_graph_edges.size() << " edges are : " << endl;
		for (Vec2i e : m_adj_region_graph_edges)
			cout << eid++ << ": " << e[0] << "," << e[1] << endl;
	}

	void HandleIsloatedNodes() {
		for (int i = 1; i < m_regions.size(); i++) {
			bool hasParent = false;
			for (int j = 0; j < m_adj_region_graph_edges.size(); j++) {
				if (m_adj_region_graph_edges[j][1] == i) {
					hasParent = true;
					break;
				}
			}
			if (hasParent == false)
				m_adj_region_graph_edges.push_back(Vec2i(0, m_regions[i].m_region_id));
		}
	}

	void GetValidRegionSupportingTrees() {
		//1. Get all valid region supporting trees
		int tree_depth = 3;
		vector<vector<Vec2i>> all_spanning_trees;
		while (1) {
			Graph Gx(m_regions.size(), m_adj_region_graph_edges, tree_depth, m_regions.size() / 3);
			Gx.SetXjunctions(m_xjunction.Conver2Arrrint4());
			all_spanning_trees = Gx.GetAllSpanningTrees();
			if (!all_spanning_trees.empty() || tree_depth >= 8) break;
			tree_depth++;
		}

		tree_depth = 3;
		while (all_spanning_trees.size() < 2) {
			Graph Gx(m_regions.size(), m_adj_region_graph_edges, tree_depth, m_regions.size() / 2);
			Gx.SetXjunctions(m_xjunction.Conver2Arrrint4());
			all_spanning_trees = Gx.GetAllSpanningTrees();
			if (all_spanning_trees.size() > 1 || tree_depth >= 8)break;
			tree_depth++;
		}
		cout << endl << "all spanning tree cnt: " << all_spanning_trees.size() << endl;
		
		int k = 0;
		for (int i = 0; i < all_spanning_trees.size(); i++) {
			Tree tree_(m_regions.size(), all_spanning_trees[i]);
			if (tree_.SatisfyAllXjunctionConstrains(m_xjunction.m_possible_configs)) {
				tree_.m_id = k++;
				m_valid_region_support_trees.push_back(tree_);
			}
		}
		cout << endl << "valid spanning tree cnt: " << m_valid_region_support_trees.size() << endl;
	}
};