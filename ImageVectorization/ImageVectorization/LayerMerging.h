#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include "Region.h"
#include "Tree.h"
#include "Utility.h"
#include "Graph.h"
#include "Object.h"

using namespace std;
using namespace cv;

class LayerMerging {
private:
	vector<Region> m_regions;
	vector<Vec4i> m_xjunctions;
	Graph m_reg_support_dag;
	vector<vector<Object>> m_layer_objects; //a layer may contain several objects

public:
	LayerMerging() {}
	LayerMerging(vector<Region> regions, Tree tree) {
		m_regions = regions;
		m_xjunctions = tree.m_meeted_xjconstrains;
		m_reg_support_dag = Graph(tree.m_vn, tree.GetEdgeList());

		m_layer_objects.resize(tree.GetTreeDepth() + 1);
		for (int i = 0; i < tree.m_vn; i++) {
			int depth = tree.GetDepthOf(i);
			set<int> rids; rids.insert(i);
			m_layer_objects[depth].push_back(Object(m_regions[i].m_region_id, depth, rids));
		}
	}

	void Release() {
		m_regions.swap(vector<Region>());
	}

	vector<vector<Object>> GetLayerObject() {
		return m_layer_objects;
	}

	//find the object that contains region(rid) in layer(layer_id)
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
		return Vec2i(-1, -1); // not found
	}

	//find the object that contains 2 regions(rid1 & rid2) concurrently
	Vec2i FindObjContainsRegionPair(int rid1, int rid2) {
		for (int i = 1; i < m_layer_objects.size(); i++) {
			for (int j = 0; j < m_layer_objects[i].size(); j++) {
				Object& obj = m_layer_objects[i][j];
				if (obj.covered_rids.find(rid1) != obj.covered_rids.end() &&
					obj.covered_rids.find(rid2) != obj.covered_rids.end())
					return Vec2i(i, j);
			}
		}
		return Vec2i(-1, -1); // not found
	}

	//check if xj has been merged before when merge other x-junctions
	bool HasBeenMergedBefore(Vec4i xj) {
		Vec2i pos1 = FindObjContainsRegionPair(xj[1], xj[3]);
		if (pos1[0] == -1) return false;

		//if it has been merged, we need add an implicit edge
		if (m_reg_support_dag.ExistEdge(xj[0], xj[1])) m_reg_support_dag.add_edge(xj[2], xj[3]);
		else if (m_reg_support_dag.ExistEdge(xj[2], xj[3])) m_reg_support_dag.add_edge(xj[0], xj[1]);

		return true;
	}

	//v1 -> v2, v3 -> v4, check if there are two regions to be merged in xj
	bool ExistRegionsToMerge(Vec4i xj, Vec2i& to_merge_rids) {
		int rid1 = xj[0], rid2 = xj[1], rid3 = xj[2], rid4 = xj[3];
		if (m_reg_support_dag.ExistEdge(rid1, rid2) && m_reg_support_dag.ExistEdge(rid3, rid4)) {
			to_merge_rids = Vec2i(rid2, rid4);
			return true;
		}
		return false;
	}

	void DownObjectInTheGraph(int rid, int r1_layer_id) {
		vector<int> r_sons = m_reg_support_dag.GetSucceessorsOf(rid);
		if (r_sons.empty()) return;

		for (int son_rid : r_sons) {
			Vec2i son_obj_pos = FindObjContainsRegion(son_rid);
			int son_obj_layer_id = son_obj_pos[0];
			int son_obj_id = son_obj_pos[1];
			Object son_obj = m_layer_objects[son_obj_layer_id][son_obj_id];

			//has been moved to next layer
			if (son_obj_layer_id == r1_layer_id + 1)
				break;

			int next_layer_id = son_obj_layer_id + 1;
			//if next layer id > current max layer depth, add a new layer
			if (m_layer_objects.size() - 1 < next_layer_id)
				m_layer_objects.push_back(vector<Object>());

			//add to next layer
			m_layer_objects[next_layer_id].push_back(son_obj);
			m_layer_objects[son_obj_layer_id].erase(m_layer_objects[son_obj_layer_id].begin() + son_obj_id);

			//recursively down son regions
			for (int new_son_id : son_obj.covered_rids)
				DownObjectInTheGraph(new_son_id, next_layer_id);
		}
	}

	void LiftRegion(int r1_id, int r1_layer_id, int r2_id, int r2_layer_id) {
		int obj_id_of_r1 = FindObjContainsRegion(r1_id, r1_layer_id)[1];
		int obj_id_of_r2 = FindObjContainsRegion(r2_id, r2_layer_id)[1];
		Object& obj1 = m_layer_objects[r1_layer_id][obj_id_of_r1];
		Object& obj2 = m_layer_objects[r2_layer_id][obj_id_of_r2];

		//merge lower obj1 to high higher obj2
		if (obj_id_of_r1 != -1) {
			obj2.covered_rids.insert(obj1.covered_rids.begin(), obj1.covered_rids.end());
			m_layer_objects[r1_layer_id].erase(m_layer_objects[r1_layer_id].begin() + obj_id_of_r1);

			//if r1 has son nodes and in lower layer, the son nodes should down to next layer
			if (r1_layer_id < r2_layer_id)
				DownObjectInTheGraph(r1_id, r2_layer_id);
		}
		//r1 may be merged into other obj before,and cannot found in r1_layer_id, thus find it in r2_layer_id
		else {
			obj_id_of_r1 = FindObjContainsRegion(r1_id, r2_layer_id)[1];
			Object& obj1 = m_layer_objects[r2_layer_id][obj_id_of_r1];

			obj2.covered_rids.insert(obj1.covered_rids.begin(), obj1.covered_rids.end());
			m_layer_objects[r2_layer_id].erase(m_layer_objects[r2_layer_id].begin() + obj_id_of_r1);
		}
	}

	bool MergeRegionsInAnXjunction(Vec4i xj) {
		if (HasBeenMergedBefore(xj))
			return true;

		Vec2i to_merge_rids;
		if (!ExistRegionsToMerge(xj, to_merge_rids))
			return false;

		int r1_id = to_merge_rids[0], r1_depth = FindObjContainsRegion(r1_id)[0];
		int r2_id = to_merge_rids[1], r2_depth = FindObjContainsRegion(r2_id)[0];
		if (r1_depth < r2_depth)
			LiftRegion(r1_id, r1_depth, r2_id, r2_depth);
		else
			LiftRegion(r2_id, r2_depth, r1_id, r1_depth);

		return true;
	}

	vector<int> IncludeChildrenRegion(int u) {
		//top layer's object
		if (m_reg_support_dag.HasNoSuccessorsAt(u)) {
			vector<int> region_ids(1, u);
			return region_ids;
		}

		//recursively include descendants' region
		vector<int> obj_region_ids(1, m_regions[u].m_region_id);
		vector<int> u_succ_verts = m_reg_support_dag.GetSucceessorsOf(u);
		for (int i = 0; i < u_succ_verts.size(); i++) {
			int v = u_succ_verts[i];
			vector<int> rids = IncludeChildrenRegion(v);
			obj_region_ids.insert(obj_region_ids.end(), rids.begin(), rids.end());
		}

		//store the regions into u's object
		if (u != 0) {
			Vec2i pos = FindObjContainsRegion(u);
			int layer_id = pos[0], obj_id = pos[1];
			Object& obj = m_layer_objects[layer_id][obj_id];
			obj.covered_rids.insert(obj_region_ids.begin(), obj_region_ids.end());
		}

		return obj_region_ids;
	}

	void MergeUnderneathObjsThatSupportTheSameObj() {
		for (int i = 1; i < m_layer_objects.size(); i++) {
			vector<Object>& objects = m_layer_objects[i];
			for (int j = objects.size() - 2; j >= 0; j--) {

				//check if there are same regions both in obj_j and obj_k (k > j)
				for (int k = j + 1; k < objects.size(); k++) {
					//their region union of these two objs
					set<int> union_set = objects[j].covered_rids;
					union_set.insert(objects[k].covered_rids.begin(), objects[k].covered_rids.end());

					//if union_set.size() < |obj_j| + |obj_k|, these two obj have the same regions
					if (union_set.size() < objects[j].covered_rids.size() + objects[k].covered_rids.size()) {
						objects[j].covered_rids.insert(objects[k].covered_rids.begin(), objects[k].covered_rids.end());
						m_layer_objects[i].erase(m_layer_objects[i].begin() + k);
					}
				}
			}
		}
	}

	//peform layer merging process
	void DetermineLayerRange() {
		set<int> meeted_xj_ids; int it_n = 0;
		while (meeted_xj_ids.size() < m_xjunctions.size()) {
			for (int i = 0; i < m_xjunctions.size(); i++) {
				//the i-th x-junction has been merged
				if (meeted_xj_ids.find(i) != meeted_xj_ids.end()) continue;
				//merge the current x-junction, may be failed, if so, skip this one first.
				if (MergeRegionsInAnXjunction(m_xjunctions[i]))
					meeted_xj_ids.insert(i);
			}
			if (++it_n > m_xjunctions.size()) break;
		}
		IncludeChildrenRegion(0);
		MergeUnderneathObjsThatSupportTheSameObj();
	}

	//check 2 merged layer configutaions are the same one
	bool LayerConfigurationEquals(LayerMerging& Lm) {
		if (m_layer_objects.size() != Lm.m_layer_objects.size())
			return false;
		for (int i = 0; i < m_layer_objects.size(); i++) {
			if (m_layer_objects[i].size() != Lm.m_layer_objects[i].size())
				return false;
			for (int j = 0; j < m_layer_objects[i].size(); j++) {
				if (m_layer_objects[i][j].covered_rids.size() != Lm.m_layer_objects[i][j].covered_rids.size())
					return false;
				vector<int> rids_1(m_layer_objects[i][j].covered_rids.begin(), m_layer_objects[i][j].covered_rids.end());
				vector<int> rids_2(Lm.m_layer_objects[i][j].covered_rids.begin(), Lm.m_layer_objects[i][j].covered_rids.end());
				for (int k = 0; k < rids_1.size(); k++)
					if (rids_1[k] != rids_2[k])
						return false;
			}
		}
		return true;
	}
};