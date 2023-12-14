#pragma once

#include<iostream>
#include<vector>
#include<algorithm>
#include<set>
#include<utility>
#include<queue>
#include<numeric>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

static bool vec3icmp_(Vec2i v1, Vec2i v2) {
    if (v1[0] != v2[0]) return v1[0] < v2[0];
    else return v1[1] < v2[1];
}

struct edge {
    int u, v;
    double w;
    int eid;
};

class Graph {
private:
    int  n, max_node_cnt_in_L1;
    vector<edge> edges;
    vector<vector<int>> edge_p;
    vector<bool> nec;
    vector<vector<Vec2i>> trees;
    vector<array<int, 4>> xjs;
    vector<vector<int>> xj_p;
    int max_tree_depth;

public:
    Graph() {}
    Graph(int const _n, 
        vector<Vec2i> graph_edges = vector<Vec2i>(),
        int tree_depth = 4,
        int L1_cnt = 0) {
        n = _n;
        edge_p.resize(_n);
        xj_p.resize(_n);
        max_node_cnt_in_L1 = L1_cnt;

        for (int i = 0; i < graph_edges.size(); i++) {
            Vec2i e = graph_edges[i];
            add_edge(e[0], e[1]);
        }

        max_tree_depth = tree_depth;
    }

    void SetXjunctions(vector<array<int, 4>> const& _xjs) {
        xjs.clear();
        for (int u = 0;u < n;++u)
            xj_p[u].clear();
        for (auto const& xj : _xjs)
            add_x_junction(xj);
    }

    void add_x_junction(array<int, 4> const& x_junction) {
        for (auto u : x_junction)
            xj_p[u].push_back(xjs.size());
        xjs.push_back(x_junction);
    }

    void add_edge(int u, int v, double w = 0) {
        if (!ExistEdge(u, v)) {
            int eid = edges.size();
            edge_p[u].push_back(eid);
            edges.push_back({ u,v,w,eid });
        }
    }

    bool x_junction_test_depth(array<int, 4> depth) {
        sort(depth.begin(), depth.end());
        return !((depth[0] == depth[1] || depth[2] == depth[3]) && depth[1] == depth[2]);
    }

    bool x_junction_test_edge(array<int, 4> const& xj, set<pair<int, int>> const& s) {
        array<array<int, 2>, 2> p;
        p[0][0] = xj[0], p[0][1] = xj[1], p[1][0] = xj[3], p[1][1] = xj[2];
        for (int i = 0;i < 2;++i) {
            if (s.count({ p[i][0],p[i][1] }) && s.count({ p[i ^ 1][1],p[i ^ 1][0] }))
                return false;
            if (s.count({ p[0][i],p[1][i] }) && s.count({ p[1][i ^ 1],p[0][i ^ 1] }))
                return false;
            if (s.count({ p[i][0],p[i ^ 1][1] }) || s.count({ p[i ^ 1][1],p[i][0] }))
                return false;
        }
        return true;
    }

    bool x_junction_test(array<int, 4> const& xj, vector<int> const& depth, set<pair<int, int>> const& s) {
        if (!x_junction_test_depth({ depth[xj[0]],depth[xj[1]],depth[xj[2]],depth[xj[3]] }))
            return false;
        if (!x_junction_test_edge(xj, s))
            return false;
        return true;
    }

    bool x_junction_test(int const u, vector<int> const& depth, vector<int> const& xj_cnt, vector<int> const& choices) {
        set<pair<int, int>> s;
        for (auto const& eid : choices)
            s.insert({ edges[eid].u,edges[eid].v });
        for (auto i : xj_p[u])
            if (xj_cnt[i] >= 3 && !x_junction_test(xjs[i], depth, s))
                return false;
        return true;
    }

    void enum_tree(int const u, vector<int>& cand, int const left, vector<int>& depth, int cnt, vector<int>& xj_cnt, int cnt2, vector<int>& choices) {
        if (++cnt == n) {
            vector<Vec2i> tree;
            for (auto const& eid : choices)
                tree.push_back(Vec2i(edges[eid].u, edges[eid].v));
            sort(tree.begin(), tree.end(), vec3icmp_);
            trees.push_back(tree);
            return;
        }
        int from = cand.size();
        for (auto const eid : edge_p[u])
            if (!depth[edges[eid].v])
                cand.push_back(eid);

        for (int k = left; k < cand.size(); ++k) {
            auto const& e = edges[cand[k]];
            if (!depth[e.v] && depth[e.u] <= max_tree_depth && (cnt2 < max_node_cnt_in_L1 || depth[e.u] != 1)) {
                choices.push_back(cand[k]);
                depth[e.v] = depth[e.u] + 1;
                for (auto i : xj_p[e.v])
                    ++xj_cnt[i];
                if (x_junction_test(e.v, depth, xj_cnt, choices))
                    enum_tree(e.v, cand, k + 1, depth, cnt, xj_cnt, cnt2 + (depth[e.v] == 2), choices);
                for (auto i : xj_p[e.v])
                    --xj_cnt[i];
                depth[e.v] = 0;
                choices.pop_back();
                if (nec[cand[k]])
                    break;
            }
            else if (nec[cand[k]])
                break;
        }
        cand.erase(cand.begin() + from, cand.end());
    }

    bool check_connectivity(int const root, int const ban_eid) {
        vector<bool> visit(n, 0);
        visit[root] = 1;
        for (queue<int> q({ root });!q.empty();q.pop()) {
            int const u = q.front();
            for (auto const eid : edge_p[u]) {
                if (eid == ban_eid)
                    continue;
                if (!visit[edges[eid].v]) {
                    visit[edges[eid].v] = 1;
                    q.push(edges[eid].v);
                }
            }
        }
        return std::accumulate(visit.begin(), visit.end(), 0) == n;
    }

    void enum_tree(int const root) {
        nec = vector<bool>(edges.size(), 0);
        for (int eid = 0;eid < edges.size();++eid)
            nec[eid] = !check_connectivity(root, eid);
        vector<int> cand(0), choices(0);
        vector<int> depth(n, 0);
        vector<int> xj_cnt(xjs.size(), 0);
        depth[root] = 1;
        enum_tree(root, cand, 0, depth, 0, xj_cnt, 0, choices);
    }

    vector<vector<Vec2i>> GetAllSpanningTrees() {
        enum_tree(0);
        return trees;
    }

    //some others methods===========================================================
    bool HasNoSuccessorsAt(int u) {
        return edge_p[u].empty();
    }

    bool ExistEdge(int u, int v) {
        for (auto const eid : edge_p[u])
            if (edges[eid].v == v)
                return true;
        return false;
    }

    vector<int> GetSucceessorsOf(int u) {
        vector<int> verts;
        for (auto const eid : edge_p[u])
            verts.push_back(edges[eid].v);
        return verts;
    }

    vector<int> GetPrecursorsOf(int u) {
        vector<int> verts;
        for (int i = 0; i < edges.size(); i++)
            if (edges[i].v == u)
                verts.push_back(edges[i].u);
        return verts;
    }
};