#pragma once

#include "Utility.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include "nlopt.h"
#include "Object.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace autodiff;

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorMat;
#define PI 3.141592653

static dual E_recon(Vec2d& pos, vector<int>& oids, Vec3d& real_color, ObjectParams& params, double w_d, int pix_cnt);
static dual E_gamut(Vec2d& pos, vector<int>& oids, ObjectParams& params, double w_g, int pix_cnt);
static double global_loss_function(unsigned n, const double* x, double* grad, void* data);

class LayerParameterOptimization {
public:
	vector<Vec3d> m_img;
	vector<PixPassedObjects> m_pix_covered_objects;
	ObjectParams m_params;
	map<int, int> m_obj_lid_map;
	double m_w_recon, m_w_gamut, m_recon_gamut_loss;

public:
	LayerParameterOptimization(
		vector<Vec3d>& img,
		vector<PixPassedObjects>& pix_covered_objects,
		int layer_cnt,
		map<int, int>& obj_lid_map,
		double w_r = 20,
		double w_g = 10) {

		m_img = img;
		m_pix_covered_objects = pix_covered_objects;
		m_obj_lid_map = obj_lid_map;

		m_w_recon = w_r;
		m_w_gamut = w_g;
	}

	double CalculateLossAndGradientOfPix(int k) {
		int pix_cnt = m_pix_covered_objects.size();

		int pid = m_pix_covered_objects[k].pix_id;
		Vec2d pos = m_pix_covered_objects[k].coord;

		vector<int> covered_objects = m_pix_covered_objects[k].covered_objects;
		Vec3d real_color = m_img[pid];

		dual e_data = E_recon(pos, covered_objects, real_color, m_params, m_w_recon, pix_cnt);
		for (int i = 0; i < covered_objects.size(); i++) {
			int oid = covered_objects[i];
			for (int j = 9 * oid; j < 9 * (oid + 1); j++)
				m_params.gradients[j] += (double)(derivative(E_recon, wrt(m_params.vars[j]), at(pos, covered_objects, real_color, m_params, m_w_recon, pix_cnt)));
		}

		dual e_gamut = E_gamut(pos, covered_objects, m_params, m_w_gamut, pix_cnt);
		for (int i = 0; i < covered_objects.size(); i++) {
			int oid = covered_objects[i];
			for (int j = 9 * oid; j < 9 * (oid + 1); j++)
				m_params.gradients[j] += (double)(derivative(E_gamut, wrt(m_params.vars[j]), at(pos, covered_objects, m_params, m_w_gamut, pix_cnt)));
		}
		return (double)e_data + (double)e_gamut;
	}

	//x[0]:¦È, x[1]:dr, x[2]:dg, x[3]:db, x[4]: da,x[5]:r0, x[6]:g0, x[7]:b0, x[8]:a0
	ObjectParams CalculateLayerObjectParameters() {
		int obj_n = m_obj_lid_map.size();
		m_params.Initialize(obj_n);
		int n = obj_n * 9;
		double* x  = new double[n];
		double* lb = new double[n];
		double* ub = new double[n];
		for (int i = 0; i < n; i++) {
			x[i] = 0.5;//rand() * 1.0 / RAND_MAX; // (double)m_params.vars[i];
			lb[i] = -1, ub[i] = 1;
		}
		for (int i = 0; i < obj_n; i++) {
			lb[9 * i] = 0, ub[9 * i] = 2 * PI;
		}
		
		for (auto it = m_obj_lid_map.begin(); it != m_obj_lid_map.end(); it++) {
			int oid = it->first;
			int lid = it->second;
			// let the bottom shape opaque
			if (lid == 1) {
				x[9 * oid + 4] = 0;
				x[9 * oid + 8] = 1;
				lb[9 * oid + 4] = 0.00, ub[9 * oid + 4] = 0.01;
				lb[9 * oid + 8] = 0.99, ub[9 * oid + 8] = 1.00;
			}
		}
		double f_min, tol = 1e-5;
		nlopt_opt opter = nlopt_create(NLOPT_LD_LBFGS, n);
		nlopt_set_lower_bounds(opter, lb);
		nlopt_set_upper_bounds(opter, ub);
		nlopt_set_min_objective(opter, global_loss_function, this);
		nlopt_set_maxeval(opter, 1000);
		nlopt_set_xtol_rel(opter, tol);

		nlopt_result result = nlopt_optimize(opter, x, &f_min);
		if (result) {
			for (int i = 0; i < n; i++)
				m_params.vars[i] = x[i];
		}
		m_recon_gamut_loss = f_min;
		return m_params;
	}
};

dual E_recon(Vec2d& pos, vector<int>& oids, Vec3d& real_color, ObjectParams& params, double w_d, int pix_cnt) {
	dual w_data = w_d, N = pix_cnt;
	dual bg_r = 1, bg_g = 1, bg_b = 1, blend_r, blend_g, blend_b;
	double x = pos[0], y = pos[1];
	for (int i = 0; i < oids.size(); i++) {
		int oid = oids[i];
		int k = 9 * oid;
		dual r = cos(params.vars[k]) * params.vars[k + 1] * x + sin(params.vars[k]) * params.vars[k + 1] * y + params.vars[k + 5];
		dual g = cos(params.vars[k]) * params.vars[k + 2] * x + sin(params.vars[k]) * params.vars[k + 2] * y + params.vars[k + 6];
		dual b = cos(params.vars[k]) * params.vars[k + 3] * x + sin(params.vars[k]) * params.vars[k + 3] * y + params.vars[k + 7];
		dual a = cos(params.vars[k]) * params.vars[k + 4] * x + sin(params.vars[k]) * params.vars[k + 4] * y + params.vars[k + 8];

		blend_r = a * r + (1 - a) * bg_r;
		blend_g = a * g + (1 - a) * bg_g;
		blend_b = a * b + (1 - a) * bg_b;

		bg_r = blend_r;
		bg_g = blend_g;
		bg_b = blend_b;
	}
	dual diff = (blend_r - real_color[0]) * (blend_r - real_color[0]) +
				(blend_g - real_color[1]) * (blend_g - real_color[1]) +
				(blend_b - real_color[2]) * (blend_b - real_color[2]);
	diff = (w_data / N) * diff;
	return diff;
}

dual E_gamut(Vec2d& pos, vector<int>& oids, ObjectParams& params, double w_g, int pix_cnt) {
	dual w_gamut = w_g, N = pix_cnt;
	double x = pos[0], y = pos[1];
	dual diff = 0;
	for (int i = 0; i < oids.size(); i++) {
		int oid = oids[i];
		int k = 9 * oid;
		dual a_max = (i == 0 )? (dual)1.0 : (dual)0.8;
		dual r = cos(params.vars[k]) * params.vars[k + 1] * x + sin(params.vars[k]) * params.vars[k + 1] * y + params.vars[k + 5];
		dual g = cos(params.vars[k]) * params.vars[k + 2] * x + sin(params.vars[k]) * params.vars[k + 2] * y + params.vars[k + 6];
		dual b = cos(params.vars[k]) * params.vars[k + 3] * x + sin(params.vars[k]) * params.vars[k + 3] * y + params.vars[k + 7];
		dual a = cos(params.vars[k]) * params.vars[k + 4] * x + sin(params.vars[k]) * params.vars[k + 4] * y + params.vars[k + 8];

		dual clip_r = r; if (r < 0) clip_r = 0; else if (r > 1) clip_r = 1;
		dual clip_g = g; if (g < 0) clip_g = 0; else if (g > 1) clip_g = 1;
		dual clip_b = b; if (b < 0) clip_b = 0; else if (b > 1) clip_b = 1;
		dual clip_a = a; if (a < 0) clip_a = 0; else if (a > a_max) clip_a = a_max;

		dual err = (r - clip_r) * (r - clip_r) + (g - clip_g) * (g - clip_g) + (b - clip_b) * (b - clip_b) + (a - clip_a) * (a - clip_a);
		diff += w_gamut / N * err;
	}
	return diff;
}

double global_loss_function(unsigned n, const double* x, double* grad, void* data) {
	LayerParameterOptimization* pLPO = (LayerParameterOptimization*)data;
	double error = 0;
	if (grad) {
		for (int i = 0; i < n; i++) {
			pLPO->m_params.gradients[i] = 0;
			pLPO->m_params.vars[i] = x[i];
		}
		for (int i = 0; i < pLPO->m_pix_covered_objects.size(); i++)
			error += pLPO->CalculateLossAndGradientOfPix(i);

		for (int i = 0; i < n; i++)
			grad[i] = pLPO->m_params.gradients[i];
	}
	return error;
}