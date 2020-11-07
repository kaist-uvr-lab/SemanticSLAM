
#ifndef UVR_SLAM_DIRECT_METHOD_H
#define UVR_SLAM_DIRECT_METHOD_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/types/types_six_dof_expmap.h"
#include <opencv2/core/eigen.hpp>

namespace g2o {

	using namespace Eigen;
	//typedef BlockSolver< BlockSolverTraits<4, 2> > BlockSolver_4_2;
		
	class EdgeDirectXYZOnlyPose : public BaseUnaryEdge<1, Matrix<double,1,1>, VertexSE3Expmap> {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		EdgeDirectXYZOnlyPose(){}
		bool read(std::istream& is);
		bool write(std::ostream& os) const;
		void computeError();
		virtual void linearizeOplus();
		bool isDepthPositive() {
			const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
			return (v1->estimate().map(Xw))(2)>0.0;
		}
		bool isInImage() {
			const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
			Vector2d est = cam_project(v1->estimate().map(Xw));
			int x = (int)est(0);
			int y = (int)est(1);
			return x >= 0 && x < w && y >= 0 && y < h;
		}
		Vector2d cam_project(const Vector3d & trans_xyz) const;
		Vector3d Xw;
		cv::Mat gra2, dx, dy; // 포즈를 추정하려는 이미지에 대한 것
		double fx, fy, cx, cy;
		int w, h;
	};

}

#endif