
#ifndef UVR_SLAM_PLANEBA_H
#define UVR_SLAM_PLANEBA_H
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

	class PlaneVertex : public BaseVertex<6, Vector6d> {

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		PlaneVertex();
		bool read(std::istream& is);
		bool write(std::ostream& os) const;
		virtual void setToOriginImpl() {
			_estimate.fill(0.);
		}
		virtual void oplusImpl(const double* update_) {

			/*Eigen::Map<const Eigen::Vector4d> v(update_);*/
			Eigen::Map<const Vector6d> v(update_);
			//std::cout << "update param = " << v<<std::endl<<"before param="<<_estimate << std::endl;
			_estimate += v;
			//setEstimate(v + estimate());
			//std::cout << "after = " << estimate() << std::endl;

		}


		//Vector6d Lw;

	private:
	};

	class BAEdgeOnlyMapPoint : public BaseUnaryEdge<2, Vector2d, VertexSBAPointXYZ> {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		BAEdgeOnlyMapPoint();
		bool read(std::istream& is);
		bool write(std::ostream& os) const;
		
		void SetPose(cv::Mat _R, cv::Mat _t) {
			R << _R.at<float>(0, 0), _R.at<float>(0, 1), _R.at<float>(0, 2),
				_R.at<float>(1, 0), _R.at<float>(1, 1), _R.at<float>(1, 2),
				_R.at<float>(2, 0), _R.at<float>(2, 1), _R.at<float>(2, 2);
			t[0] = _t.at<float>(0);
			t[1] = _t.at<float>(1);
			t[2] = _t.at<float>(2);
			//std::cout << "t::" << t << std::endl;
		}

		void computeError() {
			const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
			Vector2d obs(_measurement);
			_error = obs - cam_project(R*v2->estimate()+t);
		}

		bool isDepthPositive() {
			const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
			return (R*v2->estimate() + t)(2)>0.0;
		}

		float Depth() {
			const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
			return (R*v2->estimate() + t)(2);
		}

		virtual void linearizeOplus();
		Vector2d cam_project(const Vector3d & trans_xyz) {
			Vector2d res;
			Vector2d proj;
			proj(0) = trans_xyz(0) / trans_xyz(2);
			proj(1) = trans_xyz(1) / trans_xyz(2);
			res[0] = proj[0] * fx + cx;
			res[1] = proj[1] * fy + cy;
			return res;
		}

		Eigen::Matrix<double, 3, 3> R;
		Eigen::Matrix<double, 3, 1> t;
		double fx, fy, cx, cy;
	};

	class PlaneBAEdgeOnlyMapPoint : public BaseUnaryEdge<1, double, VertexSBAPointXYZ> {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		PlaneBAEdgeOnlyMapPoint();
		bool read(std::istream& is);
		bool write(std::ostream& os) const;
		void computeError();
		virtual void linearizeOplus();
		Vector3d normal;
		double dist;
		Vector3d Xw;

	};

	class PlaneBAEdge : public BaseBinaryEdge<1, double, PlaneVertex, VertexSBAPointXYZ> {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		PlaneBAEdge();
		bool read(std::istream& is);

		bool write(std::ostream& os) const;

		void computeError();
		virtual void linearizeOplus();
	};

	class PlaneCorrelationBAEdge : public BaseBinaryEdge<1, double, PlaneVertex, PlaneVertex> {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		PlaneCorrelationBAEdge();
		bool read(std::istream& is);
		bool write(std::ostream& os) const;
		void computeError();
		virtual void linearizeOplus();
		int type;
	};
}

#endif