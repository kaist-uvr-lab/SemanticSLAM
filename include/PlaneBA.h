
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