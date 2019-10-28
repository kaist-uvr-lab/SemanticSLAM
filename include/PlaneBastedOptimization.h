

#ifndef UVR_SLAM_PLANEBASEDOPTIMIZATION_H
#define UVR_SLAM_PLANEBASEDOPTIMIZATION_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Vertex.h>
#include <Edge.h>
#include <PoseGraphOptimization.h>


using namespace cv;

namespace UVR_SLAM {
	
	class PlaneMapPointVertex : public GraphOptimizer::Vertex {
	public:
		PlaneMapPointVertex() :Vertex() {}
		PlaneMapPointVertex(double _depth, int _size) :mDepth(_depth), Vertex(0, _size, false) {
			SetParam();
		}
		virtual ~PlaneMapPointVertex() {}
	public:
		void SetParam() {
			param = Eigen::VectorXd::Ones(1)*mDepth;
		}
		void UpdateParam() {
			param += d;
			ResetHessian();
		}
		void RestoreData() {
			mDepth = param(0);
		}
		void* GetPointer() {
			return nullptr;
		};

	public:
		void SetDepth(double _d) {
			mDepth = _d;
		}
		double GetDepth() {
			RestoreData();
			return mDepth;
		}

	private:
		double mDepth;
	};

	class PlaneEdgeOnlyPose : public GraphOptimizer::Edge {
	public:
		PlaneEdgeOnlyPose() :Edge() {
			residual = Eigen::VectorXd(1);
		};
		PlaneEdgeOnlyPose(int rsize) :Edge(rsize) {
			residual = Eigen::VectorXd(rsize);
			Ximg = Eigen::Vector3d();
			Pw = Eigen::Vector4d();
		}
		virtual ~PlaneEdgeOnlyPose() {};
	public:
		double CalcError();
		void CalcJacobian();
		void SetHessian() {
			Eigen::MatrixXd I = w*info;
			mvpVertices[0]->SetHessian(I, residual);
		}

	public:
		Eigen::Matrix3d K, Kinv;
		Eigen::Vector3d Xworld, Xcam, Ximg;
		Eigen::Vector4d Pw, Pc;
		double mdDepth;
	};
	
	class PlaneEdgeOnlyPoseNMap : public GraphOptimizer::Edge {
	public:
		PlaneEdgeOnlyPoseNMap() :Edge() {
			residual = Eigen::VectorXd(1);
		}
		PlaneEdgeOnlyPoseNMap(int rsize) :Edge(rsize) {
			residual = Eigen::VectorXd(rsize);
			Ximg = Eigen::Vector3d();
			Pw = Eigen::Vector4d();
		}
		virtual ~PlaneEdgeOnlyPoseNMap() {}
	public:
		double CalcError();
		void CalcJacobian();
		void SetHessian() {
			Eigen::MatrixXd I = w*info;
			mvpVertices[0]->SetHessian(I, residual);
			mvpVertices[1]->SetHessian(I, residual);

			mvSubHs[0] = mvpVertices[0]->GetJacobian().transpose()*mvpVertices[1]->GetJacobian();

		}

		//이 아래것들이 제대로 이용이 되는가?
		/*
		void SetSubspaceHessian(cv::Mat& HgmHmmInv, cv::Mat& Hmg) {}
		void EraseEdge() {}
		void SetEdge() {}
		*/

		//vertex 0 pose
		//vertex 1 depth
		//plane - variable or vertex 2
	public:
		Eigen::Matrix3d K, Kinv;
		Eigen::Vector3d Xworld, Xcam, Ximg;
		Eigen::Vector4d Pw, Pc;
	private:

	};
}
#endif