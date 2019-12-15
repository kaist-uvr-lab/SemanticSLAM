

#ifndef UVR_SLAM_POSEGRAPHOPTIMIZATION_H
#define UVR_SLAM_POSEGRAPHOPTIMIZATION_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <MatrixOperator.h>
#include <Vertex.h>
#include <Edge.h>

using namespace cv;

namespace UVR_SLAM {
	//class Pose;

	class EdgePoseNMap : public GraphOptimizer::Edge{
	public:
		EdgePoseNMap() :Edge(){}
		EdgePoseNMap(int rsize) :Edge(rsize) {
			Xw = Eigen::Vector3d();
		}
		virtual ~EdgePoseNMap() {}
	public:
		virtual double CalcError();
		virtual void CalcJacobian();
		virtual void SetHessian();
	public:
		int GetIndex();
		bool GetDepth();
	public:
		Eigen::Vector3d Xw, Xcam;
		Eigen::Matrix3d K;
		double fx, fy;
		double invz;
		bool bDepth;
	private:

	};

	class PoseOptimizationEdge : public GraphOptimizer::Edge {
	public:
		//생생자 및 virtual 함수
		PoseOptimizationEdge() :Edge()
		{
			//mpFrame = (UVR::Frame*)_v1->GetPointer();
			//mpMapPoint = (UVR::MapPoint*)_v2->GetPointer();
			//mpVertex1 = _v1;
			//mpVertex2 = _v2;
		}
		PoseOptimizationEdge(cv::Mat _Xw, int rsize);
		virtual ~PoseOptimizationEdge() {}
		
		virtual double CalcError();
		virtual void CalcJacobian();
		void SetHessian() {
			Eigen::MatrixXd I = w*info;
			mvpVertices[0]->SetHessian(I, residual);
		}
		
	public:
		//오직 이 클래스만 이용
		int GetIndex() {
			return index;
		}
		bool GetDepth() {
			return bDepth;
		}

	public:
		Eigen::Vector3d Xw, Xcam;
		Eigen::Matrix3d K;
		double fx, fy;
	private:
		//Frame* mpFrame;
		int index; //mvpMPs index

		double invz;
		bool bDepth;

		Eigen::MatrixXd HgmHmmInv;

	};
	class FrameVertex : public GraphOptimizer::Vertex {
	public:
		FrameVertex() :Vertex() {}
		//변경해야 함. pF를 바로 갖지 않고, Pose 정보만 넘겨 받음.
		//이로 인해 다른 곳에서 에러가 생기는 것을 막긴 막아야 함.
		//즉 Pose가 vertex와 frame 사이에 존재하게 됨.
		//이건 pose optimization 시에 사용 됨.

		//calibratioin 시에 이용하기 위함.
		//body vertex는 imu 데이터가 들어감.
		//calib vertex는 I, 0
		FrameVertex(cv::Mat _R, cv::Mat _t, int _size);
		//FrameVertex(UVR_SLAM::Pose* pPose, int _size);
		virtual ~FrameVertex();

		//data도 아예 변경되면 됨.
		//앞의 j는 프로젝션에 대한 것이고
		//뒤의 data도 두가지 버전이 존재하니 각각 상위 단계에서 계산하여 주면 됨.
	public:
		void SetParam();
		void RestoreData();
		//d의 부호를 변경하면 역행렬로 업데이트 할 수 있음.
		//right인지 left인지에 따라서 다르다
		void UpdateParam();
		//연결된 포즈 정보를 업데이트 하기 위한 것
		//포즈 이용 전에 항상 해줘야 함.

		void Accept();
		void Reject();
		void* GetPointer();
	
	public:
		Eigen::Matrix3d GetRotationMatrix();
		Eigen::Vector3d GetTranslationMatrix();
		//UVR_SLAM::Pose* GetPose();

		void SetUpdateSide(bool bSide) {
			bLeft = bSide;
			//true : left
			//false : right
		}
		/*
		void ResetHessian() {
		Vertex::ResetHessian();
		//subH = cv::Mat::zeros(6, 6, CV_64FC1); //calibration 시 Tcb와 연결을 위해
		}
		*/
		//cv::Mat subH, subHold;
	public:
		cv::Mat Rmat, Tmat;
	private:
		Eigen::Matrix3d R, Rold;
		Eigen::Vector3d t, Told;
		//UVR_SLAM::Pose* mpPose;
		bool bLeft;
	};

	class MapPointVertex : public GraphOptimizer::Vertex {
	public:
		MapPointVertex();
		MapPointVertex(cv::Mat _Xw, int idx,int _size);
		virtual ~MapPointVertex();
	public:
		//이 vertex에서만 이용 하는 것.
		void SetParam();
		void RestoreData();
		void* GetPointer();
	public:

		cv::Mat Xw; //restore시 접근
	private:
	};
}
#endif