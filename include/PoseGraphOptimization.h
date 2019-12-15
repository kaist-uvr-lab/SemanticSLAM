

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
		//������ �� virtual �Լ�
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
		//���� �� Ŭ������ �̿�
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
		//�����ؾ� ��. pF�� �ٷ� ���� �ʰ�, Pose ������ �Ѱ� ����.
		//�̷� ���� �ٸ� ������ ������ ����� ���� ���� ���ƾ� ��.
		//�� Pose�� vertex�� frame ���̿� �����ϰ� ��.
		//�̰� pose optimization �ÿ� ��� ��.

		//calibratioin �ÿ� �̿��ϱ� ����.
		//body vertex�� imu �����Ͱ� ��.
		//calib vertex�� I, 0
		FrameVertex(cv::Mat _R, cv::Mat _t, int _size);
		//FrameVertex(UVR_SLAM::Pose* pPose, int _size);
		virtual ~FrameVertex();

		//data�� �ƿ� ����Ǹ� ��.
		//���� j�� �������ǿ� ���� ���̰�
		//���� data�� �ΰ��� ������ �����ϴ� ���� ���� �ܰ迡�� ����Ͽ� �ָ� ��.
	public:
		void SetParam();
		void RestoreData();
		//d�� ��ȣ�� �����ϸ� ����ķ� ������Ʈ �� �� ����.
		//right���� left������ ���� �ٸ���
		void UpdateParam();
		//����� ���� ������ ������Ʈ �ϱ� ���� ��
		//���� �̿� ���� �׻� ����� ��.

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
		//subH = cv::Mat::zeros(6, 6, CV_64FC1); //calibration �� Tcb�� ������ ����
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
		//�� vertex������ �̿� �ϴ� ��.
		void SetParam();
		void RestoreData();
		void* GetPointer();
	public:

		cv::Mat Xw; //restore�� ����
	private:
	};
}
#endif