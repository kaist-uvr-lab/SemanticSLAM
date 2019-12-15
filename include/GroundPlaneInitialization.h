#ifndef UVR_SLAM_GROUND_PLANE_INITIALIZATION_H
#define UVR_SLAM_GROUND_PLANE_INITIALIZATION_H
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <MatrixOperator.h>
#include <Vertex.h>
#include <Edge.h>

//Vertex�� Edge�� ������ �ʿ���.
//�ʱ� �ڼ��� ��Ī ������ �� �� ����ȭ �˰����� ���ؼ� ����� ã�� �����̴�.
namespace UVR_SLAM {
	class PlaneInitEdge : public GraphOptimizer::Edge {
	public:
		PlaneInitEdge();
		PlaneInitEdge(cv::Mat _Xcam,cv::Mat _R, cv::Mat _t,int rsize);
		virtual ~PlaneInitEdge();
	public:
		virtual double CalcError();
		virtual void CalcJacobian();
		virtual void SetHessian();
	public:
		
	public:

		Eigen::Vector3d Xcam, Xplane;
		Eigen::Matrix3d K;
		double fx, fy;
		Eigen::Matrix3d R;
		Eigen::Vector3d t;

		double invz;
		bool bDepth;
		double Xdepth;

	private:
		double tempDepth;
		Eigen::Vector3d Normal;
		double Dist;
	};

	class PlaneInitVertex : public GraphOptimizer::Vertex {
	public:
		PlaneInitVertex();
		PlaneInitVertex(cv::Mat _P, int _size);
		virtual ~PlaneInitVertex();
	public:
		void SetParam();
		void RestoreData();
		void* GetPointer();
		void UpdateParam();
	public:
		//Eigen::Matrix3d GetRotationMatrix();
		//Eigen::Vector3d GetTranslationMatrix();
	private:
		void Normalize();
	public:
		cv::Mat matPlane;
	private:
		
	};
}

#endif