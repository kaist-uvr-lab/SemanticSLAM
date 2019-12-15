#ifndef UVR_SLAM_PLANE_ESTIMATOR_H
#define UVR_SLAM_PLANE_ESTIMATOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <SegmentationData.h>
#include <mutex>

namespace UVR_SLAM {

	class System;
	class Frame;
	class FrameWindow;
	class Initializer;
	class MapPoint;
	class PlaneInformation {
	public:
		PlaneInformation() :bInit(false) { }
		virtual ~PlaneInformation(){}
		bool bInit;
		cv::Mat matPlaneParam;
		int mnPlaneID;
		ObjectType mnPlaneType; //�ٴ�, ��, õ��
	};

	class PlaneEstimator {
	public:
		PlaneEstimator();
		PlaneEstimator(cv::Mat K, cv::Mat K2,int w, int h);
		virtual ~PlaneEstimator();
	public:
		void Run();
		void SetSystem(System* pSystem);
		void SetFrameWindow(FrameWindow* pFrameWindow);
		void SetTargetFrame(Frame* pFrame);
		void SetInitializer(Initializer* pInitializer);
		void SetBoolDoingProcess(bool b, int ptype);
		bool isDoingProcess();
	private:
		double calcPlaneDistance(cv::Mat P, cv::Mat X);
		float calcCosineSimilarity(cv::Mat p1, cv::Mat p2);
		bool calcUnitNormalVector(cv::Mat& X);
		void reversePlaneSign(cv::Mat& param);
		//�ٴ� ��� ã�� ��
		bool PlaneInitialization(PlaneInformation* pPlane, std::set<MapPoint*> spMPs, int ransac_trial, float thresh_distance, float thresh_ratio);
		//�ٴ� ����� �������� ��, õ�� ����� ã�� ��
		bool PlaneInitialization(UVR_SLAM::PlaneInformation* pPlane, cv::Mat GroundPlane, int type, std::set<UVR_SLAM::MapPoint*> spMPs, int ransac_trial, float thresh_distance, float thresh_ratio);
		cv::Mat FlukerLineProjection(cv::Mat P1, cv::Mat P2, cv::Mat R, cv::Mat t, float& m);
	private:

		int mnWidth, mnHeight;
		cv::Mat mK, mK2;//
		System* mpSystem;
		std::mutex mMutexDoingProcess;
		bool mbDoingProcess;
		int mnProcessType;
		Frame* mpLayoutFrame;
		Frame* mpTargetFrame;
		FrameWindow* mpFrameWindow;
		Initializer* mpInitializer;
	};
}
#endif