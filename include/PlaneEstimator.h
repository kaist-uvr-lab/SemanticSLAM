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
	class MapPoint;
	class PlaneInformation {
	public:
		cv::Mat matPlaneParam;
		int mnPlaneID;
		ObjectType mnPlaneType; //¹Ù´Ú, º®, ÃµÀå
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
		void SetBoolDoingProcess(bool b, int ptype);
		bool isDoingProcess();
	private:
		bool calcUnitNormalVector(cv::Mat& X);
		void reversePlaneSign(cv::Mat& param);
		bool PlaneInitialization(PlaneInformation* pPlane, std::set<MapPoint*> spMPs, int ransac_trial, float thresh_distance, float thresh_ratio);
	private:
		int mnWidth, mnHeight;
		cv::Mat mK, mK2;//
		System* mpSystem;
		std::mutex mMutexDoingProcess;
		bool mbDoingProcess;
		int mnProcessType;
		Frame* mpTargetFrame;
		FrameWindow* mpFrameWindow;
	};
}
#endif