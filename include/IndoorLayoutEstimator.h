
#ifndef UVR_SLAM_INDOOR_LAYOUT_ESTIMATOR_H
#define UVR_SLAM_INDOOR_LAYOUT_ESTIMATOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
#include <Frame.h>
#include <JSONConverter.h>

namespace UVR_SLAM {

	class System;
	class FrameWindow;
	class IndoorLayoutEstimator {
	public:
		IndoorLayoutEstimator();
		IndoorLayoutEstimator(int nWidth, int nHeight);
		virtual ~IndoorLayoutEstimator();
	public:
		void ObjectLabeling();
		void SetSegmentationMask(cv::Mat segmented);
		void Run();
		void SetSystem(System* pSystem);
		void SetFrameWindow(FrameWindow* pFrameWindow);
		void SetTargetFrame(Frame* pFrame);
		void SetBoolDoingProcess(bool b);
		bool isDoingProcess();
	private:
		//std::vector<cv::Vec3b> mVecLabelColors;
		std::vector<cv::Mat> mVecLabelMasks;
		int mnWidth, mnHeight;
	private:
		System* mpSystem;
		std::mutex mMutexDoingProcess;
		bool mbDoingProcess;
		Frame* mpTargetFrame;
		FrameWindow* mpFrameWindow;
	};
}

#endif