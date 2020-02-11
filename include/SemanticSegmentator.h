
#ifndef UVR_SLAM_SEMANTIC_SEGMENTATOR_H
#define UVR_SLAM_SEMANTIC_SEGMENTATOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
#include <Frame.h>
#include <JSONConverter.h>

namespace UVR_SLAM {

	class System;
	class FrameWindow;
	class PlaneEstimator;
	class SemanticSegmentator {
	public:
		SemanticSegmentator();
		SemanticSegmentator(std::string _ip, int _port, int nWidth, int nHeight);
		SemanticSegmentator(const std::string & strSettingPath);
		virtual ~SemanticSegmentator();
	public:
		void InsertKeyFrame(UVR_SLAM::Frame *pKF);
		bool CheckNewKeyFrames();
		void ProcessNewKeyFrame();

		void Run();
		void SetSystem(System* pSystem);
		void SetFrameWindow(FrameWindow* pFrameWindow);
		void SetPlaneEstimator(PlaneEstimator* pEstimator);
		void SetTargetFrame(Frame* pFrame);
		void SetBoolDoingProcess(bool b);
		bool isDoingProcess();
		bool isRun();
	public:
		void ObjectLabeling();
		void ObjectLabeling(cv::Mat masked, int ratio);
		void SetSegmentationMask(cv::Mat segmented);
	private:
		//std::vector<cv::Vec3b> mVecLabelColors;
		std::vector<cv::Mat> mVecLabelMasks;
		int mnWidth, mnHeight;
	private:
		std::queue<UVR_SLAM::Frame*> mKFQueue;
		std::mutex mMutexNewKFs;
		bool mbOn;
		System* mpSystem;
		std::mutex mMutexDoingProcess;
		bool mbDoingProcess;
		Frame* mpTargetFrame;
		FrameWindow* mpFrameWindow;
		PlaneEstimator* mpPlaneEstimator;
		std::string ip;
		int port;
	};
}

#endif