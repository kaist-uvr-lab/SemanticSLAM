
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
	class LocalMapper;
	class Visualizer;
	class Map;
	class SemanticSegmentator {
	public:
		SemanticSegmentator();
		SemanticSegmentator(std::string _ip, int _port, int nWidth, int nHeight);
		SemanticSegmentator(System* pSystem, const std::string & strSettingPath);
		virtual ~SemanticSegmentator();
	public:

		void InsertKeyFrame(UVR_SLAM::Frame *pKF);
		bool CheckNewKeyFrames();
		void ProcessNewKeyFrame();
		void Init();
		void Run();
		void SetTargetFrame(Frame* pFrame);
		void SetBoolDoingProcess(bool b);
		bool isDoingProcess();
		bool isRun();
	public:
		//바닥과 벽에서 노이즈를 조금 걸러내보기 위함.
		void ImageLabeling(cv::Mat masked, cv::Mat& labeld);
		void SetSegmentationMask(cv::Mat segmented);
	private:
		//std::vector<cv::Vec3b> mVecLabelColors;
		std::vector<cv::Mat> mVecLabelMasks;
		int mnWidth, mnHeight;
		float cx, cy;
	private:
		std::queue<UVR_SLAM::Frame*> mKFQueue;
		std::mutex mMutexNewKFs;
		bool mbOn;
		Map* mpMap;
		System* mpSystem;
		Visualizer* mpVisualizer;
		std::mutex mMutexDoingProcess;
		bool mbDoingProcess;
		Frame* mpTargetFrame, *mpPrevTargetFrame;
		FrameWindow* mpFrameWindow;
		PlaneEstimator* mpPlaneEstimator;
		LocalMapper* mpLocalMapper;
		std::string ip;
		int port;
	};
}

#endif