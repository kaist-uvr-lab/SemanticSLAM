#ifndef UVR_SLAM_MAP_OPTIMIZER_H
#define UVR_SLAM_MAP_OPTIMIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <SegmentationData.h>
#include <mutex>

namespace UVR_SLAM {
	class System;
	class Frame;
	class MapPoint;
	class Map;
	
	class LoopCloser;
	class SemanticSegmentator;
	class PlaneEstimator;

	class Visualizer;

	class MapOptimizer {
	public:
		MapOptimizer(System* pSystem);
		virtual ~MapOptimizer();
		void Init();
		void InsertKeyFrame(UVR_SLAM::Frame *pKF);
		bool CheckNewKeyFrames();
		void ProcessNewKeyFrame();
		void SetDoingProcess(bool b);
		bool isDoingProcess();
		void Run();
		void RunWithMappingServer();
	public:
		bool isStopBA();
		void StopBA(bool b);
	public:
		std::vector<UVR_SLAM::MapPoint*> mvDeletingMPs;
	private:
		std::queue<UVR_SLAM::Frame*> mKFQueue;
		int mnWidth, mnHeight;
		cv::Mat mK;
		std::mutex mMutexStopBA;
		bool mbStopBA;
		System* mpSystem;
		std::mutex mMutexDoingProcess, mMutexNewKFs;
		bool mbDoingProcess;
		Frame* mpTargetFrame;

		Map* mpMap;
		Visualizer* mpVisualizer;
		SemanticSegmentator* mpSegmentator;
		PlaneEstimator* mpPlaneEstimator;
		LoopCloser* mpLoopCloser;
	};
}

#endif
