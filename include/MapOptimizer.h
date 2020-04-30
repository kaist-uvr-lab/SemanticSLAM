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
	class Visualizer;
	class FrameWindow;
	class MapPoint;
	class Map;
	class MapOptimizer {
	public:
		MapOptimizer(std::string strPath, Map* pMap);
		virtual ~MapOptimizer();
	
		void InsertKeyFrame(UVR_SLAM::Frame *pKF);
		bool CheckNewKeyFrames();
		void ProcessNewKeyFrame();
		void SetDoingProcess(bool b);
		bool isDoingProcess();
		void Run();
		void SetSystem(System* pSystem);
		void SetVisualizer(Visualizer* pVis);
		void SetFrameWindow(FrameWindow* pFrameWindow);
	public:
		bool isStopBA();
		void StopBA(bool b);
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
		Visualizer* mpVisualizer;
		FrameWindow* mpFrameWindow;
		Map* mpMap;
	};
}

#endif
