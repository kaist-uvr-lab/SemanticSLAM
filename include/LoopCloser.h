#ifndef UVR_SLAM_LOOP_CLOSER_H
#define UVR_SLAM_LOOP_CLOSER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {
	class Frame;
	class System;
	class Map;

	class LoopCloser {
	public:
		LoopCloser();
		LoopCloser(int w, int h, cv::Mat K, Map* pMap);
		virtual ~LoopCloser();

		void SetSystem(System* pSystem);
		void SetBoolProcessing(bool b);
		bool isProcessing();
		void InsertKeyFrame(UVR_SLAM::Frame *pKF);
		bool CheckNewKeyFrames();
		void ProcessNewKeyFrame();
		void Run();

	private:
		std::queue<UVR_SLAM::Frame*> mKFQueue;
	private:
		std::mutex mMutexNewKFs, mMutexLoopClosing, mMutexProcessing;
		int mnWidth;
		int mnHeight;
		cv::Mat mK;
		System* mpSystem;
		Map* mpMap;
		bool mbProcessing;
		float mfTime;
		Frame* mpTargetFrame;
	};
}
#endif