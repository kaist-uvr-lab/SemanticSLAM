#ifndef UVR_SLAM_SERVER_MAP_OPTIMIZER_H
#define UVR_SLAM_SERVER_MAP_OPTIMIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {
	class System;
	class Frame;
	class MapPoint;
	class User;
	class ServerMap;

	class ServerMapOptimizer{
	public:
		ServerMapOptimizer(System* pSystem);
		virtual ~ServerMapOptimizer();
		void InsertKeyFrame(std::pair<Frame*, std::string> pairInfo);
		int KeyframesInQueue();
		bool CheckNewKeyFrames();
		void AcquireFrame();
		void ProcessNewKeyFrame();
		bool isDoingProcess();
		void SetDoingProcess(bool flag);
		void RunWithMappingServer();
		
	private:
		std::queue<std::pair<Frame*, std::string>> mQueue;
		std::mutex mMutexQueue;
		std::pair<Frame*, std::string> mPairFrameInfo;
		User* mpTargetUser;
		ServerMap* mpTargetMap;
		Frame* mpTargetFrame;

		std::mutex mMutexDoingProcess;
		bool mbDoingProcess;

		System* mpSystem;
	};
}

#endif