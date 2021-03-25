#ifndef UVR_SLAM_SERVER_MAPPER_H
#define UVR_SLAM_SERVER_MAPPER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {
	
	class System;
	class MapPoint;
	class Frame;
	class LoopCloser;
	class ServerMapOptimizer;
	class Matcher;
	class User;
	class ServerMap;

	class ServerMapper {
		public:
			ServerMapper(System* pSystem);
			virtual ~ServerMapper();
			void Init();
			void InsertKeyFrame(std::pair<Frame*, std::string> pairInfo);
			int KeyframesInQueue();
			bool CheckNewKeyFrames();
			void AcquireFrame();
			void ProcessNewKeyFrame();
			bool isDoingProcess();
			void SetDoingProcess(bool flag);
			void RunWithMappingServer();
			void NewMapPointMarginalization();
			void ComputeNeighborKFs(Frame* pKF);
			void ConnectNeighborKFs(Frame* pKF, std::map<UVR_SLAM::Frame*, int> mpCandiateKFs, int thresh);
			void CreateMapPoints();
			void SendData(Frame* pF, std::string user, std::string map);
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
			Matcher* mpMatcher;
			ServerMapOptimizer* mpMapOptimizer;
			LoopCloser* mpLoopCloser;

	};
}

#endif