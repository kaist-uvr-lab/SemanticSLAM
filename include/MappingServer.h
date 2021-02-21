#ifndef UVR_MAPPING_SERVER_H
#define UVR_MAPPING_SERVER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
namespace UVR_SLAM {
	class LoopCloser;
	class MapOptimizer;
	class LocalMapper;
	class Visualizer;
	class MapPoint;
	class System;
	class Frame;
	class Matcher;
	class Initializer;
	class Map;
	class MappingServer {
	public:
		MappingServer();
		MappingServer(System* pSys);
		virtual ~MappingServer();
		void Init();
		void Reset();
		bool CheckNewFrame();
		void AcquireFrame();
		void InsertFrame(std::pair<int, int> pairInfo);
		void ProcessNewFrame();
		void RunWithMappingServer();
	private:
		System* mpSystem;
		Map* mpMap;
		Matcher* mpMatcher;
		LocalMapper* mpLocalMapper;
		LoopCloser* mpLoopCloser;
		MapOptimizer* mpMapOptimizer;
		Visualizer* mpVisualizer;
		Initializer* mpInitializer;

		int mnWidth, mnHeight;
		cv::Mat mK, mInvK;

		std::queue<std::pair<int, int>> mQueue;
		std::mutex mMutexQueue;
		std::pair<int, int> mPairFrameInfo;

		bool mbInitialized;
		int mnReferenceID;
		std::map<int, UVR_SLAM::Frame*> mmFrames;
		std::map<int, UVR_SLAM::Frame*> mmKeyFrames;
	};
}
#endif