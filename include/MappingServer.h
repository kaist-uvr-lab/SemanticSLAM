#ifndef UVR_MAPPING_SERVER_H
#define UVR_MAPPING_SERVER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
namespace UVR_SLAM {
	class LoopCloser;
	class MapOptimizer;
	class ServerMapper;
	class Visualizer;
	class MapPoint;
	class System;
	class Data;
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
		int KeyframesInQueue();
		bool CheckNewFrame();
		void AcquireFrame();
		void InsertFrame(Data* pData);
		void ProcessNewFrame();
		void ProcessNewFrame2();
		void RunWithMappingServer();
	private:
		System* mpSystem;
		Map* mpMap;
		Matcher* mpMatcher;
		ServerMapper* mpLocalMapper;
		LoopCloser* mpLoopCloser;
		MapOptimizer* mpMapOptimizer;
		Visualizer* mpVisualizer;
		Initializer* mpInitializer;

		std::queue<Data*> mQueue;
		std::mutex mMutexQueue;
		Data* mpData;
	};
}
#endif