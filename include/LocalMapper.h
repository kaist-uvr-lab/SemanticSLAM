#ifndef UVR_SLAM_LOCAL_MAPPER_H
#define UVR_SLAM_LOCAL_MAPPER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {
	class Frame;
	class MatchInfo;
	class PlaneEstimator;
	class SemanticSegmentator;
	class Matcher;
	class MapPoint;
	class System;
	class Map;
	class MapOptimizer;
	class Visualizer;
	class LocalMapper {
	public:
		LocalMapper();
		LocalMapper(Map* pMap, int w, int h);
		virtual ~LocalMapper();
	public:
		void Reset();
		void Run();
		void SetSystem(System* pSystem);
		void SetPlaneEstimator(PlaneEstimator* pPlaneEstimator);
		void SetLayoutEstimator(SemanticSegmentator* pEstimator);
		void SetMatcher(Matcher* pMatcher);
		void SetMapOptimizer(MapOptimizer* pMapOptimizer);
		void SetVisualizer(Visualizer* pVis);
		////////////////
		void InsertKeyFrame(UVR_SLAM::Frame *pKF);
		//void InterruptLocalMapping();
		bool CheckNewKeyFrames();
		void ProcessNewKeyFrame();
		bool isDoingProcess();
		void CalculateKFConnections();
	private:
		void CreateMapPoints(MatchInfo* pCurrMatchInfo, cv::Mat& debug);
		////////////////////////////////////////
		void FuseMapPoints();
		void FuseMapPoints(int nn);
		int CreateMapPoints();
		int CreateMapPoints(Frame* pCurrKF, Frame* pLastKF);
		void NewMapPointMarginalization();
		void UpdateKFs();
		void UpdateMPs();
		void DeleteMPs();
		void KeyframeMarginalization();
		int Test();
	private:
		bool Triangulate(cv::Point2f pt1, cv::Point2f pt2, cv::Mat P1, cv::Mat P2, cv::Mat& X3D);
		bool CheckDepth(float depth);
		bool CheckReprojectionError(cv::Mat x3D, cv::Mat K, cv::Point2f pt, float thresh);
		bool CheckScaleConsistency(cv::Mat x3D, cv::Mat Ow1, cv::Mat Ow2, float fRatioFactor, float fScaleFactor1, float fScaleFactor2);
		void SetDoingProcess(bool flag);
	public:
		bool isStopLocalMapping();
		void StopLocalMapping(bool flag);
	public:
		
	private:

		//queue¿Í mvpNewMPs Ãß°¡
		std::queue<UVR_SLAM::Frame*> mKFQueue;
		
		std::mutex mMutexNewKFs, mMutexStopLocalMapping;
		bool mbStopBA, mbStopLocalMapping;
		std::mutex mMutexDoingProcess;
		bool mbDoingProcess;

		std::vector<MapPoint*> mvpDeletedMPs;
		SemanticSegmentator* mpSegmentator;
		System* mpSystem;
		Map* mpMap;
		MapOptimizer* mpMapOptimizer;
		PlaneEstimator* mpPlaneEstimator;
		Visualizer* mpVisualizer;
		Frame* mpTargetFrame, *mpPrevKeyFrame, *mpPPrevKeyFrame;
		Matcher* mpMatcher;
		int mnWidth, mnHeight;
	};
}

#endif
