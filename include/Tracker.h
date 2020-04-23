#ifndef UVR_SLAM_TRACKER_H
#define UVR_SLAM_TRACKER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <Initializer.h>
#include <Matcher.h>
#include <Frame.h>

namespace UVR_SLAM {
	class FrameWindow;
	class SemanticSegmentator;
	class PlaneEstimator;
	class LocalMapper;
	class Visualizer;
	class System;
	class Map;
	class MapOptimizer;
	class Tracker {
	public:

		Tracker();
		Tracker(int w, int h, cv::Mat _K);
		Tracker(Map* pMap, std::string strPath);
		virtual ~Tracker();
	public:
		//void Run();
		void Tracking(Frame* pPrev, Frame* pCurr);
		bool isInitialized();
	public:
		void SetSystem(System*);
		void SetFrameWindow(FrameWindow* pWindow);
		void SetMatcher(Matcher* pMatcher);
		void SetSegmentator(SemanticSegmentator* pSegmentator);
		void SetInitializer(Initializer* pInitializer);
		void SetLocalMapper(LocalMapper* pLocalMapper);
		void SetPlaneEstimator(PlaneEstimator* pEstimator);
		void SetMapOptimizer(MapOptimizer* pMapOptimizer);
		void SetVisualizer(Visualizer* pVis);
	private:
		bool CheckNeedKeyFrame(Frame* pCurr);
		void UpdateMatchingInfo(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<cv::Point2f> vpPts, std::vector<bool> vbInliers, std::vector<int> vnIDXs, std::vector<int> vnMPIDXs);
		void CalcVisibleCount(UVR_SLAM::Frame* pF); 
		void CalcMatchingCount(UVR_SLAM::Frame* pF, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<cv::Point2f> vpPts, std::vector<bool> vbInliers);
		void CalcMatchingCount(UVR_SLAM::Frame* pF);
		void CalcMatchingCount(UVR_SLAM::Frame* pF, std::vector<UVR_SLAM::MapPoint*> vDenseMPs, std::vector<std::pair<int, cv::Point2f>> vPairs, std::vector<bool> vbInliers);
	private:
		int mnMaxFrames, mnMinFrames;
		int mnWidth, mnHeight;
		int mnMatching;
		cv::Mat mK, mK2;
		bool mbInitializing;
		bool mbFirstFrameAfterInit;
		bool mbInitilized;
		System* mpSystem;
		Map* mpMap;
		Frame* mpRefKF;
		Matcher* mpMatcher;
		Initializer* mpInitializer;
		SemanticSegmentator* mpSegmentator;
		PlaneEstimator* mpPlaneEstimator;
		MapOptimizer* mpMapOptimizer;
		FrameWindow* mpFrameWindow;
		LocalMapper* mpLocalMapper;
		Visualizer* mpVisualizer;
	};
}

#endif

