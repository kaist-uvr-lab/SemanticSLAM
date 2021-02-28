#ifndef UVR_SLAM_TRACKER_H
#define UVR_SLAM_TRACKER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace UVR_SLAM {
	class SemanticSegmentator;
	class PlaneEstimator;
	class LocalMapper;
	class Visualizer;
	class FrameVisualizer;
	class CandidatePoint;
	class MapPoint;
	class System;
	class Frame;
	class Matcher;
	class Initializer;
	class Map;
	class Tracker {
	public:

		Tracker();
		Tracker(int w, int h, cv::Mat _K);
		Tracker(System* pSys, std::string strPath);
		virtual ~Tracker();
	public:
		//void Run();
		void Tracking(Frame* pPrev, Frame* pCurr);
		bool isInitialized();
	public:
		void Init();
		
	private:
		UVR_SLAM::Frame* CheckNeedKeyFrame(Frame* pCurr, Frame* pPrev);
		bool CheckNeedKeyFrame(Frame* pCurr, bool &bOpt1, bool &bOpt2, bool &bOpt3, bool &bOpt4);
		int UpdateMatchingInfo(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr, std::vector<UVR_SLAM::CandidatePoint*> vpCPs, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<cv::Point2f> vpPts, std::vector<bool> vbInliers, std::vector<int> vnIDXs, std::vector<int> vnMPIDXs);
		int UpdateMatchingInfo(UVR_SLAM::Frame* pCurr, std::vector<UVR_SLAM::CandidatePoint*> vpCPs,  std::vector<cv::Point2f> vpPts, std::vector<bool> vbInliers);
	private:
		int mnMaxFrames, mnMinFrames;
		int mnThreshMinCPs, mnThreshMinMPs, mnThreshDiff, mnThreshDiffPose;
		int mnMapPointMatching, mnPointMatching;
		int mnPrevMapPointMatching, mnPrevPointMatching;
		cv::Mat mK2, mD;
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
		LocalMapper* mpLocalMapper;
		Visualizer* mpVisualizer;
		FrameVisualizer* mpFrameVisualizer;
	};
}

#endif

