

#ifndef UVR_SLAM_OPTIMIZATION_H
#define UVR_SLAM_OPTIMIZATION_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
//#include <Vertex.h>
#include <InitialData.h>

namespace UVR_SLAM {
	//class Edge;
	//class Vertex;
	//class Optimizer;
	//class PlaneEdgeOnlyPoseNMap;
	class MapOptimizer;
	class CandidatePoint;
	class FrameWindow;
	class System;
	class Frame;
	class Map;
	class MapPoint;
	class PlaneProcessInformation;
	class Optimization {
	public:

		////////////////////
		////200411
		static int PoseOptimization(Map* pMap, Frame *pFrame, std::vector<UVR_SLAM::CandidatePoint*> vpCPs, std::vector<cv::Point2f> vpPts, std::vector<bool>& vbInliers, cv::Mat prevGra);
		static int PoseOptimization(UVR_SLAM::Map* pMap, Frame *pFrame, std::vector<UVR_SLAM::CandidatePoint*> vpCPs, std::vector<cv::Point2f> vpPts, std::vector<bool>& vbInliers);
		static int PoseOptimization(Frame *pFrame, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<cv::Point2f> vpPts, std::vector<bool>& vbInliers, std::vector<int> vnIDXs);
		static void OpticalLocalBundleAdjustment(UVR_SLAM::Map* pMap, UVR_SLAM::MapOptimizer* pMapOptimizer, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<UVR_SLAM::Frame*> vpKFs, std::vector<UVR_SLAM::Frame*> vpFixedKFs);
		static void OpticalLocalBundleAdjustment(UVR_SLAM::MapOptimizer* pMapOptimizer, UVR_SLAM::Frame* pKF, UVR_SLAM::FrameWindow* pWindow);
		////200411
		////////////////////
		static void InitBundleAdjustment(const std::vector<UVR_SLAM::Frame*> &vpKFs, const std::vector<UVR_SLAM::MapPoint *> &vpMP, int nIterations);
		static void PoseRecoveryOptimization(Frame* pCurrKF, Frame* pPrevKF, Frame* pPPrevKF, std::vector<cv::Point2f> vCurrPTs, std::vector<cv::Point2f> vPrevPTs, std::vector<cv::Point2f> vPPrevPTs, std::vector<cv::Mat>& vP3Ds);
		static void LocalOptimization(System* pSystem, Map* pMap, Frame* pCurrKF, std::vector<cv::Mat>& vX3Ds, std::vector<CandidatePoint*> vpCPs, std::vector<bool>& vbInliers, std::vector<bool>& vbInliers2, float scale, float fMedianDepth, float fMeanDepth, float fStdDev);
	};
}

#endif //ANDROIDOPENCVPLUGINPROJECT_OPTIMIZER_H