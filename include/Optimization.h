

#ifndef UVR_SLAM_OPTIMIZATION_H
#define UVR_SLAM_OPTIMIZATION_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
//#include <Vertex.h>
#include <InitialData.h>
#include <LoopCloser.h>

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
	class MatchInfo;
	class Map;
	class MapPoint;
	class PlaneProcessInformation;
	class Optimization {
	public:
		/////////////////
		////Loop Closing 관련
		// if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
		void static OptimizeEssentialGraph(Map* pMap, Frame* pLoopKF, Frame* pCurKF,
			const LoopCloser::KeyFrameAndPose &NonCorrectedSim3,
			const LoopCloser::KeyFrameAndPose &CorrectedSim3,
			const std::map<Frame *, std::set<Frame *> > &LoopConnections,
			const bool &bFixScale);

		// if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono)
		static int OptimizeSim3(Frame* pKF1, Frame* pKF2, std::vector<MapPoint *> &vpMatches1,
			g2o::Sim3 &g2oS12, const float th2, const bool bFixScale);
		////Loop Closing 관련

		/////////
		//201225
		static int PoseOptimization(UVR_SLAM::Map* pMap, Frame *pFrame, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<cv::Point2f> vpPts, std::vector<bool>& vbInliers, std::vector<float> vInvLevelSigma2);
		
		static int PoseOptimization(UVR_SLAM::Map* pMap, Frame *pFrame, std::vector<UVR_SLAM::CandidatePoint*> vpCPs, std::vector<cv::Point2f> vpPts, std::vector<bool>& vbInliers, std::vector<float> vInvLevelSigma2);
		
		
		static int PlanarPoseRefinement(UVR_SLAM::Map* pMap, std::vector<MapPoint*> vpPlanarMPs, std::vector<Frame*> vpKFs);
		static int ObjectPointRefinement(UVR_SLAM::Map* pMap, std::vector<MapPoint*> vpObjectMPs, std::vector<Frame*> vpKFs);
		static bool ObjectPointRefinement(UVR_SLAM::Map* pMap, UVR_SLAM::MapPoint* pMP, std::vector<Frame*> vpKFs, 
			std::set<UVR_SLAM::Frame*> spKFs, int thMinKF, float thHuberMono);
		static bool PointRefinement(UVR_SLAM::Map* pMap, UVR_SLAM::Frame* pCurrKF, UVR_SLAM::CandidatePoint* pCP, cv::Mat X3D, 
			std::map<UVR_SLAM::MatchInfo*, int> observations, std::set<UVR_SLAM::Frame*> spKFs, int thMinKF, float thHuberMono);
		//201225
		/////////


		////////////////////
		////200411
		static int PoseOptimization(Frame *pFrame, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<cv::Point2f> vpPts, std::vector<bool>& vbInliers, std::vector<int> vnIDXs);
		static void OpticalLocalBundleAdjustment(UVR_SLAM::Map* pMap, UVR_SLAM::MapOptimizer* pMapOptimizer, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<UVR_SLAM::Frame*> vpKFs, std::vector<UVR_SLAM::Frame*> vpFixedKFs);
		static void OpticalLocalBundleAdjustment(UVR_SLAM::MapOptimizer* pMapOptimizer, UVR_SLAM::Frame* pKF, UVR_SLAM::FrameWindow* pWindow);
		////200411
		////////////////////
		static void InitBundleAdjustment(const std::vector<UVR_SLAM::Frame*> &vpKFs, const std::vector<UVR_SLAM::MapPoint *> &vpMP, int nIterations);
		static void PoseRecoveryOptimization(Frame* pCurrKF, Frame* pPrevKF, Frame* pPPrevKF, std::vector<cv::Point2f> vCurrPTs, std::vector<cv::Point2f> vPrevPTs, std::vector<cv::Point2f> vPPrevPTs, std::vector<cv::Mat>& vP3Ds);
		static void LocalOptimization(System* pSystem, Map* pMap, Frame* pCurrKF, std::vector<cv::Mat>& vX3Ds, std::vector<CandidatePoint*> vpCPs, std::vector<bool>& vbInliers);
		static void LocalOptimization(System* pSystem, Map* pMap, Frame* pCurrKF, std::vector<cv::Mat>& vX3Ds, std::vector<CandidatePoint*> vpCPs, std::vector<bool>& vbInliers, std::vector<bool>& vbInliers2, float scale, float fMedianDepth, float fMeanDepth, float fStdDev);
		static void LocalOptimization(System* pSystem, Map* pMap, Frame* pCurrKF);
	};
}

#endif //ANDROIDOPENCVPLUGINPROJECT_OPTIMIZER_H