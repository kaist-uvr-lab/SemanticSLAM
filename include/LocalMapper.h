#ifndef UVR_SLAM_LOCAL_MAPPER_H
#define UVR_SLAM_LOCAL_MAPPER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {
	class System;
	class MapPoint;
	class CandidatePoint;
	class Frame;
	class MatchInfo;
	class Map;

	class Matcher;
	class MapOptimizer;
	class LoopCloser;
	class SemanticSegmentator;
	class PlaneEstimator;
	
	class Visualizer;

	class LocalMapper {
	public:
		LocalMapper();
		LocalMapper(System* pSystem, std::string strPath, int w, int h);
		virtual ~LocalMapper();
		void Init();
	public:
		void Reset();
		void Run();
		
		////////////////
		void SetInitialKeyFrame(UVR_SLAM::Frame* pKF1, UVR_SLAM::Frame* pKF2);
		void InsertKeyFrame(UVR_SLAM::Frame *pKF);
		//void InterruptLocalMapping();
		bool CheckNewKeyFrames();
		void ProcessNewKeyFrame();
		bool isDoingProcess();
		void CalculateKFConnections();

	private:
		int CreateMapPoints(Frame* pCurrKF, std::vector<cv::Point2f> vMatchCurrPts, std::vector<CandidatePoint*> vMatchPrevCPs, double& ftime, cv::Mat& debugMatch);
		int MappingProcess(Map* pMap, Frame* pCurrKF, Frame* pPrevKF, 
			std::vector<cv::Point2f>& vMappingPrevPts, std::vector<cv::Point2f>& vMappingCurrPts, std::vector<CandidatePoint*>& vMappingCPs,
			std::vector<cv::Point2f>  vMatchedPrevPts, std::vector<cv::Point2f>  vMatchedCurrPts, std::vector<CandidatePoint*>  vMatchedCPs, 
			double& dtime, cv::Mat& debugging);
		int CreateMapPoints(std::vector<cv::Point2f> vMatchCurrPts, std::vector<CandidatePoint*> vMatchCPs, double& ftime, cv::Mat& debugMatch);
		int RecoverPose(Frame* pCurrKF, Frame* pPrevKF, std::vector<cv::Point2f> vMatchPrevPts, std::vector<cv::Point2f> vMatchCurrPts, std::vector<CandidatePoint*> vPrevCPs, cv::Mat& R, cv::Mat& T, double& ftime, cv::Mat& prevImg, cv::Mat& currImg);
		int RecoverPose(Frame* pCurrKF, Frame* pPrevKF, Frame* pPPrevKF, std::vector<cv::Point2f> vCurrPts, std::vector<cv::Point2f> vPrevPts, std::vector<cv::Point2f> vPPrevPts, std::vector<CandidatePoint*> vpCPs, std::vector<bool>& vbInliers, cv::Mat& R, cv::Mat& T, double& ftime, 
			cv::Mat& currImg, cv::Mat& prevImg, cv::Mat& pprevImg);
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

		System* mpSystem;
		Map* mpMap;
		MapOptimizer* mpMapOptimizer;
		LoopCloser* mpLoopCloser;
		SemanticSegmentator* mpSegmentator;
		PlaneEstimator* mpPlaneEstimator;
		Visualizer* mpVisualizer;
		
		Frame* mpTargetFrame, *mpPrevKeyFrame, *mpPPrevKeyFrame;
		Matcher* mpMatcher;

		int mnWidth, mnHeight;
		cv::Mat mK, mInvK;
	};
}

#endif
