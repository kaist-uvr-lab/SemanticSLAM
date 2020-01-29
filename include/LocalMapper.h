#ifndef UVR_SLAM_LOCAL_MAPPER_H
#define UVR_SLAM_LOCAL_MAPPER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {
	class Frame;
	class FrameWindow;
	class PlaneEstimator;
	class SemanticSegmentator;
	class Matcher;
	class MapPoint;
	class System;
	class LocalMapper {
	public:
		LocalMapper();
		LocalMapper(int w, int h);
		virtual ~LocalMapper();
	public:
		void Run();
		void SetSystem(System* pSystem);
		void SetPlaneEstimator(PlaneEstimator* pPlaneEstimator);
		void SetLayoutEstimator(SemanticSegmentator* pEstimator);
		void SetFrameWindow(FrameWindow* pFrameWindow);
		void SetMatcher(Matcher* pMatcher);
		////////////////
		void InsertKeyFrame(UVR_SLAM::Frame *pKF);
		//void InterruptLocalMapping();
		bool CheckNewKeyFrames();
		void ProcessNewKeyFrame();
		bool isDoingProcess();
		void CalculateKFConnections();
	private:
		void FuseMapPoints();
		void FuseMapPoints(int nn);
		int CreateMapPoints();
		int CreateMapPoints(Frame* pCurrKF, Frame* pLastKF);
		void NewMapPointMaginalization();
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
		std::list<UVR_SLAM::MapPoint*> mlpNewMPs;
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
		PlaneEstimator* mpPlaneEstimator;
		FrameWindow* mpFrameWindow;
		Frame* mpTargetFrame, *mpPrevKeyFrame, *mpPPrevKeyFrame;
		Matcher* mpMatcher;
		int mnWidth, mnHeight;
	};
}

#endif
