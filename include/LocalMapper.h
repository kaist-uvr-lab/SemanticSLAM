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
	class LocalMapper {
	public:
		LocalMapper();
		LocalMapper(int w, int h);
		virtual ~LocalMapper();
	public:
		void Run();
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
		int CreateMapPoints();
		int CreateMapPoints(Frame* pCurrKF, Frame* pLastKF);
		void NewMapPointMaginalization();
		void UpdateKFs();
		void UpdateMPs();
		void DeleteMPs();
		void KeyframeMarginalization();
		int Test();
	private:
		cv::Mat Triangulate(cv::Point2f pt1, cv::Point2f pt2, cv::Mat P1, cv::Mat P2);
		bool CheckDepth(float depth);
		bool CheckReprojectionError(cv::Mat x3D, cv::Mat K, cv::Point2f pt, float thresh);
		bool CheckScaleConsistency(cv::Mat x3D, cv::Mat Ow1, cv::Mat Ow2, float fRatioFactor, float fScaleFactor1, float fScaleFactor2);
		void SetDoingProcess(bool flag);
	private:

		//queue¿Í mvpNewMPs Ãß°¡
		std::queue<UVR_SLAM::Frame*> mKFQueue;
		std::list<UVR_SLAM::MapPoint*> mlpNewMPs;
		std::mutex mMutexNewKFs;
		bool mbStopBA;
		std::mutex mMutexDoingProcess;
		bool mbDoingProcess;

		std::vector<MapPoint*> mvpDeletedMPs;
		SemanticSegmentator* mpSegmentator;
		PlaneEstimator* mpPlaneEstimator;
		FrameWindow* mpFrameWindow;
		Frame* mpTargetFrame, *mpPrevKeyFrame;
		Matcher* mpMatcher;
		int mnWidth, mnHeight;
	};
}

#endif
