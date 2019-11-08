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
		void SetTargetFrame(Frame* pFrame);
		void SetMatcher(Matcher* pMatcher);
		void SetBoolDoingProcess(bool b);
		bool isDoingProcess();
	private:
		int CreateMapPoints(Frame* pCurrKF, Frame* pLastKF);
		void NewMapPointMaginalization(int nFrameCount);
		void UpdateMPs();
		void DeleteMPs();
	private:
		cv::Mat Triangulate(cv::Point2f pt1, cv::Point2f pt2, cv::Mat P1, cv::Mat P2);
		bool CheckDepth(float depth);
		bool CheckReprojectionError(cv::Mat x3D, cv::Mat K, cv::Point2f pt, float thresh);
		bool CheckScaleConsistency(cv::Mat x3D, cv::Mat Ow1, cv::Mat Ow2, float fRatioFactor, float fScaleFactor1, float fScaleFactor2);
	private:
		std::vector<MapPoint*> mvpDeletedMPs;
		SemanticSegmentator* mpSegmentator;
		PlaneEstimator* mpPlaneEstimator;
		FrameWindow* mpFrameWindow;
		Frame* mpTargetFrame;
		Matcher* mpMatcher;
		std::mutex mMutexDoingProcess;
		bool mbDoingProcess;
		int mnWidth, mnHeight;
	};
}

#endif
