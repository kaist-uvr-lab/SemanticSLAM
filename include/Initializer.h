
#ifndef UVR_SLAM_INITIALIZER_H
#define UVR_SLAM_INITIALIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <InitialData.h>
#include <Frame.h>
#include <Matcher.h>
#include <MapPoint.h>
#include <Optimization.h>

using namespace cv;

namespace UVR_SLAM {
	class LocalMapper;
	class FrameWindow;
	class System;
	class Map;
	class SemanticSegmentator;
	class PlaneEstimator;
	class Visualizer;
	class Initializer {
	public:
		Initializer();
		Initializer(cv::Mat _K);
		Initializer(System* pSystem, Map* pMap, cv::Mat _K);
		virtual ~Initializer();

////////////////////////////
//////200410
	public:
		void SetCandidatePose(cv::Mat F, std::vector<std::pair<cv::Point2f, cv::Point2f>> Matches, std::vector<UVR_SLAM::InitialData*>& vCandidates);
		void CheckRT(std::vector<std::pair<cv::Point2f, cv::Point2f>> Matches, InitialData* candidate, float th2);
	private:
		bool Triangulate(cv::Point2f pt1, cv::Point2f pt2, cv::Mat P1, cv::Mat P2, cv::Mat& x3D);
		bool CheckCreatedPoints(cv::Mat X3D, cv::Point2f kp1, cv::Point2f kp2, cv::Mat O1, cv::Mat O2, cv::Mat R1, cv::Mat t1, cv::Mat R2, cv::Mat t2, float& cosParallax, float th1, float th2);
//////200410
////////////////////////////

	public:
		void Init();
		void Reset();
		void SetLocalMapper(LocalMapper* pMapper);
		void SetMatcher(Matcher* pMatcher);
		void SetFrameWindow(FrameWindow* pWindow);
		void SetSegmentator(SemanticSegmentator* pEstimator);
		void SetPlaneEstimator(PlaneEstimator* pEstimator);
		void SetVisualizer(Visualizer* pVis);
		bool Initialize(Frame* pFrame, bool& bReset, int w, int h);
	private:
		void SetCandidatePose(cv::Mat F, std::vector<cv::DMatch> Matches, std::vector<UVR_SLAM::InitialData*>& vCandidates);
		int SelectCandidatePose(std::vector<UVR_SLAM::InitialData*>& vCandidates);
		void DecomposeE(cv::Mat E, cv::Mat &R1, cv::Mat& R2, cv::Mat& t1, cv::Mat& t2);
		void CheckRT(std::vector<cv::DMatch> Matches, InitialData* candidate, float th2);
	public:
		Frame* mpInitFrame1, *mpInitFrame2, *mpTempFrame;
	private:
		System* mpSystem;
		SemanticSegmentator* mpSegmentator;
		PlaneEstimator* mpPlaneEstimator;
		FrameWindow* mpFrameWindow;
		LocalMapper* mpLocalMapper;
		Visualizer* mpVisualizer;
		Map* mpMap;
		bool mbInit;
		Matcher* mpMatcher;
		cv::Mat mK;
		
	};
}

#endif

