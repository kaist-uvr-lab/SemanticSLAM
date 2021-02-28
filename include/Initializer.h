
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
	class System;
	class Map;
	class SemanticSegmentator;
	class PlaneEstimator;
	class Visualizer;
	class Initializer {
	public:
		Initializer();
		Initializer(cv::Mat _K);
		Initializer(System* pSystem);
		virtual ~Initializer();

	public:
		void Init();
		void Reset();
		bool Initialize(Frame* pFrame, bool& bReset, int w, int h);
	public:
		Frame* mpInitFrame1, *mpInitFrame2, *mpTempFrame;
	private:
		System* mpSystem;
		SemanticSegmentator* mpSegmentator;
		PlaneEstimator* mpPlaneEstimator;
		LocalMapper* mpLocalMapper;
		Visualizer* mpVisualizer;
		Map* mpMap;
		bool mbInit;
		Matcher* mpMatcher;
	};
}

#endif

