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
	class Tracker {
	public:

		Tracker();
		Tracker(int w, int h, cv::Mat _K);
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
		void SetVisualizer(Visualizer* pVis);
	private:
		void CalcVisibleCount(UVR_SLAM::Frame* pF);
		void CalcMatchingCount(UVR_SLAM::Frame* pF);
	private:
		int mnWidth, mnHeight;
		cv::Mat mK;
		bool mbInitializing;
		bool mbFirstFrameAfterInit;
		bool mbInitilized;
		System* mpSystem;
		Matcher* mpMatcher;
		Initializer* mpInitializer;
		SemanticSegmentator* mpSegmentator;
		PlaneEstimator* mpPlaneEstimator;
		FrameWindow* mpFrameWindow;
		LocalMapper* mpLocalMapper;
		Visualizer* mpVisualizer;
	};
}

#endif

