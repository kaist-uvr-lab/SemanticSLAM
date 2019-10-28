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
	class LocalMapper;
	class System;
	class Tracker {
	public:

		Tracker();
		Tracker(int w, int h, cv::Mat _K);
		virtual ~Tracker();
	public:
		//void Run();
		void Tracking(Frame* pPrev, Frame* pCurr, bool & bInit);
		bool isInitialized();
	public:
		void SetSystem(System*);
		void SetFrameWindow(FrameWindow* pWindow);
		void SetMatcher(Matcher* pMatcher);
		void SetInitializer(Initializer* pInitializer);
		void SetLocalMapper(LocalMapper* pLocalMapper);
	private:
		void CalcVisibleCount(UVR_SLAM::Frame* pF);
		void CalcMatchingCount(UVR_SLAM::Frame* pF);
	private:
		int mnWidth, mnHeight;
		cv::Mat mK;
		bool mbInit;
		System* mpSystem;
		Matcher* mpMatcher;
		Initializer* mpInitializer;
		FrameWindow* mpFrameWindow;
		LocalMapper* mpLocalMapper;
	};
}

#endif

