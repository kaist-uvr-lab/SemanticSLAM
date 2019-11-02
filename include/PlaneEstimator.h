#ifndef UVR_SLAM_PLANE_ESTIMATOR_H
#define UVR_SLAM_PLANE_ESTIMATOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {

	class System;
	class Frame;
	class FrameWindow;
	class PlaneEstimator {
	public:
		PlaneEstimator();
		PlaneEstimator(int w, int h);
		virtual ~PlaneEstimator();
	public:
		void Run();
		void SetSystem(System* pSystem);
		void SetFrameWindow(FrameWindow* pFrameWindow);
		void SetTargetFrame(Frame* pFrame);
		void SetBoolDoingProcess(bool b);
		bool isDoingProcess();
	private:
		int mnWidth, mnHeight;
		System* mpSystem;
		std::mutex mMutexDoingProcess;
		bool mbDoingProcess;
		Frame* mpTargetFrame;
		FrameWindow* mpFrameWindow;
	};
}
#endif