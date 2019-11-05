#ifndef UVR_SLAM_VISUALIZER_H
#define UVR_SLAM_VISUALIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {
	class Frame;
	class FrameWindow;

	class Visualizer {
	public:
		Visualizer();
		Visualizer(int w, int h, int scale);
		virtual ~Visualizer();

	public:
		void Run();
		void SetFrameWindow(FrameWindow* pFrameWindow);
		void SetTargetFrame(Frame* pFrame);
		void SetBoolDoingProcess(bool b);
		bool isDoingProcess();
	private:
		FrameWindow* mpFrameWindow;
		Frame* mpTargetFrame;
		std::mutex mMutexDoingProcess;
		bool mbDoingProcess;
		int mnWidth, mnHeight;
	public:
		void Init();
	private:
		cv::Mat mVisualized2DMap, mVisTrajectory, mVisMapPoints, mVisPoseGraph;
		cv::Point2f mVisMidPt, mVisPrevPt;
		int mnVisScale;
	};
}
#endif