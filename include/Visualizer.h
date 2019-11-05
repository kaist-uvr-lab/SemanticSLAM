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
		void SetFrameMatching(Frame* pF1, Frame* pF2, std::vector<cv::DMatch> vMatchInfos);
	private:
		//SetFrameMatching에 이용할 데이터.
		Frame* mpMatchingFrame1,*mpMatchingFrame2;
		std::vector<cv::DMatch> mvMatchInfos;
		bool mbFrameMatching;
		void VisualizeFrameMatching();
		
	private:
		int mnFontFace;//2
		double mfFontScale;//1.2
		//void PutText(cv::Mat src, std::string txt, , int x, int y);
		//cv::putText(myImage, myText, myPoint, myFontFace, myFontScale, Scalar::all(255));
	private:
		cv::Mat mVisualized2DMap, mVisTrajectory, mVisMapPoints, mVisPoseGraph;
		cv::Point2f mVisMidPt, mVisPrevPt;
		int mnVisScale;
	};
}
#endif