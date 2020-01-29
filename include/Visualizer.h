#ifndef UVR_SLAM_VISUALIZER_H
#define UVR_SLAM_VISUALIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {
	class Frame;
	class MapPoint;
	class FrameWindow;
	class System;

	class Visualizer {
	public:
		Visualizer();
		Visualizer(int w, int h, int scale);
		virtual ~Visualizer();

	public:
		void Run();
		void SetFrameWindow(FrameWindow* pFrameWindow);
		void SetTargetFrame(Frame* pFrame);
		void SetSystem(System* pSystem);
		void SetBoolDoingProcess(bool b);
		bool isDoingProcess();
	private:
		System* mpSystem;
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
		bool mbTracking; //같이 수행되어도 됨
		void VisualizeFrameMatching();
		void VisualizeTracking();
	private:
		
	private:
		int mnFontFace;//2
		double mfFontScale;//1.2
		//void PutText(cv::Mat src, std::string txt, , int x, int y);
		//cv::putText(myImage, myText, myPoint, myFontFace, myFontScale, Scalar::all(255));
	private:
		cv::Mat mVisualized2DMap, mVisTrajectory, mVisMapPoints, mVisPoseGraph;
		cv::Point2f mVisMidPt, mVisPrevPt;
		
	//update scale
	public:
		int GetScale();
		void SetScale(int s);
		static void CallBackFunc(int event, int x, int y, int flags, void* userdata);
	private:
		int mnVisScale;
		std::mutex mMutexScale;
	//local tracknig results
	public:
		void SetMPs(std::vector<UVR_SLAM::MapPoint*> vpMPs);
		std::vector<UVR_SLAM::MapPoint*> GetMPs();
	private:
		std::mutex mMutexFrameMPs;
		std::vector<UVR_SLAM::MapPoint*> mvpFrameMPs;
	};
}
#endif