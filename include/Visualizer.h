#ifndef UVR_SLAM_VISUALIZER_H
#define UVR_SLAM_VISUALIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {
	class Frame;
	class MapPoint;
	class PlaneProcessInformation;
	class PlaneInformation;
	class PlaneEstimator;
	class System;
	class Map;

	class Visualizer {
	public:
		Visualizer();
		Visualizer(System* pSystem);
		virtual ~Visualizer();
	public:
		void Run();
		void RunWithMappingServer();
		void SetBoolDoingProcess(bool b);
		bool isDoingProcess();
	private:
		System* mpSystem;
		Map* mpMap;
		std::mutex mMutexDoingProcess;
		bool mbDoingProcess;
		int mnWidth, mnHeight;
		int mnDisplayX, mnDisplayY;
//////////////////////
////output시각화
	public:
		void SetOutputImage(cv::Mat out, int type);
		cv::Mat GetOutputImage(int type);
		bool isOutputTypeChanged(int type);
		int mnWindowImgCols, mnWindowImgRows;
	private:
		std::vector<cv::Mat> mvOutputImgs;
		std::vector<cv::Rect> mvRects;
		cv::Mat mOutputImage;
		std::vector<bool> mvOutputChanged;
		std::mutex mMutexOutput;
////output시각화
//////////////////////
	public:
		void Init();
	private:
		int mnFontFace;//2
		double mfFontScale;//1.2
		//void PutText(cv::Mat src, std::string txt, , int x, int y);
		//cv::putText(myImage, myText, myPoint, myFontFace, myFontScale, Scalar::all(255));
	private:
		cv::Mat mVisPoseGraph;
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
	///////바닥 확인용
	public:
		void AddPlaneInfo(PlaneProcessInformation* pPlaneInfo);
		std::vector<PlaneProcessInformation*> GetPlaneInfos();
	private:
		std::mutex mMutexPlaneInfo;
		std::vector<PlaneProcessInformation*> mvpPlaneInfos;
	///////바닥 확인용
	///////평면 관련
	public:
		PlaneEstimator* mpPlaneEstimator;
	private:
	///////평면 관련
	};
}
#endif