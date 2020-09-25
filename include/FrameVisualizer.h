#ifndef UVR_SLAM_FRAME_VISUALIZER_H
#define UVR_SLAM_FRAME_VISUALIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {
	class Frame;
	class MatchInfo;
	class MapPoint;
	class System;
	class Map;
	class Visualizer;
	class FrameVisualizer {
	public:
		FrameVisualizer();
		FrameVisualizer(int w, int h, cv::Mat K, Map* pMap);
		virtual ~FrameVisualizer();

		void SetSystem(System* pSystem);
		void SetVisualizer(Visualizer* pVis);
		void SetFrameMatchingInformation(Frame* pKF, Frame* pF, float time);
		//void GetFrameMatchingInformation(Frame* pKF, Frame* pF, std::vector<UVR_SLAM::MapPoint*>& vMPs, std::vector<cv::Point2f>& vPts, std::vector<bool>& vbInliers);
		void SetBoolVisualize(bool b);
		bool isVisualize();
		void Run();

	private:
		std::mutex mMutexFrameVisualizer;
		int mnWidth;
		int mnHeight;
		cv::Mat mK;
		System* mpSystem;
		Map* mpMap;
		Visualizer* mpVisualizer;
		bool mbVisualize;
		float mfTime;

		Frame* mpKeyFrame;
		Frame* mpFrame;
		std::vector<cv::Point2f> mvMatchingPTs;
		std::vector<UVR_SLAM::MapPoint*> mvpMatchingMPs;
		std::vector<bool> mvbMatchingInliers;

	};
}
#endif