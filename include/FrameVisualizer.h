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
		FrameVisualizer(System* pSystem);
		void Init();
		virtual ~FrameVisualizer();

		void SetFrameMatchingInformation(Frame* pKF, Frame* pF, float time);
		//void GetFrameMatchingInformation(Frame* pKF, Frame* pF, std::vector<UVR_SLAM::MapPoint*>& vMPs, std::vector<cv::Point2f>& vPts, std::vector<bool>& vbInliers);
		void SetBoolVisualize(bool b);
		bool isVisualize();
		void Run();

	private:
		std::mutex mMutexFrameVisualizer;
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