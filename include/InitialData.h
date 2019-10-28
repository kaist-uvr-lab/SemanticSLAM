

#pragma once
#ifndef UVR_SLAM_INITIAL_DATA_H
#define UVR_SLAM_INITIAL_DATA_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace UVR_SLAM {
	class InitialData {
	public:
		InitialData();
		InitialData(int _minGood);
		virtual ~InitialData();

		void SetRt(cv::Mat _R, cv::Mat _t);
		std::vector<cv::Mat> mvX3Ds;
		std::vector<bool> vbTriangulated;
		cv::Mat R;
		cv::Mat t;
		cv::Mat R0, t0;
		int nGood;
		int nMinGood;
		float parallax;
	};
}

#endif