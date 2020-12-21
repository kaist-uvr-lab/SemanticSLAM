#ifndef UVR_SLAM_LBP_PROCESSOR_H
#define UVR_SLAM_LBP_PROCESSOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <mutex>
#include "lbplibrary.hpp"

namespace UVR_SLAM {
	
	class LocalBinaryPatternProcessor {
	public:
		LocalBinaryPatternProcessor();
		LocalBinaryPatternProcessor(int r, int n, int m, int d);
		virtual ~LocalBinaryPatternProcessor();
		unsigned long long GetID(cv::Mat hist);
		cv::Mat ConvertDescriptor(cv::Mat src);
		cv::Mat ConvertHistogram(cv::Mat src, cv::Rect rect);
	private:
		int mnRadius, mnNeighbor;
		int mnNumPatterns;
		int mnDiscrete;
		int mnNumID;
		unsigned long long mlMaxID;
		lbplibrary::LBP* mpLBP;
	};
}
#endif
