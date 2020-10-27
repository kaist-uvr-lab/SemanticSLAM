#ifndef UVR_SLAM_FRAME_GRID_H
#define UVR_SLAM_FRAME_GRID_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace UVR_SLAM {

	class CandidatePoint;
	class FrameGrid {
	public:
		FrameGrid();
		FrameGrid(cv::Point2f base, cv::Rect r);
		FrameGrid(cv::Point2f base, int size);
		virtual ~FrameGrid();
	public:
		cv::Mat CalcGradientImage(cv::Mat src);
		bool CalcActivePoint(cv::Mat src, int gthresh, cv::Point2f& pt);
	public:
		int mnLabel;
		int mnPlane;
		bool mbMatched;
		CandidatePoint* mpCP;
		cv::Rect rect;
		cv::Point2f basePt;
		cv::Point2f pt;
	private:
	};

}
#endif