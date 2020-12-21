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
		bool CalcActivePoints(cv::Mat src, int gthresh, int& localthresh, cv::Point2f& pt);
	public:
		int mnLabel;
		int mnPlane;
		bool mbMatched;
		CandidatePoint* mpCP;
		cv::Mat mDescLBP, mHistLBP;
		cv::Mat mDescLBP2;
		unsigned char mCharCode;
		cv::Rect rect;
		cv::Point2f basePt;
		cv::Point2f pt;
		int mnMaxIDX;
		std::vector<cv::Point2f> vecPTs;
		////새로 추가한 것
		//오브젝트는 일단 대표 오브젝트만 고려?
		FrameGrid* mpPrev, *mpNext;
		cv::Mat mObjCount, mObjArea;
		/*std::map<int, int> mmObjCounts;
		std::map<int, float> mmObjAreas;*/
	private:
	};

}
#endif