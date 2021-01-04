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
		FrameGrid(cv::Point2f base, cv::Rect r, int level);
		FrameGrid(cv::Point2f base, int size, int level);
		virtual ~FrameGrid();
	public:
		cv::Mat CalcGradientImage(cv::Mat src);
		bool CalcActivePoints(cv::Mat src, int gthresh, int& localthresh, cv::Point2f& pt);
	public:
		int mnLevel;
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
	class FrameGridKey {
	public:
		FrameGridKey(FrameGrid* key1, FrameGrid* key2);
		virtual ~FrameGridKey(); 

		FrameGrid* mpKey1;
		FrameGrid* mpKey2;
		bool operator <(const FrameGridKey& key) const
		{
			if (mpKey1 != key.mpKey1) {
				return mpKey1 < key.mpKey1;
			}
			else {
				return mpKey2 < key.mpKey2;
			}
		}
	private:
	};

	class FrameGridLevelKey {
	public:
		FrameGridLevelKey(cv::Point2f pt, int level);
		virtual ~FrameGridLevelKey();
		bool operator <(const FrameGridLevelKey& key)const
		{
			if (_level == key._level) {
				return _pt.x == key._pt.x ? _pt.y < key._pt.y : _pt.x < key._pt.x;
			}
			else
				return _level < key._level;
		}
		cv::Point2f _pt;
		int _level;
	};

}
#endif