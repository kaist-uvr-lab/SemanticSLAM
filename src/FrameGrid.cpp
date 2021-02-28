#include <FrameGrid.h>
#include <Frame.h>
#include <CandidatePoint.h>
#include <SegmentationData.h>

namespace UVR_SLAM {
	FrameGrid::FrameGrid(){}
	FrameGrid::FrameGrid(cv::Point2f base, cv::Rect r, int level):basePt(std::move(base)), rect(std::move(r)),mbMatched(false), mpPrev(nullptr), mpNext(nullptr), mnLevel(level){
		mObjCount = cv::Mat::zeros(1, ObjectColors::mvObjectLabelColors.size(), CV_32SC1);
		mObjArea = cv::Mat::zeros(1, ObjectColors::mvObjectLabelColors.size(), CV_32FC1);
	}FrameGrid::FrameGrid(cv::Point2f base, int nsize, int level) : basePt(std::move(base)), mbMatched(false), mpPrev(nullptr), mpNext(nullptr), mnLevel(level) {
		rect = cv::Rect(basePt, std::move(cv::Point2f(basePt.x + nsize, basePt.y + nsize)));
		mObjCount = cv::Mat::zeros(1, ObjectColors::mvObjectLabelColors.size(), CV_32SC1);
		mObjArea = cv::Mat::zeros(1, ObjectColors::mvObjectLabelColors.size(), CV_32FC1);
	}
	FrameGrid::~FrameGrid(){}

	cv::Mat FrameGrid::CalcGradientImage(cv::Mat src) {
		cv::Mat temp = src(rect);
		cv::Mat edge;
		cv::cvtColor(temp, edge, CV_BGR2GRAY);
		edge.convertTo(edge, CV_8UC1);
		cv::Mat matDY, matDX, matGradient;
		cv::Sobel(edge, matDX, CV_64FC1, 1, 0, 3);
		cv::Sobel(edge, matDY, CV_64FC1, 0, 1, 3);
		matDX = abs(matDX);
		matDY = abs(matDY);
		matDX.convertTo(matDX, CV_8UC1);
		matDY.convertTo(matDY, CV_8UC1);
		matGradient = (matDX + matDY) / 2.0;
		return matGradient;
	}
	
	FrameGridKey::FrameGridKey(FrameGrid* key1, FrameGrid* key2):mpKey1(key1), mpKey2(key2){}
	FrameGridKey::~FrameGridKey(){}
	FrameGridLevelKey::FrameGridLevelKey(cv::Point2f pt, int level) : _pt(pt), _level(level) {}
	FrameGridLevelKey::~FrameGridLevelKey() {}
}