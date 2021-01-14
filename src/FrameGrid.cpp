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
	bool FrameGrid::CalcActivePoints(MatchInfo* pF, cv::Mat src, int gthresh, int& localthresh, cv::Point2f& pt, cv::Mat& occupied, int r) {
		std::vector<uchar> vecFromMat;
		for (int y = 0; y < src.rows; y++) {
			for (int x = 0; x < src.cols; x++) {
				int val = src.at<uchar>(y, x);
				if (val == 0)
					continue;
				vecFromMat.push_back(val);
			}
		}
		if (vecFromMat.size() < 5)
			return false;
		std::nth_element(vecFromMat.begin(), vecFromMat.begin() + vecFromMat.size() / 2, vecFromMat.end());

		cv::Point2f gridTempRect(5, 5);

		////median 값 계산
		int median = (int)vecFromMat[vecFromMat.size() / 2];
		localthresh = median + gthresh;
		
		////그리드 내 모든 포인트를 추가하는 경우
		int maxval = median;
		std::vector<std::pair<int, cv::Point2f>> vCandidateCPs;
		for (int y = 0; y < src.rows; y++) {
			for (int x = 0; x < src.cols; x++) {
				if (vCandidateCPs.size() >= 20)
					break;
				int val = src.at<uchar>(y, x);
				if(val > localthresh){
					
					cv::Point2f pt(x+ basePt.x, y+ basePt.y);
					vCandidateCPs.push_back(std::make_pair(val, pt));
					
				}//thresh
			}//for x
		}//fory
		std::sort(vCandidateCPs.begin(), vCandidateCPs.end(), [](const std::pair<int, cv::Point2f>&x, const std::pair<int, cv::Point2f>& y) {
			if (x.first == y.first)
				return x.second.x != y.second.x ? x.second.x > y.second.x : x.second.y > y.second.y;
			else
				return x.first > y.first;
		});
		for (size_t i = 0; i < vCandidateCPs.size(); i++) {
			auto pt = vCandidateCPs[i].second;
			auto val = vCandidateCPs[i].first;
			if (mvPTs.size() == 5)
				break;
			if (!pF->CheckOpticalPointOverlap(occupied, pt, r, 10)) {
				continue;
			}
			cv::rectangle(occupied, pt - gridTempRect, pt + gridTempRect, cv::Scalar(255, 0, 0), -1);
			auto pCP = new UVR_SLAM::CandidatePoint(pF->mpRefFrame);
			int idx = pF->AddCP(pCP, pt);
			mvPTs.push_back(pt);
			mvpCPs.push_back(pCP);
		}

		//std::cout << mvPTs.size() << std::endl;
		if (mvPTs.size() > 0)
			return true;
		return false;
		////그리드 내 모든 포인트를 추가하는 경우

		//////max 값 하나만 선택시
		/*double minVal, maxVal;
		cv::Point maxPt;
		cv::Point gPt;
		cv::minMaxLoc(src, &minVal, &maxVal, NULL, &gPt);

		int resVal = (int)maxVal;
		if (resVal > localthresh) {
			pt = cv::Point2f(gPt.x + basePt.x, gPt.y + basePt.y);
			vecPTs.push_back(pt);
			return true;
		}
		else
			return false;*/
		//////max 값 하나만 선택시
	}

	FrameGridKey::FrameGridKey(FrameGrid* key1, FrameGrid* key2):mpKey1(key1), mpKey2(key2){}
	FrameGridKey::~FrameGridKey(){}
	FrameGridLevelKey::FrameGridLevelKey(cv::Point2f pt, int level) : _pt(pt), _level(level) {}
	FrameGridLevelKey::~FrameGridLevelKey() {}
}