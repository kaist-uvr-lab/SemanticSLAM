#include <FrameGrid.h>
#include <CandidatePoint.h>
#include <SegmentationData.h>

namespace UVR_SLAM {
	FrameGrid::FrameGrid(){}
	FrameGrid::FrameGrid(cv::Point2f base, cv::Rect r):basePt(std::move(base)), rect(std::move(r)),mbMatched(false), mpPrev(nullptr), mpNext(nullptr){
		mObjCount = cv::Mat::zeros(1, ObjectColors::mvObjectLabelColors.size(), CV_32SC1);
		mObjArea = cv::Mat::zeros(1, ObjectColors::mvObjectLabelColors.size(), CV_32FC1);
	}FrameGrid::FrameGrid(cv::Point2f base, int nsize) : basePt(std::move(base)), mbMatched(false), mpPrev(nullptr), mpNext(nullptr) {
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
	bool FrameGrid::CalcActivePoints(cv::Mat src, int gthresh, int& localthresh, cv::Point2f& pt) {
		std::vector<uchar> vecFromMat;
		cv::Mat mReshaped = src.reshape(0, 1); // spread Input Mat to single row
		mReshaped.copyTo(vecFromMat); // Copy Input Mat to vector vecFromMat
		//std::cout << "before::" << (int)vecFromMat[0] <<" "<< (int)vecFromMat[vecFromMat.size() / 2] << " " << (int)vecFromMat[vecFromMat.size() - 1] << std::endl;;
		std::nth_element(vecFromMat.begin(), vecFromMat.begin() + vecFromMat.size() / 2, vecFromMat.end());
		//std::cout <<"gra::"<< (int)vecFromMat[0]<<" "<< (int)vecFromMat[vecFromMat.size() / 2] << " " << (int)vecFromMat[vecFromMat.size() - 1] << std::endl;;
		
		

		/*std::cout << "test::" << std::endl;
		std::cout << "max::"<<maxVal << std::endl;
		std::cout << "min::" << minVal << std::endl;
		std::cout << "median::" << (int)vecFromMat[vecFromMat.size() / 2] << std::endl;
		for (int i = 0; i < vecFromMat.size(); i++) {
			std::cout <<i<<"="<< vecFromMat.size() / 2 <<"::"<< (int)vecFromMat[i] << std::endl;
		}*/
		////median 값 계산
		int median = (int)vecFromMat[vecFromMat.size() / 2];
		localthresh = median + gthresh;
		
		////그리드 내 모든 포인트를 추가하는 경우
		int maxval = median;
		for (int y = 0; y < src.rows; y++) {
			for (int x = 0; x < src.cols; x++) {
				int val = src.at<uchar>(y, x);
				if(val > localthresh){
					cv::Point2f tpt(x, y);
					//auto tpt = cv::Point2f(gPt.x + basePt.x, gPt.y + basePt.y);
					auto apt = tpt + basePt;
					vecPTs.push_back(apt);
					if (val > maxval) {
						pt = apt;
						maxval = val;
						mnMaxIDX = vecPTs.size() - 1; //vecpt+basept를 하면 됨.
					}//max
				}//thresh
			}//for x
		}//fory

		if (vecPTs.size() > 0)
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
}