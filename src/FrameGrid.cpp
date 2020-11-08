#include <FrameGrid.h>
#include <CandidatePoint.h>

namespace UVR_SLAM {
	FrameGrid::FrameGrid(){}
	FrameGrid::FrameGrid(cv::Point2f base, cv::Rect r):basePt(std::move(base)), rect(std::move(r)),mbMatched(false){
		
	}FrameGrid::FrameGrid(cv::Point2f base, int nsize) : basePt(std::move(base)), mbMatched(false) {
		rect = cv::Rect(basePt, std::move(cv::Point2f(basePt.x + nsize, basePt.y + nsize)));
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
	bool FrameGrid::CalcActivePoints(cv::Mat src, int gthresh, cv::Point2f& pt) {
		std::vector<uchar> vecFromMat;
		cv::Mat mReshaped = src.reshape(0, 1); // spread Input Mat to single row
		mReshaped.copyTo(vecFromMat); // Copy Input Mat to vector vecFromMat
		//std::cout << "before::" << (int)vecFromMat[0] <<" "<< (int)vecFromMat[vecFromMat.size() / 2] << " " << (int)vecFromMat[vecFromMat.size() - 1] << std::endl;;
		std::nth_element(vecFromMat.begin(), vecFromMat.begin() + vecFromMat.size() / 2, vecFromMat.end());
		//std::cout <<"gra::"<< (int)vecFromMat[0]<<" "<< (int)vecFromMat[vecFromMat.size() / 2] << " " << (int)vecFromMat[vecFromMat.size() - 1] << std::endl;;
		
		double minVal, maxVal;
		cv::Point maxPt;
		cv::Point gPt;
		cv::minMaxLoc(src, &minVal, &maxVal, NULL, &gPt);
		/*std::cout << "test::" << std::endl;
		std::cout << "max::"<<maxVal << std::endl;
		std::cout << "min::" << minVal << std::endl;
		std::cout << "median::" << (int)vecFromMat[vecFromMat.size() / 2] << std::endl;
		for (int i = 0; i < vecFromMat.size(); i++) {
			std::cout <<i<<"="<< vecFromMat.size() / 2 <<"::"<< (int)vecFromMat[i] << std::endl;
		}*/
		int median = (int)vecFromMat[vecFromMat.size() / 2];
		int thresh = median + gthresh;
		int maxval = median;
		for (int y = 0; y < src.rows; y++) {
			for (int x = 0; x < src.cols; x++) {
				int val = src.at<uchar>(y, x);
				if(val > thresh){
					cv::Point2f tpt(x, y);
					//auto tpt = cv::Point2f(gPt.x + basePt.x, gPt.y + basePt.y);
					vecPTs.push_back(tpt+ basePt);
					if (val > maxval) {
						pt = tpt+basePt;
						maxval = val;
						mnMaxIDX = vecPTs.size() - 1; //vecpt+basept¸¦ ÇÏ¸é µÊ.
					}//max
				}//thresh
			}//for x
		}//fory
		if (vecPTs.size() > 0)
			return true;
		return false;
		/*int resVal = (int)maxVal;
		if (resVal > thresh) {
			pt = cv::Point2f(gPt.x + basePt.x, gPt.y + basePt.y);
			return true;
		}
		else
			return false;*/
	}
}