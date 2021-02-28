#include "FrameVisualizer.h"
#include "Frame.h"
#include "System.h"
#include "Matcher.h"
#include "MapPoint.h"
#include "Visualizer.h"
#include "CandidatePoint.h"
#include "Map.h"
#include "FrameGrid.h"
#include "DepthFilter.h"
#include "ZMSSD.h"

namespace UVR_SLAM {
	UVR_SLAM::FrameVisualizer::FrameVisualizer(){}
	UVR_SLAM::FrameVisualizer::FrameVisualizer(System* pSys):mpSystem(pSys), mbVisualize(false){
	}
	UVR_SLAM::FrameVisualizer::~FrameVisualizer(){}

	void UVR_SLAM::FrameVisualizer::Init(){
		mpMap = mpSystem->mpMap;
		mpVisualizer = mpSystem->mpVisualizer;
	}
	float vmin = 0.001;
	float vmax = 4.0;
	cv::Scalar ConvertDepthToColor(float v) {
		float dv;

		if (v < vmin)
			v = vmin;
		if (v > vmax)
			v = vmax;
		dv = vmax - vmin;
		float r = 1.0, g = 1.0, b = 1.0;
		if (v < (vmin + 0.25 * dv)) {
			r = 0;
			g = 4 * (v - vmin) / dv;
		}
		else if (v < (vmin + 0.5 * dv)) {
			r = 0;
			b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
		}
		else if (v < (vmin + 0.75 * dv)) {
			r = 4 * (v - vmin - 0.5 * dv) / dv;
			b = 0;
		}
		else {
			g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
			b = 0;
		}
		return cv::Scalar(r * 255, g * 255, b * 255);
	}
	void UVR_SLAM::FrameVisualizer::Run(){

		cv::Scalar color1(255, 255, 0);
		cv::Scalar color2(255,0,255);
		cv::Scalar color3(0, 255, 255);
		cv::Scalar color4(0, 0, 255);
		cv::Scalar color5(0, 255, 0);
		cv::Scalar color6(255, 0, 0);
		
		while (1) {

			if (isVisualize()) {
				//std::cout << "FrameVisualizer::Start" << std::endl;

				

				Frame* pKF = mpKeyFrame;
				Frame* pF = mpFrame;

				cv::Mat vis = pF->GetOriginalImage().clone();


				std::vector<cv::Point2f> vTrackingPTs = mpMap->GetTrackingPoints();
				cv::Scalar tColor(255, 0, 0);
				for (size_t i = 0, iend = vTrackingPTs.size(); i < iend; i++) {
					auto pt = vTrackingPTs[i];
					cv::circle(vis, pt, 3, tColor, -1);//ObjectColors::mvObjectLabelColors[label]
				}

				
				std::stringstream ss;
				ss << "Traking = "<<mpKeyFrame->mnKeyFrameID<<", "<<mpFrame->mnFrameID<<"="<< vTrackingPTs.size() << "::" <<mfTime<< "::";
				cv::rectangle(vis, cv::Point2f(0, 0), cv::Point2f(vis.cols, 30), cv::Scalar::all(0), -1);
				cv::putText(vis, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));

				/*cv::Mat res = cv::Mat::zeros(mnHeight * 2, mnWidth, CV_8UC3);
				cv::Rect rect1 = cv::Rect(0, 0, mnWidth, mnHeight);
				cv::Rect rect2 = cv::Rect(0, mnHeight, mnWidth, mnHeight);
				vis.copyTo(res(rect1));
				kfImg.copyTo(res(rect2));*/
				
				cv::Mat resized;
				cv::resize(vis, resized, cv::Size(vis.cols/2, vis.rows/2));
				mpVisualizer->SetOutputImage(resized, 0);
				
				/*cv::Mat resized2;
				cv::resize(vis2, resized2, cv::Size(vis.cols / 2, vis.rows / 2));
				mpVisualizer->SetOutputImage(resized2, 3);*/
				SetBoolVisualize(false);
			}//visualize
		}
	}

	void FrameVisualizer::SetFrameMatchingInformation(Frame* pKF, Frame* pF, float fTime) {
		std::unique_lock<std::mutex> lock(mMutexFrameVisualizer);
		mpKeyFrame = pKF;
		mpFrame = pF;
		/*
		mvpMatchingMPs.resize(vbInliers.size());
		mvMatchingPTs.resize(vbInliers.size());
		mvbMatchingInliers.resize(vbInliers.size());
		std::copy(vMPs.begin(), vMPs.end(), mvpMatchingMPs.begin());
		std::copy(vPts.begin(), vPts.end(), mvMatchingPTs.begin());
		std::copy(vbInliers.begin(), vbInliers.end(), mvbMatchingInliers.begin());*/
		mfTime = fTime;
		mbVisualize = true;
	}

	bool FrameVisualizer::isVisualize() {
		std::unique_lock<std::mutex> lock(mMutexFrameVisualizer);
		return mbVisualize;
	}
	void FrameVisualizer::SetBoolVisualize(bool b) {
		std::unique_lock<std::mutex> lock(mMutexFrameVisualizer);
		mbVisualize = b;
	}
}