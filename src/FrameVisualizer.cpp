#include "FrameVisualizer.h"
#include "Frame.h"
#include "System.h"
#include "MapPoint.h"
#include "Visualizer.h"
#include "CandidatePoint.h"
#include "Map.h"

namespace UVR_SLAM {
	UVR_SLAM::FrameVisualizer::FrameVisualizer(){}
	UVR_SLAM::FrameVisualizer::FrameVisualizer(int w, int h, cv::Mat K, Map* pMap):mnWidth(w), mnHeight(h), mK(K), mbVisualize(false){
		mpMap = pMap;
	}
	UVR_SLAM::FrameVisualizer::~FrameVisualizer(){}

	void UVR_SLAM::FrameVisualizer::SetSystem(System* pSystem){
		mpSystem = pSystem;
	}
	void UVR_SLAM::FrameVisualizer::SetVisualizer(Visualizer* pVis) {
		mpVisualizer = pVis;
	}
	void UVR_SLAM::FrameVisualizer::Run(){
	
		while (1) {

			cv::Scalar color1(255, 255, 0);
			cv::Scalar color2(0, 255, 255);

			if (isVisualize()) {
				//std::cout << "FrameVisualizer::Start" << std::endl;
				Frame* pKF = mpKeyFrame;
				Frame* pF = mpFrame;

				cv::Mat vis = pF->GetOriginalImage();
				cv::Mat kfImg = pKF->GetOriginalImage();
				auto pKFMatch = pKF->mpMatchInfo;
				//vis.convertTo(vis, CV_8UC3);
				cv::Mat R = pF->GetRotation();
				cv::Mat t = pF->GetTranslation();
				std::vector<MapPoint*> vpMPs;
				int nMatch = 0;
				int nCP = pF->mpMatchInfo->GetNumCPs();
				for (int i = 0; i < nCP; i++) {
					auto pCPi = pF->mpMatchInfo->mvpMatchingCPs[i];
					auto pMPi = pCPi->GetMP();
					if (!pMPi || pMPi->isDeleted() || !pMPi->GetQuality())
						continue;
					
					cv::Point2f p2D;
					cv::Mat pCam;
					bool b = pMPi->Projection(p2D, pCam, R, t, mK, mnWidth, mnHeight);
					auto pt = pF->mpMatchInfo->mvMatchingPts[i];
					nMatch++;
					int label = pMPi->GetLabel();
					cv::circle(vis, p2D, 3, ObjectColors::mvObjectLabelColors[label], -1);
					cv::line(vis, p2D, pt, color2, 2);
				}
				
				std::stringstream ss;
				ss << "Traking = "<<mpKeyFrame->GetKeyFrameID()<<", "<<mpFrame->GetFrameID()<<", "<< nMatch << "::" <<mfTime<< "::";
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