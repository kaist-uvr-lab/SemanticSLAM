#include "FrameVisualizer.h"
#include "Frame.h"
#include "System.h"
#include "MapPoint.h"
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
	void UVR_SLAM::FrameVisualizer::Run(){
	
		while (1) {
			if (isVisualize()) {
				//std::cout << "FrameVisualizer::Start" << std::endl;
				Frame* pKF = mpKeyFrame;
				Frame* pF = mpFrame;

				cv::Mat vis = pF->GetOriginalImage();
				//vis.convertTo(vis, CV_8UC3);
				cv::Mat R = pF->GetRotation();
				cv::Mat t = pF->GetTranslation();
				
				int nMatch = 0;
				for (int i = 0; i < mvpMatchingMPs.size(); i++) {
					UVR_SLAM::MapPoint* pMPi = mvpMatchingMPs[i];
					if (!pMPi || pMPi->isDeleted())
						continue;
					cv::Point2f p2D;
					cv::Mat pCam;
					bool b = pMPi->Projection(p2D, pCam, R, t, mK, mnWidth, mnHeight);

					int label = 0;// mpRefKF->mpMatchInfo->mvObjectLabels[vnIDXs[i]];
					int pid = pMPi->GetPlaneID();
					int type = pMPi->GetRecentLayoutFrameID();
					cv::Scalar color(150, 150, 0);
					if (pid > 0 && label == 150) {
						color = cv::Scalar(0, 0, 255);
					}
					else if (pid > 0 && label == 100) {
						color = cv::Scalar(0, 255, 0);
					}
					//else if (pid > 0 && label == 255) {
					else if (label == 255) {
						color = cv::Scalar(255, 0, 0);
						//color = UVR_SLAM::ObjectColors::mvObjectLabelColors[pid];
					}
					if (pid <= 0)
						color /= 2;

					if (mvbMatchingInliers[i])
						cv::circle(vis, p2D, 2, color, -1);
					nMatch++;
					//cv::line(vis, p2D, mvMatchingPTs[i], color, 1);
				}
				std::stringstream ss;
				ss << "Traking = "<<mpKeyFrame->GetKeyFrameID()<<", "<<mpFrame->GetFrameID()<<", "<< nMatch << "::" <<mfTime<< "::";
				cv::rectangle(vis, cv::Point2f(0, 0), cv::Point2f(vis.cols, 30), cv::Scalar::all(0), -1);
				cv::putText(vis, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
				cv::imshow("Output::Tracking", vis);
				//std::cout << "FrameVisualizer::End" << std::endl;
				SetBoolVisualize(false);
			}//visualize
		}
	}

	void FrameVisualizer::SetFrameMatchingInformation(Frame* pKF, Frame* pF, std::vector<UVR_SLAM::MapPoint*> vMPs, std::vector<cv::Point2f> vPts, std::vector<bool> vbInliers, float fTime) {
		std::unique_lock<std::mutex> lock(mMutexFrameVisualizer);
		mpKeyFrame = pKF;
		mpFrame = pF;
		mvpMatchingMPs.resize(vbInliers.size());
		mvMatchingPTs.resize(vbInliers.size());
		mvbMatchingInliers.resize(vbInliers.size());
		std::copy(vMPs.begin(), vMPs.end(), mvpMatchingMPs.begin());
		std::copy(vPts.begin(), vPts.end(), mvMatchingPTs.begin());
		std::copy(vbInliers.begin(), vbInliers.end(), mvbMatchingInliers.begin());
		mfTime = fTime;
		mbVisualize = true;
	}
	void FrameVisualizer::GetFrameMatchingInformation(Frame* pKF, Frame* pF, std::vector<UVR_SLAM::MapPoint*>& vMPs, std::vector<cv::Point2f>& vPts, std::vector<bool>& vbInliers) {

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