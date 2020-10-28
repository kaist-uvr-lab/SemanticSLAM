#include "FrameVisualizer.h"
#include "Frame.h"
#include "System.h"
#include "MapPoint.h"
#include "Visualizer.h"
#include "CandidatePoint.h"
#include "Map.h"

namespace UVR_SLAM {
	UVR_SLAM::FrameVisualizer::FrameVisualizer(){}
	UVR_SLAM::FrameVisualizer::FrameVisualizer(System* pSys, int w, int h, cv::Mat K):mpSystem(pSys), mnWidth(w), mnHeight(h), mK(K), mbVisualize(false){
	}
	UVR_SLAM::FrameVisualizer::~FrameVisualizer(){}

	void UVR_SLAM::FrameVisualizer::Init(){
		mpMap = mpSystem->mpMap;
		mpVisualizer = mpSystem->mpVisualizer;
	}
	float vmin = 0.001;
	float vmax = 1.0;
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
		
		while (1) {

			if (isVisualize()) {
				//std::cout << "FrameVisualizer::Start" << std::endl;
				Frame* pKF = mpKeyFrame;
				Frame* pF = mpFrame;

				cv::Mat vis = pF->GetOriginalImage().clone();
				cv::Mat vis2 = pF->GetOriginalImage().clone();
				cv::Mat kfImg = pKF->GetOriginalImage().clone();
				auto pKFMatch = pKF->mpMatchInfo;
				//vis.convertTo(vis, CV_8UC3);
				cv::Mat R = pF->GetRotation();
				cv::Mat t = pF->GetTranslation();
				std::vector<MapPoint*> vpMPs;
				{
					std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateCP);
					mpSystem->cvUseCreateCP.wait(lock, [&] {return mpSystem->mbCreateCP; });
				}
				int nMatch = 0;
				for (size_t i = 0, iend = pF->mpMatchInfo->mvbMapPointInliers.size(); i < iend; i++){
					auto pCPi = pF->mpMatchInfo->mvpMatchingCPs[i];
					auto pt = pF->mpMatchInfo->mvMatchingPts[i];
					int nCP = pCPi->GetNumSize();

					if(nCP > mpSystem->mnThreshMinKF)
						cv::circle(vis2, pt, 3, color2, -1);
					else
						cv::circle(vis2, pt, 3, color1, -1);

					auto pMPi = pCPi->GetMP();
					bool bMP = pMPi && !pMPi->isDeleted() && pMPi->GetQuality();

					if(bMP){
						cv::Point2f p2D;
						cv::Mat pCam;
						float depth;
						bool b = pMPi->Projection(p2D, pCam, depth,R, t, mK, mnWidth, mnHeight);
						nMatch++;
						int label = pMPi->GetLabel();
						if(pF->mpMatchInfo->mvbMapPointInliers[i]){
							cv::circle(vis2, p2D, 3, ObjectColors::mvObjectLabelColors[label], -1);//
							cv::line(vis2, p2D, pt, color3, 2);
						}else{
							cv::circle(vis2, p2D, 5, color4, -1);//
						}

						//Depth
						cv::Scalar c = ConvertDepthToColor(depth);
						cv::circle(vis, p2D, 3, c, -1);//ObjectColors::mvObjectLabelColors[label]
					}
				}
				
				std::stringstream ss;
				ss << "Traking = "<<mpKeyFrame->mnKeyFrameID<<", "<<mpFrame->mnFrameID<<"="<< pF->mpMatchInfo->mvpMatchingCPs.size() <<"::"<< nMatch << "::" <<mfTime<< "::";
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