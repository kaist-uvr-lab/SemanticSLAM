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
				////////전전전 프레임의 매칭 포인트가 epi line을 제대로 따르는지 테스트
				//{
				//	Frame* pPrevFrame = nullptr;
				//	Frame* pPPrevFrame = nullptr;
				//	Frame* pPPPrevFrame = nullptr;
				//	pPrevFrame = pF->mpPrev;
				//	if (pPrevFrame) {
				//		pPPrevFrame = pPrevFrame->mpPrev;
				//	}
				//	if (pPPrevFrame) {
				//		pPPPrevFrame = pPPrevFrame->mpPrev;
				//	}
				//	if (pPPPrevFrame) {
				//		cv::Mat R, t, P;
				//		pF->GetPose(R, t);
				//		cv::hconcat(R, t, P);
				//		auto pMatcher = mpSystem->mpMatcher;
				//		auto mpGrids = pF->mmpFrameGrids;
				//		auto mbGrids = pF->mmbFrameGrids;
				//		int nRectWidth = mpGrids.begin()->second->rect.width;
				//		int nRectHeight = mpGrids.begin()->second->rect.height;
				//		int nRectWidth2 = nRectWidth*2;
				//		int nRectHeight2 = nRectHeight *2;
				//		cv::Mat base = cv::Mat::zeros(nRectHeight, nRectWidth, CV_8UC3);
				//		cv::rectangle(base, cv::Rect(0, 0, nRectWidth, nRectHeight), cv::Scalar(255, 0, 0), -1);
				//		cv::Mat base2 = cv::Mat::zeros(nRectHeight, nRectWidth, CV_8UC3);
				//		cv::rectangle(base2, cv::Rect(0, 0, nRectWidth, nRectHeight), cv::Scalar(0, 255, 0), -1);

				//		cv::Mat prevImg = pPPPrevFrame->GetOriginalImage().clone();
				//		cv::Mat currImg = pF->GetOriginalImage().clone();
				//		
				//		cv::Mat invK = mK.inv();
				//		cv::Mat Rrel, Trel;
				//		pF->GetRelativePoseFromTargetFrame(pPPPrevFrame, Rrel, Trel);
				//		float fx = mK.at<float>(0, 0);
				//		float fy = mK.at<float>(1, 1);
				//		float cx = mK.at<float>(0, 2);
				//		float cy = mK.at<float>(1, 2);

				//		/////zmssd
				//		int patch_size = UVR_SLAM::ZMSSD::patch_size_;
				//		int patch_half_size = patch_size / 2;
				//		int patch_area = patch_size * patch_size;
				//		int mzssd_thresh = 2000 * patch_area;
				//		cv::Point2f patch_pt(patch_half_size, patch_half_size);
				//		/////zmssd

				//		for (auto iter = mpGrids.begin(), iend = mpGrids.end(); iter != iend; iter++) {
				//			auto pt = iter->first;
				//			auto pCurrGrid = iter->second;
				//			if (!pCurrGrid)
				//				continue;
				//			if (!mbGrids[pt])
				//				continue;
				//			bool bPrevGrid = false;
				//			auto tempGrid = pCurrGrid->mpPrev;
				//			if (tempGrid) { //prev
				//				tempGrid = tempGrid->mpPrev;
				//			}
				//			if(tempGrid)	//pprev
				//				tempGrid = tempGrid->mpPrev;
				//			if (tempGrid) {
				//				auto pPrevGrid = tempGrid;

				//				auto currRect = pCurrGrid->rect;
				//				auto prevRect = pPrevGrid->rect;
				//				cv::Mat temp1 = prevImg(prevRect);
				//				cv::Mat temp2 = currImg(currRect);

				//				auto prevPt = pPrevGrid->pt;
				//				auto currPt = pCurrGrid->pt;

				//				cv::Mat ray = mpSystem->mInvK*(cv::Mat_<float>(3, 1) << prevPt.x, prevPt.y, 1.0);
				//				float z_min, z_max;
				//				z_min = 0.01f;
				//				z_max = 1.0f;
				//				cv::Point2f XimgMin, XimgMax;
				//				pMatcher->ComputeEpiLinePoint(XimgMin, XimgMax, ray, z_min, z_max, Rrel, Trel, mK);
				//				cv::Mat lineEqu = pMatcher->ComputeLineEquation(XimgMin, XimgMax);
				//				bool bEpiConstraints = pMatcher->CheckLineDistance(lineEqu, currPt, 1.0);

				//				////라인 포인트 그리드를 이용하여 획득 과정
				//				auto extendedPt = pF->GetExtendedRect(currPt, nRectWidth);
				//				bool bSlope = abs(lineEqu.at<float>(0) / lineEqu.at<float>(1)) > 1.0;
				//				cv::Point2f uv_best;
				//				float zmssd_best = mzssd_thresh;
				//				float a = lineEqu.at <float>(0);
				//				float b = lineEqu.at <float>(1);
				//				float c = lineEqu.at <float>(2);
				//				cv::Point2f sPt3, ePt3;
				//				
				//				//////ref pt in previous image
				//				auto refLeftPt = prevPt - patch_pt;
				//				auto refRightPt = prevPt + patch_pt;
				//				bool bRefLeft = pF->isInImage(refLeftPt.x, refLeftPt.y, 10);
				//				bool bRefRight = pF->isInImage(refRightPt.x, refRightPt.y, 10);
				//				SSD* refZMSSD = nullptr;
				//				if (bRefLeft && bRefRight) {
				//					cv::Rect refRect(prevPt - patch_pt, prevPt + patch_pt);
				//					refZMSSD = new SSD(pPPPrevFrame->GetOriginalImage()(refRect).clone());
				//				}
				//				//////ref pt in previous image
				//				
				//				//라인 포인트 그리드를 이용하여 획득 과정
				//				if (bSlope) {
				//					sPt3 = pMatcher->CalcLinePoint(extendedPt.y, lineEqu, bSlope);
				//					ePt3 = pMatcher->CalcLinePoint(extendedPt.y + nRectHeight2, lineEqu, bSlope);

				//					if (refZMSSD) {
				//						for (float i = sPt3.y, iend = ePt3.y; i < iend; i+=0.1) {
				//							float x = (-b*i - c) / a;
				//							cv::Point2f pt(x, i);
				//							auto leftPt = pt - patch_pt;
				//							auto rightPt = pt + patch_pt;
				//							bool bPatchLeft = pF->isInImage(leftPt.x, leftPt.y, 10);
				//							bool bPatchRight = pF->isInImage(rightPt.x, rightPt.y, 10);
				//							if (bPatchLeft && bPatchRight) {
				//								cv::Rect patchRect(leftPt, rightPt);
				//								float val = refZMSSD->computeScore(pF->GetOriginalImage()(patchRect).clone());
				//								if (val <zmssd_best) {
				//									uv_best = pt;
				//									zmssd_best = val;
				//								}
				//							}//if patch
				//							//cv::circle(currImg, pt, 3, cv::Scalar(255, 255, 0), 1);
				//						}
				//					}

				//				}else{
				//					sPt3 = pMatcher->CalcLinePoint(extendedPt.x, lineEqu, bSlope);
				//					ePt3 = pMatcher->CalcLinePoint(extendedPt.x+ nRectWidth2, lineEqu, bSlope);
				//					if (refZMSSD) {
				//						for (float i = sPt3.x, iend = ePt3.x; i < iend; i+= 0.1) {
				//							float y = (-a*i - c) / b;
				//							cv::Point2f pt(i,y);
				//							auto leftPt = pt - patch_pt;
				//							auto rightPt = pt + patch_pt;
				//							bool bPatchLeft = pF->isInImage(leftPt.x, leftPt.y, 10);
				//							bool bPatchRight = pF->isInImage(rightPt.x, rightPt.y, 10);
				//							if (bPatchLeft && bPatchRight) {
				//								cv::Rect patchRect(leftPt, rightPt);
				//								float val = refZMSSD->computeScore(pF->GetOriginalImage()(patchRect).clone());
				//								if (val < zmssd_best) {
				//									uv_best = pt;
				//									zmssd_best = val;
				//								}
				//							}//if patch
				//							//cv::circle(currImg, pt, 3, cv::Scalar(255, 255, 0), 1);
				//						}
				//					}
				//				}
				//				//라인 포인트 그리드를 이용하여 획득 과정

				//				if (zmssd_best < mzssd_thresh) {//mzssd_thresh
				//					//cv::line(currImg, prevPt, uv_best, cv::Scalar(0, 255, 255), 1);
				//					cv::circle(currImg, uv_best, 3, cv::Scalar(0, 0, 255), -1);
				//				}
				//				
				//				auto pCP = pCurrGrid->mpCP;
				//				////사각형 시각화
				//				/*if (!pCP) {
				//					cv::addWeighted(temp1, 0.5, base2, 0.5, 0.0, temp1);
				//					cv::addWeighted(temp2, 0.5, base2, 0.5, 0.0, temp2);
				//				}
				//				else {
				//					cv::addWeighted(temp1, 0.5, base, 0.5, 0.0, temp1);
				//					cv::addWeighted(temp2, 0.5, base, 0.5, 0.0, temp2);
				//				}*/
				//				////사각형 시각화

				//				///////시드 테스트
				//				//{
				//				//	auto pSeed = pCP->mpSeed;
				//				//	if (pSeed) {

				//				//		/*cv::Mat X3D;
				//				//		float depth;
				//				//		pCP->CreateMapPoint(X3D, depth, mK, mpSystem->mInvK, P, R, t, currPt);*/

				//				//		float z_inv_min = pSeed->mu + sqrt(pSeed->sigma2);
				//				//		float z_inv_max = max(pSeed->mu - sqrt(pSeed->sigma2), 0.00000001f);
				//				//		float z_min2 = 1. / z_inv_min;
				//				//		float z_max2 = 1. / z_inv_max;
				//				//		cv::Point2f XimgMin2, XimgMax2;
				//				//		cv::Mat Rrel2, Trel2;
				//				//		pF->GetRelativePoseFromTargetFrame(pCP->mpRefKF, Rrel2, Trel2);
				//				//		pMatcher->ComputeEpiLinePoint(XimgMin2, XimgMax2, pSeed->ray, z_min2, z_max2, Rrel2, Trel2, mK);
				//				//		cv::Mat lineEqu2 = pMatcher->ComputeLineEquation(XimgMin2, XimgMax2);
				//				//		bool bEpiConstraints1 = pMatcher->CheckLineDistance(lineEqu2, currPt, 1.0);

				//				//		bool bSlope = abs(lineEqu2.at<float>(0) / lineEqu2.at<float>(1)) > 1.0;
				//				//		cv::Point2f sPt2, ePt2;
				//				//		if (bSlope) {
				//				//			sPt2 = pMatcher->CalcLinePoint(currRect.y, lineEqu2, bSlope);
				//				//			ePt2 = pMatcher->CalcLinePoint(currRect.y + currRect.height, lineEqu2, bSlope);
				//				//		}
				//				//		else {
				//				//			sPt2 = pMatcher->CalcLinePoint(currRect.x, lineEqu2, bSlope);
				//				//			ePt2 = pMatcher->CalcLinePoint(currRect.x + currRect.width, lineEqu2, bSlope);
				//				//		}
				//				//		if (bEpiConstraints1) {
				//				//			cv::line(currImg, sPt2, ePt2, cv::Scalar(0, 0, 255), 1);
				//				//		}
				//				//		else {
				//				//			cv::line(currImg, sPt2, ePt2, cv::Scalar(255, 0, 255), 1);
				//				//		}

				//				//		////depth 추정
				//				//		/*cv::Mat temp1 = Rrel2*pSeed->ray;
				//				//		cv::Mat temp2 = mpSystem->mInvK*(cv::Mat_<float>(3, 1) << currPt.x, currPt.y, 1.0);
				//				//		cv::Mat A = cv::Mat::zeros(3, 2, CV_32FC1);
				//				//		temp1.copyTo(A.col(0));
				//				//		temp2.copyTo(A.col(1));
				//				//		cv::Mat AtA = A.t()*A;
				//				//		if (cv::determinant(AtA) >= 0.000001) {
				//				//			cv::Mat depthMat = AtA.inv()*A.t()*Trel2;
				//				//			float depth = fabs(depthMat.at<float>(0));
				//				//			cv::Mat Xw = Rrel2*pSeed->ray*depth + Trel2;
				//				//			temp1 = mK*Xw;
				//				//			float d1 = temp1.at<float>(2);
				//				//			cv::Point2f est(temp1.at<float>(0) / d1, temp1.at<float>(1) / d1);
				//				//			cv::circle(currImg, est, 3, cv::Scalar(255, 255, 0), -1);
				//				//		}*/
				//				//		////depth 추정
				//				//		/*cv::Point2f est(0,0);
				//				//		float d = pSeed->ComputeDepth(est, currPt, Rrel2, Trel2, mK);*/
				//				//		//std::cout << est << std::endl;
				//				//	}
				//				//}
				//				///////시드 테스트

				//				cv::circle(prevImg, pPrevGrid->pt, 2, cv::Scalar(255, 0, 0),-1);
				//				cv::circle(currImg, pCurrGrid->pt, 2, cv::Scalar(255, 0, 0), -1);
				//				
				//				/*cv::line(prevImg, prevPt, pF->GetGridBasePt(prevPt, nRectWidth), cv::Scalar(0, 255, 0), 1);
				//				cv::line(currImg, currPt, pF->GetGridBasePt(currPt, nRectWidth), cv::Scalar(0, 255, 0), 1);*/

				//				if (bEpiConstraints) {
				//					cv::line(currImg, sPt3, ePt3, cv::Scalar(0, 255, 0), 1);
				//				}
				//				else {
				//					cv::line(currImg, sPt3, ePt3, cv::Scalar(0, 255, 255), 1);
				//				}

				//			}
				//		}//for

				//		 cv::Mat debugMatch = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
				//		 cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
				//		 cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
				//		 prevImg.copyTo(debugMatch(mergeRect1));
				//		 currImg.copyTo(debugMatch(mergeRect2));
				//		 cv::moveWindow("Output::MatchTest2", mpSystem->mnDisplayX+prevImg.cols*2, mpSystem->mnDisplayY);
				//		 cv::imshow("Output::MatchTest2", debugMatch); cv::waitKey(1);

				//	}//if
				//}
				////////전전전 프레임의 매칭 포인트가 epi line을 제대로 따르는지 테스트


				///////그리드 관련 시각화. 현재 막음
				////오브젝트 정보가 얼마나 전달이 되는지 확인함.
				//auto mpGrids = pF->mmpFrameGrids;
				//auto mbGrids = pF->mmbFrameGrids;
				//if (mpGrids.size() > 0){
				//	int nRectWidth = mpGrids.begin()->second->rect.width;
				//	int nRectHeight = mpGrids.begin()->second->rect.height;

				//	cv::Mat base = cv::Mat::zeros(nRectHeight, nRectWidth, CV_8UC3);
				//	cv::rectangle(base, cv::Rect(0, 0, nRectWidth, nRectHeight), cv::Scalar(255, 0, 0), -1);
				//	cv::Mat base2 = cv::Mat::zeros(nRectHeight, nRectWidth, CV_8UC3);
				//	cv::rectangle(base2, cv::Rect(0, 0, nRectWidth, nRectHeight), cv::Scalar(0, 255, 0), -1);
				//	cv::Mat base3 = cv::Mat::zeros(nRectHeight, nRectWidth, CV_8UC3);

				//	for (auto iter = mpGrids.begin(), iend = mpGrids.end(); iter != iend; iter++) {
				//		auto pt = iter->first;
				//		auto pGrid = iter->second;
				//		if (!pGrid)
				//			continue;
				//		
				//		auto rect = pGrid->rect;
				//		cv::Mat temp = vis2(rect);
				//		
				//		if (pGrid->mmObjCounts.size() > 0) {
				//			auto iter = pGrid->mmObjCounts.begin();
				//			int label = iter->first;
				//			cv::rectangle(base3, cv::Rect(0, 0, nRectWidth, nRectHeight), ObjectColors::mvObjectLabelColors[label], -1);
				//			cv::addWeighted(temp, 0.5, base3, 0.5, 0.0, temp);
				//		}

				//		/*if (!mbGrids[pt])
				//			continue;
				//		auto pCP = pGrid->mpCP;
				//		if (!pCP) {
				//			cv::addWeighted(temp, 0.5, base2, 0.5, 0.0, temp);
				//		}
				//		else {
				//			cv::addWeighted(temp, 0.5, base, 0.5, 0.0, temp);
				//		}*/
				//	}
				//}
				///////그리드 관련 시각화
				
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
				
				cv::Mat resized2;
				cv::resize(vis2, resized2, cv::Size(vis.cols / 2, vis.rows / 2));
				mpVisualizer->SetOutputImage(resized2, 3);
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