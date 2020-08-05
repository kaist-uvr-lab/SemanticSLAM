#include "CandidatePoint.h"
#include "Frame.h"
#include "Map.h"
#include "MapPoint.h"

namespace  UVR_SLAM{
	CandidatePoint::CandidatePoint():octave(0), bCreated(false) {}
	CandidatePoint::CandidatePoint(MatchInfo* pRefKF, int aoct):mpRefKF(pRefKF), octave(aoct), bCreated(false), mbDelete(false){

	}
	CandidatePoint::~CandidatePoint(){}

	std::map<MatchInfo*, int> CandidatePoint::GetFrames() {
		std::unique_lock<std::mutex> lockMP(mMutexCP);
		return std::map<UVR_SLAM::MatchInfo*, int>(mmpFrames.begin(), mmpFrames.end());
	}
	void CandidatePoint::AddFrame(MatchInfo* pF, cv::Point2f pt) {
		std::unique_lock<std::mutex> lockMP(mMutexCP);
		auto res = mmpFrames.find(pF);
		if (res == mmpFrames.end()) {
			int idx = pF->AddCP(this, pt);
			mmpFrames.insert(std::pair<UVR_SLAM::MatchInfo*, int>(pF, idx));
			mnConnectedFrames++;
		}
	}
	void CandidatePoint::RemoveFrame(UVR_SLAM::MatchInfo* pKF){
		{
			std::unique_lock<std::mutex> lockMP(mMutexCP);
			auto res = mmpFrames.find(pKF);
			if (res != mmpFrames.end()) {
				int idx = res->second;
				res = mmpFrames.erase(res);
				mnConnectedFrames--;
				pKF->RemoveCP(idx);
				if (this->mpRefKF == mpRefKF) {
					mpRefKF = mmpFrames.begin()->first;
				}
				if (mnConnectedFrames < 3)
					mbDelete = true;
			}
		}
		if (mbDelete) {
			Delete();
		}
	}
	void CandidatePoint::Delete() {
		{
			std::unique_lock<std::mutex> lockMP(mMutexCP);
			mbDelete = true;
			for (auto iter = mmpFrames.begin(); iter != mmpFrames.end(); iter++) {
				auto* pF = iter->first;
				auto idx = iter->second;
				pF->RemoveCP(idx);
			}
			//mnConnectedFrames = 0;
			mmpFrames.clear();
		}
	}

	int CandidatePoint::GetPointIndexInFrame(MatchInfo* pF) {
		std::unique_lock<std::mutex> lock(mMutexCP);
		auto res = mmpFrames.find(pF);
		if (res == mmpFrames.end())
			return -1;
		else
			return res->second;
	}

	float CandidatePoint::CalcParallax(cv::Mat Rkf1c, cv::Mat Rkf2c, cv::Point2f pt1, cv::Point2f pt2, cv::Mat invK) {
		cv::Mat xn1 = (cv::Mat_<float>(3, 1) << pt1.x, pt1.y, 1.0);
		cv::Mat xn2 = (cv::Mat_<float>(3, 1) << pt2.x, pt2.y, 1.0);
		cv::Mat ray1 = Rkf1c*invK*xn1;
		cv::Mat ray2 = Rkf2c*invK*xn2;
		float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1)*cv::norm(ray2));
		return cosParallaxRays;
	}
	cv::Mat CandidatePoint::Triangulate(cv::Point2f pt1, cv::Point2f pt2, cv::Mat P1, cv::Mat P2) {

		cv::Mat A(4, 4, CV_32F);
		A.row(0) = pt1.x*P1.row(2) - P1.row(0);
		A.row(1) = pt1.y*P1.row(2) - P1.row(1);
		A.row(2) = pt2.x*P2.row(2) - P2.row(0);
		A.row(3) = pt2.y*P2.row(2) - P2.row(1);

		cv::Mat u, w, vt;
		cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
		cv::Mat x3D = vt.row(3).t();
		x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
		return x3D;
	}

	bool CandidatePoint::CheckDepth(float depth) {
		if (depth < 0)
			return false;
		return true;
	}

	bool CandidatePoint::CheckReprojectionError(cv::Mat x3D, cv::Mat K, cv::Point2f pt, float thresh) {
		cv::Mat reproj1 = K*x3D;
		reproj1 /= x3D.at<float>(2);
		float squareError1 = (reproj1.at<float>(0) - pt.x)*(reproj1.at<float>(0) - pt.x) + (reproj1.at<float>(1) - pt.y)*(reproj1.at<float>(1) - pt.y);
		if (squareError1>thresh)
			return false;
		return true;
	}
	bool CandidatePoint::DelayedTriangulate(Map* pMap, MatchInfo* pMatch, cv::Point2f pt, MatchInfo* pPPrevMatch, MatchInfo* pPrevMatch, cv::Mat& debug) {
		
		int mnWidth = debug.cols / 3;
		cv::Point2f ptLeft1 = cv::Point2f(mnWidth, 0);
		cv::Point2f ptLeft2 = cv::Point2f(mnWidth * 2, 0);
		
		/////////
		float minParallax = FLT_MAX;
		cv::Point2f minPt;
		cv::Mat minP;
		std::vector<cv::Mat> vPs, vRs, vTs;
		std::vector<cv::Point2f> vPts;
		std::vector<MatchInfo*> vpMatches;
		auto targetKF = pMatch->mpRefFrame;
		cv::Mat K = targetKF->mK.clone();
		cv::Mat invK = K.inv();;
		cv::Mat Rt, Tt, Pt;
		targetKF->GetPose(Rt, Tt);
		cv::hconcat(Rt, Tt, Pt);
		//Rt = Rt.t();

		for (auto iter = mmpFrames.begin(); iter != mmpFrames.end(); iter++) {
			
			auto pKF = iter->first->mpRefFrame;
			auto tempInfo = iter->first;
			int tempIdx = iter->second;
			auto tempPt = tempInfo->GetCPPt(tempIdx);;

			cv::Mat tempR, tempT, tempP;
			pKF->GetPose(tempR, tempT);
			cv::hconcat(tempR, tempT, tempP);

			//calcparalalx
			float val = CalcParallax(Rt.t(), tempR.t(), pt, tempPt, invK);
			
			if (val < minParallax) {
				minParallax = val;
				minPt = tempPt;
				minP = tempP.clone();
			}
			vPs.push_back(tempP);
			vPts.push_back(tempPt);
			vRs.push_back(tempR);
			vTs.push_back(tempT);
			vpMatches.push_back(tempInfo);
		}

		vPs.push_back(Pt);
		vRs.push_back(Rt);
		vTs.push_back(Tt);
		vPts.push_back(pt);
		vpMatches.push_back(pMatch);
		//std::cout << "MIN::" << minParallax <<"::"<<mmpFrames.size()<< std::endl;
		if (minParallax < 0.9995f) {
			//cv::Mat P;
			//cv::Mat K;
			cv::Mat x3D = Triangulate(minPt, pt, K*minP, K*Pt);
		
			std::vector<bool> vbInliers(mmpFrames.size(), false);
			bool bSuccess = true;
			for (int i = 0; i < vPs.size(); i++) {
				cv::Mat xcam = vRs[i] * x3D + vTs[i];
				float depth = xcam.at<float>(2);
				if (CheckDepth(depth) && CheckReprojectionError(xcam, K, vPts[i], 9.0)) {
					vbInliers[i] = true;
				}else{
					bSuccess = false;
					this->RemoveFrame(vpMatches[i]);
					//break;
				}
			}
			//float fRatio = ((float)nSuccess / mmpFrames.size());

			////////시각화
			cv::Point2f pppt = cv::Point2f(0, 0);
			cv::Point2f ppt = cv::Point2f(0, 0);
			cv::Point2f cpt = pt+ ptLeft2;
			int pppidx = this->GetPointIndexInFrame(pPPrevMatch);
			if (pppidx >= 0) {
				pppt = pPPrevMatch->GetCPPt(pppidx);
			}
			int ppidx = this->GetPointIndexInFrame(pPrevMatch);
			if (ppidx >= 0) {
				ppt = pPrevMatch->GetCPPt(ppidx)+ptLeft1;
			}
			/*int cidx = this->GetPointIndexInFrame(pMatch);
			if (cidx >= 0) {
				cpt = pMatch->GetCPPt(cidx)+ ptLeft2;
			}*/
			cv::Scalar color;
			if (bSuccess) {
				color = cv::Scalar(0,0,255);
			}
			else {
				color = cv::Scalar(0, 255, 0);
			}
			circle(debug, pppt, 5, color);
			circle(debug, ppt, 5, color);
			circle(debug, cpt, 5, color);
			////////시각화

			if (bSuccess) {
				int label = 0;
				int octave = 0;
				////labeling 결과까지
				auto pMP = new UVR_SLAM::MapPoint(pMap, targetKF, x3D, cv::Mat(), label, octave);
				for (int i = 0; i < vPs.size(); i++) {
					pMP->AddFrame(vpMatches[i], vPts[i]);
				}
				/*
				pMP->AddFrame(pCurrKF->mpMatchInfo, pt1);
				pMP->AddFrame(pPrevKF->mpMatchInfo, pt2);
				pMP->AddFrame(pPPrevKF->mpMatchInfo, pt3);*/
				//std::cout << "CreateMP::" << nSuccess << std::endl;
				return true;
			}
			else {

			}
		}
		return false;
	}
}


