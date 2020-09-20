#include "CandidatePoint.h"
#include "Frame.h"
#include "Map.h"
#include "MapPoint.h"

namespace  UVR_SLAM{
	CandidatePoint::CandidatePoint():octave(0), bCreated(false), mbDelete(false){
		mpMapPoint = nullptr;
	}
	CandidatePoint::CandidatePoint(MatchInfo* pRefKF, int alabel, int aoct):mpRefKF(pRefKF), label(alabel), octave(aoct), bCreated(false), mbDelete(false){
		mpMapPoint = nullptr;
		mnFirstID = pRefKF->mpRefFrame->GetFrameID();
	}
	CandidatePoint::~CandidatePoint(){}

	std::map<MatchInfo*, int> CandidatePoint::GetFrames() {
		std::unique_lock<std::mutex> lockMP(mMutexCP);
		return std::map<UVR_SLAM::MatchInfo*, int>(mmpFrames.begin(), mmpFrames.end());
	}
	
	int CandidatePoint::GetLabel(){
		std::unique_lock<std::mutex> lockMP(mMutexLabel);
		return label;
	}
	void CandidatePoint::SetLabel(int a){
		std::unique_lock<std::mutex> lockMP(mMutexLabel);
		label = a;
	}

	int CandidatePoint::GetNumSize() {
		std::unique_lock<std::mutex> lockMP(mMutexCP);
		return mnConnectedFrames;
	}

	/*void CandidatePoint::AddFrame(MatchInfo* pF, cv::Point2f pt) {
		std::unique_lock<std::mutex> lockMP(mMutexCP);
		auto res = mmpFrames.find(pF);
		if (res == mmpFrames.end()) {
			int idx = pF->AddCP(this, pt);
			mmpFrames.insert(std::pair<UVR_SLAM::MatchInfo*, int>(pF, idx));
			mnConnectedFrames++;
		}
	}*/

	void CandidatePoint::ConnectToFrame(MatchInfo* pF, int idx) {
		std::unique_lock<std::mutex> lockMP(mMutexCP);
		auto res = mmpFrames.find(pF);
		if (res == mmpFrames.end()) {
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
	cv::Mat CandidatePoint::Triangulate(cv::Point2f pt1, cv::Point2f pt2, cv::Mat P1, cv::Mat P2, bool& bRank) {

		cv::Mat A(4, 4, CV_32F);
		A.row(0) = pt1.x*P1.row(2) - P1.row(0);
		A.row(1) = pt1.y*P1.row(2) - P1.row(1);
		A.row(2) = pt2.x*P2.row(2) - P2.row(0);
		A.row(3) = pt2.y*P2.row(2) - P2.row(1);

		cv::Mat u, w, vt;
		cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

		//rank 확인
		cv::Mat nonZeroSingularValues = w > 0.0001;
		int rank = countNonZero(nonZeroSingularValues);
		if (rank < 4) {
			//std::cout << "non singular matrix in triangulate in CP" << std::endl;
			bRank = false;
		}
		//rank 확인

		cv::Mat x3D = vt.row(3).t();
		x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
		return x3D.clone();
	}
	void CandidatePoint::CreateMapPoint(cv::Mat& X3D, cv::Mat K, cv::Mat invK, cv::Mat Pcurr, cv::Mat Rcurr, cv::Mat Tcurr, cv::Point2f ptCurr, bool& bProjec, bool& bParallax, cv::Mat& debug) {
		cv::Point2f ptBottom = cv::Point2f(0, 480);
		MatchInfo* pFirst;
		int idx;
		{
			std::unique_lock<std::mutex> lockMP(mMutexCP);
			pFirst = mmpFrames.begin()->first;
			idx = mmpFrames.begin()->second;
		}
		cv::Mat P, R, t, Rt;
		auto pTargetKF = pFirst->mpRefFrame;
		pTargetKF->GetPose(R, t);
		cv::hconcat(R, t, P);
		Rt = R.t();
		auto ptFirst = pFirst->GetPt(idx);
		float val = CalcParallax(Rt, Rcurr.t(), ptFirst, ptCurr, invK);

		if (val >= 0.9998) {
			bParallax = false;
			bProjec = false;
			return;
		}
		bool bRank = true;
		X3D = Triangulate(ptFirst, ptCurr, K*P, K*Pcurr, bRank);

		bool bd1, bd2;
		auto pt1 = Projection(X3D, R, t, K, bd1); //first
		auto pt2 = Projection(X3D, Rcurr, Tcurr, K, bd2); //curr

		cv::line(debug, pt1, ptFirst, cv::Scalar(255, 0, 255), 1);
		cv::line(debug, pt2+ ptBottom, ptCurr+ ptBottom, cv::Scalar(255, 0, 255), 1);

		if (bRank && bd1 && bd2 && CheckReprojectionError(pt1, ptFirst, 9.0) && CheckReprojectionError(pt2, ptCurr, 9.0)){
			bProjec = true;
		}
		else {
			bProjec = false;
		}
		return;
	}
	bool CandidatePoint::CheckDepth(float depth) {
		return depth > 0.0;
	}
	bool CandidatePoint::CheckReprojectionError(cv::Point2f pt1, cv::Point2f pt2, float thresh) {
		auto diffPt = pt1 - pt2;
		float dist = diffPt.dot(diffPt);
		return dist < thresh;
	}
	bool CandidatePoint::CheckReprojectionError(cv::Mat x3D, cv::Mat K, cv::Point2f pt, float thresh) {
		cv::Mat reproj1 = K*x3D;
		reproj1 /= x3D.at<float>(2);
		float squareError1 = (reproj1.at<float>(0) - pt.x)*(reproj1.at<float>(0) - pt.x) + (reproj1.at<float>(1) - pt.y)*(reproj1.at<float>(1) - pt.y);
		return squareError1 < thresh;
	}
	cv::Point2f CandidatePoint::Projection(cv::Mat Xw, cv::Mat R, cv::Mat T, cv::Mat K, bool& bDepth) {
		cv::Mat Xcam = R * Xw + T;
		cv::Mat Ximg = K*Xcam;
		float depth = Ximg.at < float>(2);
		bDepth = depth > 0.0;
		return cv::Point2f(Ximg.at<float>(0) / Ximg.at < float>(2), Ximg.at<float>(1) / Ximg.at < float>(2));
	}
	bool CandidatePoint::DelayedTriangulate(Map* pMap, MatchInfo* pMatch, cv::Point2f pt, MatchInfo* pPPrevMatch, MatchInfo* pPrevMatch, cv::Mat K, cv::Mat invK, cv::Mat& debug) {
		
		int mnWidth = debug.cols / 3;
		cv::Point2f ptLeft1 = cv::Point2f(mnWidth, 0);
		cv::Point2f ptLeft2 = cv::Point2f(mnWidth * 2, 0);
		
		/////////
		float minParallax = 2.0;
		cv::Point2f minPt;
		cv::Mat minP;
		std::vector<cv::Mat> vPs, vRs, vTs;
		std::vector<cv::Point2f> vPts;
		std::vector<MatchInfo*> vpMatches;
		auto targetKF = pMatch->mpRefFrame;
		cv::Mat Rt, Tt, Pt;
		targetKF->GetPose(Rt, Tt);
		cv::hconcat(Rt, Tt, Pt);
		//Rt = Rt.t();

		for (auto iter = mmpFrames.begin(); iter != mmpFrames.end(); iter++) {
			
			auto pKF = iter->first->mpRefFrame;
			auto tempInfo = iter->first;
			int tempIdx = iter->second;
			auto tempPt = tempInfo->GetPt(tempIdx);;

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

		if(mnConnectedFrames > 10)
			circle(debug, pt + ptLeft2, 5, cv::Scalar(255, 255, 0));

		if (minParallax <= 0.9998f){
			//cv::Mat P;
			//cv::Mat K;
			bool bRank = true;
			cv::Mat x3D = Triangulate(minPt, pt, K*minP, K*Pt, bRank);
		
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
				pppt = pPPrevMatch->GetPt(pppidx);
			}
			int ppidx = this->GetPointIndexInFrame(pPrevMatch);
			if (ppidx >= 0) {
				ppt = pPrevMatch->GetPt(ppidx)+ptLeft1;
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
				auto pMP = new UVR_SLAM::MapPoint(pMap, targetKF, this, x3D, cv::Mat(), label, octave);
				for (int i = 0; i < vPs.size(); i++) {
					//pMP->AddFrame(vpMatches[i], vPts[i]);
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
		else {
			//std::cout << minParallax << std::endl;
			//circle(debug, pt + ptLeft2, 5, cv::Scalar(255,255,0));
		}
		return false;
	}
}


