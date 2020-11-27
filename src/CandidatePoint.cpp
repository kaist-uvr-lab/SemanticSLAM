#include "CandidatePoint.h"
#include "Frame.h"
#include "Map.h"
#include "MapPoint.h"
#include <DepthFilter.h>

static int nCandidatePointID = 0;

namespace  UVR_SLAM{
	CandidatePoint::CandidatePoint():octave(0), bCreated(false), mbDelete(false),mpSeed(nullptr),
		mnLastVisibleFrameID(-1), mnLastMatchingFrameID(-1), mnCandidatePointID(++nCandidatePointID)		
	{
		mpMapPoint = nullptr;
	}
	CandidatePoint::CandidatePoint(Frame* pRefKF, int alabel, int aoct):mpRefKF(pRefKF), label(alabel), octave(aoct), bCreated(false), mbDelete(false), mpSeed(nullptr),
		mnLastVisibleFrameID(-1), mnLastMatchingFrameID(-1), mnCandidatePointID(++nCandidatePointID)
	{
		mpMapPoint = nullptr;
		mnFirstID = mpRefKF->mnFrameID;
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
		
		mmnObjectLabelHistory[a]++;
		int maxVal = 0;
		int maxLabel;
		for (auto iter = mmnObjectLabelHistory.begin(), eiter = mmnObjectLabelHistory.end(); iter != eiter; iter++) {
			if (iter->second > maxVal) {
				maxVal = iter->second;
				maxLabel = iter->first;
				//std::cout << this->mnCandidatePointID << "::" << maxLabel << "=" << maxVal << "::" << iter->first << ", " << iter->second << std::endl;
			}
		}
		{
			std::unique_lock<std::mutex> lockMP(mMutexLabel);
			label = maxLabel;
		}
		
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

	void CandidatePoint::ConnectFrame(MatchInfo* pF, int idx) {
		std::unique_lock<std::mutex> lockMP(mMutexCP);
		auto res = mmpFrames.find(pF);
		if (res == mmpFrames.end()) {
			if (mmpFrames.size() == 0){
				this->mpRefKF = pF->mpRefFrame;
				////seed����
				/*auto pt = pF->mvMatchingPts[idx];
				cv::Mat a = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
				this->mpSeed = new Seed(std::move(pF->mpSystem->mInvK*a), pF->mpRefFrame->mfMedianDepth, pF->mpRefFrame->mfMinDepth);*/
				////seed����
			}
			mmpFrames.insert(std::pair<UVR_SLAM::MatchInfo*, int>(pF, idx));
			mnConnectedFrames++;
		}
		else {
			//std::cout << "cp::connect::error::" << mnCandidatePointID << std::endl;
		}
	}
	
	void CandidatePoint::DisconnectFrame(UVR_SLAM::MatchInfo* pKF)
	{
		{
			std::unique_lock<std::mutex> lockMP(mMutexCP);
			auto res = mmpFrames.find(pKF);
			if (res != mmpFrames.end()) {
				int idx = res->second;
				res = mmpFrames.erase(res);
				mnConnectedFrames--;
				//pKF->RemoveCP(idx);
				if (this->mpRefKF == mpRefKF) {
					mpRefKF = mmpFrames.begin()->first->mpRefFrame;
				}
				/*if (mnConnectedFrames < 3)
					mbDelete = true;*/
			}
		}
		/*if (mbDelete) {
			Delete();
		}*/
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
		if (mpMapPoint && !mpMapPoint->isDeleted())
			mpMapPoint->Delete(); //���⼭ ���� �� �� �� ����.
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

		//rank Ȯ��
		cv::Mat nonZeroSingularValues = w > 0.0001;
		int rank = countNonZero(nonZeroSingularValues);
		if (rank < 4) {
			//std::cout << "non singular matrix in triangulate in CP" << std::endl;
			bRank = false;
		}
		//rank Ȯ��

		cv::Mat x3D = vt.row(3).t();
		x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
		return x3D.clone();
	}
	bool CandidatePoint::CreateMapPoint(cv::Mat& X3D, float& fDepth, cv::Mat K, cv::Mat invK, cv::Mat Pcurr, cv::Mat Rcurr, cv::Mat Tcurr, cv::Point2f ptCurr) {
		MatchInfo* pFirst;
		pFirst = mpRefKF->mpMatchInfo;
		int idx = this->GetPointIndexInFrame(pFirst);
		
		cv::Mat Pref, Rref, Tref;
		mpRefKF->GetPose(Rref, Tref);
		cv::hconcat(Rref, Tref, Pref);
		auto ptFirst = pFirst->mvMatchingPts[idx];
		bool bRank = true;
		X3D = Triangulate(ptFirst, ptCurr, K*Pref, K*Pcurr, bRank);

		bool bd1, bd2;
		float fDepth1, fDepth2;
		auto pt1 = Projection(X3D, Rref, Tref, K, fDepth, bd1); //first
		auto pt2 = Projection(X3D, Rcurr, Tcurr, K, fDepth2, bd2); //curr

		if (bRank && bd1 && bd2 && CheckReprojectionError(pt1, ptFirst, 9.0) && CheckReprojectionError(pt2, ptCurr, 9.0)) {
			return true;
		}
		return false;
	}
	void CandidatePoint::CreateMapPoint(cv::Mat& X3D, cv::Mat K, cv::Mat invK, cv::Mat Pcurr, cv::Mat Rcurr, cv::Mat Tcurr, cv::Point2f ptCurr, bool& bProjec, bool& bParallax, cv::Mat& debug) {
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
		auto ptFirst = pFirst->mvMatchingPts[idx];
		float val = CalcParallax(Rt, Rcurr.t(), ptFirst, ptCurr, invK);

		if (val >= 0.9998) {
			bParallax = false;
			bProjec = false;
			return;
		}
		bool bRank = true;
		X3D = Triangulate(ptFirst, ptCurr, K*P, K*Pcurr, bRank);

		bool bd1, bd2;
		float fDepth1, fDepth2;
		auto pt1 = Projection(X3D, R, t, K, fDepth1, bd1); //first
		auto pt2 = Projection(X3D, Rcurr, Tcurr, K, fDepth2, bd2); //curr

		/*cv::line(debug, pt1, ptFirst, cv::Scalar(255, 0, 255), 1);
		cv::line(debug, pt2+ ptBottom, ptCurr+ ptBottom, cv::Scalar(255, 0, 255), 1);*/

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
	cv::Point2f CandidatePoint::Projection(cv::Mat Xw, cv::Mat R, cv::Mat T, cv::Mat K, float& fDepth, bool& bDepth) {
		cv::Mat Xcam = R * Xw + T;
		cv::Mat Ximg = K*Xcam;
		fDepth = Ximg.at < float>(2);
		bDepth = fDepth > 0.0;
		return cv::Point2f(Ximg.at<float>(0) / fDepth, Ximg.at<float>(1) / fDepth);
	}

	/////////////////////////////
	////MP ����
	MapPoint* CandidatePoint::GetMP() {
		std::unique_lock<std::mutex> lockMP(mMutexCP);
		return mpMapPoint;
	}
	void CandidatePoint::SetMapPoint(MapPoint* pMP) {
		std::unique_lock<std::mutex> lockMP(mMutexCP);
		bCreated = true;
		mpMapPoint = pMP;
	}
	void CandidatePoint::ResetMapPoint() {
		std::unique_lock<std::mutex> lockMP(mMutexCP);
		bCreated = false;
		mpMapPoint = nullptr;

		/*mnFail = 0;
		mnSuccess = 0;
		mnTotal = 0;

		mbOptimized = false;
		mbLowQuality = true;*/

		/*mnFirstMapPointID = -1;
		mnVisibleFrameID;
		mnLastFrameID;*/

	}
	void CandidatePoint::SetLastSuccessFrame(int id) {
		std::unique_lock<std::mutex> lockMP(mMutexCP);
		mnLastMatchingFrameID = id;
	}
	int CandidatePoint::GetLastSuccessFrame() {
		std::unique_lock<std::mutex> lockMP(mMutexCP);
		return mnLastMatchingFrameID;
	}
	void CandidatePoint::SetLastVisibleFrame(int id) {
		std::unique_lock<std::mutex> lockMP(mMutexCP);
		mnLastVisibleFrameID = id;
	}
	int CandidatePoint::GetLastVisibleFrame() {
		std::unique_lock<std::mutex> lockMP(mMutexCP);
		return mnLastVisibleFrameID;
	}
	////MP ����
	/////////////////////////////
}


