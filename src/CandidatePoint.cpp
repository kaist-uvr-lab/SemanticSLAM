#include "CandidatePoint.h"
#include "Frame.h"
#include "Map.h"
#include "MapPoint.h"
#include <DepthFilter.h>

static int nCandidatePointID = 0;

namespace  UVR_SLAM{
	CandidatePoint::CandidatePoint():octave(0), bCreated(false), mbDelete(false),mpSeed(nullptr),
		mnLastVisibleFrameID(-1), mnLastMatchingFrameID(-1), mnCandidatePointID(++nCandidatePointID), mnConnectedFrames(0)
	{
		mpMapPoint = nullptr;
	}
	CandidatePoint::CandidatePoint(Frame* pRefKF, int alabel, int aoct):mpRefKF(pRefKF), label(alabel), octave(aoct), bCreated(false), mbDelete(false), mpSeed(nullptr),
		mnLastVisibleFrameID(-1), mnLastMatchingFrameID(-1), mnCandidatePointID(++nCandidatePointID), mnConnectedFrames(0)
	{
		mpMapPoint = nullptr;
		mnFirstID = mpRefKF->mnFrameID;
	}
	CandidatePoint::~CandidatePoint(){}
		
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

		//rank 犬牢
		cv::Mat nonZeroSingularValues = w > 0.0001;
		int rank = countNonZero(nonZeroSingularValues);
		if (rank < 4) {
			//std::cout << "non singular matrix in triangulate in CP" << std::endl;
			bRank = false;
		}
		//rank 犬牢

		cv::Mat x3D = vt.row(3).t();
		x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
		return x3D.clone();
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
	////MP 包府
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
	////MP 包府
	/////////////////////////////
}


