#include <Tracker.h>
#include <System.h>
#include <Map.h>
#include <Plane.h>
#include <Frame.h>
#include <Matcher.h>
#include <Initializer.h>
#include <LocalMapper.h>
#include <ORBextractor.h>
#include <SemanticSegmentator.h>
#include <SegmentationData.h>
#include <PlaneEstimator.h>
#include <FrameVisualizer.h>
#include <Visualizer.h>
#include <CandidatePoint.h>
#include <MapPoint.h>
#include <FrameGrid.h>
#include <MapGrid.h>
#include <DepthFilter.h>
#include <ZMSSD.h>
#include "lbplibrary.hpp"

//std::vector<cv::Vec3b> UVR_SLAM::ObjectColors::mvObjectLabelColors;
UVR_SLAM::Tracker::Tracker() {}
UVR_SLAM::Tracker::Tracker(int w, int h, cv::Mat K):mnWidth(w), mnHeight(h), mK(K), mbInitializing(false), mbFirstFrameAfterInit(false), mbInitilized(false){}
UVR_SLAM::Tracker::Tracker(System* pSys, std::string strPath) : mpSystem(pSys), mbInitializing(false), mbFirstFrameAfterInit(false), mbInitilized(false) {
	FileStorage fs(strPath, FileStorage::READ);
	
	cv::Mat DistCoef(4, 1, CV_32F);
	DistCoef.at<float>(0) = fs["Camera.k1"];
	DistCoef.at<float>(1) = fs["Camera.k2"];
	DistCoef.at<float>(2) = fs["Camera.p1"];
	DistCoef.at<float>(3) = fs["Camera.p2"];
	const float k3 = fs["Camera.k3"];
	if (k3 != 0)
	{
		DistCoef.resize(5);
		DistCoef.at<float>(4) = k3;
	}
	DistCoef.copyTo(mD);

	float fps = fs["Camera.fps"];
	mnMaxFrames = 4;// 10;//fps;
	mnMinFrames = 4; //fps / 3;//3

	mnThreshMinCPs	 = fs["Tracker.MinCP"];
	mnThreshMinMPs	 = fs["Tracker.MinMP"];
	mnThreshDiff	 = fs["Tracker.MinDiff"];
	mnThreshDiffPose = fs["Tracker.MinPoseHandle"];

	//mK2 = (cv::Mat_<float>(3, 3) << fx, 0, 0, 0, fy, 0, -fy*cx, -fx*cy, fx*fy); //line projection
	fs.release();
}
UVR_SLAM::Tracker::~Tracker() {}

bool UVR_SLAM::Tracker::isInitialized() {
	return mbInitilized;
}
void UVR_SLAM::Tracker::Init() {
	mK = mpSystem->mK.clone();
	mnWidth = mpSystem->mnWidth;
	mnHeight = mpSystem->mnHeight;
	mpMap = mpSystem->mpMap;
	mpVisualizer = mpSystem->mpVisualizer;
	mpFrameVisualizer = mpSystem->mpFrameVisualizer;
	mpMatcher = mpSystem->mpMatcher;
	mpInitializer = mpSystem->mpInitializer;
	mpSegmentator = mpSystem->mpSegmentator;
	mpLocalMapper = mpSystem->mpLocalMapper;
	mpPlaneEstimator = mpSystem->mpPlaneEstimator;
}
bool UVR_SLAM::Tracker::CheckNeedKeyFrame(Frame* pCurr, bool &bNeedCP, bool &bNeedMP, bool &bNeedPoseHandle, bool &bNeedNewKF) {
	///////////////
	//keyframe process

	bool bDoingMapping = !mpLocalMapper->isDoingProcess();
	bool bMaxFrames = pCurr->mnFrameID >= mpRefKF->mnFrameID + mnMaxFrames;//mnMinFrames;
	bool bMinFrames = pCurr->mnFrameID >= mpRefKF->mnFrameID + mnMinFrames;
	bNeedCP = bMinFrames;
	bNeedMP = bMinFrames;
	bNeedNewKF = bMaxFrames;

	return bDoingMapping && (bNeedCP || bNeedMP || bNeedNewKF);;
	int nHalf = mpSystem->mnRadius;
	int nSize = nHalf * 2;
	int a = mnWidth / nSize;
	int b = mnHeight / nSize;
	int nTotal = a*b;
	float fRatioCP = ((float)mnPointMatching) / nTotal;
	float fRatioMP = ((float)mnMapPointMatching) / nTotal;

	//int nDiffCP = abs(mnPointMatching - mnPrevPointMatching);
	//int nDiffMP = abs(mnMapPointMatching - mnPrevMapPointMatching);
	//int nPoseFail = abs(mnPointMatching - mnMapPointMatching);
	//bool bDiffCP = nDiffCP > mnThreshDiff;
	//bool bDiffMP = nDiffMP > mnThreshDiff;
	//bool bPoseFail = mnMapPointMatching < 80;//nPoseFail > mnThreshDiffPose;
	//bool bMatchMP = mnMapPointMatching < mnThreshMinMPs;
	//bool bMatchCP = mnPointMatching < mnThreshMinCPs;
	//bool bDoingMapping = !mpLocalMapper->isDoingProcess();
	//bool bMaxFrames = pCurr->mnFrameID >= mpRefKF->mnFrameID + mnMaxFrames;//mnMinFrames;
	//bool bMinFrames = pCurr->mnFrameID >= mpRefKF->mnFrameID + mnMinFrames;
	//bNeedCP = bDiffCP || bMatchCP;
	//bNeedMP = (bDiffMP || bMatchMP) && bMinFrames;
	//bNeedPoseHandle = bPoseFail;
	//bNeedNewKF = bMinFrames;
	
	
	bNeedCP = fRatioCP < 0.2f;
	bNeedMP = fRatioMP < 0.1f && bMinFrames;
	bNeedPoseHandle = fRatioMP < 0.1f;
	bNeedNewKF = bMinFrames;
	return bDoingMapping && (bNeedCP || bNeedMP || bNeedPoseHandle || bNeedNewKF);
}
UVR_SLAM::Frame* UVR_SLAM::Tracker::CheckNeedKeyFrame(Frame* pCurr, Frame* pPrev) {

	///////////////
	//keyframe process
	int nDiffCP = abs(mnPointMatching - mnPrevPointMatching);
	int nDiffMP = abs(mnMapPointMatching - mnPrevMapPointMatching);
	int nPoseFail = abs(mnPointMatching - mnMapPointMatching);

	//bool bDiff = nDiff > 50;
	bool bDiffCP = nDiffCP > 50;
	bool bDiffMp = nDiffMP > 50;
	bool bPoseFail = nPoseFail > 100;

	//1 : rotation angle
	bool bDoingMapping = !mpLocalMapper->isDoingProcess();
	bool bRotation = pCurr->CalcDiffAngleAxis(mpRefKF) > 10.0;
	bool bMaxFrames = pCurr->mnFrameID >= mpRefKF->mnFrameID + mnMaxFrames;//mnMinFrames;
	bool bMinFrames = pCurr->mnFrameID < mpRefKF->mnFrameID + mnMinFrames;

	bool bMatchMapPoint = mnMapPointMatching < 200;
	bool bMatchPoint = mnPointMatching < 350;
	
	//if ((bRotation || bMatchMapPoint || bMatchPoint || bMaxFrames) && bDoingSegment)
	/*if ((bRotation || bMatchMapPoint || bKF || bAVG || bMaxFrames) && !bMinFrames && bDoingMapping)
	{
	if(pCurr->CheckBaseLine(mpRefKF))
	return true;
	return false;
	}
	else
	return false;*/
	UVR_SLAM::Frame* pRes = nullptr;
	if (!bMinFrames && bDoingMapping) {
		//if (bRotation || bMaxFrames) {
		//	pRes = pCurr;
		//}
		//else if (bMatchMapPoint || bDiff) {//bKF
		//	pRes = pPrev;
		//}
		//else
		//	pRes = pCurr;
		pRes = pCurr;
		return pRes;
		//baseline이 중요하지 않은듯 여기서는
		/*if (pRes->CheckBaseLine(mpRefKF))
			return pRes;
		else
			return nullptr;*/
	}
	else
		return nullptr;
}
float fThresh_depth_filter = 100.;
//bool bRefKF = false;
void UVR_SLAM::Tracker::Tracking(Frame* pPrev, Frame* pCurr) {
	if(!mbInitializing){
		bool bReset = false;
		mbInitializing = mpInitializer->Initialize(pCurr, bReset, mnWidth, mnHeight);
		
		if (bReset){
			mpSystem->Reset();
		}
		//mbInit = bInit;
		mbFirstFrameAfterInit = false;
		
		if (mbInitializing){
			mpRefKF = pCurr;
			mbInitilized = true;
			mpSystem->SetBoolInit(true);
			mnPrevMapPointMatching = pCurr->mpMatchInfo->GetNumMPs();
			mnPrevPointMatching = mnPrevMapPointMatching;
			std::cout << "INIT::" << mnPrevMapPointMatching << ", " << mnPrevPointMatching << std::endl;
		}
	}
	else {
		
		std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////Optical Flow Matching
		////MatchInfo 설정
		/*pCurr->mpMatchInfo = new UVR_SLAM::MatchInfo(mpSystem, pCurr, pPrev, mnWidth, mnHeight);
		{
			std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateCP);
			mpSystem->cvUseCreateCP.wait(lock, [&] {return mpSystem->mbCreateCP;});
		}*/
		pCurr->mfMedianDepth = pPrev->mfMedianDepth;
		pCurr->mfRange = pPrev->mfRange;
		float fMaxDepth = pCurr->mfMeanDepth + pCurr->mfRange;
		////MatchInfo 설정
		//초기 매칭 테스트
		std::vector<UVR_SLAM::MapPoint*> vpTempMPs;
		std::vector<cv::Point2f> vTempCurrPts2;
		std::vector<bool> vbTempInliers;

		std::vector<UVR_SLAM::CandidatePoint*> vpTempCPs;
		std::vector<cv::Point2f> vTempCurrPts, vTempPrevPts;
		
		std::vector<int> vnTempIDXs, vnMPIDXs;
		cv::Mat debugImg;

		///////정보 추출
		auto vpKFs = mpMap->GetWindowFramesVector(1);
		std::vector<cv::Point2f> vRefPTs;
		std::vector<MapPoint*> vpRefMPs;
		int nFrameID = pCurr->mnFrameID;
		int nFirstMatch = 0;
		cv::Mat overlap = cv::Mat::zeros(mnHeight, mnWidth, CV_8UC1);
		for (int k = vpKFs.size()-1, kend = k-1; k > kend && k >= 0; k--) {
			//std::cout << "tracking::id::" << vpKFs[k]->mnFrameID << std::endl;
			auto vpRefTempMPs = vpKFs[k]->GetMapPoints();
			vRefPTs.clear();
			vpRefMPs.clear();
			for (size_t i = 0, iend = vpRefTempMPs.size(); i < iend; i++) {
				auto pMPi = vpRefTempMPs[i];
				if (!pMPi || pMPi->isDeleted() || pMPi->mnTrackingID == nFrameID)
					continue;
				pMPi->mnTrackingID = nFrameID;
				vRefPTs.push_back(vpKFs[k]->mvPts[i]);
				vpRefMPs.push_back(pMPi);
			}
			if(vRefPTs.size()>5){
				mnPointMatching = mpMatcher->OpticalFlowMatching(nFrameID, vpKFs[k]->GetOriginalImage(), pCurr->GetOriginalImage(), vRefPTs, vpRefMPs, vTempCurrPts, vTempCurrPts2, vpTempMPs, vbTempInliers, overlap);
				if (nFirstMatch == 0)
					nFirstMatch = mnPointMatching;
			}
		}
		
		std::chrono::high_resolution_clock::time_point tracking_a = std::chrono::high_resolution_clock::now();
		/*{
			std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateMP);
			mpSystem->cvUseCreateMP.wait(lock, [&] {return mpSystem->mbCreateMP; });
		}*/
		
		cv::Mat prevR, prevT;
		
		vpKFs.back()->GetPose(prevR, prevT);
		pCurr->SetPose(prevR, prevT);
		mnMapPointMatching = Optimization::PoseOptimization(mpMap, pCurr, vpTempMPs, vTempCurrPts2, vbTempInliers, mpSystem->mpORBExtractor->GetInverseScaleSigmaSquares());
		
		
		mpMap->ClearReinit();
		std::vector<cv::Point2f> vTrackingPTs;
		for (size_t i = 0, iend = vpTempMPs.size(); i < iend; i++) {
			auto pMPi = vpTempMPs[i];
			if (!pMPi || pMPi->isDeleted())
				continue;
			//pMPi->IncreaseVisible();
			if (vbTempInliers[i]){
				//pMPi->IncreaseFound();
				mpMap->AddReinit(pMPi->GetWorldPos());
				vTrackingPTs.push_back(vTempCurrPts2[i]);
			}
		}
		mpMap->SetTrackingPoints(vTrackingPTs);

		//{
		//	//입력 프레임에서 현재 프레임으로 프로젝션되는 매트릭스
		//	auto pKF = vpKFs[vpKFs.size() - 1];
		//	cv::Mat refImg = pKF->GetOriginalImage().clone();// mpRefKF->GetOriginalImage().clone();
		//	cv::Mat currImg = pCurr->GetOriginalImage().clone();
		//	cv::Rect mergeRect1 = cv::Rect(0, 0, refImg.cols, refImg.rows);
		//	cv::Rect mergeRect2 = cv::Rect(refImg.cols, 0, refImg.cols, refImg.rows);
		//	cv::Mat debugMatch = cv::Mat::zeros(refImg.rows, refImg.cols * 2, refImg.type());
		//	cv::Point2f ptBottom = cv::Point2f(refImg.cols, 0);
		//	refImg.copyTo(debugMatch(mergeRect1));
		//	currImg.copyTo(debugMatch(mergeRect2));
		//	
		//	int res = 0;
		//	int nBad = 0;
		//	cv::Mat R, t;
		//	pCurr->GetPose(R, t);
		//	for (int i = 0; i < vbTempInliers.size(); i++) {
		//		/*if (!vbTempInliers[i]) {
		//			continue;
		//		}*/
		//		
		//		auto currPt = vTempCurrPts2[i];
		//		auto pMPi = vpTempMPs[i];
		//		if (i < nFirstMatch) {
		//			auto prevPt = vTempCurrPts[i];
		//			
		//			cv::line(debugMatch, prevPt, currPt + ptBottom, cv::Scalar(255, 255, 0), 1);
		//			cv::Mat temp = mK*(R*pMPi->GetWorldPos() + t);
		//			float depth = temp.at<float>(2);
		//			cv::Point2f proj(temp.at<float>(0) / depth, temp.at<float>(1) / depth);
		//			if (vbTempInliers[i]){
		//				cv::circle(debugMatch, prevPt, 3, cv::Scalar(0, 255, 255), -1);
		//				cv::circle(debugMatch, currPt + ptBottom, 3, cv::Scalar(0, 255, 255), -1);
		//			}
		//			else{
		//				cv::circle(debugMatch, prevPt, 3, cv::Scalar(0, 255, 0), -1);
		//				cv::circle(debugMatch, currPt + ptBottom, 3, cv::Scalar(0, 255, 0), -1);
		//			}
		//			cv::circle(debugMatch, proj + ptBottom, 4, cv::Scalar(0, 0, 255), 1);
		//		}
		//		
		//	}
		//	//imshow("super::ref", refImg);
		//	imshow("tracking::opticalflow", debugMatch); cv::waitKey(1);
		//}
		
		

		////트래킹 결과 갱신
		//int nMP = UpdateMatchingInfo(pCurr, vpTempCPs, vTempCurrPts, vbTempInliers);
		///////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////키프레임 체크
		bool bNeedCP, bNeedMP, bNeedPoseHandle, bNeedNewKF;
		auto bNewKF = CheckNeedKeyFrame(pCurr, bNeedCP, bNeedMP, bNeedPoseHandle, bNeedNewKF);
		bNeedPoseHandle = false;
		if (bNewKF) {
			/*if (bNeedCP) {
				std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateCP);
				mpSystem->mbCreateCP = false;
			}  
			if (bNeedMP) {
				std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateMP);
				mpSystem->mbCreateMP = false;
			}*/
			if(bNeedNewKF)
				mpRefKF = pCurr;
			mpLocalMapper->InsertKeyFrame(pCurr, bNeedCP, bNeedMP, bNeedPoseHandle, bNeedNewKF);
		}
		mnPrevPointMatching = mnPointMatching;
		mnPrevMapPointMatching = mnMapPointMatching;
		////////Visualization & 시간 계산
		std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
		auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_a - tracking_start).count();
		double t1 = duration1 / 1000.0;
		auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
		double t2 = duration2 / 1000.0;

		//std::cout << "tracking::feature::" << vpKFs.back()->mnFrameID <<"::"<< t2 << "=" << vTempCurrPts2.size() << "::" << nFirstMatch << ", " << mnMapPointMatching << std::endl;

		//std::cout << "tracking::time::" << t2 << std::endl;
		///////시각화
		if (!mpFrameVisualizer->isVisualize()) {
			mpFrameVisualizer->SetFrameMatchingInformation(vpKFs.back(), pCurr, t2);//vpTempMPs, vpTempPts, vbTempInliers,
		}

		/////////트래킹 결과 이미지 저장                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
		//visualizer thread
		if (!mpVisualizer->isDoingProcess()) {
			mpVisualizer->SetMatchInfo(pCurr->mpMatchInfo);
			mpVisualizer->SetBoolDoingProcess(true);
		}
		//visualizer thread
	}
}

//void UVR_SLAM::Tracker::Run() {
//	while (1) {
//		////Frame 정보와 Queue 정보를 받아야 함.
//		if (mbInit) {
//
//		}		else {
//
//		}
//	}
//}

////MP와 PT가 대응함.
////pPrev가 mpRefKF가 됨
//매칭정보에 비율을 아예 추가하기
////이제 그리드 정보가 존재하고 그 안에 포인트가 있으면 넘겨도 되는거 아님??
int UVR_SLAM::Tracker::UpdateMatchingInfo(UVR_SLAM::Frame* pCurr, std::vector<UVR_SLAM::CandidatePoint*> vpCPs, std::vector<cv::Point2f> vpPts, std::vector<bool> vbInliers) {
	
	auto pMatchInfo = pCurr->mpMatchInfo;
	
	for (size_t i = 0, iend = vpCPs.size(); i < iend; i++) {
		auto pCP = vpCPs[i];
		auto pt = vpPts[i];
		auto pMP = pCP->GetMP();
		bool bMP = pMP && !pMP->isDeleted();
		if (bMP) {
			pMP->IncreaseVisible();
		}
		if (vbInliers[i]) {
			if(bMP)
				pMP->IncreaseFound();
			if (pMatchInfo->CheckOpticalPointOverlap(pt, mpSystem->mnRadius) < 0) {
				int idx = pMatchInfo->AddCP(pCP, pt);
				//pCP->ConnectFrame(pMatchInfo, idx);
			}
		}
	}
	
	return 0;
}

int UVR_SLAM::Tracker::UpdateMatchingInfo(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr, std::vector<UVR_SLAM::CandidatePoint*> vpCPs, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<cv::Point2f> vpPts, std::vector<bool> vbInliers, std::vector<int> vnIDXs, std::vector<int> vnMPIDXs) {
	auto pMatchInfo = pCurr->mpMatchInfo;
	auto pPrevMatchInfo = pPrev->mpMatchInfo;
	int nCurrID = pCurr->mnFrameID;
	int nres = 0;
	int nFail = 0;

	for (int i = 0; i < vpPts.size(); i++) {
		auto pCP = vpCPs[i];
		auto pMP = vpMPs[i];
		if (!vbInliers[i]){
			//pMP->AddFail();
			nFail++;
			continue;
		}
		int prevIdx = vnIDXs[i];
		auto pt = vpPts[i];
		if (pMatchInfo->CheckOpticalPointOverlap(pt, mpSystem->mnRadius) < 0) {
			//pMP->AddSuccess();
			//pMP->SetLastSuccessFrame(pCurr->GetFrameID());
			pMatchInfo->AddCP(pCP, pt);
			nres++;
		}
	}

	pMatchInfo->mfLowQualityRatio = ((float)nFail)/ vpPts.size();
	//std::cout << "Tracking::ID=" << pPrev->GetKeyFrameID() <<", "<< nCurrID << " matching = " << nres <<", Quality = "<< pMatchInfo->mfLowQualityRatio << std::endl;
	return nres;
}