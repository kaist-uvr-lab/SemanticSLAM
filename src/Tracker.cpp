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
#include <DepthFilter.h>
#include <ZMSSD.h>
#include "lbplibrary.hpp"

//std::vector<cv::Vec3b> UVR_SLAM::ObjectColors::mvObjectLabelColors;
UVR_SLAM::Tracker::Tracker() {}
UVR_SLAM::Tracker::Tracker(int w, int h, cv::Mat K):mnWidth(w), mnHeight(h), mK(K), mbInitializing(false), mbFirstFrameAfterInit(false), mbInitilized(false){}
UVR_SLAM::Tracker::Tracker(System* pSys, std::string strPath) : mpSystem(pSys), mbInitializing(false), mbFirstFrameAfterInit(false), mbInitilized(false) {
	FileStorage fs(strPath, FileStorage::READ);

	float fx = fs["Camera.fx"];
	float fy = fs["Camera.fy"];
	float cx = fs["Camera.cx"];
	float cy = fs["Camera.cy"];

	mK = cv::Mat::eye(3, 3, CV_32F);
	mK.at<float>(0, 0) = fx;
	mK.at<float>(1, 1) = fy;
	mK.at<float>(0, 2) = cx;
	mK.at<float>(1, 2) = cy;

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
	mnMaxFrames = 3;// 10;//fps;
	mnMinFrames = 3; //fps / 3;//3

	mnThreshMinCPs	 = fs["Tracker.MinCP"];
	mnThreshMinMPs	 = fs["Tracker.MinMP"];
	mnThreshDiff	 = fs["Tracker.MinDiff"];
	mnThreshDiffPose = fs["Tracker.MinPoseHandle"];

	mnWidth = fs["Image.width"];
	mnHeight = fs["Image.height"];
	mK2 = (cv::Mat_<float>(3, 3) << fx, 0, 0, 0, fy, 0, -fy*cx, -fx*cy, fx*fy); //line projection
	fs.release();
}
UVR_SLAM::Tracker::~Tracker() {}

bool UVR_SLAM::Tracker::isInitialized() {
	return mbInitilized;
}
void UVR_SLAM::Tracker::Init() {
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
		pCurr->mpMatchInfo = new UVR_SLAM::MatchInfo(mpSystem, pCurr, pPrev, mnWidth, mnHeight);
		{
			std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateCP);
			mpSystem->cvUseCreateCP.wait(lock, [&] {return mpSystem->mbCreateCP;});
		}
		pCurr->mfMedianDepth = pPrev->mfMedianDepth;
		pCurr->mfRange = pPrev->mfRange;
		////MatchInfo 설정
		//초기 매칭 테스트
		std::vector<UVR_SLAM::MapPoint*> vpTempMPs;
		std::vector<cv::Point2f> vTempCurrPts2;
		std::vector<bool> vbTempInliers;

		std::vector<UVR_SLAM::CandidatePoint*> vpTempCPs;
		std::vector<cv::Point2f> vTempCurrPts, vTempPrevPts;
		
		std::vector<int> vnTempIDXs, vnMPIDXs;
		cv::Mat debugImg;
		cv::Mat overlap = cv::Mat::zeros(pCurr->mnHeight, pCurr->mnWidth, CV_8UC1);
		//mnPointMatching = mpMatcher->OpticalMatchingForTracking(pPrev, pCurr, vpTempCPs, vTempPrevPts, vTempCurrPts, vbTempInliers, vnTempIDXs);
		mnPointMatching = mpMatcher->OpticalMatchingForTracking(pPrev, pCurr, vpTempMPs, vTempCurrPts2, vbTempInliers, 
			vpTempCPs, vTempPrevPts, vTempCurrPts, vnTempIDXs);


		std::chrono::high_resolution_clock::time_point tracking_a = std::chrono::high_resolution_clock::now();
		{
			std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateMP);
			mpSystem->cvUseCreateMP.wait(lock, [&] {return mpSystem->mbCreateMP; });
		}
		cv::Mat prevR, prevT;
		pPrev->GetPose(prevR, prevT);
		pCurr->SetPose(prevR, prevT);
		//mnMapPointMatching = Optimization::PoseOptimization(mpMap, pCurr, vpTempCPs, vTempCurrPts, vbTempInliers, mpSystem->mpORBExtractor->GetInverseScaleSigmaSquares());
		mnMapPointMatching = Optimization::PoseOptimization(mpMap, pCurr, vpTempMPs, vTempCurrPts2, vbTempInliers, mpSystem->mpORBExtractor->GetInverseScaleSigmaSquares());

		int N2 = 0;
		////local map
		{
			//local keyframe
			int nCurrID = pCurr->mnFrameID;
			std::vector<Frame*> vpLocalFrames = mpMap->GetWindowFramesVector(1);
			/*for (size_t i = 0, iend = vpTempCPs.size(); i < iend; i++) {
				auto pCPi = vpTempCPs[i];
				auto pMPi = pCPi->GetMP();
				if (!pMPi || pMPi->isDeleted())
					continue;
				auto frames = pMPi->GetConnedtedFrames();
				for (auto iter = frames.begin(), itend = frames.end(); iter != itend; iter++) {
					auto pKFi = iter->first->mpRefFrame;
					if (pKFi->isDeleted())
						continue;
					if (pKFi->mnLocalMapFrameID == nCurrID)
						continue;
					pKFi->mnLocalMapFrameID = nCurrID;
					vpLocalFrames.push_back(pKFi);
				}
			}*/
			
			std::vector<MapPoint*> vpLocalMap;
			for (size_t i = 0, iend = vpLocalFrames.size(); i < iend; i++) {
				auto pKFi = vpLocalFrames[i];
				auto vpCPs = pKFi->mpMatchInfo->mvpMatchingCPs;
				for (size_t i = 0, iend = vpCPs.size(); i < iend; i++) {
					auto pCPi = vpCPs[i];
					auto pMPi = pCPi->GetMP();
					if (!pMPi || pMPi->isDeleted() || pMPi->mnLocalMapID == nCurrID || pMPi->mnTrackingID == nCurrID || pCPi->mnTrackingFrameID == nCurrID)
						continue;
					pMPi->mnLocalMapID = nCurrID;
					vpLocalMap.push_back(pMPi);
				}
			}
			
			int nGridSize = mpSystem->mnRadius * 2;
			int maxLvl = 0;
			int searchSize = 5;
			cv::Mat R, t;
			pCurr->GetPose(R, t);
			
			for (size_t i = 0, iend = vpLocalMap.size(); i < iend; i++) {
				auto pMPi = vpLocalMap[i];
				if (!pMPi->mbLastMatch)
					continue;
				cv::Mat proj = mK*(R*pMPi->GetWorldPos() + t);
				float depth = proj.at<float>(2);
				cv::Point projPt(proj.at<float>(0) / depth, proj.at<float>(1) / depth);
				if (!pCurr->isInImage(projPt.x, projPt.y, 10.0))
					continue;
				cv::Point2f left1(projPt.x - nGridSize, projPt.y - nGridSize);
				if (left1.x < 0 || left1.y < 0)
					continue;
				cv::Point2f right1(projPt.x + nGridSize, projPt.y + nGridSize);
				if (right1.x >= mnWidth || right1.y >= mnHeight)
					continue;

				//두 렉트 보기
				cv::Rect rect1(left1, right1);
				cv::Mat img1 = pCurr->GetOriginalImage()(rect1);

				////매칭
				std::vector<uchar> status;
				std::vector<float> err;
				std::vector<cv::Point2f> prevPts, currPts;
				prevPts.push_back(pMPi->mLastMatchPoint);

				/*if(pMPi->mLastMatchPatch.empty())
				std::cout << "??????" << std::endl;*/
				if (pMPi->mLastMatchPatch.channels() != 3 || pMPi->mLastMatchPatch.rows != nGridSize * 2)
					std::cout << "??????" << std::endl;
				if(img1.size() != pMPi->mLastMatchPatch.size())
					std::cout << "??????" << img1.size <<"::"<<pMPi->mLastMatchPatch.size()<< std::endl;
				cv::calcOpticalFlowPyrLK(pMPi->mLastMatchPatch, img1, prevPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);
				if (!status[0])
					continue;

				cv::Point2f currPt = currPts[0] + left1;
				auto prevPt = pMPi->mLastMatchPoint + pMPi->mLastMatchBasePt;
				//auto pCPi = vpLocalCPs[i];
				vpTempMPs.push_back(pMPi);
				vTempCurrPts2.push_back(currPt);
				//vTempPrevPts.push_back(prevPt);
				//vpTempCPs.push_back(pCPi);
				vbTempInliers.push_back(true);
				//vnTempIDXs.push_back(-1);
				N2++;
				////Epiconstraints
				//cv::Mat Rrel, Trel;
				//pCurr->GetRelativePoseFromTargetFrame(pMPi->mLastFrame, Rrel, Trel);
				//auto prevPt = pMPi->mLastMatchPoint + pMPi->mLastMatchBasePt;
				//cv::Mat ray = mpSystem->mInvK*(cv::Mat_<float>(3, 1) << prevPt.x, prevPt.y, 1.0);
				//float z_min, z_max;
				//z_min = 0.01f;
				//z_max = 1.0f;
				//cv::Point2f XimgMin, XimgMax;
				//mpMatcher->ComputeEpiLinePoint(XimgMin, XimgMax, ray, z_min, z_max, Rrel, Trel, mK); //ray,, Rrel, Trel
				//cv::Mat lineEqu = mpMatcher->ComputeLineEquation(XimgMin, XimgMax);
				//bool bEpiConstraints = mpMatcher->CheckLineDistance(lineEqu, currPt, 1.0);

				//if (!bEpiConstraints) {
				//	//pCP->mnTrackingFrameID = -1;
				//	continue;
				//}
				////Epiconstraints
				
				//////이미 그리드에 포함되어 있는지 확인하는 단계
				//auto gridPt = pPrev->GetGridBasePt(currPt, nGridSize);
				//if (pCurr->mmbFrameGrids[gridPt]) {
				//	pCPi->mnTrackingFrameID = -1;
				//	continue;
				//}
				//////이미 그리드에 포함되어 있는지 확인하는 단계

				//if (pCurr->mpMatchInfo->CheckOpticalPointOverlap(currPt, mpSystem->mnRadius) > -1) {
				//	pCPi->mnTrackingFrameID = -1;
				//	continue;
				//}

				////////grid 추가
				//cv::Point right2(gridPt.x + nGridSize, gridPt.y + nGridSize);
				//if (right2.x >= mnWidth || right2.y >= mnHeight)
				//	continue;
				//auto rect = cv::Rect(gridPt, right2);
				//pCurr->mmbFrameGrids[gridPt] = true;
				//auto currGrid = new FrameGrid(gridPt, rect, 0);
				//pCurr->mmpFrameGrids[gridPt] = currGrid;
				//pCurr->mmpFrameGrids[gridPt]->mpCP = pCPi;
				//pCurr->mmpFrameGrids[gridPt]->pt = currPt;
				//
				//pCurr->mpMatchInfo->AddCP(pCPi, currPt);

				
				
				/*if (pMPi->isInFrame(mpRefKF->mpMatchInfo)) {
					int pidx = pMPi->GetPointIndexInFrame(mpRefKF->mpMatchInfo);
					cv::Point2f prevPt = mpRefKF->mpMatchInfo->mvMatchingPts[pidx];
					
					cv::circle(cImg, currPt, 2, cv::Scalar(255), -1);
					cv::circle(pImg, prevPt, 2, cv::Scalar(255), -1);
				}*/
			}
			//imshow("local map::p", pImg); imshow("local map::c", cImg); cv::waitKey(1);
		}
		std::cout << "tracker:;1::" <<N2<< std::endl;
		if (N2 > 10) {
			//mnMapPointMatching = Optimization::PoseOptimization(mpMap, pCurr, vpTempCPs, vTempCurrPts, vbTempInliers, mpSystem->mpORBExtractor->GetInverseScaleSigmaSquares());
			mnMapPointMatching = Optimization::PoseOptimization(mpMap, pCurr, vpTempMPs, vTempCurrPts2, vbTempInliers, mpSystem->mpORBExtractor->GetInverseScaleSigmaSquares());
		}
			
		std::cout << "tracker:;2" << std::endl;

		/////////////////Matching Validation
		int nFinal = 0;
		int nFinal2 = 0;
		auto pMatchInfo = pCurr->mpMatchInfo;
		int nGridSize = mpSystem->mnRadius * 2;
		float fGridDistThresh = nGridSize*nGridSize * 4;
		mpMap->ClearReinit();
		for (size_t i = 0, iend = vpTempMPs.size(); i < iend; i++) {
			if (!vbTempInliers[i])
				continue;
			auto pMPi = vpTempMPs[i];
			if (!pMPi || pMPi->isDeleted())
				continue;
			auto currPt = vTempCurrPts2[i];
			auto pCPi = vpTempMPs[i]->mpCP;

			////이미 그리드에 포함되어 있는지 확인하는 단계
			auto gridPt = pPrev->GetGridBasePt(currPt, nGridSize);
			if (pCurr->mmbFrameGrids[gridPt]) {
				vbTempInliers[i] = false;
				pCPi->mnTrackingFrameID = -1;
				continue;
			}
			////이미 그리드에 포함되어 있는지 확인하는 단계

			if (pMatchInfo->CheckOpticalPointOverlap(currPt, mpSystem->mnRadius) > -1) {
				vbTempInliers[i] = false;
				pCPi->mnTrackingFrameID = -1;
				continue;
			}

			//////grid 추가
			cv::Point right2(gridPt.x + nGridSize, gridPt.y + nGridSize);
			if (right2.x >= mnWidth || right2.y >= mnHeight)
				continue;
			auto rect = cv::Rect(gridPt, right2);
			pCurr->mmbFrameGrids[gridPt] = true;
			auto currGrid = new FrameGrid(gridPt, rect, 0);
			pCurr->mmpFrameGrids[gridPt] = currGrid;
			pCurr->mmpFrameGrids[gridPt]->mpCP = pCPi;
			pCurr->mmpFrameGrids[gridPt]->pt = currPt;

			pMatchInfo->AddCP(pCPi, currPt);

			cv::Point pt = currPt;
			cv::Point2f left1(pt.x - nGridSize, pt.y - nGridSize);
			cv::Point2f right1(pt.x + nGridSize, pt.y + nGridSize);
			if (left1.x >= 0 && left1.y >= 0 && right1.x < mnWidth && right1.y < mnHeight) {
				pMPi->mbLastMatch = true;
				cv::Rect recta(left1, right1);
				pMPi->mLastMatchPatch = pCurr->GetOriginalImage()(recta);
				pMPi->mLastMatchPoint = currPt - left1;
				pMPi->mLastMatchBasePt = left1;
				pMPi->mLastFrame = pCurr;
			}
			mpMap->AddReinit(pMPi->GetWorldPos());
			nFinal2++;
		}

		cv::Mat Rrel, Trel;
		pCurr->GetRelativePoseFromTargetFrame(pPrev, Rrel, Trel);
		
		for (size_t i = 0, iend = vpTempCPs.size(); i < iend; i++) {
			auto pCP = vpTempCPs[i];
			auto currPt = vTempCurrPts[i];
			auto prevPt = vTempPrevPts[i];//mpRefKF->mpMatchInfo->mvMatchingPts[vnTempIDXs[i]];//vTempPrevPts[i];
			auto pMP = pCP->GetMP();
			bool bMP = pMP && !pMP->isDeleted();
			int prevIDX = vnTempIDXs[i];

			/////////check epipolar constraints
			cv::Mat ray = mpSystem->mInvK*(cv::Mat_<float>(3, 1) << prevPt.x, prevPt.y, 1.0);
			float z_min, z_max;
			z_min = 0.01f;
			z_max = 1.0f;
			cv::Point2f XimgMin, XimgMax;
			mpMatcher->ComputeEpiLinePoint(XimgMin, XimgMax, ray, z_min, z_max, Rrel, Trel, mK); //ray,, Rrel, Trel
			cv::Mat lineEqu = mpMatcher->ComputeLineEquation(XimgMin, XimgMax);
			bool bEpiConstraints = mpMatcher->CheckLineDistance(lineEqu, currPt, 1.0);
			if (!bEpiConstraints) {
				pCP->mnTrackingFrameID = -1;
				continue;
			}

			////이미 그리드에 포함되어 있는지 확인하는 단계
			auto gridPt = pPrev->GetGridBasePt(currPt, nGridSize);
			if (pCurr->mmbFrameGrids[gridPt]) {
				pCP->mnTrackingFrameID = -1;
				continue;
			}
			////이미 그리드에 포함되어 있는지 확인하는 단계

			if (pMatchInfo->CheckOpticalPointOverlap(currPt, mpSystem->mnRadius) > -1) {
				pCP->mnTrackingFrameID = -1;
				continue;
			}

			//////grid 추가
			cv::Point right2(gridPt.x + nGridSize, gridPt.y + nGridSize);
			if (right2.x >= mnWidth || right2.y >= mnHeight)
				continue;
			auto rect = cv::Rect(gridPt, right2);
			pCurr->mmbFrameGrids[gridPt] = true;
			auto currGrid = new FrameGrid(gridPt, rect, 0);
			pCurr->mmpFrameGrids[gridPt] = currGrid;
			pCurr->mmpFrameGrids[gridPt]->mpCP = pCP;
			pCurr->mmpFrameGrids[gridPt]->pt = currPt;

			pMatchInfo->AddCP(pCP, currPt, vnTempIDXs[i]);

			if (bMP) {
				cv::Point pt = currPt;
				cv::Point2f left1(pt.x - nGridSize, pt.y - nGridSize);
				cv::Point2f right1(pt.x + nGridSize, pt.y + nGridSize);
				if (left1.x >= 0 && left1.y >= 0 && right1.x < mnWidth && right1.y < mnHeight) {
					pMP->mbLastMatch = true;
					cv::Rect recta(left1, right1);
					pMP->mLastMatchPatch = pCurr->GetOriginalImage()(recta);
					pMP->mLastMatchPoint = currPt-left1;
					pMP->mLastMatchBasePt = left1;
					pMP->mLastFrame = pCurr;
				}
			}

			/*
			auto prevGridPt = pPrev->GetGridBasePt(prevPt, nGridSize);
			auto prevGrid = pPrev->mmpFrameGrids[prevGridPt];
			if (!prevGrid) {
			std::cout << "tracking::????" << std::endl;
			}
			prevGrid->mpNext = currGrid;
			currGrid->mpPrev = prevGrid;

			currGrid->mObjArea = prevGrid->mObjArea.clone();
			currGrid->mObjCount = prevGrid->mObjCount.clone();*/

			/*for (auto oiter = prevGrid->mmObjCounts.begin(), oiend = prevGrid->mmObjCounts.end(); oiter != oiend; oiter++) {
				auto label = oiter->first;
				auto count = oiter->second;
				currGrid->mmObjCounts[label] = count;
			}*/
			//////grid 추가
			if(bMP)
				nFinal++;
			/*cv::circle(prevImg, prevPt, 3, cv::Scalar(255, 0, 0), -1);
			cv::circle(currImg, currPt, 3, cv::Scalar(255, 0, 0), -1);*/
		}
		
		std::cout << "tracker::3=" <<nFinal<<":"<< nFinal2 << std::endl;

		////트래킹 결과 갱신
		//int nMP = UpdateMatchingInfo(pCurr, vpTempCPs, vTempCurrPts, vbTempInliers);
		///////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////키프레임 체크
		bool bNeedCP, bNeedMP, bNeedPoseHandle, bNeedNewKF;
		auto bNewKF = CheckNeedKeyFrame(pCurr, bNeedCP, bNeedMP, bNeedPoseHandle, bNeedNewKF);
		bNeedPoseHandle = false;
		if (true) {
			if (bNeedCP) {
				std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateCP);
				mpSystem->mbCreateCP = false;
			}  
			if (bNeedMP) {
				std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateMP);
				mpSystem->mbCreateMP = false;
			}
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

		///////시각화
		if (!mpFrameVisualizer->isVisualize()) {
			mpFrameVisualizer->SetFrameMatchingInformation(mpRefKF, pCurr, t2);//vpTempMPs, vpTempPts, vbTempInliers,
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