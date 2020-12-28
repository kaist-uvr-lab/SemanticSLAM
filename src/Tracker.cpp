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
	mnMaxFrames = 5;// 10;//fps;
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
	
	bool bDoingMapping = !mpLocalMapper->isDoingProcess();
	bool bMaxFrames = pCurr->mnFrameID >= mpRefKF->mnFrameID + mnMaxFrames;//mnMinFrames;
	bool bMinFrames = pCurr->mnFrameID >= mpRefKF->mnFrameID + mnMinFrames;
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
		cv::Mat currFrame = pCurr->matFrame.clone();
		
		lbplibrary::LBP* lbp2 = new lbplibrary::ELBP(2, 8);
		//{
		//	cv::Mat mImgLBP;
		//	lbplibrary::LBP* lbp = new lbplibrary::SCSLBP(2, 4);
		//	lbp->run(pCurr->matFrame, mImgLBP); //종류에 따라서 변경이 필요함.
		//	std::cout << "size:" << mImgLBP.size() << std::endl;
		//	cv::normalize(mImgLBP, mImgLBP, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		//	cv::imshow("varlbp", mImgLBP);
		//	lbplibrary::show_histogram("lbp_hist::VAR", mImgLBP);
		//}
		//{
		//	cv::Mat mImgLBP2;
		//	lbplibrary::LBP* lbp = new lbplibrary::ELBP(2, 8);
		//	lbp->run(pCurr->matFrame, mImgLBP2);
		//	cv::normalize(mImgLBP2, mImgLBP2, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		//	cv::imshow("elbp", mImgLBP2);
		//	lbplibrary::show_histogram("lbp_hist::ELBP", mImgLBP2);
		//}
		std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////Optical Flow Matching
		////MatchInfo 설정
		mpRefKF->SetRecentTrackedFrameID(pCurr->mnFrameID);
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
		std::vector<UVR_SLAM::CandidatePoint*> vpTempCPs;
		std::vector<cv::Point2f> vTempCurrPts, vTempPrevPts;
		std::vector<cv::Point3f> vpTempPts2;
		std::vector<uchar> vcInliers;
		std::vector<bool> vbTempInliers;// = std::vector<bool>(pPrev->mvpMatchingMPs.size(), false);
		std::vector<int> vnIDXs, vnMPIDXs;
		cv::Mat debugImg;
		cv::Mat overlap = cv::Mat::zeros(pCurr->mnHeight, pCurr->mnWidth, CV_8UC1);
		mnPointMatching = mpMatcher->OpticalMatchingForTracking(pPrev, pCurr, vpTempCPs, vTempPrevPts, vTempCurrPts, vbTempInliers);

		//////grid matching test
		//{
		//	std::map < cv::Point2f, FrameGrid*, Point2fLess> mpGrids;
		//	std::map < FrameGrid*, std::vector<cv::Point2f>> mGridPTs;
		//	std::map < FrameGrid*, std::vector<size_t>> mGridIDXs;
		//	int nGridSize = mpSystem->mnRadius * 2 * 8;
		//	//포인트를 나눌 그리드 설정. 이건 미리 되어도 될 듯.
		//	for (int x = 0; x < mnWidth; x += nGridSize) {
		//		for(int y = 0; y < mnHeight; y += nGridSize){
		//			cv::Point2f ptLeft(x, y);
		//			cv::Point2f ptRight(x + nGridSize, y + nGridSize);

		//			if (ptRight.x > mnWidth || ptRight.y > mnHeight) {
		//				//cv::circle(testImg, ptLeft, 3, cv::Scalar(255, 255, 0), -1);
		//				continue;
		//			}
		//			cv::Rect rect(ptLeft, ptRight);
		//			auto pGrid = new FrameGrid(std::move(ptLeft), std::move(rect));
		//			mpGrids.insert(std::make_pair(ptLeft, pGrid));
		//			mGridPTs.insert(std::make_pair(pGrid, std::vector<cv::Point2f>()));
		//			mGridIDXs.insert(std::make_pair(pGrid, std::vector<size_t>()));
		//		}
		//	}
		//	//포인트 추가
		//	for (size_t i = 0, iend = pPrev->mpMatchInfo->mvMatchingPts.size(); i < iend; i++) {
		//		auto pt = pPrev->mpMatchInfo->mvMatchingPts[i];
		//		auto basePt = pPrev->GetGridBasePt(pt, nGridSize);
		//		auto grid = mpGrids[basePt];
		//		if (!grid)
		//			continue;
		//		mGridPTs[grid].push_back(pt - basePt);
		//		mGridIDXs[grid].push_back(i);
		//	}
		//	//그리드별 매칭 수행
		//	int nCurrFrameID = pCurr->mnFrameID;
		//	int maxLvl = 3;
		//	int searchSize = 25;
		//	std::vector<cv::Point2f> matchPrevPTs, matchCurrPTs;
		//	for (auto iter = mGridPTs.begin(), iend = mGridPTs.end(); iter != iend; iter++) {
		//		auto grid = iter->first;
		//		auto rect = grid->rect;
		//		auto basePt = grid->basePt;
		//		auto prevPts = iter->second;
		//		auto vIDXs = mGridIDXs[grid];

		//		if (prevPts.size() == 0)
		//			continue;

		//		cv::Mat prevImg = pPrev->GetOriginalImage()(rect);
		//		cv::Mat currImg = pCurr->GetOriginalImage()(rect);

		//		std::vector<uchar> status;
		//		std::vector<float> err;
		//		std::vector<cv::Point2f> currPts;
		//		cv::calcOpticalFlowPyrLK(prevImg, currImg, prevPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);

		//		for (size_t i = 0, iend2 = prevPts.size(); i < iend2; i++) {
		//			if (status[i] == 0) {
		//				continue;
		//			}
		//			cv::Point2f prevPt = prevPts[i] + basePt;
		//			cv::Point2f currPt = currPts[i] + basePt;
		//			if (!pPrev->isInImage(currPt.x, currPt.y, 20))
		//				continue;
		//			int idx = vIDXs[i];
		//			auto pCP = pPrev->mpMatchInfo->mvpMatchingCPs[idx];
		//			if (pCP->mnTrackingFrameID == nCurrFrameID)
		//				continue;
		//			pCP->mnTrackingFrameID = nCurrFrameID;
		//			vpTempCPs.push_back(pCP);
		//			vTempPrevPts.push_back(prevPt);
		//			vTempCurrPts.push_back(currPt);
		//			vbTempInliers.push_back(true);
		//		}
		//	}
		//	mnPointMatching = vTempCurrPts.size();
		//}
		//////grid matching test

		std::chrono::high_resolution_clock::time_point tracking_a = std::chrono::high_resolution_clock::now();
		{
			std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateMP);
			mpSystem->cvUseCreateMP.wait(lock, [&] {return mpSystem->mbCreateMP; });
		}
		cv::Mat prevR, prevT;
		mpRefKF->GetPose(prevR, prevT);
		pCurr->SetPose(prevR, prevT);
		mnMapPointMatching = Optimization::PoseOptimization(mpMap, pCurr, vpTempCPs, vTempCurrPts, vbTempInliers, mpSystem->mpORBExtractor->GetInverseScaleSigmaSquares());
		
		//////////////////디스크립션 매칭 테스트
		//cv::Mat img1 = pPrev->GetOriginalImage().clone();
		//cv::Mat img2 = pCurr->GetOriginalImage().clone();
		//std::vector<cv::KeyPoint> vpTempKPs, vpTempKPs2;
		//for (size_t i = 0, iend = pPrev->mpMatchInfo->mvMatchingPts.size(); i < iend; i++) {
		//	cv::KeyPoint kp(pPrev->mpMatchInfo->mvMatchingPts[i], 3.0);
		//	vpTempKPs.push_back(kp);
		//}

		//for (size_t i = 0, iend = vTempCurrPts.size(); i < iend; i++) {
		//	cv::KeyPoint kp2(vTempCurrPts[i], 3.0);
		//	vpTempKPs2.push_back(kp2);
		//	cv::circle(img1, vTempPrevPts[i], 5, cv::Scalar(255, 0, 0));
		//	cv::circle(img2, vTempCurrPts[i], 5, cv::Scalar(255, 0, 0));
		//}

		//cv::Ptr<cv::Feature2D> feature = cv::ORB::create();
		//cv::Mat desc, desc2;
		//
		//feature->compute(img1, vpTempKPs, desc);
		//feature->compute(img2, vpTempKPs2, desc2);

		//std::vector< std::vector<cv::DMatch> > matches;
		//std::vector<cv::DMatch> vMatches;
		//auto matcher = DescriptorMatcher::create("BruteForce-Hamming");
		//matcher->knnMatch(desc, desc2, matches, 2);
		//
		//for (unsigned long i = 0; i < matches.size(); i++) {
		//	if (matches[i][0].distance < 0.8f * matches[i][1].distance) {
		//		vMatches.push_back(matches[i][0]);
		//		cv::circle(img1, vTempPrevPts[matches[i][0].trainIdx], 3, cv::Scalar(255, 0, 255), -1);
		//		cv::circle(img2, vTempCurrPts[matches[i][0].queryIdx], 3, cv::Scalar(255, 0, 255), -1);
		//	}
		//}
		//imshow("a", img1); imshow("b", img2);
		//waitKey(1);
		//std::cout << desc2.rows << ", " << vMatches.size() << std::endl;
		////////////////디스크립션 매칭 테스트

		//////Ref-Curr 매칭
		//{
		//	int nGridSize = mpSystem->mnRadius * 2 * 4;
		//	std::map < cv::Point2f, FrameGrid*, Point2fLess> mapRefGrids, mapCurrGrids;
		//	std::map<FrameGridKey, std::vector<cv::Point2f>> mapFrameGridAndKeyPoints;
		//	std::map<FrameGridKey, std::vector<MapPoint*>>   mapFrameGridAndMapPoints;
		//	cv::Mat R, t;
		//	pCurr->GetPose(R, t);
		//	for (int x = 0; x < mnWidth; x += nGridSize) {
		//		for(int y = 0; y < mnHeight; y += nGridSize){
		//			cv::Point2f ptLeft(x, y);
		//			cv::Point2f ptRight(x + nGridSize, y + nGridSize);

		//			if (ptRight.x > mnWidth || ptRight.y > mnHeight) {
		//				//cv::circle(testImg, ptLeft, 3, cv::Scalar(255, 255, 0), -1);
		//				continue;
		//			}
		//			cv::Rect rect(ptLeft, ptRight);
		//			auto pGrid1 = new FrameGrid((ptLeft), (rect));
		//			auto pGrid2 = new FrameGrid((ptLeft), (rect));
		//			mapRefGrids.insert(std::make_pair(ptLeft, pGrid1));
		//			mapCurrGrids.insert(std::make_pair(ptLeft, pGrid2));
		//		}
		//	}
		//	for (size_t i = 0, iend = mpRefKF->mpMatchInfo->mvpMatchingCPs.size(); i < iend; i++) {
		//		auto pCP = mpRefKF->mpMatchInfo->mvpMatchingCPs[i];
		//		auto pMP = pCP->GetMP();
		//		if (!pMP || pMP->isDeleted())
		//			continue;
		//		auto pt = mpRefKF->mpMatchInfo->mvMatchingPts[i];
		//		auto refBasePt = mpRefKF->GetGridBasePt(pt, nGridSize);

		//		cv::Mat Xw = pMP->GetWorldPos();
		//		cv::Mat temp = mK*(R*Xw + t);
		//		float depth = temp.at<float>(2);
		//		cv::Point2f projPt(temp.at<float>(0) / depth, temp.at<float>(1) / depth);
		//		auto currBasePt = pCurr->GetGridBasePt(projPt, nGridSize);

		//		auto refGrid = mapRefGrids[refBasePt];
		//		if (!refGrid)
		//			continue;
		//		auto currGrid = mapCurrGrids[currBasePt];
		//		if (!currGrid)
		//			continue;

		//		FrameGridKey key(refGrid, currGrid);
		//		mapFrameGridAndKeyPoints[key].push_back(pt - refBasePt);
		//		mapFrameGridAndMapPoints[key].push_back(pMP);
		//	}
		//	//매칭
		//	cv::Mat refImg = mpRefKF->GetOriginalImage().clone();
		//	cv::Mat currImg = pCurr->GetOriginalImage().clone();

		//	cv::Mat Rrel, Trel;
		//	pCurr->GetRelativePoseFromTargetFrame(mpRefKF, Rrel, Trel);

		//	int nCurrFrameID = pCurr->mnFrameID;
		//	int maxLvl = 3;
		//	int searchSize = 10;
		//	std::vector<cv::Point2f> matchPrevPTs, matchCurrPTs;
		//	for (auto iter = mapFrameGridAndKeyPoints.begin(), iend = mapFrameGridAndKeyPoints.end(); iter != iend; iter++) {
		//		auto key = iter->first;
		//		auto refGrid = key.mpKey1;
		//		auto currGrid = key.mpKey2;
		//		auto refRect = refGrid->rect;
		//		auto currRect = currGrid->rect;
		//		auto refBasePt = refGrid->basePt;
		//		auto currBasePt = currGrid->basePt;
		//		auto refPts = iter->second;
		//		auto refMPs = mapFrameGridAndMapPoints[key];

		//		if (refPts.size() == 0)
		//			continue;

		//		cv::Mat refRectImg = mpRefKF->GetOriginalImage()(refRect);
		//		cv::Mat currRectImg = pCurr->GetOriginalImage()(currRect);

		//		std::vector<uchar> status;
		//		std::vector<float> err;
		//		std::vector<cv::Point2f> currPts;
		//		cv::calcOpticalFlowPyrLK(refRectImg, currRectImg, refPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);

		//		for (size_t i = 0, iend2 = refPts.size(); i < iend2; i++) {
		//			if (status[i] == 0) {
		//				continue;
		//			}
		//			cv::Point2f refPt = refPts[i] + refBasePt;
		//			cv::Point2f currPt = currPts[i] + currBasePt;
		//			if (!pPrev->isInImage(currPt.x, currPt.y, 20))
		//				continue;
		//			auto pMP = refMPs[i];
		//			
		//			cv::Mat ray = mpSystem->mInvK*(cv::Mat_<float>(3, 1) << refPt.x, refPt.y, 1.0);
		//			float z_min, z_max;
		//			z_min = 0.01f;
		//			z_max = 1.0f;
		//			cv::Point2f XimgMin, XimgMax;
		//			cv::Mat Rcr, Tcr;
		//			mpMatcher->ComputeEpiLinePoint(XimgMin, XimgMax, ray, z_min, z_max, Rrel, Trel, mK); //ray,, Rrel, Trel
		//			cv::Mat lineEqu = mpMatcher->ComputeLineEquation(XimgMin, XimgMax);
		//			bool bEpiConstraints = mpMatcher->CheckLineDistance(lineEqu, currPt, 1.0);
		//			if (!bEpiConstraints) {
		//				cv::circle(refImg, refPt, 3, cv::Scalar(255, 0, 255), -1);
		//				cv::circle(currImg, currPt, 3, cv::Scalar(255, 0, 255), -1);
		//			}
		//			else {
		//				cv::circle(refImg, refPt, 3, cv::Scalar(255, 255, 0), -1);
		//				cv::circle(currImg, currPt, 3, cv::Scalar(255, 255, 0), -1);
		//			}

		//			/*auto pCP = pPrev->mpMatchInfo->mvpMatchingCPs[idx];
		//			if (pCP->mnTrackingFrameID == nCurrFrameID)
		//				continue;
		//			pCP->mnTrackingFrameID = nCurrFrameID;
		//			vpTempCPs.push_back(pCP);
		//			vTempPrevPts.push_back(prevPt);
		//			vTempCurrPts.push_back(currPt);
		//			vbTempInliers.push_back(true);*/
		//		}
		//	}
		//	cv::imshow("KF-KF::ref", refImg);
		//	cv::imshow("KF-KF::curr", currImg);
		//	cv::waitKey(1);
		//}
		//////Ref-Curr 매칭

		//에피폴라 관련 파라메터
		cv::Mat Rrel, Trel;
		pCurr->GetRelativePoseFromTargetFrame(pPrev, Rrel, Trel);
		//에피폴라 관련 파라메터

		//////그리드 파라메터
		auto pMatchInfo = pCurr->mpMatchInfo;
		int nGridSize = mpSystem->mnRadius*2;
		float fGridDistThresh = nGridSize*nGridSize*4;
		////그리드 파라메터
		for (size_t i = 0, iend = vpTempCPs.size(); i < iend; i++) {
			auto pCP = vpTempCPs[i];
			auto currPt = vTempCurrPts[i];
			auto prevPt = vTempPrevPts[i];
			auto pMP = pCP->GetMP();
			bool bMP = pMP && !pMP->isDeleted();
			bool bInlier = vbTempInliers[i];
			if (!bInlier) {
				continue;
			}

			/////////check epipolar constraints 
			cv::Mat ray = mpSystem->mInvK*(cv::Mat_<float>(3, 1) << prevPt.x, prevPt.y, 1.0);
			float z_min, z_max;
			z_min = 0.01f;
			z_max = 1.0f;
			cv::Point2f XimgMin, XimgMax;
			cv::Mat Rcr, Tcr;
			mpMatcher->ComputeEpiLinePoint(XimgMin, XimgMax, ray, z_min, z_max, Rrel, Trel, mK); //ray,, Rrel, Trel
			cv::Mat lineEqu = mpMatcher->ComputeLineEquation(XimgMin, XimgMax);
			bool bEpiConstraints = mpMatcher->CheckLineDistance(lineEqu, currPt, 1.0);
			vbTempInliers[i] = bEpiConstraints;
			if (!bEpiConstraints) {
				continue;
			}

			////이미 그리드에 포함되어 있는지 확인하는 단계
			auto gridPt = pPrev->GetGridBasePt(currPt, nGridSize);
			if (pCurr->mmbFrameGrids[gridPt]) {
				vbTempInliers[i] = false;
				continue;
			}
			////이미 그리드에 포함되어 있는지 확인하는 단계

			//////grid 추가
			auto rect = cv::Rect(gridPt, std::move(cv::Point2f(gridPt.x + nGridSize, gridPt.y + nGridSize)));
			pCurr->mmbFrameGrids[gridPt] = true;
			auto currGrid = new FrameGrid(gridPt, rect);
			pCurr->mmpFrameGrids[gridPt] = currGrid;
			pCurr->mmpFrameGrids[gridPt]->mpCP = pCP;
			pCurr->mmpFrameGrids[gridPt]->pt = currPt;

			pMatchInfo->AddCP(pCP, currPt);

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

		}

		////트래킹 결과 갱신
		//int nMP = UpdateMatchingInfo(pCurr, vpTempCPs, vTempCurrPts, vbTempInliers);
		///////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////키프레임 체크
		bool bNeedCP, bNeedMP, bNeedPoseHandle, bNeedNewKF;
		auto bNewKF = CheckNeedKeyFrame(pCurr, bNeedCP, bNeedMP, bNeedPoseHandle, bNeedNewKF);
		if (true) {
			if (bNeedCP) {
				std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateCP);
				mpSystem->mbCreateCP = false;
			}
			/*if (bNeedMP) {
				std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateMP);
				mpSystem->mbCreateMP = false;
			}*/
			if(bNeedNewKF || bNeedMP)
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