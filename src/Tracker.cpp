#include <Tracker.h>
#include <System.h>
#include <Map.h>
#include <Plane.h>
#include <FrameWindow.h>
#include <Matcher.h>
#include <Initializer.h>
#include <LocalMapper.h>
#include <SemanticSegmentator.h>
#include <SegmentationData.h>
#include <PlaneEstimator.h>
#include <Visualizer.h>

//std::vector<cv::Vec3b> UVR_SLAM::ObjectColors::mvObjectLabelColors;

UVR_SLAM::Tracker::Tracker() {}
UVR_SLAM::Tracker::Tracker(int w, int h, cv::Mat K):mnWidth(w), mnHeight(h), mK(K), mbInitializing(false), mbFirstFrameAfterInit(false), mbInitilized(false){}
UVR_SLAM::Tracker::Tracker(Map* pMap, std::string strPath) : mbInitializing(false), mbFirstFrameAfterInit(false), mbInitilized(false) {
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

	float fps = fs["Camera.fps"];
	mnMaxFrames = fps;
	mnMinFrames = fps / 7 ;//3

	mnWidth = fs["Image.width"];
	mnHeight = fs["Image.height"];
	mK2 = (cv::Mat_<float>(3, 3) << fx, 0, 0, 0, fy, 0, -fy*cx, -fx*cy, fx*fy); //line projection
	fs.release();

	mpMap = pMap;
}
UVR_SLAM::Tracker::~Tracker() {}

bool UVR_SLAM::Tracker::isInitialized() {
	return mbInitilized;
}

void UVR_SLAM::Tracker::SetSystem(System* pSystem) {
	mpSystem = pSystem;
}
void UVR_SLAM::Tracker::SetMapOptimizer(MapOptimizer* pMapOptimizer) {
	mpMapOptimizer = pMapOptimizer;
}
void UVR_SLAM::Tracker::SetVisualizer(Visualizer* pVis) {
	mpVisualizer = pVis;
}
void UVR_SLAM::Tracker::SetSegmentator(SemanticSegmentator* pSegmentator) {
	mpSegmentator = pSegmentator;
}
void UVR_SLAM::Tracker::SetMatcher(UVR_SLAM::Matcher* pMatcher){	
	mpMatcher = pMatcher;
}
void UVR_SLAM::Tracker::SetInitializer(UVR_SLAM::Initializer* pInitializer){
	mpInitializer = pInitializer;
}
void UVR_SLAM::Tracker::SetFrameWindow(UVR_SLAM::FrameWindow* pWindow) {
	mpFrameWindow = pWindow;
}
void UVR_SLAM::Tracker::SetLocalMapper(UVR_SLAM::LocalMapper* pLocalMapper) {
	mpLocalMapper = pLocalMapper;
}
void UVR_SLAM::Tracker::SetPlaneEstimator(UVR_SLAM::PlaneEstimator* pEstimator) {
	mpPlaneEstimator = pEstimator;
}

bool UVR_SLAM::Tracker::CheckNeedKeyFrame(Frame* pCurr) {
	int nMinObs = 3;
	if (mpFrameWindow->GetLocalMapFrames().size() <= 2)
		nMinObs = 2;
	//int nRefMatches = mpFrameWindow->TrackedMapPoints(nMinObs);
	int nRefMatches = mpRefKF->TrackedMapPoints(nMinObs);
	
	float thRefRatio = 0.9f;

	//bool bLocalMappingIdle = !mpLocalMapper->isDoingProcess();
	
	//기존 방식대로 최소 프레임을 만족하면 무조건 추가.
	//트래킹 수가 줄어들면 바로 추가.
	int nLastID = mpRefKF->GetFrameID();
	bool c1 = pCurr->GetFrameID() >= nLastID + mnMinFrames; //최소한의 조건
	bool c2 = mnMatching < 800; //pCurr->mpMatchInfo->mvMatchingPts.size() < 800;//mnMatching < 500;
	bool c3 = false;//mnMatching < mpFrameWindow->mnLastMatches*0.8;
	if (c2) { //c1 || c2 || c3
		/*if (!bLocalMappingIdle)
		{
			mpLocalMapper->StopLocalMapping(true);
			return false;
		}*/
		return true;
	}
	else
		return false;

	


	////초기조건들
	//bool c1a = pCurr->GetFrameID() >= nLastID + mnMaxFrames; //무조건 추가되는 상황
	//bool c1b = pCurr->GetFrameID() >= nLastID + mnMinFrames; //최소한의 조건
	//bool c2 = false; mnMatching < nRefMatches*thRefRatio && mnMatching > 20; //매칭 퀄리티를 유지하기 위한 것. 

	////std::cout << "CheckNeedKeyFrame::Ref = " << nRefMatches << ", " << mnMatching << ", " << mpFrameWindow->mnLastMatches << "::IDLE = " << bLocalMappingIdle <<"::C2 = "<<c2<<", "<<nLastID<< std::endl;

	//if ((c1b || c2)&& !bLocalMappingIdle) {
	//	//interrupt
	//	mpLocalMapper->StopLocalMapping(true);
	//}

	//if ((c1a || c1b || c2) && !mpLocalMapper->isDoingProcess()) {
	//	//KF 동작 중 멈추는 과정이 필요함.
	//	//local mapping이 동작하는 와중에도 KF가 추가됨.
	//	return true;
	//}
	//else
	//	return false;

	//return false;
}

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
		}
		
	}
	else {
		std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
		
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////Optical Flow Matching
		////MatchInfo 설정
		pCurr->mpMatchInfo = new UVR_SLAM::MatchInfo(pCurr, mpRefKF, mnWidth, mnHeight);
		pCurr->SetPose(pPrev->GetRotation(), pPrev->GetTranslation());
		////MatchInfo 설정
		//초기 매칭 테스트
		std::vector<UVR_SLAM::MapPoint*> vpTempMPs;
		std::vector<cv::Point2f> vpTempPts;
		std::vector<bool> vbTempInliers;// = std::vector<bool>(pPrev->mvpMatchingMPs.size(), false);
		std::vector<int> vnIDXs, vnMPIDXs;
		cv::Mat debugImg;
		cv::Mat overlap = cv::Mat::zeros(pCurr->GetOriginalImage().size(), CV_8UC1);
		int nMatch = mpMatcher->OpticalMatchingForTracking(pPrev, pCurr, vpTempMPs, vpTempPts, vbTempInliers, vnIDXs, vnMPIDXs, overlap, debugImg); //pCurr
		mnMatching = Optimization::PoseOptimization(pCurr, vpTempMPs, vpTempPts, vbTempInliers, vnMPIDXs);
		pCurr->SetInliers(mnMatching);
		UpdateMatchingInfo(pPrev, pCurr, vpTempMPs, vpTempPts, vbTempInliers, vnIDXs, vnMPIDXs);
		
		//키프레임 체크
		float angle = mpRefKF->CalcDiffZ(pCurr);
		if (CheckNeedKeyFrame(pCurr)) {
			if (!mpSegmentator->isDoingProcess()) {
				pCurr->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
				mpRefKF = pCurr;
				//mpRefKF->Init(mpSystem->mpORBExtractor, mpSystem->mK, mpSystem->mD);
				//mpRefKF->mpMatchInfo->SetKeyFrame();
				mpLocalMapper->InsertKeyFrame(pCurr);
				mpSegmentator->InsertKeyFrame(pCurr);
				mpFrameWindow->AddFrame(pCurr);
			}
			/*if (!mpSegmentator->isDoingProcess() && !mpPlaneEstimator->isDoingProcess() && !mpRefKF->GetBoolMapping()) {
				std::cout << "insert key frame" << std::endl;
				pCurr->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
				mpSegmentator->InsertKeyFrame(pCurr);
				mpLocalMapper->InsertKeyFrame(pCurr);
				mpPlaneEstimator->InsertKeyFrame(pCurr);
				mpRefKF = pCurr;
			}*/
		}
		////////Optical Flow Matching

		////////Visualization & 시간 계산
		std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
		double tttt = duration / 1000.0;
		
		cv::Mat vis = pCurr->GetOriginalImage();
		vis.convertTo(vis, CV_8UC3);
		cv::Mat R = pCurr->GetRotation();
		cv::Mat t = pCurr->GetTranslation();
		for (int i = 0; i < vpTempPts.size(); i++) {
			int label = pPrev->mpMatchInfo->mvObjectLabels[vnIDXs[i]];
			if (label == 150) {
				cv::circle(debugImg, vpTempPts[i], 3, cv::Scalar(255, 0, 255), 1);
			}
			else if (label == 255) {
				cv::circle(debugImg, vpTempPts[i], 3, cv::Scalar(0, 255, 255), 1);
			}
		}
		imshow("Output::Matching", debugImg);
		for (int i = 0; i < vpTempMPs.size(); i++) {
			UVR_SLAM::MapPoint* pMPi = vpTempMPs[i];
			if (!pMPi || pMPi->isDeleted())
				continue;
			/*if (!vbTempInliers[vnMPIDXs[i]])
				continue;*/
			cv::Point2f p2D;
			cv::Mat pCam;
			bool b = pMPi->Projection(p2D, pCam, R, t, mK, mnWidth, mnHeight);
			if (!b || !vbTempInliers[vnMPIDXs[i]]) {
				cv::line(vis, p2D, vpTempPts[vnMPIDXs[i]], cv::Scalar(0, 0, 255), 1);
			}
			else {
				cv::line(vis, p2D, vpTempPts[vnMPIDXs[i]], cv::Scalar(255, 255, 0), 1);
			}
			
			//cv::circle(vis, p2D, 2, cv::Scalar(255, 0, 0), -1);
		}
		std::stringstream ss;
		ss << "Traking = " << mnMatching <<", "<< tttt;
		cv::rectangle(vis, cv::Point2f(0, 0), cv::Point2f(vis.cols, 30), cv::Scalar::all(0), -1);
		cv::putText(vis, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
		cv::imshow("Output::Tracking", vis);
		////////Visualization & 시간 계산
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//visualizer thread
		mpVisualizer->SetMPs(vpTempMPs);
		if (!mpVisualizer->isDoingProcess()) {
			mpVisualizer->SetBoolDoingProcess(true);
		}
		//visualizer thread
		cv::waitKey(1);
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

//사용X
void UVR_SLAM::Tracker::CalcVisibleCount(UVR_SLAM::Frame* pF) {
	for (int i = 0; i < pF->mvKeyPoints.size(); i++) {
		UVR_SLAM::MapPoint* pMP = pF->mvpMPs[i];
		if (!pMP)
			continue;
		pMP->IncreaseVisible();
	}
}

void UVR_SLAM::Tracker::CalcMatchingCount(UVR_SLAM::Frame* pF) {

	//update tracked status
	pF->mTrackedDescriptor = cv::Mat::zeros(0, pF->matDescriptor.cols, pF->matDescriptor.type());
	pF->mNotTrackedDescriptor = cv::Mat::zeros(0, pF->matDescriptor.cols, pF->matDescriptor.type());

	for (int i = 0; i < pF->mvKeyPoints.size(); i++) {
		bool bMatch = true;
		//if (!pF->GetBoolInlier(i))
		//continue;
		UVR_SLAM::MapPoint* pMP = pF->mvpMPs[i];
		if (!pMP){
			pF->mNotTrackedDescriptor.push_back(pF->matDescriptor.row(i));
			pF->mvNotTrackedIdxs.push_back(i);
			continue;
		}
		if (pMP->isDeleted()){
			pF->mNotTrackedDescriptor.push_back(pF->matDescriptor.row(i));
			pF->mvNotTrackedIdxs.push_back(i);
			continue;
		}
		pMP->IncreaseVisible();
		if (pF->mvbMPInliers[i]) {
			pMP->IncreaseFound();
			pF->mTrackedDescriptor.push_back(pF->matDescriptor.row(i));
			pF->mvTrackedIdxs.push_back(i);
			pMP->SetDescriptor(pF->matDescriptor.row(i));

			////floor, wall descriptor update
			//if (pF->mLabelStatus.at<uchar>(i) == 0) {
			//	auto type = pMP->GetObjectType();
			//	switch (type) {
			//	case ObjectType::OBJECT_FLOOR:
			//		pF->mLabelStatus.at<uchar>(i) = (int)ObjectType::OBJECT_FLOOR;
			//		pF->mPlaneDescriptor.push_back(pF->matDescriptor.row(i));
			//		pF->mPlaneIdxs.push_back(i);
			//		break;
			//	case ObjectType::OBJECT_WALL:
			//		pF->mLabelStatus.at<uchar>(i) = (int)ObjectType::OBJECT_WALL;
			//		pF->mWallDescriptor.push_back(pF->matDescriptor.row(i));
			//		pF->mWallIdxs.push_back(i);
			//		break;
			//	}
			//}
			
		}
		else {
			pF->mNotTrackedDescriptor.push_back(pF->matDescriptor.row(i));
			pF->mvNotTrackedIdxs.push_back(i);
			pF->mvpMPs[i] = nullptr;
		}
	}
}

void UVR_SLAM::Tracker::CalcMatchingCount(UVR_SLAM::Frame* pF, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<cv::Point2f> vpPts, std::vector<bool> vbInliers) {

	for (int i = 0; i < vpMPs.size(); i++) {
		bool bMatch = true;
		UVR_SLAM::MapPoint* pMPi = vpMPs[i];
		if (!pMPi || pMPi->isDeleted())
			continue;
		pMPi->IncreaseVisible();
		if (vbInliers[i]) {
			pMPi->IncreaseFound();
			pF->mvpMatchingMPs.push_back(pMPi);
			pF->mvMatchingPts.push_back(vpPts[i]);
		}
	}
	//for (int i = 0; i < pF->mvKeyPoints.size(); i++) {
	//	bool bMatch = true;
	//	//if (!pF->GetBoolInlier(i))
	//	//continue;
	//	UVR_SLAM::MapPoint* pMP = pF->mvpMPs[i];
	//	if (!pMP) {
	//		pF->mNotTrackedDescriptor.push_back(pF->matDescriptor.row(i));
	//		pF->mvNotTrackedIdxs.push_back(i);
	//		continue;
	//	}
	//	if (pMP->isDeleted()) {
	//		pF->mNotTrackedDescriptor.push_back(pF->matDescriptor.row(i));
	//		pF->mvNotTrackedIdxs.push_back(i);
	//		continue;
	//	}
	//	pMP->IncreaseVisible();
	//	if (pF->mvbMPInliers[i]) {
	//		pMP->IncreaseFound();
	//		pF->mTrackedDescriptor.push_back(pF->matDescriptor.row(i));
	//		pF->mvTrackedIdxs.push_back(i);
	//		pMP->SetDescriptor(pF->matDescriptor.row(i));

	//		////floor, wall descriptor update
	//		//if (pF->mLabelStatus.at<uchar>(i) == 0) {
	//		//	auto type = pMP->GetObjectType();
	//		//	switch (type) {
	//		//	case ObjectType::OBJECT_FLOOR:
	//		//		pF->mLabelStatus.at<uchar>(i) = (int)ObjectType::OBJECT_FLOOR;
	//		//		pF->mPlaneDescriptor.push_back(pF->matDescriptor.row(i));
	//		//		pF->mPlaneIdxs.push_back(i);
	//		//		break;
	//		//	case ObjectType::OBJECT_WALL:
	//		//		pF->mLabelStatus.at<uchar>(i) = (int)ObjectType::OBJECT_WALL;
	//		//		pF->mWallDescriptor.push_back(pF->matDescriptor.row(i));
	//		//		pF->mWallIdxs.push_back(i);
	//		//		break;
	//		//	}
	//		//}

	//	}
	//	else {
	//		pF->mNotTrackedDescriptor.push_back(pF->matDescriptor.row(i));
	//		pF->mvNotTrackedIdxs.push_back(i);
	//		pF->mvpMPs[i] = nullptr;
	//	}
	//}
}


void UVR_SLAM::Tracker::CalcMatchingCount(UVR_SLAM::Frame* pF, std::vector<UVR_SLAM::MapPoint*> vDenseMPs, std::vector<std::pair<int, cv::Point2f>> vPairs, std::vector<bool> vbInliers) {

	//update tracked status
	pF->mTrackedDescriptor = cv::Mat::zeros(0, pF->matDescriptor.cols, pF->matDescriptor.type());
	pF->mNotTrackedDescriptor = cv::Mat::zeros(0, pF->matDescriptor.cols, pF->matDescriptor.type());

	for (int i = 0; i < pF->mvKeyPoints.size(); i++) {
		bool bMatch = true;
		//if (!pF->GetBoolInlier(i))
		//continue;
		UVR_SLAM::MapPoint* pMP = pF->mvpMPs[i];
		if (!pMP) {
			pF->mNotTrackedDescriptor.push_back(pF->matDescriptor.row(i));
			pF->mvNotTrackedIdxs.push_back(i);
			continue;
		}
		if (pMP->isDeleted()) {
			pF->mNotTrackedDescriptor.push_back(pF->matDescriptor.row(i));
			pF->mvNotTrackedIdxs.push_back(i);
			continue;
		}
		pMP->IncreaseVisible();
		if (pF->mvbMPInliers[i]) {
			pMP->IncreaseFound();
			pF->mTrackedDescriptor.push_back(pF->matDescriptor.row(i));
			pF->mvTrackedIdxs.push_back(i);
			pMP->SetDescriptor(pF->matDescriptor.row(i));
		}
		else {
			pF->mNotTrackedDescriptor.push_back(pF->matDescriptor.row(i));
			pF->mvNotTrackedIdxs.push_back(i);
			pF->mvpMPs[i] = nullptr;
		}
	}
	for (int i = 0; i < vPairs.size(); i++) {
		if (!vbInliers[i])
			continue;
		auto idx = vPairs[i].first;
		auto pt = vPairs[i].second;
		UVR_SLAM::MapPoint* pMPi = vDenseMPs[idx];
		if (!pMPi)
			continue;
		if (pMPi->isDeleted())
			continue;
		pMPi->IncreaseVisible();
		if (vbInliers[i]) {
			pMPi->IncreaseFound();
		}
	}
}

void UVR_SLAM::Tracker::UpdateMatchingInfo(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<cv::Point2f> vpPts, std::vector<bool> vbInliers, std::vector<int> vnIDXs, std::vector<int> vnMPIDXs) {
	//std::cout << "tracking::update::" << vpMPs.size() << ", " << vpPts.size() << ", " << vbInliers.size() << ", " << vnIDXs.size() << std::endl;
	auto pMatchInfo = pCurr->mpMatchInfo;
	auto pPrevMatchInfo = pPrev->mpMatchInfo;
	//std::cout << "tracking::update::" << pPrevMatchInfo->mvnMatchingPtIDXs.size() << ", " << vnIDXs.size() << std::endl;
	std::vector<UVR_SLAM::MapPoint*> vpTempMPs = std::vector<UVR_SLAM::MapPoint*>(vpPts.size(), nullptr);
	//pMatchInfo->mvpMatchingMPs = std::vector<UVR_SLAM::MapPoint*>(vpPts.size(), nullptr);
	for (int i = 0; i < vpMPs.size(); i++) {
		int idx = vnMPIDXs[i];
		if (vbInliers[idx]) {
			vpTempMPs[idx] = vpMPs[i];
		}
	}
	int nres = 0;
	for (int i = 0; i < vpPts.size(); i++) {
		if (!vbInliers[i]){
			continue;
		}
		auto pt = vpPts[i];
		auto pMP = vpTempMPs[i];
		if (!pMatchInfo->CheckPt(pt)) {
			/*if (pPrev->mpMatchInfo->mvnMatchingPtIDXs.size() <= vnIDXs[i])
				std::cout << "tracking::update::error::" << vnIDXs[i] << std::endl;*/
			//curr->mpMatchInfo->mpTargetFrame->mpMatchInfo->mvMatchingPts[prev->mpMatchInfo->mvnMatchingPtIDXs[i]]
			//pMatchInfo->AddMatchingPt(pt, pMP, pPrevMatchInfo->mvnMatchingPtIDXs[vnIDXs[i]], pPrevMatchInfo->mvObjectLabels[vnIDXs[i]]);
			pMatchInfo->AddMatchingPt(pt, pMP, pPrevMatchInfo->mvnMatchingPtIDXs[vnIDXs[i]], mpRefKF->mpMatchInfo->mvObjectLabels[pPrev->mpMatchInfo->mvnMatchingPtIDXs[vnIDXs[i]]]);
			if (pMP) {
				nres++;
			}
		}
	}
	//std::cout << "tracking::update::" << nres << std::endl;
}