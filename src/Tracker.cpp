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
	bool c2 = mnMatching < 100;
	bool c3 = false;//mnMatching < mpFrameWindow->mnLastMatches*0.8;
	if (c1 || c2 || c3) {
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
		//std::unique_lock<std::mutex> lock(mpSystem->mMutexUseLocalMap);
		//
		//while (!mpSystem->mbLocalMapUpdateEnd){
		//	mpSystem->cvUseLocalMap.wait(lock);
		//}
		//mpSystem->mbTrackingEnd = false;
		//
		////pCurr->SetPose(pPrev->GetRotation(), pPrev->GetTranslation());
		//pCurr->SetPose(mpFrameWindow->GetRotation(), mpFrameWindow->GetTranslation());

		//int nLocalMapID = mpFrameWindow->GetLastFrameID();
		//auto mvpLocalMPs = mpFrameWindow->GetLocalMap();
		//cv::Mat mLocalMapDesc = mpFrameWindow->GetLocalMapDescriptor();
		//std::vector<cv::DMatch> mvMatchInfo;
		//mpSystem->mbTrackingEnd = true;
		//lock.unlock();
		//mpSystem->cvUseLocalMap.notify_one();
		
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		////Optical Flow Matching
		//초기 매칭 테스트
		std::vector<UVR_SLAM::MapPoint*> vpTempMPs;
		std::vector<cv::Point2f> vpTempPts;
		std::vector<bool> vbTempInliers;// = std::vector<bool>(pPrev->mvpMatchingMPs.size(), false);
		//int nMatch = mpMatcher->OpticalMatchingForTracking(mpRefKF, pCurr, vpTempMPs, vpTempPts, vbTempInliers); //pCurr
		std::cout << "tracking::start" << std::endl;
		cv::Mat overlap = cv::Mat::zeros(pCurr->GetOriginalImage().size(), CV_8UC1);
		std::cout << pPrev->mvMatchingPts.size() << std::endl;
		int nMatch = mpMatcher->OpticalMatchingForTracking(pPrev, pCurr, vpTempMPs, vpTempPts, vbTempInliers, overlap); //pCurr
		std::cout << "tracking::1" << std::endl;
		if (mpRefKF->GetBoolMapping() && !mpPlaneEstimator->isDoingProcess()) {
			mpRefKF->SetBoolMapping(false);
			mpMatcher->OpticalMatchingForTracking(mpRefKF, pCurr, vpTempMPs, vpTempPts, vbTempInliers, overlap);
		}
		mnMatching = Optimization::PoseOptimization(pCurr, vpTempMPs, vpTempPts, vbTempInliers);
		pCurr->SetInliers(mnMatching);
		mpFrameWindow->SetPose(pCurr->GetRotation(), pCurr->GetTranslation());
		CalcMatchingCount(pCurr, vpTempMPs, vpTempPts, vbTempInliers);
		//키프레임 체크
		if (CheckNeedKeyFrame(pCurr)) {
			if (!mpSegmentator->isDoingProcess() && !mpPlaneEstimator->isDoingProcess() && !mpRefKF->GetBoolMapping()) {
				std::cout << "insert key frame" << std::endl;
				pCurr->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
				mpSegmentator->InsertKeyFrame(pCurr);
				mpLocalMapper->InsertKeyFrame(pCurr);
				mpPlaneEstimator->InsertKeyFrame(pCurr);
				mpRefKF = pCurr;
			}
		}

		//visualizer에 맵포인트 설정
		mpVisualizer->SetMPs(vpTempMPs);
		std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
		double tttt = duration / 1000.0;

		//일단 테스트
		cv::Mat vis = pCurr->GetOriginalImage();
		vis.convertTo(vis, CV_8UC3);

		cv::Mat R = pCurr->GetRotation();
		cv::Mat t = pCurr->GetTranslation();

		for (int i = 0; i < vpTempMPs.size(); i++) {
			UVR_SLAM::MapPoint* pMPi = vpTempMPs[i];
			if (!pMPi || pMPi->isDeleted())
				continue;
			if (!vbTempInliers[i])
				continue;
			cv::Point2f p2D;
			cv::Mat pCam;
			pMPi->Projection(p2D, pCam, R, t, mK, mnWidth, mnHeight);
			cv::line(vis, p2D, vpTempPts[i], cv::Scalar(255, 255, 0), 2);
		}
		
		std::stringstream ss;
		ss << "Traking = " << mnMatching <<", "<< tttt;
		cv::rectangle(vis, cv::Point2f(0, 0), cv::Point2f(vis.cols, 30), cv::Scalar::all(0), -1);
		cv::putText(vis, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
		////Optical Flow Matching
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//이전 프레임에 포함된 맵포인트와 현재 프레임의 키포인트를 매칭하는 과정.
		//빠른 속도를 위해 이전 프레임에서 추적되는 맵포인트를 디스크립터로 만듬.
		//int nInitMatching = mpMatcher->MatchingWithPrevFrame(pPrev, pCurr, mvMatchInfo);
		//std::cout << "tracker::initmatching::" << nInitMatching << std::endl;
		//mnMatching = Optimization::PoseOptimization(pCurr);
		//std::cout << "tracker::Pose::" << mnMatching << std::endl;
		/////////////////dense test
		////pPrev로 테스트해보기. 바로바로 다음 프레임에 트래킹 결과 추가하기
		//mpRefKF = mpMap->GetCurrFrame();
		//auto mvpDenseMPs =  mpRefKF->GetDenseVectors();
		//std::cout << "tracker::dense matching::copy::" << std::endl;
		//cv::Mat debugging;
		//std::vector<std::pair<int, cv::Point2f>> vPairs;
		//std::vector<bool> vbInliers;
		//int mnDenseMatching = mpMatcher->DenseMatchingWithEpiPolarGeometry(pCurr, mpRefKF, mvpDenseMPs, vPairs, mpSystem->mnPatchSize, mpSystem->mnHalfWindowSize, debugging);
		//vbInliers = std::vector<bool>(vPairs.size(), true);
		///*std::stringstream ss;
		//ss << mpSystem->GetDirPath(0) << "/tracking/" << mpRefKF->GetKeyFrameID() << "_" << pCurr->GetFrameID() << ".jpg";
		//imwrite(ss.str(), debugging);*/
		//std::cout << "tracker::dense matching::" << std::endl;
		/////////////////dense test
		//mnMatching = mpMatcher->MatchingWithLocalMap(pCurr, mvpLocalMPs, mLocalMapDesc, 5.0);
		//std::cout << "tracker::localmap::matching::" << mnMatching << std::endl;
		//
		////std::vector<cv::DMatch> tempMatches;
		////mpMatcher->MatchingWithEpiPolarGeometry(mpRefKF, pCurr, tempMatches);

		////////////////
		///*cv::Mat debugging;
		//mpMatcher->DenseMatchingWithEpiPolarGeometry(mpRefKF, pCurr,mpSystem->mnPatchSize, mpSystem->mnHalfWindowSize, debugging);
		//std::string base = mpSystem->GetDirPath(0);
		//std::stringstream ssss;
		//ssss << base << "/dense/dense_" << mpRefKF->GetKeyFrameID() << "_" << pCurr->GetFrameID() << ".jpg";
		//imwrite(ssss.str(), debugging);*/
		///////dense

		//mnMatching = Optimization::PoseOptimization(pCurr, mvpDenseMPs, vPairs, vbInliers);
		//pCurr->SetInliers(mnM);
		//mpFrameWindow->SetPose(pCurr->GetRotation(), pCurr->GetTranslation());
		//CalcMatchingCount(pCurr, mvpDenseMPs, vPairs, vbInliers);

		float angle = mpRefKF->CalcDiffZ(pCurr);
		//std::cout << "angle : " << angle << std::endl;
		/*if (CheckNeedKeyFrame(pCurr)) {
			if (!mpSegmentator->isDoingProcess()) {
				mpSegmentator->InsertKeyFrame(pCurr);
			}
		}*/

		/*if (!mpSegmentator->isDoingProcess()) {
			mpSegmentator->InsertKeyFrame(pCurr);
		}*/
		////update tracking results
		mpFrameWindow->mnLastMatches = mnMatching;

		////일단 테스트
		//cv::Mat vis = pCurr->GetOriginalImage();
		//vis.convertTo(vis, CV_8UC3);
		//
		//auto mvpMPs = pCurr->GetMapPoints();
		////auto mvpOPs = pCurr->GetObjectVector();
		//cv::Mat R = pCurr->GetRotation();
		//cv::Mat t = pCurr->GetTranslation();
		//
		//for (int i = 0; i < vpTempMPs.size(); i++) {
		//	UVR_SLAM::MapPoint* pMPi = vpTempMPs[i];
		//	if (!pMPi || pMPi->isDeleted())
		//		continue;
		//	if (!vbTempInliers[i])
		//		continue;
		//	cv::Point2f p2D;
		//	cv::Mat pCam;
		//	pMPi->Projection(p2D, pCam, R, t, mK, mnWidth, mnHeight);
		//	cv::line(vis, p2D, vpTempPts[i], cv::Scalar(255, 255, 0), 2);
		//}

		//for (int i = 0; i < mvpMPs.size(); i++) {
		//	UVR_SLAM::MapPoint* pMP = mvpMPs[i];
		//	
		//	if (!pMP){
		//		//cv::circle(vis, pCurr->mvKeyPoints[i].pt, 1, cv::Scalar(0, 0, 255), -1);
		//		continue;
		//	}
		//	if (pMP->isDeleted()) {
		//		pCurr->mvbMPInliers[i] = false;
		//		continue;
		//	}
		//	cv::Point2f p2D;
		//	cv::Mat pCam;
		//	pMP->Projection(p2D, pCam, R, t, mK, mnWidth, mnHeight);

		//	if (!pCurr->mvbMPInliers[i]){
		//		//if (pMP->GetPlaneID() > 0) {
		//		//	//circle(vis, p2D, 4, cv::Scalar(255, 0, 255), 2);
		//		//}
		//	}
		//	else {

		//		cv::line(vis, p2D, pCurr->mvKeyPoints[i].pt, cv::Scalar(255, 255, 0), 2);
		//		
		//		int nObservations = pMP->GetConnedtedFrames().size();
		//		if (nObservations > 13)
		//			cv::circle(vis, pCurr->mvKeyPoints[i].pt, 2, cv::Scalar(0, 255, 0), -1);
		//		else if (nObservations > 10)
		//			cv::circle(vis, pCurr->mvKeyPoints[i].pt, 2, cv::Scalar(0, 255, 255), -1);
		//		else if (nObservations > 7)
		//			cv::circle(vis, pCurr->mvKeyPoints[i].pt, 2, cv::Scalar(0, 0, 255), -1);
		//		else if (nObservations > 5)
		//			cv::circle(vis, pCurr->mvKeyPoints[i].pt, 2, cv::Scalar(255, 0, 255), -1);
		//		else if(nObservations > 3)
		//			cv::circle(vis, pCurr->mvKeyPoints[i].pt, 2, cv::Scalar(255, 0, 0), -1);
		//		else
		//			cv::circle(vis, pCurr->mvKeyPoints[i].pt, 2, cv::Scalar(255, 255, 0), -1);
		//		UVR_SLAM::ObjectType type = pMP->GetObjectType();
		//		
		//		/*if (type != OBJECT_NONE)
		//			circle(vis, p2D, 3, UVR_SLAM::ObjectColors::mvObjectLabelColors[type], -1);
		//		if (pMP->GetMapPointType() == MapPointType::PLANE_MP) {
		//			circle(vis, p2D, 2, cv::Scalar(255, 0, 255), -1);
		//		}*/
		//	}
		//}
		/*for (int i = 0; i < vPairs.size(); i++) {
			if (!vbInliers[i])
				continue;
			auto idx = vPairs[i].first;
			auto pt = vPairs[i].second;
			mvpMPs.push_back(mvpDenseMPs[idx]);
			cv::circle(vis, pt, 2, cv::Scalar(0, 0, 0), -1);
		}*/

		////////////////////////////////////////////////////////////////////////////////
		////////////////line test
		///*auto lines = mpRefKF->Getlines();
		//bool bLine = lines.size() > 0;
		//bool bPlane = mpRefKF->mvpPlanes.size() > 0;
		//if (bLine && bPlane) {
		//	
		//	auto plane = mpRefKF->mvpPlanes[0];
		//	cv::Mat normal1;
		//	float dist1;
		//	plane->GetParam(normal1, dist1);
		//	cv::Mat K = mK.clone();
		//	K.at<float>(0, 0) /= 2.0;
		//	K.at<float>(1, 1) /= 2.0;
		//	K.at<float>(0, 2) /= 2.0;
		//	K.at<float>(1, 2) /= 2.0;

		//	cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
		//	R.copyTo(T.rowRange(0, 3).colRange(0, 3));
		//	t.copyTo(T.col(3).rowRange(0, 3));

		//	cv::Mat planeParam = plane->matPlaneParam.clone();
		//	cv::Mat invT = T.inv();
		//	cv::Mat invP = invT.t()*planeParam;
		//	cv::Mat invK = K.inv();

		//	cv::Mat R, t;
		//	pCurr->GetPose(R, t);
		//	
		//	for (int i = 0; i < lines.size(); i++) {
		//		cv::Mat param = UVR_SLAM::PlaneInformation::PlaneWallEstimator(lines[i], normal1, invP, invT, invK);
		//		float m;
		//		cv::Mat mLine = UVR_SLAM::PlaneInformation::FlukerLineProjection(planeParam, param, R, t, mK2, m);
		//		cv::Point2f sPt, ePt;
		//		UVR_SLAM::PlaneInformation::CalcFlukerLinePoints(sPt, ePt, 0.0, mnHeight, mLine);
		//		cv::line(vis, sPt, ePt, cv::Scalar(0, 255, 0), 3);
		//		std::cout << "tttt::" << param.t() << std::endl;
		//	}
		//}*/
		////////////////line test

		//////////////////////////////////////////////////////////////////////////
		////////////////Wall Line TEST
		//auto wallParams = mpMap->GetWallPlanes();//mpRefKF->GetWallParams();
		//int nkid = mpRefKF->GetKeyFrameID();
		//if (wallParams.size() > 0 && mpRefKF->mvpPlanes.size() > 0) {
		//	auto plane = mpRefKF->mvpPlanes[0];
		//	cv::Mat planeParam = plane->matPlaneParam.clone();
		//	for (int i = 0; i < wallParams.size(); i++) {
		//		int nid = wallParams[i]->GetRecentKeyFrameID();
		//		if (nid + 2 < nkid)
		//			continue;
		//		float m;
		//		cv::Mat mLine = UVR_SLAM::PlaneInformation::FlukerLineProjection(wallParams[i]->GetParam(), planeParam, R, t, mK2, m);
		//		cv::Point2f sPt, ePt;
		//		UVR_SLAM::PlaneInformation::CalcFlukerLinePoints(sPt, ePt, 0.0, mnHeight, mLine);
		//		cv::line(vis, sPt, ePt, cv::Scalar(0, 255, 0), 3);
		//		//std::cout << "tttt::" << wallParams[i].t() << std::endl;
		//	}
		//}
		////////////////Wall Line TEST
		//////////////////////////////////////////////////////////////////////////
		//

		////속도 및 에러 출력
		///*std::stringstream ss;
		//ss << std::setw(5) << "Tracker TIME : " << tttt << " || " << mnMatching<<" Local Map : "<<mvpLocalMPs.size();
		//mpSystem->SetTrackerString(ss.str());*/
		
		cv::imshow("Output::Tracking", vis);
		//mpVisualizer->SetMPs(pCurr->GetMapPoints());
		
		//mpVisualizer->SetMPs(mvpMPs);

		//visualizer thread
		if (!mpVisualizer->isDoingProcess()) {
			mpVisualizer->SetBoolDoingProcess(true);
		}
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