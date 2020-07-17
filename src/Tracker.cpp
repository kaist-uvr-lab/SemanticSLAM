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
	mnMaxFrames = fps;
	mnMinFrames = fps / 3 ;//3

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

	//1 : rotation angle
	bool bRotation = pCurr->CalcDiffAngleAxis(mpRefKF) > 10.0;
	bool bMaxFrames = pCurr->GetFrameID() >= mpRefKF->mnFrameID + mnMinFrames;
	bool bDoingSegment = !mpSegmentator->isDoingProcess();
	
	bool bMatchMapPoint = mnMapPointMatching < 600;
	bool bMatchPoint = mnPointMatching < 1200;

	if ((bRotation || bMatchMapPoint || bMatchPoint || bMaxFrames) && bDoingSegment)
	{
		if(pCurr->CheckBaseLine(mpRefKF))
			return true;
		return false;
	}
	else
		return false;


	///////////////////////////











	//2 : point && map point
	//mnPointMatching && mnMapPointMatching;

	int nMinObs = 3;
	if (mpFrameWindow->GetLocalMapFrames().size() <= 2)
		nMinObs = 2;
	//int nRefMatches = mpFrameWindow->TrackedMapPoints(nMinObs);
	int nRefMatches = mpRefKF->TrackedMapPoints(nMinObs);
	



	

	//bool bLocalMappingIdle = !mpLocalMapper->isDoingProcess();
	float thRefRatio = 0.9f;
	//기존 방식대로 최소 프레임을 만족하면 무조건 추가.
	//트래킹 수가 줄어들면 바로 추가.
	int nLastID = mpRefKF->GetFrameID();
	bool c1 = pCurr->GetFrameID() >= nLastID + mnMinFrames; //일정프레임이 지나면 무조건 추가
	bool c2 = mnMapPointMatching < 800; //pCurr->mpMatchInfo->mvMatchingPts.size() < 800;//mnMatching < 500;
	bool c3 = false;//mnMatching < mpFrameWindow->mnLastMatches*0.8;
	if ( c2 ) { //c1 || c2 || c3
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
			mpMap->AddTraFrame(mpInitializer->mpInitFrame1);
			mpMap->AddTraFrame(pCurr);
		}
	}
	else {
		std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
		std::cout << "tracker::start" << std::endl;
		//std::cout << mpMap->mpFirstKeyFrame->mpPlaneInformation->GetFloorPlane()->GetParam() << std::endl << std::endl << std::endl;
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////Optical Flow Matching
		////MatchInfo 설정
		mpRefKF->SetRecentTrackedFrameID(pCurr->GetFrameID());
		pCurr->mpMatchInfo = new UVR_SLAM::MatchInfo(pCurr, mpRefKF, mnWidth, mnHeight);
		cv::Mat prevR, prevT;
		pPrev->GetPose(prevR, prevT);
		pCurr->SetPose(prevR, prevT);
		mpMap->AddTraFrame(pCurr);
		////MatchInfo 설정
		//초기 매칭 테스트
		std::vector<UVR_SLAM::MapPoint*> vpTempMPs;
		std::vector<cv::Point2f> vpTempPts, vpTempPts1;
		std::vector<cv::Point3f> vpTempPts2;
		std::vector<uchar> vcInliers;
		std::vector<bool> vbTempInliers;// = std::vector<bool>(pPrev->mvpMatchingMPs.size(), false);
		std::vector<int> vnIDXs, vnMPIDXs;
		cv::Mat debugImg;
		cv::Mat overlap = cv::Mat::zeros(pCurr->GetOriginalImage().size(), CV_8UC1);
		mnPointMatching = mpMatcher->OpticalMatchingForTracking(pPrev, pCurr, vpTempMPs, vpTempPts, vpTempPts1, vpTempPts2, vbTempInliers, vnIDXs, vnMPIDXs, overlap, debugImg); //pCurr
		std::chrono::high_resolution_clock::time_point tracking_a = std::chrono::high_resolution_clock::now();
		
		////////KF-F matching test
		cv::Mat dddddbug;
		mpMatcher->OpticalMatchingForTracking2(mpRefKF, pCurr, dddddbug);
		/*
		cv::Mat imgKFNF;
		mpMatcher->OpticalKeyframeAndFrameMatchingForTracking(mpRefKF, mpRefKF->mpMatchInfo->mpTargetFrame, imgKFNF);
		*/
		std::stringstream ssdira;
		ssdira << mpSystem->GetDirPath(0) << "/kfmatching/" <<mpRefKF->GetFrameID()<<"_"<< pCurr->GetFrameID() << "_tracking.jpg";
		imwrite(ssdira.str(), dddddbug);
		////////KF-F matching test

		//graph-based0.
		mnMapPointMatching = Optimization::PoseOptimization(pCurr, vpTempMPs, vpTempPts, vbTempInliers, vnMPIDXs);
		////solvepnp
		/*mnMatching = 0;
		cv::Mat rvec, tvec;
		cv::Rodrigues(prevR, rvec);
		cv::solvePnPRansac(vpTempPts2, vpTempPts1, mK, mD, rvec, tvec, true, 100, 4.0, 0.99, vcInliers);
		cv::Rodrigues(rvec, prevR);
		prevR.convertTo(prevR, CV_32FC1);
		tvec.convertTo(tvec, CV_32FC1);
		pCurr->SetPose(prevR, tvec);
		for (int i = 0; i < vcInliers.size(); i++)
		{
			if (vcInliers[i]) {
				mnMatching++;
			}
			else {
				vbTempInliers[vnMPIDXs[i]] = false;
			}
		}*/
		////solvepnp
		//pCurr->SetInliers(mnMatching); //이용X
		int nMP = UpdateMatchingInfo(pPrev, pCurr, vpTempMPs, vpTempPts, vbTempInliers, vnIDXs, vnMPIDXs);
		
		///////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////키프레임 체크
		if (CheckNeedKeyFrame(pCurr)) {
			auto pNewKF = pCurr;
			pNewKF->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
			mpRefKF = pNewKF;
			mpLocalMapper->InsertKeyFrame(pNewKF);
			mpSegmentator->InsertKeyFrame(pNewKF);
			mpPlaneEstimator->InsertKeyFrame(pNewKF);
		}
		////////////////////이전 버전
		//float angle = (mpRefKF->CalcDiffAngleAxis(pCurr));
		//if (CheckNeedKeyFrame(pCurr)) {
		//	if (!mpSegmentator->isDoingProcess() && pCurr->CheckBaseLine(mpRefKF)) {
		//		pCurr->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
		//		mpRefKF = pCurr;
		//		//mpRefKF->Init(mpSystem->mpORBExtractor, mpSystem->mK, mpSystem->mD);
		//		//mpRefKF->mpMatchInfo->SetKeyFrame();
		//		mpLocalMapper->InsertKeyFrame(pCurr);
		//		mpSegmentator->InsertKeyFrame(pCurr);
		//		mpPlaneEstimator->InsertKeyFrame(pCurr);
		//		//mpFrameWindow->AddFrame(pCurr);
		//	}
		//	/*if (!mpSegmentator->isDoingProcess() && !mpPlaneEstimator->isDoingProcess() && !mpRefKF->GetBoolMapping()) {
		//		std::cout << "insert key frame" << std::endl;
		//		pCurr->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
		//		mpSegmentator->InsertKeyFrame(pCurr);
		//		mpLocalMapper->InsertKeyFrame(pCurr);
		//		mpPlaneEstimator->InsertKeyFrame(pCurr);
		//		mpRefKF = pCurr;
		//	}*/
		//}
		/////////////////////////////////////////////키프레임 체크
		///////////////////////////////////////////////////////////////////////////////
		////////Optical Flow Matching

		///////////////////SAVE TRACKING RESULTS
		/*std::stringstream ssdir;
		ssdir << mpSystem->GetDirPath(0) << "/kfmatching/" <<mpRefKF->GetFrameID()<<"_"<< pCurr->GetFrameID() << "_tracking.jpg";
		imwrite(ssdir.str(), imgKFNF);*/
		///////////////////SAVE TRACKING RESULTS

		std::cout << "tracker::end" << std::endl;

		////////Visualization & 시간 계산
		std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
		
		auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_a - tracking_start).count();
		double t1 = duration1 / 1000.0;
		auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_a).count();
		double t2 = duration2 / 1000.0;

		//////시각화1
		cv::Mat vis = pCurr->GetOriginalImage();
		//vis.convertTo(vis, CV_8UC3);
		cv::Mat R = pCurr->GetRotation();
		cv::Mat t = pCurr->GetTranslation();
		//for (int i = 0; i < vpTempPts.size(); i++) {
		//	int label = pPrev->mpMatchInfo->mvObjectLabels[vnIDXs[i]];
		//	if (label == 150) {
		//		cv::circle(debugImg, vpTempPts[i], 3, cv::Scalar(255, 0, 255), 1);
		//	}
		//	else if (label == 255) {
		//		cv::circle(debugImg, vpTempPts[i], 3, cv::Scalar(0, 255, 255), 1);
		//	}
		//}
		//////시각화1
		//imshow("Output::Matching", debugImg);
		/*for (int i = 0; i < vpTempPts.size(); i++) {
			cv::circle(vis, vpTempPts[i], 2, cv::Scalar(255, 0, 0), 1);
		}*/

		//////시각화2
		for (int i = 0; i < vpTempMPs.size(); i++) {
			UVR_SLAM::MapPoint* pMPi = vpTempMPs[i];
			if (!pMPi || pMPi->isDeleted())
				continue;
			/*if (!vbTempInliers[vnMPIDXs[i]])
				continue;*/
			cv::Point2f p2D;
			cv::Mat pCam;
			bool b = pMPi->Projection(p2D, pCam, R, t, mK, mnWidth, mnHeight);

			int label = pPrev->mpMatchInfo->mvObjectLabels[vnIDXs[vnMPIDXs[i]]];
			int pid = pMPi->GetPlaneID();
			int type = pMPi->GetRecentLayoutFrameID();
			cv::Scalar color(150, 150, 0);
			if (pid > 0 && label == 150) {
				color = cv::Scalar(0, 0, 255);
			}
			else if (pid > 0 && label == 100) {
				color = cv::Scalar(0, 255, 0);
			}
			//else if (pid > 0 && label == 255) {
			else if (label == 255) {
				color = cv::Scalar(255, 0, 0);
				//color = UVR_SLAM::ObjectColors::mvObjectLabelColors[pid];
			}
			if (pid <= 0)
				color /= 2;
			if(vbTempInliers[vnMPIDXs[i]])
				cv::circle(vis, p2D, 2, color, -1);
			/*if (!b || !vbTempInliers[vnMPIDXs[i]]) {
				cv::line(vis, p2D, vpTempPts[vnMPIDXs[i]], cv::Scalar(0, 0, 255), 1);
			}
			else {
				cv::line(vis, p2D, vpTempPts[vnMPIDXs[i]], cv::Scalar(255, 255, 0), 1);
			}*/
			cv::line(vis, p2D, vpTempPts[vnMPIDXs[i]], color, 1);
			//cv::circle(vis, p2D, 2, cv::Scalar(255, 0, 0), -1);
		}
		////////시각화2
		std::stringstream ss;
		ss << "Traking = " << mnMapPointMatching <<", "<< mnPointMatching <<"::"<< t1<<", "<<t2<<"::";
		cv::rectangle(vis, cv::Point2f(0, 0), cv::Point2f(vis.cols, 30), cv::Scalar::all(0), -1);
		cv::putText(vis, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
		cv::imshow("Output::Tracking", vis);
		////////Visualization & 시간 계산
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//visualizer thread
		mpVisualizer->SetMatchInfo(pCurr->mpMatchInfo);
		//mpVisualizer->SetMPs(vpTempMPs);
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

int UVR_SLAM::Tracker::UpdateMatchingInfo(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<cv::Point2f> vpPts, std::vector<bool> vbInliers, std::vector<int> vnIDXs, std::vector<int> vnMPIDXs) {
	auto pMatchInfo = pCurr->mpMatchInfo;
	auto pPrevMatchInfo = pPrev->mpMatchInfo;
	
	std::vector<UVR_SLAM::MapPoint*> vpTempMPs = std::vector<UVR_SLAM::MapPoint*>(vpPts.size(), nullptr);
	
	////////////////////////////평면 관려 기능
	/////////////즉각적으로 맵포인트를 생성하기 위함
	//auto pTargetMatchInfo = pCurr->mpMatchInfo->mpTargetFrame->mpMatchInfo;
	//auto pTargetTargetMatchInfo = pTargetMatchInfo->mpTargetFrame->mpMatchInfo;
	//auto pPlane = pTargetMatchInfo->mpRefFrame->mpPlaneInformation->GetFloorPlane();
	////auto pPlaneInfo = pPrevMatchInfo->mpTargetFrame->mpPlaneInformation->
	//cv::Mat invP, invT, invK;
	//pTargetMatchInfo->mpRefFrame->mpPlaneInformation->Calculate();
	//pTargetMatchInfo->mpRefFrame->mpPlaneInformation->GetInformation(invP, invT, invK);
	//int nTargetMatch = pTargetMatchInfo->nMatch;
	//int nTargetTargetMatch = pTargetTargetMatchInfo->nMatch;
	////////현재 카메라 자세 확인용
	///*cv::Mat R, t;
	//pCurr->GetPose(R, t);*/
	/////////////즉각적으로 맵포인트를 생성하기 위함
	////////////////////////////평면 관려 기능

	int nInstantMP = 0;
	////매칭된 맵포인트를 복사
	for (int i = 0; i < vpMPs.size(); i++) {
		int idx = vnMPIDXs[i];
		if (vbInliers[idx]) {
			vpTempMPs[idx] = vpMPs[i];
		}
	}
	////매칭된 맵포인트를 복사
	int nres = 0;
	for (int i = 0; i < vpPts.size(); i++) {
		if (!vbInliers[i]){
			continue;
		}
		int prevIdx = vnIDXs[i];
		auto pt = vpPts[i];
		auto pMP = vpTempMPs[i];
		if (!pMatchInfo->CheckPt(pt)) {
			///////새로운 MP 즉성 생성 테스트
			//int targetIdx = pPrev->mpMatchInfo->mvnMatchingPtIDXs[vnIDXs[i]];
			//int label = pTargetMatchInfo->mvObjectLabels[targetIdx];
			//if (!pMP && label == 150 && targetIdx < nTargetMatch) {
			//	//현재 포인트의 이전 이전 키프레임까지 연결
			//	int targettargetIdx = pTargetMatchInfo->mvnTargetMatchingPtIDXs[targetIdx];
			//	//std::cout << "1::" << tempIdx << ", " << nPrevPrevMatch << std::endl;
			//	//std::cout << tempIdx << ", " << pTargetTargetMatchInfo->mvnMatchingPtIDXs.size() << ", " << pTargetTargetMatchInfo->mvpMatchingMPs.size() << std::endl;
			//	//int targettargetIdx = pTargetTargetMatchInfo->mvnMatchingPtIDXs[tempIdx];
			//	//std::cout << tempIdx << "," << targettargetIdx << std::endl;
			//	if (!pTargetTargetMatchInfo->mvpMatchingMPs[targettargetIdx]) {
			//		cv::Mat a = UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(pt, invP, invT, invK);
			//		pMP = new MapPoint(pTargetTargetMatchInfo->mpRefFrame, a, cv::Mat(), UVR_SLAM::MapPointType::PLANE_DENSE_MP);
			//		auto pt1 = pTargetMatchInfo->mvMatchingPts[targetIdx];
			//		auto pt2 = pTargetTargetMatchInfo->mvMatchingPts[targettargetIdx];
			//		pMP->AddFrame(pTargetMatchInfo, targetIdx);
			//		pMP->AddFrame(pTargetTargetMatchInfo, targettargetIdx);
			//		nInstantMP++;
			//	}

			//	//pMP = new MapPoint(pPrevMatchInfo->mpTargetFrame, a, cv::Mat(), UVR_SLAM::MapPointType::PLANE_DENSE_MP);
			//	/*cv::Mat tmp = R*a + t;
			//	tmp = mK*tmp;
			//	cv::Point2f tpt(tmp.at<float>(0) / tmp.at<float>(2), tmp.at<float>(1) / tmp.at<float>(2));
			//	tpt -= pt;
			//	float diff = sqrt(tpt.dot(tpt));*/
			//	//std::cout << "tracking::new instant mp::" << diff << std::endl;
			//}
			/////////새로운 MP 즉성 생성 테스트

			/*if (pPrev->mpMatchInfo->mvnMatchingPtIDXs.size() <= vnIDXs[i])
				std::cout << "tracking::update::error::" << vnIDXs[i] << std::endl;*/
			//curr->mpMatchInfo->mpTargetFrame->mpMatchInfo->mvMatchingPts[prev->mpMatchInfo->mvnMatchingPtIDXs[i]]
			//pMatchInfo->AddMatchingPt(pt, pMP, pPrevMatchInfo->mvnMatchingPtIDXs[vnIDXs[i]], pPrevMatchInfo->mvObjectLabels[vnIDXs[i]]);
			pMatchInfo->AddMatchingPt(pt, pMP, pPrevMatchInfo->mvnMatchingPtIDXs[vnIDXs[i]], mpRefKF->mpMatchInfo->mvObjectLabels[pPrev->mpMatchInfo->mvnMatchingPtIDXs[vnIDXs[i]]]
				, mpRefKF->mpMatchInfo->mvnOctaves[pPrev->mpMatchInfo->mvnMatchingPtIDXs[vnIDXs[i]]]);
			if (pMP) {
				nres++;
			}
		}
	}
	return nInstantMP;
}