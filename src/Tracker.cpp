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
#include <FrameVisualizer.h>
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
	mnMaxFrames = 5;// 10;//fps;
	mnMinFrames = 3; //fps / 3;//3

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
void UVR_SLAM::Tracker::SetFrameVisualizer(FrameVisualizer* pVis) {
	mpFrameVisualizer = pVis;
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

UVR_SLAM::Frame* UVR_SLAM::Tracker::CheckNeedKeyFrame(Frame* pCurr, Frame* pPrev) {
	
	///////////////
	//keyframe process
	int Nref = mpRefKF->mpMatchInfo->GetMatchingSize();
	int Ncurr = pCurr->mpMatchInfo->GetMatchingSize();

	float avg = ((float)mpRefKF->mpMatchInfo->mnTotalMatch) / mpRefKF->mpMatchInfo->mvpMatchInfos.size();

	mpRefKF->mpMatchInfo->mnTotalMatch += Ncurr;
	mpRefKF->mpMatchInfo->mvpMatchInfos.push_back(pCurr->mpMatchInfo);

	int nDiff = mnPointMatching - mnMapPointMatching;
	bool bDiff = nDiff > 50;
	//bool bKF = (((float)Ncurr) / Nref) < 0.7f;
	//bool bAVG = Ncurr < avg*0.7;

	//1 : rotation angle
	bool bDoingMapping = !mpLocalMapper->isDoingProcess();
	bool bRotation = pCurr->CalcDiffAngleAxis(mpRefKF) > 10.0;
	bool bMaxFrames = pCurr->GetFrameID() >= mpRefKF->mnFrameID + mnMaxFrames;//mnMinFrames;
	bool bMinFrames = pCurr->GetFrameID() < mpRefKF->mnFrameID + mnMinFrames;
	bool bDoingSegment = !mpSegmentator->isDoingProcess();

	bool bMatchMapPoint = mnMapPointMatching < 200;
	//bool bMatchPoint = mnPointMatching < 1200;

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
	if (!bMinFrames && bDoingMapping)
		return pCurr;
	if (!bMinFrames && bDoingMapping) {
		if (bRotation || bMaxFrames) {
			pRes = pCurr;
		}
		else if (bMatchMapPoint || bDiff) {//bKF
			pRes = pPrev;
		}
		else
			pRes = pCurr;
		//return pRes;
		if (pRes->CheckBaseLine(mpRefKF))
			return pRes;
		else
			return nullptr;
	}
	else
		return nullptr;
	///////////////////////////

	/////////////////
	////keyframe process
	///*int Nref = mpRefKF->mpMatchInfo->GetMatchingSize();
	//int Ncurr = pCurr->mpMatchInfo->GetMatchingSize();
	//float avg = ((float)mpRefKF->mpMatchInfo->mnTotalMatch) / mpRefKF->mpMatchInfo->mvpMatchInfos.size();
	//mpRefKF->mpMatchInfo->mnTotalMatch += Ncurr;
	//mpRefKF->mpMatchInfo->mvpMatchInfos.push_back(pCurr->mpMatchInfo);*/

	//if (mpRefKF->mpMatchInfo->mnMaxMatch < mnMapPointMatching) {
	//	mpRefKF->mpMatchInfo->mnMaxMatch = mnMapPointMatching;
	//	mpRefKF->mpMatchInfo->mpNextFrame = pCurr;
	//}

	//int nDiff = mnPointMatching - mnMapPointMatching;
	//bool bDiff = nDiff > 50;
	////bool bKF = (((float)Ncurr) / Nref) < 0.7f;
	////bool bAVG = Ncurr < avg*0.7;

	////1 : rotation angle
	//bool bDoingMapping = !mpLocalMapper->isDoingProcess();
	//bool bRotation = pCurr->CalcDiffAngleAxis(mpRefKF) > 10.0;
	//bool bMaxFrames = pCurr->GetFrameID() >= mpRefKF->mnFrameID + mnMaxFrames;//mnMinFrames;
	//bool bMinFrames = pCurr->GetFrameID() < mpRefKF->mnFrameID + mnMinFrames;
	//bool bDoingSegment = !mpSegmentator->isDoingProcess();
	//
	//bool bMatchMapPoint = mnMapPointMatching < 200;
	////bool bMatchPoint = mnPointMatching < 1200;

	////if ((bRotation || bMatchMapPoint || bMatchPoint || bMaxFrames) && bDoingSegment)
	///*if ((bRotation || bMatchMapPoint || bKF || bAVG || bMaxFrames) && !bMinFrames && bDoingMapping)
	//{
	//	if(pCurr->CheckBaseLine(mpRefKF))
	//		return true;
	//	return false;
	//}
	//else
	//	return false;*/
	//UVR_SLAM::Frame* pRes = nullptr;
	//if (!bMinFrames && bDoingMapping) {
	//	/*if (bMaxFrames) {
	//		pRes = pCurr;
	//	}
	//	else */
	//	if (bMatchMapPoint || bDiff || bRotation) {//bKF
	//		pRes = mpRefKF->mpMatchInfo->mpNextFrame;
	//	}
	//	else
	//		pRes = pCurr;
	//	return pRes;
	//	/*if (pRes->CheckBaseLine(mpRefKF))
	//		return pRes;
	//	else
	//		return nullptr;*/
	//}
	//else
	//	return nullptr;
	/////////////////////////////











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
		return nullptr;
	}
	else
		return nullptr;

	
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

UVR_SLAM::Frame* pMatchInfoTemp1 = nullptr;
UVR_SLAM::Frame* pMatchInfoTemp2 = nullptr;
//bool bRefKF = false;
void UVR_SLAM::Tracker::Tracking(Frame* pPrev, Frame* pCurr) {

	////////////Dense Matching Test
	cv::Mat testMatchingImg;
	double dtime = 0.0;
	cv::Mat mFlow;
	if(pPrev){
		mpMatcher->DenseOpticalMatchingForTracking(pCurr, pPrev, mFlow, dtime, testMatchingImg);
		mpMap->AddFlow(pPrev->GetFrameID(), mFlow);
	}
	////////////Dense Matching Test

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
			mnMapPointMatching = pCurr->mpMatchInfo->GetMatchingSize();

			//mpRefKF->mpMatchInfo->mvTempDenseMatchPts = mpRefKF->mpMatchInfo->GetMatchingPts();
			//mpRefKF->mpMatchInfo->mvbTempDense = std::vector<bool>(mpRefKF->mpMatchInfo->mvTempDenseMatchPts.size(), true);
		}
	}
	else {

		/*if (pPrev->GetFrameID() % 3 == 0) {
			pMatchInfoTemp1 = pMatchInfoTemp2;
			pMatchInfoTemp2 = pPrev;
			if (pMatchInfoTemp1) {
				cv::Mat testMatchingImg;
				mpMatcher->TestOpticalMatchingForMapping2(pMatchInfoTemp2, pMatchInfoTemp1, pPrev, testMatchingImg);
				std::stringstream sstdir;
				sstdir << mpSystem->GetDirPath(0) << "/kfmatching/tracking_test_" << pMatchInfoTemp2->GetFrameID() << "_" << pMatchInfoTemp1->GetFrameID() << ".jpg";
				imwrite(sstdir.str(), testMatchingImg);
			}
		}*/

		std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();

		/*std::unique_lock<std::mutex> lock(mpSystem->mMutexUseLocalMap);
		while (!mpSystem->mbLocalMapUpdateEnd) {
			mpSystem->cvUseLocalMap.wait(lock);
		}
		mpSystem->mbTrackingEnd = false;*/

		std::cout << "tracker::start::" <<pCurr->GetFrameID()<< std::endl;
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
		//mnPointMatching = mpMatcher->OpticalMatchingForTracking(mpRefKF, pCurr, vpTempMPs, vpTempPts, vbTempInliers, vnIDXs, overlap, debugImg); //pCurr

		////////////Dense Matching Test
		mpRefKF->mpMatchInfo->mvFlows.push_back(mFlow.clone());
		////////////Dense Matching Test
		
		//////Optical flow
		std::chrono::high_resolution_clock::time_point matching_start = std::chrono::high_resolution_clock::now();
		std::vector<cv::Mat> currPyr, prevPyr;
		std::vector<uchar> status1, status2;
		std::vector<float> err1, err2;
		//cv::Mat pprevImg = pPPrevKF->GetOriginalImage();
		cv::Mat prevImg = mpRefKF->GetOriginalImage();
		cv::Mat currImg = pCurr->GetOriginalImage();
											  ///////debug
		cv::Point2f ptBottom1 = cv::Point2f(0, prevImg.rows);
		cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
		cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
		testMatchingImg = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
		//pprevImg.copyTo(debugging(mergeRect1));
		prevImg.copyTo(testMatchingImg(mergeRect1));
		currImg.copyTo(testMatchingImg(mergeRect2));
		/////////////
		
		auto mpTempMPs = mpRefKF->mpMatchInfo->GetMatchingMPs();
		auto mpTempPts = mpRefKF->mpMatchInfo->GetMatchingPts();
		int nDense = 0;
		int nCurrFrameID = pCurr->GetFrameID();
		int nLastFlow = mpRefKF->mpMatchInfo->mvFlows.size()-1;

		std::chrono::high_resolution_clock::time_point dense_matching_start = std::chrono::high_resolution_clock::now();
		auto mvFlows = mpMap->GetFlows(mpRefKF->GetFrameID(), pCurr->GetFrameID());
		std::vector<cv::Point2f> mvTempPts2;
		std::vector<bool> vbTempIinliers2;
		mpMatcher->DenseOpticalMatching(pCurr, mpTempPts, mvTempPts2, vbTempIinliers2, mvFlows);
		for (int i = 0; i < mpTempPts.size(); i++) {
			if (!vbTempIinliers2[i])
				continue;
			auto pMPi = mpTempMPs[i];
			if (pMPi && !pMPi->isDeleted() && pMPi->GetRecentTrackingFrameID() != nCurrFrameID && pCurr->isInFrustum(pMPi, 0.6f)) {
				pMPi->SetRecentTrackingFrameID(nCurrFrameID);
				vpTempMPs.push_back(pMPi);
				vnIDXs.push_back(i);
				vpTempPts.push_back(mvTempPts2[i]);
				vbTempInliers.push_back(true);
				circle(testMatchingImg, mpTempPts[i], 2, cv::Scalar(255, 255, 0), -1);
				circle(testMatchingImg, mvTempPts2[i]+ptBottom1, 2, cv::Scalar(255, 255, 0), -1);
			}
		}
		//for (int f = 0; f < mvFlows.size(); f++) {
		//	for (int i = 0; i < mpTempMPs.size(); i++) {
		//		auto prevPt = mpTempPts[i];
		//		cv::Point2f idxPt(prevPt.x / 2, prevPt.y / 2);
		//		cv::Vec2f fval = mvFlows[f].at<Vec2f>(idxPt);
		//		cv::Point2f currPt = cv::Point2f(fval.val[0] * 2, fval.val[1] * 2);
		//		if (!pCurr->isInImage(currPt.x, currPt.y, 10)) {
		//			//cv::circle(testMatchingImg, mpTempPts[i], 2, cv::Scalar(0, 255, 0), -1);
		//			//mpRefKF->mpMatchInfo->mvbTempDense[i] = false;
		//			continue;
		//		}
		//		mpTempPts[i] = currPt;
		//		if (f == nLastFlow) {
		//			auto pMPi = mpTempMPs[i];
		//			if (pMPi && !pMPi->isDeleted() && pMPi->GetRecentTrackingFrameID() != nCurrFrameID && pCurr->isInFrustum(pMPi, 0.6f)) {
		//				pMPi->SetRecentTrackingFrameID(nCurrFrameID);
		//				vpTempMPs.push_back(pMPi);
		//				vnIDXs.push_back(i);
		//				vpTempPts.push_back(currPt);
		//				vbTempInliers.push_back(true);
		//			}
		//		}
		//	}
		//}
		/*cv::Mat testImg = cv::Mat::zeros(mnHeight/2, mnWidth/2, CV_32FC2);
		for (int x = 0; x < mnWidth/2; x++) {
			for (int y = 0; y < mnHeight/2; y++) {
				cv::Vec2f pt(x, y);
				testImg.at<Vec2f>(y, x) = pt;
			}
		}
		testImg += mFlow;*/

		std::chrono::high_resolution_clock::time_point dense_matching_end = std::chrono::high_resolution_clock::now();
		double t_dense = std::chrono::duration_cast<std::chrono::milliseconds>(dense_matching_end - dense_matching_start).count() / 1000.0;

		std::stringstream ssaas;
		ssaas << "Dense Tracking= " << pCurr->GetFrameID() << ", " << mpRefKF->GetFrameID() << "::" << nDense << "::" << dtime<<"::"<< t_dense;
		cv::rectangle(testMatchingImg, cv::Point2f(0, 0), cv::Point2f(testMatchingImg.cols, 30), cv::Scalar::all(0), -1);
		cv::putText(testMatchingImg, ssaas.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
		
		////////////Dense Matching Test
		//mpSystem->mbTrackingEnd = true;
		std::chrono::high_resolution_clock::time_point tracking_a = std::chrono::high_resolution_clock::now();

		//graph-based0.
		mnMapPointMatching = Optimization::PoseOptimization(pCurr, vpTempMPs, vpTempPts, vbTempInliers, vnMPIDXs);
		
		//pCurr->SetInliers(mnMatching); //이용X
		int nMP = 0;
		

		////파일 이미지 저장시 이름 설정
		std::stringstream ssdira;
		ssdira << mpSystem->GetDirPath(0) << "/kfmatching/tracking_" << mpRefKF->GetKeyFrameID()<<"_"<<mpRefKF->GetFrameID() << "_" << pCurr->GetFrameID() <<"_"<<mpRefKF->mpMatchInfo->mnMaxMatch<< ".jpg";
		std::stringstream suc;
		/*suc << "Tracking::" << mnPointMatching << ", " << mnMapPointMatching << "::" << mpRefKF->mpMatchInfo->GetMatchingSize();
		cv::rectangle(debugImg, cv::Point2f(0, 0), cv::Point2f(debugImg.cols, 30), cv::Scalar::all(0), -1);
		cv::putText(debugImg, suc.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));*/
		////파일 이미지 저장시 이름 설정

		///////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////키프레임 체크
		auto pNewKF = CheckNeedKeyFrame(pCurr, pPrev);
		if (pNewKF) {
			//auto pNewKF = pCurr;
			UpdateMatchingInfo(mpRefKF, pCurr, vpTempMPs, vpTempPts, vbTempInliers, vnIDXs, vnMPIDXs);
			pNewKF->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
			mpRefKF = pNewKF;
			//mpRefKF->mpMatchInfo->mvTempDenseMatchPts = mpRefKF->mpMatchInfo->GetMatchingPts();
			//mpRefKF->mpMatchInfo->mvbTempDense = std::vector<bool>(mpRefKF->mpMatchInfo->mvTempDenseMatchPts.size(), true);
			mpLocalMapper->InsertKeyFrame(pNewKF);
			//mpSegmentator->InsertKeyFrame(pNewKF);
			//mpPlaneEstimator->InsertKeyFrame(pNewKF);
		}

		//ref, curr, matching mp, matching pt, bool

		std::cout << "tracker::end::"<<pCurr->GetFrameID()<<"::"<<mnMapPointMatching << std::endl;

		////////Visualization & 시간 계산
		std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
		
		auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_a - tracking_start).count();
		double t1 = duration1 / 1000.0;
		auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
		double t2 = duration2 / 1000.0;

		///////시각화
		if (!mpFrameVisualizer->isVisualize()) {
			mpFrameVisualizer->SetFrameMatchingInformation(mpRefKF, pCurr, vpTempMPs, vpTempPts, vbTempInliers, t2);
		}

		////////시각화1
		//cv::Mat vis = pCurr->GetOriginalImage();
		////vis.convertTo(vis, CV_8UC3);
		//cv::Mat R = pCurr->GetRotation();
		//cv::Mat t = pCurr->GetTranslation();
		//
		////////시각화2
		//cv::Point2f ptBottom(0, mnHeight);
		//for (int i = 0; i < vpTempMPs.size(); i++) {
		//	UVR_SLAM::MapPoint* pMPi = vpTempMPs[i];
		//	if (!pMPi || pMPi->isDeleted())
		//		continue;
		//	/*if (!vbTempInliers[vnMPIDXs[i]])
		//		continue;*/
		//	cv::Point2f p2D;
		//	cv::Mat pCam;
		//	bool b = pMPi->Projection(p2D, pCam, R, t, mK, mnWidth, mnHeight);

		//	int label = 0;// mpRefKF->mpMatchInfo->mvObjectLabels[vnIDXs[i]];
		//	int pid = pMPi->GetPlaneID();
		//	int type = pMPi->GetRecentLayoutFrameID();
		//	cv::Scalar color(150, 150, 0);
		//	if (pid > 0 && label == 150) {
		//		color = cv::Scalar(0, 0, 255);
		//	}
		//	else if (pid > 0 && label == 100) {
		//		color = cv::Scalar(0, 255, 0);
		//	}
		//	//else if (pid > 0 && label == 255) {
		//	else if (label == 255) {
		//		color = cv::Scalar(255, 0, 0);
		//		//color = UVR_SLAM::ObjectColors::mvObjectLabelColors[pid];
		//	}
		//	if (pid <= 0)
		//		color /= 2;
		//	if(vbTempInliers[i])
		//		cv::circle(vis, p2D, 2, color, -1);
		//	/*if (!b || !vbTempInliers[vnMPIDXs[i]]) {
		//		cv::line(vis, p2D, vpTempPts[vnMPIDXs[i]], cv::Scalar(0, 0, 255), 1);
		//	}
		//	else {
		//		cv::line(vis, p2D, vpTempPts[vnMPIDXs[i]], cv::Scalar(255, 255, 0), 1);
		//	}*/
		//	cv::line(vis, p2D, vpTempPts[i], color, 1);
		//	if(!vbTempInliers[i]){
		//		cv::line(testMatchingImg, p2D+ptBottom, vpTempPts[i]+ ptBottom, cv::Scalar(0,0,255), 1);
		//		cv::circle(testMatchingImg, p2D+ ptBottom, 2, cv::Scalar(0, 0, 255), -1);
		//	}
		//	else {
		//		cv::line(testMatchingImg, p2D + ptBottom, vpTempPts[i] + ptBottom, cv::Scalar(0, 255, 255), 1);
		//		cv::circle(testMatchingImg, p2D + ptBottom, 2, cv::Scalar(0, 255, 255), -1);
		//	}
		//	//cv::circle(vis, p2D, 2, cv::Scalar(255, 0, 0), -1);
		//}
		//////////시각화2
		//std::stringstream ss;
		//ss << "Traking = " << mnMapPointMatching << ", " << mnMapPointMatching << "::" << t1 << ", " << t2 << "::";
		//cv::rectangle(vis, cv::Point2f(0, 0), cv::Point2f(vis.cols, 30), cv::Scalar::all(0), -1);
		//cv::putText(vis, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
		///*cv::rectangle(debugImg, cv::Point2f(0, 0), cv::Point2f(vis.cols, 30), cv::Scalar::all(0), -1);
		//cv::putText(debugImg, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));*/
		//cv::imshow("Output::Tracking", vis);
		//////////Visualization & 시간 계산
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		///////테스트 이미지 저장
		//std::stringstream sstdir;
		//sstdir << mpSystem->GetDirPath(0) << "/tracking/tracking_test_" << pCurr->GetFrameID() << "_" << pPrev->GetFrameID() << ".jpg";
		///*cv::Mat ttttddddebug = cv::Mat::zeros(testMatchingImg.rows, testMatchingImg.cols * 2, testMatchingImg.type());
		//debugImg.copyTo(ttttddddebug(cv::Rect(0, 0, testMatchingImg.cols, testMatchingImg.rows)));
		//testMatchingImg.copyTo(ttttddddebug(cv::Rect(testMatchingImg.cols, 0, testMatchingImg.cols, testMatchingImg.rows)));
		//*/
		//cv::imwrite(sstdir.str(), testMatchingImg);

		/////////트래킹 결과 이미지 저장
		pCurr->mpMatchInfo->mMatchedImage = testMatchingImg.clone();
		
		/////////트래킹 결과 이미지 저장
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

////MP와 PT가 대응함.
////pPrev가 mpRefKF가 됨
int UVR_SLAM::Tracker::UpdateMatchingInfo(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<cv::Point2f> vpPts, std::vector<bool> vbInliers, std::vector<int> vnIDXs, std::vector<int> vnMPIDXs) {
	auto pMatchInfo = pCurr->mpMatchInfo;
	auto pPrevMatchInfo = pPrev->mpMatchInfo;
	
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
	
	int nres = 0;
	for (int i = 0; i < vpPts.size(); i++) {
		int prevIdx = vnIDXs[i];
		//mpRefKF->mpMatchInfo->mvbTempDense[prevIdx] = vbInliers[i];
		if (!vbInliers[i]){
			continue;
		}
		auto pt = vpPts[i];
		auto pMP = vpMPs[i];
		if (!pMatchInfo->CheckPt(pt)) {
			//circle(pMatchInfo->used, pt, 2, cv::Scalar(255), -1);
			pMatchInfo->AddMatchingPt(pt, pMP, prevIdx);
			/*pMatchInfo->AddMatchingPt(pt, pMP, pPrevMatchInfo->mvnMatchingPtIDXs[vnIDXs[i]], mpRefKF->mpMatchInfo->mvObjectLabels[pPrev->mpMatchInfo->mvnMatchingPtIDXs[vnIDXs[i]]]
				, mpRefKF->mpMatchInfo->mvnOctaves[pPrev->mpMatchInfo->mvnMatchingPtIDXs[vnIDXs[i]]]);*/
			if (pMP) {
				nres++;
			}
		}
		//dense matching에서 추가
		
	}
	pMatchInfo->mnNumMatch = mnMapPointMatching;
	return nInstantMP;
}