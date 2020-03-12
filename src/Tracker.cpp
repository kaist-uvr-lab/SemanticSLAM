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
	mnMinFrames = fps / 3;

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

	bool bLocalMappingIdle = !mpLocalMapper->isDoingProcess();
	
	//기존 방식대로 최소 프레임을 만족하면 무조건 추가.
	//트래킹 수가 줄어들면 바로 추가.
	int nLastID = mpFrameWindow->GetLastFrameID();
	bool c1 = pCurr->GetFrameID() >= nLastID + mnMinFrames; //최소한의 조건
	bool c2 = mnMatching < 100;
	bool c3 = false;//mnMatching < mpFrameWindow->mnLastMatches*0.8;
	if (c1 || c2 || c3) {
		if (!bLocalMappingIdle)
		{
			mpLocalMapper->StopLocalMapping(true);
			return false;
		}
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

void UVR_SLAM::Tracker::Tracking(Frame* pPrev, Frame* pCurr) {
	if(!mbInitializing){
		bool bReset = false;
		mbInitializing = mpInitializer->Initialize(pCurr, bReset, mnWidth, mnHeight);
		
		if (bReset)
			mpSystem->Reset();

		//mbInit = bInit;
		mbFirstFrameAfterInit = false;
		
		if (mbInitializing){
			mpRefKF = pCurr;
			mbInitilized = true;
			mpSystem->SetBoolInit(true);
		}
		
	}
	else {
		std::unique_lock<std::mutex> lock(mpSystem->mMutexUseLocalMap);
		while (!mpSystem->mbLocalMapUpdateEnd){
			mpSystem->cvUseLocalMap.wait(lock);
		}
		mpSystem->mbTrackingEnd = false;
		std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
		
		//pCurr->SetPose(pPrev->GetRotation(), pPrev->GetTranslation());
		pCurr->SetPose(mpFrameWindow->GetRotation(), mpFrameWindow->GetTranslation());

		int nLocalMapID = mpFrameWindow->GetLastFrameID();
		auto mvpLocalMPs = mpFrameWindow->GetLocalMap();
		cv::Mat mLocalMapDesc = mpFrameWindow->GetLocalMapDescriptor();
		std::vector<cv::DMatch> mvMatchInfo;
		mpSystem->mbTrackingEnd = true;
		lock.unlock();
		mpSystem->cvUseLocalMap.notify_one();

		//이전 프레임에 포함된 맵포인트와 현재 프레임의 키포인트를 매칭하는 과정.
		//빠른 속도를 위해 이전 프레임에서 추적되는 맵포인트를 디스크립터로 만듬.
		std::chrono::high_resolution_clock::time_point matching_start = std::chrono::high_resolution_clock::now();
		int nInitMatching = mpMatcher->MatchingWithPrevFrame(pPrev, pCurr, mvMatchInfo);
		int nLabelMatch = 0;
		//if (pPrev->mPlaneDescriptor.rows == 0 && mpRefKF && mpRefKF->mPlaneDescriptor.rows > 0) {
		/*if(mpRefKF && mpRefKF->mPlaneDescriptor.rows > 0){
			nLabelMatch = mpMatcher->MatchingWithLabeling(pPrev, pCurr);
		}*/

		//int nInitMatching = mpMatcher->FeatureMatchingForInitialPoseTracking(mpFrameWindow, pCurr);
		std::chrono::high_resolution_clock::time_point matching_end = std::chrono::high_resolution_clock::now();
		std::chrono::high_resolution_clock::time_point optimize_start = std::chrono::high_resolution_clock::now();
		//mnMatching = Optimization::PoseOptimization(mpFrameWindow, pCurr, mvpLocalMPs, mvbLocalMapInliers,false,4,10);
		//mnMatching = Optimization::PoseOptimization(pCurr, mvPairMatchingInfos, false, 4, 10); //이게 최근 버전
		mnMatching = Optimization::PoseOptimization(pCurr);
		std::chrono::high_resolution_clock::time_point optimize_end = std::chrono::high_resolution_clock::now();
		
		std::chrono::high_resolution_clock::time_point matching_start2 = std::chrono::high_resolution_clock::now();
		mpMatcher->MatchingWithLocalMap(pCurr, mvpLocalMPs, mLocalMapDesc, 5.0);
		//mpMatcher->FeatureMatchingForInitialPoseTracking(mpFrameWindow, pCurr, mvbLocalMapInliers);
		std::chrono::high_resolution_clock::time_point matching_end2 = std::chrono::high_resolution_clock::now();
		std::chrono::high_resolution_clock::time_point optimize_start2 = std::chrono::high_resolution_clock::now();
		//mnMatching = Optimization::PoseOptimization(mpFrameWindow, pCurr, mvpLocalMPs, mvbLocalMapInliers, false, 4, 10);
		//mnMatching = Optimization::PoseOptimization(pCurr, mvPairMatchingInfos, false, 4, 10);
		mnMatching = Optimization::PoseOptimization(pCurr);
		std::chrono::high_resolution_clock::time_point optimize_end2 = std::chrono::high_resolution_clock::now();
		//std::cout << "Track::res::"<< nInitMatching <<", "<< mnMatching<<"::"<<mpFrameWindow->GetLocalMapSize() << std::endl;
		pCurr->SetInliers(mnMatching);
		mpFrameWindow->SetPose(pCurr->GetRotation(), pCurr->GetTranslation());
		
		std::chrono::high_resolution_clock::time_point count_start = std::chrono::high_resolution_clock::now();
		CalcMatchingCount(pCurr);

		std::chrono::high_resolution_clock::time_point count_end = std::chrono::high_resolution_clock::now();
		
		std::chrono::high_resolution_clock::time_point check_start = std::chrono::high_resolution_clock::now();

		//std::cout << "cam : " << pCurr->GetCameraCenter().t() << std::endl;

		if (CheckNeedKeyFrame(pCurr)) {
			mpRefKF = pCurr;
			pCurr->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
			mpSegmentator->InsertKeyFrame(pCurr);
			mpLocalMapper->InsertKeyFrame(pCurr);
		}
		std::chrono::high_resolution_clock::time_point check_end = std::chrono::high_resolution_clock::now();

		//시간 체크
		std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end-tracking_start).count();

		auto duration_matching = std::chrono::duration_cast<std::chrono::milliseconds>(matching_end - matching_start).count();
		auto duration_optimize = std::chrono::duration_cast<std::chrono::milliseconds>(optimize_end - optimize_start).count();

		double tttt = duration / 1000.0;
		//tttt = 1.0 /tttt;

		double t_matching = duration_matching / 1000.0;
		double t_optimize = duration_optimize / 1000.0;
		//bool bBow =  mpFrameWindow->CalcFrameDistanceWithBOW(pCurr);

		auto duration_matching2 = std::chrono::duration_cast<std::chrono::milliseconds>(matching_end2 - matching_start2).count();
		double t_matching2 = duration_matching2 / 1000.0;

		auto duration_optimize2 = std::chrono::duration_cast<std::chrono::milliseconds>(optimize_end2 - optimize_start2).count();
		double t_optimize2 = duration_optimize2 / 1000.0;

		auto duration_count = std::chrono::duration_cast<std::chrono::milliseconds>(count_end - count_start).count();
		double t_count = duration_count / 1000.0;

		auto duration_check = std::chrono::duration_cast<std::chrono::milliseconds>(check_end - check_start).count();
		double t_check = duration_check / 1000.0;

		//update tracking results
		mpFrameWindow->mnLastMatches = mnMatching;

		//일단 테스트
		cv::Mat vis = pCurr->GetOriginalImage();
		//cvtColor(vis, vis, CV_RGBA2BGR);
		vis.convertTo(vis, CV_8UC3);

		auto mvpMPs = pCurr->GetMapPoints();
		//auto mvpOPs = pCurr->GetObjectVector();
		cv::Mat R = pCurr->GetRotation();
		cv::Mat t = pCurr->GetTranslation();
		for (int i = 0; i < mvpMPs.size(); i++) {
			UVR_SLAM::MapPoint* pMP = mvpMPs[i];
			
			if (!pMP){
				//cv::circle(vis, pCurr->mvKeyPoints[i].pt, 1, cv::Scalar(0, 0, 255), -1);
				continue;
			}
			if (pMP->isDeleted()) {
				pCurr->mvbMPInliers[i] = false;
				continue;
			}
			cv::Point2f p2D;
			cv::Mat pCam;
			pMP->Projection(p2D, pCam, R, t, mK, mnWidth, mnHeight);

			if (!pCurr->mvbMPInliers[i]){
				//if (pMP->GetPlaneID() > 0) {
				//	//circle(vis, p2D, 4, cv::Scalar(255, 0, 255), 2);
				//}
			}
			else {
				int nObservations = pMP->GetConnedtedFrames().size();
				/*if (nObservations > 5)
					cv::circle(vis, pCurr->mvKeyPoints[i].pt, 2, cv::Scalar(0, 0, 255), -1);
				else if(nObservations > 3)
					cv::circle(vis, pCurr->mvKeyPoints[i].pt, 2, cv::Scalar(255, 0, 0), -1);
				else
					cv::circle(vis, pCurr->mvKeyPoints[i].pt, 2, cv::Scalar(255, 0, 255), -1);*/
				UVR_SLAM::ObjectType type = pMP->GetObjectType();
				cv::line(vis, p2D, pCurr->mvKeyPoints[i].pt, cv::Scalar(255, 255, 0), 2);
				if (type != OBJECT_NONE)
					circle(vis, p2D, 3, UVR_SLAM::ObjectColors::mvObjectLabelColors[type], -1);
				if (pMP->GetMapPointType() == MapPointType::PLANE_MP) {
					circle(vis, p2D, 2, cv::Scalar(255, 0, 255), -1);
				}
			}
		}
		
		//////////////////////////////////////////////////////////////////////////////
		//////////////line test
		/*auto lines = mpRefKF->Getlines();
		bool bLine = lines.size() > 0;
		bool bPlane = mpRefKF->mvpPlanes.size() > 0;
		if (bLine && bPlane) {
			
			auto plane = mpRefKF->mvpPlanes[0];
			cv::Mat normal1;
			float dist1;
			plane->GetParam(normal1, dist1);
			cv::Mat K = mK.clone();
			K.at<float>(0, 0) /= 2.0;
			K.at<float>(1, 1) /= 2.0;
			K.at<float>(0, 2) /= 2.0;
			K.at<float>(1, 2) /= 2.0;

			cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
			R.copyTo(T.rowRange(0, 3).colRange(0, 3));
			t.copyTo(T.col(3).rowRange(0, 3));

			cv::Mat planeParam = plane->matPlaneParam.clone();
			cv::Mat invT = T.inv();
			cv::Mat invP = invT.t()*planeParam;
			cv::Mat invK = K.inv();

			cv::Mat R, t;
			pCurr->GetPose(R, t);
			
			for (int i = 0; i < lines.size(); i++) {
				cv::Mat param = UVR_SLAM::PlaneInformation::PlaneWallEstimator(lines[i], normal1, invP, invT, invK);
				float m;
				cv::Mat mLine = UVR_SLAM::PlaneInformation::FlukerLineProjection(planeParam, param, R, t, mK2, m);
				cv::Point2f sPt, ePt;
				UVR_SLAM::PlaneInformation::CalcFlukerLinePoints(sPt, ePt, 0.0, mnHeight, mLine);
				cv::line(vis, sPt, ePt, cv::Scalar(0, 255, 0), 3);
				std::cout << "tttt::" << param.t() << std::endl;
			}
		}*/
		//////////////line test

		////////////////////////////////////////////////////////////////////////
		//////////////Wall Line TEST
		auto wallParams = mpMap->GetWallPlanes();//mpRefKF->GetWallParams();
		if (wallParams.size() > 0 && mpRefKF->mvpPlanes.size() > 0) {
			auto plane = mpRefKF->mvpPlanes[0];
			cv::Mat planeParam = plane->matPlaneParam.clone();
			for (int i = 0; i < wallParams.size(); i++) {
				float m;
				cv::Mat mLine = UVR_SLAM::PlaneInformation::FlukerLineProjection(wallParams[i]->GetParam(), planeParam, R, t, mK2, m);
				cv::Point2f sPt, ePt;
				UVR_SLAM::PlaneInformation::CalcFlukerLinePoints(sPt, ePt, 0.0, mnHeight, mLine);
				cv::line(vis, sPt, ePt, cv::Scalar(0, 255, 0), 3);
				//std::cout << "tttt::" << wallParams[i].t() << std::endl;
			}
		}
		//////////////Wall Line TEST
		////////////////////////////////////////////////////////////////////////
		

		//속도 및 에러 출력
		/*std::stringstream ss;
		ss << std::setw(5) << "Tracker TIME : " << tttt << " || " << mnMatching<<" Local Map : "<<mvpLocalMPs.size();
		mpSystem->SetTrackerString(ss.str());*/
		
		cv::imshow("Output::Tracking", vis);
		
		mpVisualizer->SetMPs(pCurr->GetMapPoints());
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

			//floor, wall descriptor update
			if (pF->mLabelStatus.at<uchar>(i) == 0) {
				auto type = pMP->GetObjectType();
				switch (type) {
				case ObjectType::OBJECT_FLOOR:
					pF->mLabelStatus.at<uchar>(i) = (int)ObjectType::OBJECT_FLOOR;
					pF->mPlaneDescriptor.push_back(pF->matDescriptor.row(i));
					pF->mPlaneIdxs.push_back(i);
					break;
				case ObjectType::OBJECT_WALL:
					pF->mLabelStatus.at<uchar>(i) = (int)ObjectType::OBJECT_WALL;
					pF->mWallDescriptor.push_back(pF->matDescriptor.row(i));
					pF->mWallIdxs.push_back(i);
					break;
				}
			}
			
		}
		else {
			pF->mNotTrackedDescriptor.push_back(pF->matDescriptor.row(i));
			pF->mvNotTrackedIdxs.push_back(i);
			pF->mvpMPs[i] = nullptr;
		}
	}
}