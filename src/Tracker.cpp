#include <Tracker.h>
#include <System.h>
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
UVR_SLAM::Tracker::Tracker(std::string strPath) : mbInitializing(false), mbFirstFrameAfterInit(false), mbInitilized(false) {
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

	fs.release();
}
UVR_SLAM::Tracker::~Tracker() {}

bool UVR_SLAM::Tracker::isInitialized() {
	return mbInitilized;
}

void UVR_SLAM::Tracker::SetSystem(System* pSystem) {
	mpSystem = pSystem;
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
	if (mpFrameWindow->size() <= 2)
		nMinObs = 2;
	//int nRefMatches = mpFrameWindow->TrackedMapPoints(nMinObs);
	int nRefMatches = mpRefKF->TrackedMapPoints(nMinObs);
	
	float thRefRatio = 0.9f;

	bool bLocalMappingIdle = !mpLocalMapper->isDoingProcess();
	//std::cout << "CheckNeedKeyFrame::Ref = " << nRefMatches <<", "<< mnMatching <<", "<<mpFrameWindow->mnLastMatches<<":: "<<bLocalMappingIdle<< std::endl;
	
	int nLastID = mpFrameWindow->GetLastFrameID();
	bool c1a = pCurr->GetFrameID() >= nLastID + mnMaxFrames;
	bool c1b = pCurr->GetFrameID() >= nLastID + mnMinFrames && bLocalMappingIdle;
	bool c2 = mnMatching < nRefMatches*thRefRatio && mnMatching > 20;

	if ((c1a || c1b) && c2) {
		/*if (bLocalMappingIdle)
			return true;
		else {

			return false;
		}*/
		//KF 동작 중 멈추는 과정이 필요함.
		//local mapping이 동작하는 와중에도 KF가 추가됨.
		return true;
	}
	else
		return false;

	return false;
}

void UVR_SLAM::Tracker::Tracking(Frame* pPrev, Frame* pCurr) {
	if(!mbInitializing){
		mbInitializing = mpInitializer->Initialize(pCurr, mnWidth, mnHeight);
		//mbInit = bInit;
		mbFirstFrameAfterInit = mbInitializing;
		if (mbInitializing)
			mpRefKF = pCurr;
	}
	else {
		
		std::unique_lock<std::mutex>(mpSystem->mMutexTracking);
		std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
		////시맨틱 프레임과 매칭
		//int nLastFrameIndex = mpFrameWindow->GetLastSemanticFrameIndex();
		//if (nLastFrameIndex >= 0) {
		//	UVR_SLAM::Frame* mpSemanticFrame = mpFrameWindow->GetFrame(nLastFrameIndex);
		//	mpMatcher->FeatureMatchingWithSemanticFrames(mpSemanticFrame, pCurr);
		//}
			
		//Local Map 자료 얻기
		//local map points, descriptors, matching information, inlier vector
		std::cout << "track::1" << std::endl;
		auto mvpLocalMPs = mpFrameWindow->GetLocalMap();
		cv::Mat mLocalMapDesc = mpFrameWindow->GetLocalMapDescriptor();
		mpFrameWindow->mvMatchInfos = std::vector<cv::DMatch>(mpFrameWindow->GetLocalMapSize());
		std::vector<bool> mvbLocalMapInliers(mvpLocalMPs.size(), false);

		//mpMatcher->FeatureMatchingForInitialPoseTracking(mpFrameWindow, pFrame);
		std::vector<cv::DMatch> vMatchInfos;
		
		//std::cout << "Matching Init : " << nInitMatching << std::endl;
		std::cout << "track::2" << std::endl;
		std::chrono::high_resolution_clock::time_point matching_start = std::chrono::high_resolution_clock::now();
		int nInitMatching = mpMatcher->FeatureMatchingForInitialPoseTracking(pPrev, pCurr, mpFrameWindow, mvpLocalMPs, mLocalMapDesc, mvbLocalMapInliers, vMatchInfos);
		//int nInitMatching = mpMatcher->FeatureMatchingForInitialPoseTracking(mpFrameWindow, pCurr);
		std::chrono::high_resolution_clock::time_point matching_end = std::chrono::high_resolution_clock::now();
		std::cout << "track::3" << std::endl;
		std::chrono::high_resolution_clock::time_point optimize_start = std::chrono::high_resolution_clock::now();
		mnMatching = Optimization::PoseOptimization(mpFrameWindow, pCurr, mvpLocalMPs, mvbLocalMapInliers,false,4,10);
		std::chrono::high_resolution_clock::time_point optimize_end = std::chrono::high_resolution_clock::now();
		std::cout << "track::4" << std::endl;
		std::chrono::high_resolution_clock::time_point matching_start2 = std::chrono::high_resolution_clock::now();
		mpMatcher->FeatureMatchingForPoseTrackingByProjection(mpFrameWindow, pCurr, mvpLocalMPs, mLocalMapDesc,mvbLocalMapInliers, 5.0);
		//mpMatcher->FeatureMatchingForInitialPoseTracking(mpFrameWindow, pCurr, mvbLocalMapInliers);
		std::chrono::high_resolution_clock::time_point matching_end2 = std::chrono::high_resolution_clock::now();
		std::cout << "track::5" << std::endl;
		std::chrono::high_resolution_clock::time_point optimize_start2 = std::chrono::high_resolution_clock::now();
		mnMatching = Optimization::PoseOptimization(mpFrameWindow, pCurr, mvpLocalMPs, mvbLocalMapInliers, false, 4, 10);
		std::chrono::high_resolution_clock::time_point optimize_end2 = std::chrono::high_resolution_clock::now();
		std::cout << "track::6" << std::endl;
		//std::cout << "Track::res::"<< nInitMatching <<", "<< mnMatching<<"::"<<mpFrameWindow->GetLocalMapSize() << std::endl;
		pCurr->SetInliers(mnMatching);
		
		//슬램 초기화 최종 판단
		if (mbFirstFrameAfterInit) {
			mbFirstFrameAfterInit = false;
			if (mnMatching < 80) {

				//매칭 실패
				mbInitializing = false;
				mpSystem->Reset();
				std::cout << "Fail Initilization = " << mnMatching << std::endl;
				return;
			}
			else {
				mpSystem->SetBoolInit(true);
				mbInitilized = true;

				if (!mpSegmentator->isDoingProcess()) {
					/*UVR_SLAM::Frame* pFirst = mpFrameWindow->GetFrame(0);
					UVR_SLAM::Frame* pSecond = mpFrameWindow->GetFrame(1);
					mpRefKF = pSecond;*/

					/*pFirst->TurnOnFlag(UVR_SLAM::FLAG_SEGMENTED_FRAME);
					mpSegmentator->SetBoolDoingProcess(true);
					mpSegmentator->SetTargetFrame(pFirst);*/
				}

			}
		}
		std::cout << "track::7" << std::endl;
		std::chrono::high_resolution_clock::time_point count_start = std::chrono::high_resolution_clock::now();
		CalcMatchingCount(pCurr);
		std::chrono::high_resolution_clock::time_point count_end = std::chrono::high_resolution_clock::now();
		std::cout << "track::8" << std::endl;
		std::chrono::high_resolution_clock::time_point check_start = std::chrono::high_resolution_clock::now();
		if (CheckNeedKeyFrame(pCurr)) {
			mpRefKF = pCurr;
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
		mpFrameWindow->SetLocalMapInliers(mvbLocalMapInliers);

		/*if (mpFrameWindow->GetLastFrameID() + 10 < pCurr->GetFrameID() || (nMatching < 50)) {
			mpLocalMapper->InsertKeyFrame(pCurr);
		}*/
		/*else {
			if (!mpPlaneEstimator->isDoingProcess()) {
				mpPlaneEstimator->SetBoolDoingProcess(true, 1);
				mpPlaneEstimator->SetTargetFrame(pCurr);
			}
		}*/

		//일단 테스트
		std::cout << "track::9" << std::endl;
		cv::Mat vis = pCurr->GetOriginalImage();
		//cvtColor(vis, vis, CV_RGBA2BGR);
		vis.convertTo(vis, CV_8UC3);

		auto mvpMPs = pCurr->GetMapPoints();
		cv::Mat R = pCurr->GetRotation();
		cv::Mat t = pCurr->GetTranslation();
		for (int i = 0; i < mvpMPs.size(); i++) {
			UVR_SLAM::MapPoint* pMP = mvpMPs[i];
			if (!pMP){
				cv::circle(vis, pCurr->mvKeyPoints[i].pt, 1, cv::Scalar(0, 0, 255), -1);
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
				cv::circle(vis, pCurr->mvKeyPoints[i].pt, 2, cv::Scalar(255, 0, 255), -1);
				UVR_SLAM::ObjectType type = pMP->GetObjectType();
				cv::line(vis, p2D, pCurr->mvKeyPoints[i].pt, cv::Scalar(255, 255, 0), 2);
				if (type != OBJECT_NONE)
					circle(vis, p2D, 3, UVR_SLAM::ObjectColors::mvObjectLabelColors[type], -1);
				if (pMP->GetPlaneID() > 0) {
					circle(vis, p2D, 4, cv::Scalar(255, 0, 255), -1);
				}
			}
		}
		
		std::cout << "track::10" << std::endl;
		//속도 및 에러 출력
		std::stringstream ss;
		ss << std::setw(5) << "TIME : " << tttt << " Matching : " << mnMatching<<" Local Map : "<<mvpLocalMPs.size();
		cv::rectangle(vis, cv::Point2f(0, 0), cv::Point2f(vis.cols, 30*3), cv::Scalar::all(0), -1);
		
		cv::putText(vis, ss.str(), cv::Point2f(0, 20), 2, 0.6,cv::Scalar::all(255));
		ss.str("");
		ss << "Matching : " << t_matching << ", " << t_matching2;
		cv::putText(vis, ss.str(), cv::Point2f(0, 50), 2, 0.6, cv::Scalar::all(255));

		ss.str("");
		ss << "Optimize : " << t_optimize << ", " << t_optimize2<<"| "<<t_count<<", "<<t_check;
		cv::putText(vis, ss.str(), cv::Point2f(0, 80), 2, 0.6, cv::Scalar::all(255));

		cv::imshow("Output::Tracking", vis);

		//시각화 과정
		//std::stringstream ss;
		//ss << "../../bin/segmentation/res/img/img_" << pCurr->GetFrameID() << ".jpg";
		//cv::imwrite(ss.str(), vis);
		//cv::imwrite("../../bin/segmentation/res/tracking.jpg", vis);

		////시각화
		//cv::Mat vis2 = pCurr->GetOriginalImage();
		//for (int i = 0; i < pCurr->mvKeyPoints.size(); i++) {
		//	UVR_SLAM::ObjectType type = pCurr->GetObjectType(i);
		//	if (type != OBJECT_NONE)
		//		circle(vis2, pCurr->mvKeyPoints[i].pt, 2, ObjectColors::mvObjectLabelColors[type], -1);
		//}
		//cv::imshow("Output::Matching::SemanticFrame", vis2);
		//cv::imwrite("../../bin/segmentation/res/labeling.jpg", vis2);
		cv::waitKey(1);
		std::cout << "track::11" << std::endl;
		//visualizer thread
		if (!mpVisualizer->isDoingProcess()) {
			//mpSystem->SetVisualizeFrame(pCurr);
			mpVisualizer->SetBoolDoingProcess(true);
			mpVisualizer->SetFrameMatching(pPrev, pCurr, vMatchInfos);
		}
		std::cout << "track::12" << std::endl;
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
		if(pF->mvbMPInliers[i]){
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
}