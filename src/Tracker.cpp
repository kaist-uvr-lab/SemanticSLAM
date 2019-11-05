#include <Tracker.h>
#include <System.h>
#include <FrameWindow.h>
#include <Matcher.h>
#include <Initializer.h>
#include <LocalMapper.h>
#include <SegmentationData.h>
#include <PlaneEstimator.h>
#include <Visualizer.h>

//std::vector<cv::Vec3b> UVR_SLAM::ObjectColors::mvObjectLabelColors;

UVR_SLAM::Tracker::Tracker() {}
UVR_SLAM::Tracker::Tracker(int w, int h, cv::Mat K):mnWidth(w), mnHeight(h), mK(K), mbInitializing(false), mbFirstFrameAfterInit(false), mbInitilized(false){}
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

void UVR_SLAM::Tracker::Tracking(Frame* pPrev, Frame* pCurr) {
	if(!mbInitializing){
		mbInitializing = mpInitializer->Initialize(pCurr, mnWidth, mnHeight);
		//mbInit = bInit;
		mbFirstFrameAfterInit = mbInitializing;
	}
	else {
		
		////시맨틱 프레임과 매칭
		//int nLastFrameIndex = mpFrameWindow->GetLastSemanticFrameIndex();
		//if (nLastFrameIndex >= 0) {
		//	UVR_SLAM::Frame* mpSemanticFrame = mpFrameWindow->GetFrame(nLastFrameIndex);
		//	mpMatcher->FeatureMatchingWithSemanticFrames(mpSemanticFrame, pCurr);
		//}
			
		//FeatureMatchingWithSemanticFrames
		mpFrameWindow->mvPairMatchingInfo.clear();
		mpFrameWindow->SetVectorInlier(mpFrameWindow->LocalMapSize, false);

		//mpMatcher->FeatureMatchingForInitialPoseTracking(mpFrameWindow, pFrame);
		
		int nInitMatching = mpMatcher->FeatureMatchingForInitialPoseTracking(pPrev, pCurr, mpFrameWindow);
		//std::cout << "Matching Init : " << nInitMatching << std::endl;

		Optimization::PoseOptimization(mpFrameWindow, pCurr, false,4,5);
		int nProjection = mpMatcher->FeatureMatchingForPoseTrackingByProjection(mpFrameWindow, pCurr,10.0);
		//std::cout << "Matching projection : " << nProjection << std::endl;

		//visible
		CalcVisibleCount(pCurr);
		int nMatching =  Optimization::PoseOptimization(mpFrameWindow, pCurr, false,10,2);


		//슬램 초기화 최종 판단
		if (mbFirstFrameAfterInit) {
			mbFirstFrameAfterInit = false;
			if (nMatching < 80) {
				//매칭 실패
				mbInitializing = false;
				mpSystem->Reset();
				std::cout << "Fail Initilization" << std::endl;
				return;
			}
			else {
				mpSystem->SetBoolInit(true);
				mbInitilized = true;
			}
		}

		//matching
		CalcMatchingCount(pCurr);
		mpFrameWindow->IncrementFrameCount();


		bool bBow =  mpFrameWindow->CalcFrameDistanceWithBOW(pCurr);
		if (mpFrameWindow->GetFrameCount() > 10 &&(nMatching < 50 || bBow)) {
			if (!mpLocalMapper->isDoingProcess()) {
				mpLocalMapper->SetBoolDoingProcess(true);
				mpLocalMapper->SetTargetFrame(pCurr);
			}
		}
		else {
			if (!mpPlaneEstimator->isDoingProcess()) {
				mpPlaneEstimator->SetBoolDoingProcess(true, 2);
				mpPlaneEstimator->SetTargetFrame(pCurr);
			}
		}

		//일단 테스트

		cv::Mat vis = pCurr->GetOriginalImage();
		//cvtColor(vis, vis, CV_RGBA2BGR);
		vis.convertTo(vis, CV_8UC3);

		for (int i = 0; i < pCurr->mvKeyPoints.size(); i++) {
			if (!pCurr->GetBoolInlier(i))
				continue;
			UVR_SLAM::MapPoint* pMP = pCurr->GetMapPoint(i);
			cv::circle(vis, pCurr->mvKeyPoints[i].pt, 1, cv::Scalar(255, 0, 255), -1);
			if (pMP) {
				if (pMP->isDeleted()){
					pCurr->SetBoolInlier(false, i);
					continue;
				}
				cv::Point2f p2D;
				cv::Mat pCam;
				pMP->Projection(p2D, pCam, pCurr->GetRotation(), pCurr->GetTranslation(), mK, mnWidth, mnHeight);
				UVR_SLAM::ObjectType type = pMP->GetObjectType();
				cv::line(vis, p2D, pCurr->mvKeyPoints[i].pt, cv::Scalar(255, 255, 0), 1);
				if (type != OBJECT_NONE)
					circle(vis, p2D, 3, UVR_SLAM::ObjectColors::mvObjectLabelColors[type], -1);
				
			}
			
		}
		cv::imshow("Output::Tracking", vis);

		//시각화
		cv::Mat vis2 = pCurr->GetOriginalImage();
		for (int i = 0; i < pCurr->mvKeyPoints.size(); i++) {
			UVR_SLAM::ObjectType type = pCurr->GetObjectType(i);
			if (type != OBJECT_NONE)
				circle(vis2, pCurr->mvKeyPoints[i].pt, 2, ObjectColors::mvObjectLabelColors[type], -1);
		}
		cv::imshow("Output::Matching::SemanticFrame", vis2);

		std::stringstream ss;
		ss << "../../bin/segmentation/res/img/img_" << pCurr->GetFrameID() << ".jpg";
		cv::imwrite(ss.str(), vis);
		cv::imwrite("../../bin/segmentation/res/labeling.jpg", vis2);

		cv::waitKey(1);

		if (!mpVisualizer->isDoingProcess()) {
			//mpSystem->SetVisualizeFrame(pCurr);
			mpVisualizer->SetBoolDoingProcess(true);
		}
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
		if (!pF->GetBoolInlier(i))
			continue;
		UVR_SLAM::MapPoint* pMP = pF->GetMapPoint(i);
		if (!pMP)
			continue;
		pMP->mnVisibleCount++;
	}
}
void UVR_SLAM::Tracker::CalcMatchingCount(UVR_SLAM::Frame* pF) {
	for (int i = 0; i < pF->mvKeyPoints.size(); i++) {
		if (!pF->GetBoolInlier(i))
			continue;
		UVR_SLAM::MapPoint* pMP = pF->GetMapPoint(i);
		if (!pMP)
			continue;
		pMP->mnMatchingCount++;
	}
}