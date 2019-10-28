#include <Tracker.h>
#include <System.h>
#include <FrameWindow.h>
#include <Matcher.h>
#include <Initializer.h>
#include <LocalMapper.h>
#include <SegmentationData.h>

//std::vector<cv::Vec3b> UVR_SLAM::ObjectColors::mvObjectLabelColors;

UVR_SLAM::Tracker::Tracker() {}
UVR_SLAM::Tracker::Tracker(int w, int h, cv::Mat K):mnWidth(w), mnHeight(h), mK(K){}
UVR_SLAM::Tracker::~Tracker() {}

bool UVR_SLAM::Tracker::isInitialized() {
	return mbInit;
}

void UVR_SLAM::Tracker::SetSystem(System* pSystem) {
	mpSystem = pSystem;
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

void UVR_SLAM::Tracker::Tracking(Frame* pPrev, Frame* pCurr, bool & bInit) {
	if(!bInit){
		bInit = mpInitializer->Initialize(pCurr, mnWidth, mnHeight);
		mbInit = bInit;
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
		
		mpMatcher->FeatureMatchingForInitialPoseTracking(pPrev, pCurr, mpFrameWindow);
		
		Optimization::PoseOptimization(mpFrameWindow, pCurr,4,5);
		mpMatcher->FeatureMatchingForPoseTrackingByProjection(mpFrameWindow, pCurr,10.0);

		//visible
		CalcVisibleCount(pCurr);
		int nMatching =  Optimization::PoseOptimization(mpFrameWindow, pCurr,10,2);
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