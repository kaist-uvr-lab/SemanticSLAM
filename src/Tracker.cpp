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
#include <CandidatePoint.h>

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
	int nDiff = mnPointMatching - mnMapPointMatching;
	bool bDiff = nDiff > 50;

	//1 : rotation angle
	bool bDoingMapping = !mpLocalMapper->isDoingProcess();
	bool bRotation = pCurr->CalcDiffAngleAxis(mpRefKF) > 10.0;
	bool bMaxFrames = pCurr->GetFrameID() >= mpRefKF->mnFrameID + mnMaxFrames;//mnMinFrames;
	bool bMinFrames = pCurr->GetFrameID() < mpRefKF->mnFrameID + mnMinFrames;
	bool bDoingSegment = !mpSegmentator->isDoingProcess();

	bool bMatchMapPoint = mnMapPointMatching < 200;
	bool bMatchPoint = mnPointMatching < 500;

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
			mnMapPointMatching = pCurr->mpMatchInfo->GetNumMapPoints();
		}
	}
	else {
		std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
		//std::cout << "tracker::start::" <<pCurr->GetFrameID()<< std::endl;
		//std::cout << mpMap->mpFirstKeyFrame->mpPlaneInformation->GetFloorPlane()->GetParam() << std::endl << std::endl << std::endl;
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////Optical Flow Matching
		////MatchInfo 설정
		mpRefKF->SetRecentTrackedFrameID(pCurr->GetFrameID());
		pCurr->mpMatchInfo = new UVR_SLAM::MatchInfo(pCurr, mpRefKF, mnWidth, mnHeight);
		
		/*int Ncp = mpRefKF->mpMatchInfo->GetNumCPs();
		for (int i = 0; i < Ncp; i++) {
			mpRefKF->mpMatchInfo->mvpMatchingCPs[i]->GetFrames();
		}*/

		cv::Mat prevR, prevT;
		pPrev->GetPose(prevR, prevT);
		pCurr->SetPose(prevR, prevT);
		
		////MatchInfo 설정
		//초기 매칭 테스트
		std::vector<UVR_SLAM::MapPoint*> vpTempMPs;
		std::vector<UVR_SLAM::CandidatePoint*> vpTempCPs;
		std::vector<cv::Point2f> vpTempPts, vpTempPts1;
		std::vector<cv::Point3f> vpTempPts2;
		std::vector<uchar> vcInliers;
		std::vector<bool> vbTempInliers;// = std::vector<bool>(pPrev->mvpMatchingMPs.size(), false);
		std::vector<int> vnIDXs, vnMPIDXs;
		cv::Mat debugImg;
		cv::Mat overlap = cv::Mat::zeros(pCurr->GetOriginalImage().size(), CV_8UC1);
		mnPointMatching = mpMatcher->OpticalMatchingForTracking(mpRefKF, pCurr, vpTempCPs, vpTempMPs, vpTempPts, vbTempInliers, vnIDXs, overlap, debugImg); //pCurr

		//mpSystem->mbTrackingEnd = true;
		std::chrono::high_resolution_clock::time_point tracking_a = std::chrono::high_resolution_clock::now();

		//graph-based0.
		mnMapPointMatching = Optimization::PoseOptimization(pCurr, vpTempMPs, vpTempPts, vbTempInliers, vnMPIDXs);
		int nMP = UpdateMatchingInfo(mpRefKF, pCurr, vpTempCPs,vpTempMPs, vpTempPts, vbTempInliers, vnIDXs, vnMPIDXs);
	
		///////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////키프레임 체크
		auto pNewKF = CheckNeedKeyFrame(pCurr, pPrev);
		if (pNewKF) {
			//auto pNewKF = pCurr;
			pNewKF->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
			mpRefKF = pNewKF;
			mpLocalMapper->InsertKeyFrame(pNewKF);
		}

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

		/////////트래킹 결과 이미지 저장
		//visualizer thread
		//mpVisualizer->SetMPs(vpTempMPs);
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
int UVR_SLAM::Tracker::UpdateMatchingInfo(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr, std::vector<UVR_SLAM::CandidatePoint*> vpCPs, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<cv::Point2f> vpPts, std::vector<bool> vbInliers, std::vector<int> vnIDXs, std::vector<int> vnMPIDXs) {
	auto pMatchInfo = pCurr->mpMatchInfo;
	auto pPrevMatchInfo = pPrev->mpMatchInfo;
	int nCurrID = pCurr->GetFrameID();
	int nres = 0;
	int nLowQuality = 0;
	for (int i = 0; i < vpPts.size(); i++) {
		auto pCP = vpCPs[i];
		pCP->mnVisibleFrameID = nCurrID;
		if (!vbInliers[i]){
			pCP->AddFail();
			pCP->ComputeQuality();
			continue;
		}
		int prevIdx = vnIDXs[i];
		auto pt = vpPts[i];
		if (pMatchInfo->CheckOpticalPointOverlap(Frame::mnRadius, 10, pt) < 0) {
			pCP->AddSuccess();
			pCP->SetLastSuccessFrame(pCurr->GetFrameID());
			pMatchInfo->AddCP(pCP, pt);
			nres++;
		}
	}
	//std::cout << "Tracking::UpdateMatchingInfo::" << nres << std::endl;
	return nres;
}