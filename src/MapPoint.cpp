#include <MapPoint.h>
#include <Map.h>
#include <Frame.h>
#include <FrameWindow.h>
#include <SegmentationData.h>
#include <CandidatePoint.h>

static int nMapPointID = 0;
namespace UVR_SLAM {
	MapPoint::MapPoint()
		:p3D(cv::Mat::zeros(3, 1, CV_32FC1)), mbNewMP(true), mbSeen(false), mnVisible(0), mnFound(0), mnConnectedFrames(0), mnDenseFrames(0), mfDepth(0.0), mbDelete(false), mObjectType(OBJECT_NONE), mnPlaneID(0), mnType(MapPointType::NORMAL_MP)
		, mnFirstKeyFrameID(0), mnLocalMapID(0), mnLocalBAID(0), mnTrackedFrameID(-1), mnLayoutFrameID(-1), mnOctave(0)
		, mnLastVisibleFrameID(-1), mnLastMatchingFrameID(-1), mbLowQuality(true), mbOptimized(false), mnSuccess(0.0), mnTotal(0)
	{}
	MapPoint::MapPoint(Map* pMap, UVR_SLAM::Frame* pRefKF, CandidatePoint* pCP,cv::Mat _p3D, cv::Mat _desc, int alabel, int octave)
	: mpMap(pMap), mpRefKF(pRefKF),p3D(_p3D), desc(_desc), mbNewMP(true), mbSeen(false), mnVisible(0), mnFound(0), mnConnectedFrames(0), mnDenseFrames(0), mfDepth(0.0), mnMapPointID(++nMapPointID), mbDelete(false), mObjectType(OBJECT_NONE), mnPlaneID(0), mnType(MapPointType::NORMAL_MP)
	, mnLocalMapID(0), mnLocalBAID(0), mnTrackedFrameID(-1), mnLayoutFrameID(-1), mnOctave(octave)
	, mnLastMatchingFrameID(-1), mbLowQuality(true), mbOptimized(false), mnSuccess(0.0), mnTotal(0)
	{
		alabel = label;
		mnFirstKeyFrameID = mpRefKF->mnKeyFrameID;
		mnLastMatchingFrameID = mnLastVisibleFrameID = mnFirstKeyFrameID;
		//CP처리
		mpCP = pCP;
		mpCP->SetMapPoint(this);
		//맵처리
		mpMap->AddMap(this, label);
	}
	MapPoint::MapPoint(Map* pMap, UVR_SLAM::Frame* pRefKF, CandidatePoint* pCP, cv::Mat _p3D, cv::Mat _desc, MapPointType ntype, int alabel, int octave)
	: mpMap(pMap), mpRefKF(pRefKF), p3D(_p3D), desc(_desc), mbNewMP(true), mbSeen(false), mnVisible(0), mnFound(0), mnConnectedFrames(0), mnDenseFrames(0), mfDepth(0.0), mnMapPointID(++nMapPointID), mbDelete(false), mObjectType(OBJECT_NONE), mnPlaneID(0), mnType(ntype)
	, mnLocalMapID(0), mnLocalBAID(0), mnTrackedFrameID(-1), mnLayoutFrameID(-1), mnOctave(octave)
	, mnLastMatchingFrameID(-1), mbLowQuality(true), mbOptimized(false), mnSuccess(0.0), mnTotal(0)
	{
		alabel = label;
		mnFirstKeyFrameID = mpRefKF->mnKeyFrameID;
		mnLastMatchingFrameID = mnLastVisibleFrameID = mnFirstKeyFrameID;
		//CP처리
		mpCP = pCP;
		mpCP->SetMapPoint(this);
		//맵처리
		mpMap->AddMap(this, label);
	}
	MapPoint::~MapPoint(){}

	int MapPoint::GetRecentLocalMapID() {
		std::unique_lock<std::mutex> lockMP(mMutexRecentLocalMapID);
		return mnLocalMapID;
	}
	void MapPoint::SetRecentLocalMapID(int nLocalMapID) {
		std::unique_lock<std::mutex> lockMP(mMutexRecentLocalMapID);
		mnLocalMapID = nLocalMapID;
	}
	////트래킹 시 중복 체크하는 용도
	int MapPoint::GetRecentTrackingFrameID() {
		std::unique_lock<std::mutex> lockMP(mMutexRecentTrackedFrameID);
		return mnTrackedFrameID;
	}
	void MapPoint::SetRecentTrackingFrameID(int nFrameID) {
		std::unique_lock<std::mutex> lockMP(mMutexRecentTrackedFrameID);
		mnTrackedFrameID = nFrameID;
	}
	int MapPoint::GetRecentLayoutFrameID() {
		std::unique_lock<std::mutex> lockMP(mMutexRecentLayoutFrameID);
		return mnLayoutFrameID;
	}
	void MapPoint::SetRecentLayoutFrameID(int nFrameID) {
		std::unique_lock<std::mutex> lockMP(mMutexRecentLayoutFrameID);
		mnLayoutFrameID = nFrameID;
	}

	void MapPoint::SetPlaneID(int nid) {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		mnPlaneID = nid;
	}

	int MapPoint::GetPlaneID() {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		return mnPlaneID;
	}

	void MapPoint::SetMapPointType(MapPointType type) {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		mnType = type;
	}

	MapPointType MapPoint::GetMapPointType() {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		return mnType;
	}

	cv::Mat MapPoint::GetWorldPos() {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		return p3D.clone();
	}
	void MapPoint::SetWorldPos(cv::Mat X) {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		p3D = X.clone();
	}
	int MapPoint::GetLabel() {
		std::unique_lock<std::mutex> lockMP(mMutexLabel);
		return label;
	}
	void MapPoint::SetLabel(int a) {
		std::unique_lock<std::mutex> lockMP(mMutexLabel);
		label = a;
	}
	void MapPoint::SetObjectType(UVR_SLAM::ObjectType nType){
		std::unique_lock<std::mutex> lock(mMutexObjectType);
		mObjectType = nType;
	}
	ObjectType MapPoint::GetObjectType(){
		std::unique_lock<std::mutex> lock(mMutexObjectType);
		return mObjectType;
	}

	void MapPoint::SetNewMP(bool _b){
		mbNewMP = _b;
	}
	bool MapPoint::isNewMP(){
		return mbNewMP;
	}

	//void UVR_SLAM::MapPoint::SetDelete(bool b) {
	//	std::unique_lock<std::mutex> lockMP(mMutexMP);
	//	mbDelete = b;
	//}
	bool MapPoint::isDeleted(){
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		return mbDelete;
	}


	void MapPoint::IncreaseVisible(int n)
	{
		std::unique_lock<std::mutex> lock(mMutexMP);
		mnVisible += n;
	}

	void MapPoint::IncreaseFound(int n)
	{
		std::unique_lock<std::mutex> lock(mMutexMP);
		mnFound += n;
	}

	bool MapPoint::isInFrame(UVR_SLAM::MatchInfo* pF) {
		std::unique_lock<std::mutex> lock(mMutexMP);
		return mmpFrames.count(pF) > 0;
	}

	int MapPoint::GetPointIndexInFrame(MatchInfo* pF) {
		std::unique_lock<std::mutex> lock(mMutexMP);
		auto res = mmpFrames.find(pF);
		if (res == mmpFrames.end())
			return -1;
		else
			return res->second;
	}

	//결합되어 삭제되는 맵포인트가 실행됨
	void MapPoint::Fuse(UVR_SLAM::MapPoint* pMP) {
		if (this->mnMapPointID == pMP->mnMapPointID)
			return;
		int nvisible, nfound;
		std::map<MatchInfo*, int> obs;
		{
			//std::unique_lock<std::mutex> lock(mMutexFeatures);
			std::unique_lock<std::mutex> lock2(mMutexMP);
			obs = mmpFrames;
			mmpFrames.clear();
			nvisible = mnVisible;
			nfound = mnFound;
			mbDelete = true;
			//std::cout << "MapPoint::Fuse::" << obs.size() << std::endl;
		}
		for (std::map<MatchInfo*, int>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
		{
			// Replace measurement in keyframe
			MatchInfo* pKF = mit->first;

			if (!pMP->isInFrame(pKF))
			{
				pMP->ConnectFrame(pKF, mit->second);
			}
			else
			{
				//pKF->RemoveMP();
			}
		}
		pMP->IncreaseFound(nfound);
		pMP->IncreaseVisible(nvisible);
	}

	float MapPoint::GetFVRatio() {
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		if (mnVisible == 0)
			return 0.0;
		return ((float)mnFound) / mnVisible;
	}

	std::map<MatchInfo*, int> MapPoint::GetConnedtedFrames() {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		return std::map<UVR_SLAM::MatchInfo*, int>(mmpFrames.begin(), mmpFrames.end());
	}

	int MapPoint::GetNumConnectedFrames() {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		return mnConnectedFrames;
	}

	void MapPoint::ConnectFrame(UVR_SLAM::MatchInfo* pF, int idx) {
	
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		auto res = mmpFrames.find(pF);
		if (res == mmpFrames.end()) {
			mmpFrames.insert(std::pair<UVR_SLAM::MatchInfo*, int>(pF, idx));
			mnConnectedFrames++;
		}
		/*else {
			std::cout << "MapPoint::AddFrame::Error::" <<mpCP->mnCandidatePointID<< std::endl;
		}*/
	}
	void MapPoint::DisconnectFrame(UVR_SLAM::MatchInfo* pF){
		{
			std::unique_lock<std::mutex> lockMP(mMutexMP);
			auto res = mmpFrames.find(pF);
			if (res != mmpFrames.end()) {
				int idx = res->second;
				res = mmpFrames.erase(res);
				mnConnectedFrames--;
				if (pF->mpRefFrame == mpRefKF) {
					mpRefKF = mmpFrames.begin()->first->mpRefFrame;
				}
				if (mnConnectedFrames < 3)
					mbDelete = true;
			}
		}
		if (mbDelete){
			Delete();
		}
	}

	void MapPoint::Delete() {
		{
			std::unique_lock<std::mutex> lockMP(mMutexMP);
			mbDelete = true;
			for (auto iter = mmpFrames.begin(); iter != mmpFrames.end(); iter++) {
				auto* pF = iter->first;
				auto idx = iter->second;
				//pF->RemoveMP(idx);
			}
			mnConnectedFrames = 0;
			mmpFrames.clear();
		}
		////CP 처리
		mpCP->ResetMapPoint();
		//맵처리
		mpMap->RemoveMap(this);
		mpMap->DeleteMapPoint(this);
	}
	void MapPoint::SetDescriptor(cv::Mat _desc){
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		desc = _desc.clone();
	}
	cv::Mat MapPoint::GetDescriptor(){
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		return desc;
	}
	bool MapPoint::isSeen(){
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		return mbSeen;
	}

	bool MapPoint::Projection(cv::Point2f& _P2D, Frame* pF, int w, int h) {
		cv::Mat X3D, _Pcam;
		{
			std::unique_lock<std::mutex> lockMP(mMutexMP);
			X3D = p3D.clone();
		}
		cv::Mat R, t;
		pF->GetPose(R, t);
		_Pcam = R*X3D + t;
		cv::Mat temp = pF->mK*_Pcam;
		float depth = temp.at<float>(2);
		_P2D = cv::Point2f(temp.at<float>(0) / depth, temp.at<float>(1) / depth);
	
		bool bres = false;
		if (depth > 0.0f && (_P2D.x >= 0 && _P2D.x < w && _P2D.y >= 0 && _P2D.y < h)) {
			bres = true;
		}
		return bres;
	}

	bool MapPoint::Projection(cv::Point2f& _P2D, cv::Mat& _Pcam, cv::Mat R, cv::Mat t, cv::Mat K, int w, int h) {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		_Pcam = R*p3D + t;
		cv::Mat temp = K*_Pcam;
		_P2D = cv::Point2f(temp.at<float>(0) / temp.at<float>(2), temp.at<float>(1) / temp.at<float>(2));
		mfDepth = temp.at<float>(2);
		bool bres = false;
		if (mfDepth > 0.0f && (_P2D.x >= 0 && _P2D.x < w && _P2D.y >= 0 && _P2D.y < h)) {
			bres = true;
		}
		/*if (mfDepth < 0.0)
			std::cout <<"depth error  = "<< mfDepth << std::endl;*/
		/*if (!(_P2D.x >= 0 && _P2D.x < w && _P2D.y >= 0 && _P2D.y < h))
			std::cout << "as;dlfja;sldjfasl;djfaskl;fjasd" << std::endl;*/
		mbSeen = bres;
		return bres;
	}

	void MapPoint::UpdateNormalAndDepth()
	{
		std::map<UVR_SLAM::MatchInfo*, int> observations;
		UVR_SLAM::Frame* pRefKF;
		cv::Mat Pos;
		{
			std::unique_lock<std::mutex> lock1(mMutexMP);
			//unique_lock<mutex> lock2(mMutexPos);
			if (mbDelete)
				return;
			observations = mmpFrames;
			pRefKF = mpRefKF;
			Pos = p3D.clone();
		}

		if (observations.empty())
			return;

		cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
		int n = 0;
		for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			auto matchInfo = mit->first;
			UVR_SLAM::Frame* pKF = matchInfo->mpRefFrame;
			cv::Mat Owi = pKF->GetCameraCenter();
			cv::Mat normali = Pos - Owi;
			normal = normal + normali / cv::norm(normali);
			n++;
		}

		cv::Mat PC = Pos - pRefKF->GetCameraCenter();
		const float dist = cv::norm(PC);
		const int level = mnOctave;
		const float levelScaleFactor = pRefKF->mvScaleFactors[level];
		const int nLevels = pRefKF->mnScaleLevels;

		{
			std::unique_lock < std::mutex > lock3(mMutexMP);
			mfMaxDistance = dist*levelScaleFactor;
			mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
			mNormalVector = normal / n;
		}
	}

	int MapPoint::PredictScale(const float &currentDist, Frame* pKF)
	{
		float ratio;
		{
			std::unique_lock<std::mutex> lock(mMutexMP);
			ratio = mfMaxDistance / currentDist;
		}

		int nScale = ceil(log(ratio) / pKF->mfLogScaleFactor);
		if (nScale<0)
			nScale = 0;
		else if (nScale >= pKF->mnScaleLevels)
			nScale = pKF->mnScaleLevels - 1;

		return nScale;
	}


	float MapPoint::GetMaxDistance() {
		std::unique_lock<std::mutex> lock(mMutexMP);
		return 1.2f*mfMaxDistance;
	}
	float MapPoint::GetMinDistance(){
		std::unique_lock<std::mutex> lock(mMutexMP);
		return 0.8f*mfMinDistance;
	}

	cv::Mat MapPoint::GetNormal() {
		std::unique_lock<std::mutex> lock(mMutexMP);
		return mNormalVector;
	}

	//////////////////////////////
	/////매칭 퀄리티
	/*void MapPoint::AddFail(int n) {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		mnFail += n;
		mnTotal += n;
	}
	int MapPoint::GetFail() {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		return mnFail;
	}
	void MapPoint::AddSuccess(int n) {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		mnSuccess += n;
		mnTotal += n;
	}
	int MapPoint::GetSuccess() {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		return mnSuccess;
	}*/
	void MapPoint::SetLastSuccessFrame(int id) {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		mnLastMatchingFrameID = id;
		mnSuccess++;
	}
	int MapPoint::GetLastSuccessFrame() {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		return mnLastMatchingFrameID;
	}
	void MapPoint::SetLastVisibleFrame(int id) {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		mnLastVisibleFrameID = id;
		mnTotal++;
	}
	int MapPoint::GetLastVisibleFrame() {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		return mnLastVisibleFrameID;
	}
	void MapPoint::ComputeQuality() {

		float nS;
		int nMatchFrame, nVisibleFrame, nTotal;
		std::map<UVR_SLAM::MatchInfo*, int> tempFrames;
		{
			std::unique_lock<std::mutex> lockMP(mMutexMP);
			if (!mbOptimized)
				return;
			nS = mnSuccess;
			nMatchFrame = mnLastMatchingFrameID;
			nVisibleFrame = mnLastVisibleFrameID;
			nTotal = mnTotal;
			tempFrames = mmpFrames;
		}
		int nWidth = mpRefKF->mpMatchInfo->mnWidth;
		int nHeight = mpRefKF->mpMatchInfo->mnHeight;
		//projection test
		/*for (auto iter = tempFrames.begin(); iter != tempFrames.end(); iter++) {
			auto pKF = iter->first;
			auto idx = iter->second;
			auto pt = pKF->mvMatchingPts[idx];
			cv::Point2f pt2;
			this->Projection(pt2, pKF->mpRefFrame, nWidth, nHeight);
			cv::Point2f diffPt = pt2 - pt;
			float dist = diffPt.dot(diffPt);
			if (dist > 9.0) {
				this->DisconnectFrame(pKF);
			}
		}*/
		//projection test

		bool b = true;
		bool bFrame = (nMatchFrame + 2) < nVisibleFrame;
		float ratio = nS / nTotal;
		bool bRatio = false;//ratio < 0.3;
		bool bConnect = false;//;GetNumConnectedFrames() < 3;
		//std::cout << nS <<", "<< ratio << ", " << nVisible << ", " << nLastFrame << std::endl;
		if (bFrame || bRatio || bConnect)
			b = false;

		{
			std::unique_lock<std::mutex> lockMP(mMutexMP);
			mbLowQuality = b;
		}
	}
	bool MapPoint::GetQuality() {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		///매칭 성능이 좋으면 true, 안좋으면 false
		return mbLowQuality;
	}
	void MapPoint::SetOptimization(bool b) {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		mbOptimized = b;
	}
	bool MapPoint::isOptimized() {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		return mbOptimized;
	}
	/////매칭 퀄리티
	//////////////////////////////
}