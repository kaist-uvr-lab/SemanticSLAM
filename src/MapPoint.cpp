#include <MapPoint.h>
#include <Map.h>
#include <Frame.h>
#include <MapGrid.h>
#include <FrameWindow.h>
#include <SegmentationData.h>
#include <CandidatePoint.h>

namespace UVR_SLAM {
	MapPoint::MapPoint()
		:p3D(cv::Mat::zeros(3, 1, CV_32FC1)), mbNewMP(true), mbSeen(false), mnVisible(0), mnFound(0), mnConnectedFrames(0), mnDenseFrames(0), mfDepth(0.0), mbDelete(false), mObjectType(OBJECT_NONE), mnPlaneID(0), mnType(MapPointType::NORMAL_MP)
		, mnLoopPointForKF(0), mnCorrectedByKF(0), mnCorrectedReference(0), mnBAGlobalForKF(0)
		, mnFirstKeyFrameID(0), mnLocalMapID(-1), mnLocalBAID(0), mnTrackingID(-1), mnLayoutFrameID(-1), mnMapGridID(0), mnOctave(0)
		, mnLastVisibleFrameID(-1), mnLastMatchingFrameID(-1), mbLowQuality(true), mbOptimized(false), mnSuccess(0.0), mnTotal(0), mbLastMatch(false)
	{}
	MapPoint::MapPoint(Map* pMap, cv::Mat _p3D, cv::Mat _desc, int alabel, int octave)
		: mpMap(pMap), mpRefKF(nullptr), p3D(_p3D), desc(_desc), mbNewMP(true), mbSeen(false), mnVisible(0), mnFound(0), mnConnectedFrames(0), mnDenseFrames(0), mfDepth(0.0), mnMapPointID(UVR_SLAM::System::nMapPointID++), mbDelete(false), mObjectType(OBJECT_NONE), mnPlaneID(0), mnType(MapPointType::NORMAL_MP)
		, mnLoopPointForKF(0), mnCorrectedByKF(0), mnCorrectedReference(0), mnBAGlobalForKF(0)
		, mnLocalMapID(-1), mnLocalBAID(0), mnTrackingID(-1), mnLayoutFrameID(-1), mnMapGridID(0), mnOctave(octave)
		, mnLastMatchingFrameID(-1), mbLowQuality(true), mbOptimized(false), mnSuccess(0.0), mnTotal(0), mbLastMatch(false)
	{
		alabel = label;
		//��ó��
		mpMap->AddMap(this, label);
	}
	MapPoint::MapPoint(Map* pMap, UVR_SLAM::Frame* pRefKF, cv::Mat _p3D, cv::Mat _desc, int alabel, int octave)
		: mpMap(pMap), mpRefKF(pRefKF), p3D(_p3D), desc(_desc), mbNewMP(true), mbSeen(false), mnVisible(0), mnFound(0), mnConnectedFrames(0), mnDenseFrames(0), mfDepth(0.0), mnMapPointID(UVR_SLAM::System::nMapPointID++), mbDelete(false), mObjectType(OBJECT_NONE), mnPlaneID(0), mnType(MapPointType::NORMAL_MP)
		, mnLoopPointForKF(0), mnCorrectedByKF(0), mnCorrectedReference(0), mnBAGlobalForKF(0)
		, mnLocalMapID(-1), mnLocalBAID(0), mnTrackingID(-1), mnLayoutFrameID(-1), mnMapGridID(0), mnOctave(octave)
		, mnLastMatchingFrameID(-1), mbLowQuality(true), mbOptimized(false), mnSuccess(0.0), mnTotal(0), mbLastMatch(false)
	{
		alabel = label;
		mnFirstKeyFrameID = mpRefKF->mnKeyFrameID;
		mnLastMatchingFrameID = mnLastVisibleFrameID = mnFirstKeyFrameID;
		
		//��ó��
		mpMap->AddMap(this, label);
	}

	cv::Mat MapPoint::GetWorldPos() {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		return p3D.clone();
	}
	void MapPoint::SetWorldPos(cv::Mat X) {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		p3D = X.clone();
	}

	bool MapPoint::isDeleted() {
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

	float MapPoint::GetFVRatio() {
		std::unique_lock<std::mutex> lock(mMutexMP);
		if (mnVisible == 0)
			return 0.0;
		return ((float)mnFound) / mnVisible;
	}

	std::map<Frame*, int> MapPoint::GetObservations() {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		return std::map<Frame*, int>(mmpObservations.begin(), mmpObservations.end());
	}

	void MapPoint::AddObservation(UVR_SLAM::Frame* pF, int idx) {

		std::unique_lock<std::mutex> lockMP(mMutexMP);
		if (!mmpObservations.count(pF)) {
			mmpObservations[pF] = idx;
			mnConnectedFrames++;
		}
	}
	void MapPoint::EraseObservation(UVR_SLAM::Frame* pF) {
		bool bDelete = false;
		{
			std::unique_lock<std::mutex> lockMP(mMutexMP);
			if (mmpObservations.count(pF)) {
				mmpObservations.erase(pF);
				mnConnectedFrames--;
				if (pF == mpRefKF) {
					mpRefKF = mmpObservations.begin()->first;
				}
				if (mnConnectedFrames < 3) {
					mbDelete = true;
					bDelete = true;
				}
			}
		}
		if (bDelete) {
			DeleteMapPoint();
		}
	}

	void MapPoint::DeleteMapPoint() {
		cv::Mat apos;
		std::map<Frame*, int> observations;
		{
			std::unique_lock<std::mutex> lockMP(mMutexMP);
			mbDelete = true;
			observations = mmpObservations;
			mmpObservations.clear();
			apos = p3D;
		}
		
		for (auto iter = observations.begin(), iend = observations.end(); iter != iend; iter++) {
			auto pKF = iter->first;
			auto idx = iter->second;
			pKF->EraseMapPoint(idx);
		}

		//��ó��
		mpMap->RemoveMap(this);
		//�׸��� ó��
		/*auto key = MapGrid::ComputeKey(apos);
		auto pMapGrid = mpMap->GetMapGrid(key);
		if (pMapGrid)
			pMapGrid->RemoveMapPoint(this);*/
	}
	bool MapPoint::isInFrame(UVR_SLAM::Frame* pF) {
		std::unique_lock<std::mutex> lock(mMutexMP);
		return mmpObservations.count(pF) > 0;
	}

	int MapPoint::GetIndexInKeyFrame(Frame* pF) {
		std::unique_lock<std::mutex> lock(mMutexMP);
		if (mmpObservations.count(pF)) {
			return mmpObservations[pF];
		}
		else
			return -1;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////








	MapPoint::MapPoint(Map* pMap, UVR_SLAM::Frame* pRefKF, CandidatePoint* pCP,cv::Mat _p3D, cv::Mat _desc, int alabel, int octave)
	: mpMap(pMap), mpRefKF(pRefKF),p3D(_p3D), desc(_desc), mbNewMP(true), mbSeen(false), mnVisible(0), mnFound(0), mnConnectedFrames(0), mnDenseFrames(0), mfDepth(0.0), mnMapPointID(UVR_SLAM::System::nMapPointID++), mbDelete(false), mObjectType(OBJECT_NONE), mnPlaneID(0), mnType(MapPointType::NORMAL_MP)
	, mnLocalMapID(-1), mnLocalBAID(0), mnTrackingID(-1), mnLayoutFrameID(-1), mnMapGridID(0), mnOctave(octave)
	, mnLastMatchingFrameID(-1), mbLowQuality(true), mbOptimized(false), mnSuccess(0.0), mnTotal(0), mbLastMatch(false)
	{
		alabel = label;
		mnFirstKeyFrameID = mpRefKF->mnKeyFrameID;
		mnLastMatchingFrameID = mnLastVisibleFrameID = mnFirstKeyFrameID;
		//CPó��
		mpCP = pCP;
		mpCP->SetMapPoint(this);
		//��ó��
		mpMap->AddMap(this, label);
	}
	MapPoint::MapPoint(Map* pMap, UVR_SLAM::Frame* pRefKF, CandidatePoint* pCP, cv::Mat _p3D, cv::Mat _desc, MapPointType ntype, int alabel, int octave)
	: mpMap(pMap), mpRefKF(pRefKF), p3D(_p3D), desc(_desc), mbNewMP(true), mbSeen(false), mnVisible(0), mnFound(0), mnConnectedFrames(0), mnDenseFrames(0), mfDepth(0.0), mnMapPointID(UVR_SLAM::System::nMapPointID++), mbDelete(false), mObjectType(OBJECT_NONE), mnPlaneID(0), mnType(ntype)
	, mnLocalMapID(-1), mnLocalBAID(0), mnTrackingID(-1), mnLayoutFrameID(-1), mnMapGridID(0), mnOctave(octave)
	, mnLastMatchingFrameID(-1), mbLowQuality(true), mbOptimized(false), mnSuccess(0.0), mnTotal(0), mbLastMatch(false)
	{
		alabel = label;
		mnFirstKeyFrameID = mpRefKF->mnKeyFrameID;
		mnLastMatchingFrameID = mnLastVisibleFrameID = mnFirstKeyFrameID;
		//CPó��
		mpCP = pCP;
		mpCP->SetMapPoint(this);
		//��ó��
		mpMap->AddMap(this, label);
	}
	MapPoint::~MapPoint(){}


	////Ʈ��ŷ �� �ߺ� üũ�ϴ� �뵵
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

	void MapPoint::SetMapGridID(int id) {
		std::unique_lock<std::mutex> lock(mMutexMapGrid);
		mnMapGridID = id;
	}
	int MapPoint::GetMapGridID() {
		std::unique_lock<std::mutex> lock(mMutexMapGrid);
		return mnMapGridID;
	}

	//���յǾ� �����Ǵ� ������Ʈ�� �����
	void MapPoint::Fuse(UVR_SLAM::MapPoint* pMP) {
		if (this->mnMapPointID == pMP->mnMapPointID)
			return;
		int nvisible, nfound;
		std::map<Frame*, int> obs;
		{
			//std::unique_lock<std::mutex> lock(mMutexFeatures);
			std::unique_lock<std::mutex> lock2(mMutexMP);
			obs = mmpObservations;
			mmpObservations.clear();
			nvisible = mnVisible;
			nfound = mnFound;
			mbDelete = true;
			//std::cout << "MapPoint::Fuse::" << obs.size() << std::endl;
		}
		for (std::map<Frame*, int>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
		{
			// Replace measurement in keyframe
			Frame* pKF = mit->first;
			int idx = mit->second;
			if (!pMP->isInFrame(pKF))
			{
				pMP->AddObservation(pKF, idx);
				pKF->AddMapPoint(pMP, idx);
				//pMP->ConnectFrame(pKF, mit->second);
			}
			else
			{
				pKF->EraseMapPoint(idx);
			}
		}
		pMP->IncreaseFound(nfound);
		pMP->IncreaseVisible(nvisible);
		mpMap->RemoveMap(this);
	}
		

	int MapPoint::GetNumObservations() {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		return mnConnectedFrames;
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

	bool MapPoint::Projection(cv::Point2f& _P2D, cv::Mat& _Pcam, float& fDepth, cv::Mat R, cv::Mat t, cv::Mat K, int w, int h) {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		_Pcam = R*p3D + t;
		cv::Mat temp = K*_Pcam;
		_P2D = cv::Point2f(temp.at<float>(0) / temp.at<float>(2), temp.at<float>(1) / temp.at<float>(2));
		mfDepth = temp.at<float>(2);
		fDepth = mfDepth;
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

	////���� �ʿ�. 210228
	void MapPoint::UpdateNormalAndDepth()
	{
		std::map<UVR_SLAM::Frame*, int> observations;
		UVR_SLAM::Frame* pRefKF;
		cv::Mat Pos;
		{
			std::unique_lock<std::mutex> lock1(mMutexMP);
			//unique_lock<mutex> lock2(mMutexPos);
			if (mbDelete)
				return;
			observations = mmpObservations;
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
			cv::Mat Owi = pRefKF->GetCameraCenter();
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
	/////��Ī ����Ƽ
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
	
	void MapPoint::SetOptimization(bool b) {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		mbOptimized = b;
	}
	bool MapPoint::isOptimized() {
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		return mbOptimized;
	}
	/////��Ī ����Ƽ
	//////////////////////////////
}