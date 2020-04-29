#include <MapPoint.h>
#include <Frame.h>
#include <FrameWindow.h>
#include <SegmentationData.h>

static int nMapPointID = 0;

UVR_SLAM::MapPoint::MapPoint()
	:p3D(cv::Mat::zeros(3, 1, CV_32FC1)), mbNewMP(true), mbSeen(false), mnVisible(0), mnFound(0), mnConnectedFrames(0), mnDenseFrames(0), mfDepth(0.0), mbDelete(false), mObjectType(OBJECT_NONE), mnPlaneID(0), mnType(MapPointType::NORMAL_MP)
	, mnFirstKeyFrameID(0), mnLocalMapID(0), mnLocalBAID(0), mnTrackedFrameID(-1), mnLayoutFrameID(-1)
{}
UVR_SLAM::MapPoint::MapPoint(UVR_SLAM::Frame* pRefKF,cv::Mat _p3D, cv::Mat _desc)
:mpRefKF(pRefKF),p3D(_p3D), desc(_desc), mbNewMP(true), mbSeen(false), mnVisible(0), mnFound(0), mnConnectedFrames(0), mnDenseFrames(0), mfDepth(0.0), mnMapPointID(++nMapPointID), mbDelete(false), mObjectType(OBJECT_NONE), mnPlaneID(0), mnType(MapPointType::NORMAL_MP)
, mnFirstKeyFrameID(0), mnLocalMapID(0), mnLocalBAID(0), mnTrackedFrameID(-1), mnLayoutFrameID(-1)
{}
UVR_SLAM::MapPoint::MapPoint(UVR_SLAM::Frame* pRefKF, cv::Mat _p3D, cv::Mat _desc, MapPointType ntype)
: mpRefKF(pRefKF), p3D(_p3D), desc(_desc), mbNewMP(true), mbSeen(false), mnVisible(0), mnFound(0), mnConnectedFrames(0), mnDenseFrames(0), mfDepth(0.0), mnMapPointID(++nMapPointID), mbDelete(false), mObjectType(OBJECT_NONE), mnPlaneID(0), mnType(ntype)
, mnFirstKeyFrameID(0), mnLocalMapID(0), mnLocalBAID(0), mnTrackedFrameID(-1), mnLayoutFrameID(-1)
{}
UVR_SLAM::MapPoint::~MapPoint(){}

int UVR_SLAM::MapPoint::GetMapPointID() {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	return mnMapPointID;
}

int UVR_SLAM::MapPoint::GetRecentLocalMapID() {
	std::unique_lock<std::mutex> lockMP(mMutexRecentLocalMapID);
	return mnLocalMapID;
}
void UVR_SLAM::MapPoint::SetRecentLocalMapID(int nLocalMapID) {
	std::unique_lock<std::mutex> lockMP(mMutexRecentLocalMapID);
	mnLocalMapID = nLocalMapID;
}
int UVR_SLAM::MapPoint::GetRecentTrackingFrameID() {
	std::unique_lock<std::mutex> lockMP(mMutexRecentTrackedFrameID);
	return mnTrackedFrameID;
}
void UVR_SLAM::MapPoint::SetRecentTrackingFrameID(int nFrameID) {
	std::unique_lock<std::mutex> lockMP(mMutexRecentTrackedFrameID);
	mnTrackedFrameID = nFrameID;
}
int UVR_SLAM::MapPoint::GetRecentLayoutFrameID() {
	std::unique_lock<std::mutex> lockMP(mMutexRecentLayoutFrameID);
	return mnLayoutFrameID;
}
void UVR_SLAM::MapPoint::SetRecentLayoutFrameID(int nFrameID) {
	std::unique_lock<std::mutex> lockMP(mMutexRecentLayoutFrameID);
	mnLayoutFrameID = nFrameID;
}

void UVR_SLAM::MapPoint::SetPlaneID(int nid) {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	mnPlaneID = nid;
}

int UVR_SLAM::MapPoint::GetPlaneID() {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	return mnPlaneID;
}

void UVR_SLAM::MapPoint::SetMapPointType(MapPointType type) {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	mnType = type;
}

UVR_SLAM::MapPointType UVR_SLAM::MapPoint::GetMapPointType() {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	return mnType;
}

cv::Mat UVR_SLAM::MapPoint::GetWorldPos() {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	return p3D.clone();
}
void UVR_SLAM::MapPoint::SetWorldPos(cv::Mat X) {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	p3D = X.clone();
}

void UVR_SLAM::MapPoint::SetObjectType(UVR_SLAM::ObjectType nType){
	std::unique_lock<std::mutex> lock(mMutexObjectType);
	mObjectType = nType;
}
UVR_SLAM::ObjectType  UVR_SLAM::MapPoint::GetObjectType(){
	std::unique_lock<std::mutex> lock(mMutexObjectType);
	return mObjectType;
}

void UVR_SLAM::MapPoint::SetNewMP(bool _b){
	mbNewMP = _b;
}
bool UVR_SLAM::MapPoint::isNewMP(){
	return mbNewMP;
}

void UVR_SLAM::MapPoint::SetDelete(bool b) {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	mbDelete = b;
}
bool UVR_SLAM::MapPoint::isDeleted(){
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	return mbDelete;
}


void UVR_SLAM::MapPoint::IncreaseVisible(int n)
{
	std::unique_lock<std::mutex> lock(mMutexFeatures);
	mnVisible += n;
}

void UVR_SLAM::MapPoint::IncreaseFound(int n)
{
	std::unique_lock<std::mutex> lock(mMutexFeatures);
	mnFound += n;
}

bool UVR_SLAM::MapPoint::isInFrame(UVR_SLAM::Frame* pF) {
	std::unique_lock<std::mutex> lock(mMutexFeatures);
	return mmpFrames.count(pF);
}

//결합되어 삭제되는 맵포인트가 실행됨
void UVR_SLAM::MapPoint::Fuse(UVR_SLAM::MapPoint* pMP) {
	if (this->mnMapPointID == pMP->mnMapPointID)
		return;
	int nvisible, nfound;
	std::map<Frame*, int> obs;
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		std::unique_lock<std::mutex> lock2(mMutexMP);
		obs = mmpFrames;
		mmpFrames.clear();
		nvisible = mnVisible;
		nfound = mnFound;
		mbDelete = true;
		//std::cout << "MapPoint::Fuse::" << obs.size() << std::endl;
	}
	for (std::map<Frame*, int>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
	{
		// Replace measurement in keyframe
		Frame* pKF = mit->first;

		if (!pMP->isInFrame(pKF))
		{
			pKF->mvpMPs[mit->second] = pMP;
			pKF->mvbMPInliers[mit->second] = true;
			pMP->AddFrame(pKF, mit->second);
		}
		else
		{
			pKF->mvpMPs[mit->second] = nullptr;
			pKF->mvbMPInliers[mit->second] = false;
			//pKF->EraseMapPointMatch(mit->second);
		}
	}
	pMP->IncreaseFound(nfound);
	pMP->IncreaseVisible(nvisible);
}

void UVR_SLAM::MapPoint::FuseDenseMP(MapPoint* pMP) {
	if (this->mnMapPointID == pMP->mnMapPointID)
		return;
	int nvisible, nfound;
	std::map<Frame*, cv::Point2f> obs;
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		std::unique_lock<std::mutex> lock2(mMutexMP);
		obs = mmpDenseFrames;
		mmpDenseFrames.clear();
		nvisible = mnVisible;
		nfound = mnFound;
		mbDelete = true;
		//std::cout << "MapPoint::Fuse::" << obs.size() << std::endl;
	}
	//for (auto mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
	//{
	//	// Replace measurement in keyframe
	//	Frame* pKF = mit->first;

	//	if (!pMP->isInFrame(pKF))
	//	{
	//		pKF->mvpMPs[mit->second] = pMP;
	//		pKF->mvbMPInliers[mit->second] = true;
	//		pMP->AddFrame(pKF, mit->second);
	//	}
	//	else
	//	{

	//		pKF->mvpMPs[mit->second] = nullptr;
	//		pKF->mvbMPInliers[mit->second] = false;
	//		//pKF->EraseMapPointMatch(mit->second);
	//	}
	//}
	//pMP->IncreaseFound(nfound);
	//pMP->IncreaseVisible(nvisible);
}

float UVR_SLAM::MapPoint::GetFVRatio() {
	std::unique_lock<std::mutex> lock(mMutexFeatures);
	if (mnVisible == 0)
		return 0.0;
	return ((float)mnFound) / mnVisible;
}

std::map<UVR_SLAM::Frame*, int> UVR_SLAM::MapPoint::GetConnedtedFrames() {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	return std::map<UVR_SLAM::Frame*, int>(mmpFrames.begin(), mmpFrames.end());
}

int UVR_SLAM::MapPoint::GetNumConnectedFrames() {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	return mnConnectedFrames;
}

void UVR_SLAM::MapPoint::AddFrame(UVR_SLAM::Frame* pF, int idx) {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	auto res = mmpFrames.find(pF);
	if (res == mmpFrames.end()) {
		mmpFrames.insert(std::pair<UVR_SLAM::Frame*, int>(pF, idx));
		mnConnectedFrames++;
		pF->AddMP(this, idx);
	}
}
void UVR_SLAM::MapPoint::RemoveFrame(UVR_SLAM::Frame* pF){
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	auto res = mmpFrames.find(pF);
	if (res != mmpFrames.end()) {
		int idx = res->second;
		res = mmpFrames.erase(res);
		mnConnectedFrames--;
		pF->RemoveMP(idx);
	}
}

void UVR_SLAM::MapPoint::Delete() {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	/*if (mnType == UVR_SLAM::PLANE_DENSE_MP) {
		for (auto iter = mmpDenseFrames.begin(); iter != mmpDenseFrames.end(); iter++) {
			UVR_SLAM::Frame* pF = iter->first;
			auto pt = iter->second;
			pF->RemoveDenseMP(pt);
		}
		mmpDenseFrames.clear();
	}
	else {
		for (auto iter = mmpFrames.begin(); iter != mmpFrames.end(); iter++) {
			UVR_SLAM::Frame* pF = iter->first;
			int idx = iter->second;
			pF->RemoveMP(idx);
		}
		mmpFrames.clear();
	}*/
	for (auto iter = mmpDenseFrames.begin(); iter != mmpDenseFrames.end(); iter++) {
		UVR_SLAM::Frame* pF = iter->first;
		auto pt = iter->second;
		pF->RemoveDenseMP(pt);
	}
	mmpDenseFrames.clear();
}
void UVR_SLAM::MapPoint::SetDescriptor(cv::Mat _desc){
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	desc = _desc.clone();
}
cv::Mat UVR_SLAM::MapPoint::GetDescriptor(){
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	return desc;
}
bool UVR_SLAM::MapPoint::isSeen(){
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	return mbSeen;
}

bool UVR_SLAM::MapPoint::Projection(cv::Point2f& _P2D, cv::Mat& _Pcam, cv::Mat R, cv::Mat t, cv::Mat K, int w, int h) {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	_Pcam = R*p3D + t;
	cv::Mat temp = K*_Pcam;
	_P2D = cv::Point2f(temp.at<float>(0) / temp.at<float>(2), temp.at<float>(1) / temp.at<float>(2));
	mfDepth = temp.at<float>(2);
	bool bres = false;
	if (mfDepth > -0.001f && (_P2D.x >= 0 && _P2D.x < w && _P2D.y >= 0 && _P2D.y < h)) {
		bres = true;
	}
	/*if (mfDepth < 0.0)
		std::cout <<"depth error  = "<< mfDepth << std::endl;*/
	/*if (!(_P2D.x >= 0 && _P2D.x < w && _P2D.y >= 0 && _P2D.y < h))
		std::cout << "as;dlfja;sldjfasl;djfaskl;fjasd" << std::endl;*/
	mbSeen = bres;
	return bres;
}

void UVR_SLAM::MapPoint::UpdateNormalAndDepth()
{
	std::map<UVR_SLAM::Frame*, int> observations;
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
		UVR_SLAM::Frame* pKF = mit->first;
		cv::Mat Owi = pKF->GetCameraCenter();
		cv::Mat normali = Pos - Owi;
		normal = normal + normali / cv::norm(normali);
		n++;
	}

	cv::Mat PC = Pos - pRefKF->GetCameraCenter();
	const float dist = cv::norm(PC);
	const int level = pRefKF->mvKeyPoints[observations[pRefKF]].octave;
	const float levelScaleFactor = pRefKF->mvScaleFactors[level];
	const int nLevels = pRefKF->mnScaleLevels;

	{
		std::unique_lock < std::mutex > lock3(mMutexMP);
		mfMaxDistance = dist*levelScaleFactor;
		mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
		mNormalVector = normal / n;
	}
}

int UVR_SLAM::MapPoint::PredictScale(const float &currentDist, Frame* pKF)
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


float UVR_SLAM::MapPoint::GetMaxDistance() {
	std::unique_lock<std::mutex> lock(mMutexMP);
	return 1.2f*mfMaxDistance;
}
float UVR_SLAM::MapPoint::GetMinDistance(){
	std::unique_lock<std::mutex> lock(mMutexMP);
	return 0.8f*mfMinDistance;
}

cv::Mat UVR_SLAM::MapPoint::GetNormal() {
	std::unique_lock<std::mutex> lock(mMutexMP);
	return mNormalVector;
}
int UVR_SLAM::MapPoint::GetIndexInFrame(Frame *pKF)
{
	std::unique_lock<std::mutex> lock(mMutexMP);
	if (mmpFrames.count(pKF))
		return mmpFrames[pKF];
	else
		return -1;
}

////////////////////////////////////////////
////Dense
void UVR_SLAM::MapPoint::AddDenseFrame(UVR_SLAM::Frame* pF, cv::Point2f pt) {
	
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	auto res = mmpDenseFrames.find(pF);
	auto res2 = pF->GetDenseMP(pt);

	if (res2) {
		std::cout << "lm::updatemp::??????????" << std::endl;
	}

	if (res == mmpDenseFrames.end() && !res2) {
		//std::cout << "3" << std::endl;
		bool bAdd = pF->AddDenseMP(this, pt);
		//std::cout << "4" << std::endl;
		if (!bAdd){
			std::cout << "MP::AddDenseFrame::추가 실패" << std::endl;
			return;
		}
		mmpDenseFrames.insert(std::pair<UVR_SLAM::Frame*, cv::Point2f>(pF, pt));
		mnDenseFrames++;
		//std::cout << "5" << std::endl;
	}
	else {
		std::cout << "MP::AddDenseFrame::다른 포인트 존재" << std::endl;
	}
	//std::cout << "6" << std::endl;
}
void UVR_SLAM::MapPoint::RemoveDenseFrame(UVR_SLAM::Frame* pKF) {
	{
		std::unique_lock<std::mutex> lockMP(mMutexMP);
		auto res = mmpDenseFrames.find(pKF);
		if (res != mmpDenseFrames.end()) {
			auto pt = res->second;
			auto res2 = pKF->GetDenseMP(pt);
			if (res2 == this) {
				bool bRemove = pKF->RemoveDenseMP(pt);
				if (!bRemove) {
					std::cout << "MP::RemoveDenseFrame::제거 실패" << std::endl;
					return;
				}
				res = mmpDenseFrames.erase(res);
				mnDenseFrames--;
				
				if (mnDenseFrames < 3) {
					mbDelete = true;
				}
			}
			else {
				std::cout << "MP::RemoveDenseFrame::다른 포인트 존재" << std::endl;
			}
		}
		else {
			//std::cout << "MP::RemoveDenseFrame::포인트가 존재하지 않음" << std::endl;
		}
	}
	if(this->isDeleted()){
		this->Delete();
	}
}
std::map<UVR_SLAM::Frame*, cv::Point2f> UVR_SLAM::MapPoint::GetConnedtedDenseFrames() {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	return std::map<UVR_SLAM::Frame*, cv::Point2f>(mmpDenseFrames.begin(), mmpDenseFrames.end());
}
int UVR_SLAM::MapPoint::GetNumDensedFrames() {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	return mmpDenseFrames.size();
}