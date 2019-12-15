#include <MapPoint.h>
#include <Frame.h>
#include <FrameWindow.h>
#include <SegmentationData.h>

static int nMapPointID = 0;

UVR_SLAM::MapPoint::MapPoint()
	:p3D(cv::Mat::zeros(3, 1, CV_32FC1)), mbNewMP(true), mbSeen(false), mnVisibleCount(0), mnMatchingCount(0), mnConnectedFrames(0), mfDepth(0.0), mbDelete(false), mObjectType(OBJECT_NONE), mnPlaneID(0), mnType(MapPointType::NORMAL_MP)
{}
UVR_SLAM::MapPoint::MapPoint(cv::Mat _p3D, UVR_SLAM::Frame* pTargetKF, int idx,cv::Mat _desc)
:p3D(_p3D), desc(_desc), mbNewMP(true), mbSeen(false), mnVisibleCount(0), mnMatchingCount(0), mnConnectedFrames(0), mfDepth(0.0), mnMapPointID(++nMapPointID), mbDelete(false), mObjectType(OBJECT_NONE), mnPlaneID(0), mnType(MapPointType::NORMAL_MP)
{
	Init(pTargetKF, idx);
}
UVR_SLAM::MapPoint::MapPoint(cv::Mat _p3D, Frame* pTargetKF, int idx, cv::Mat _desc, MapPointType ntype)
	: p3D(_p3D), desc(_desc), mbNewMP(true), mbSeen(false), mnVisibleCount(0), mnMatchingCount(0), mnConnectedFrames(0), mfDepth(0.0), mnMapPointID(++nMapPointID), mbDelete(false), mObjectType(OBJECT_NONE), mnPlaneID(0), mnType(ntype)
{
	Init(pTargetKF, idx);
}
UVR_SLAM::MapPoint::~MapPoint(){}

void UVR_SLAM::MapPoint::Init(UVR_SLAM::Frame* pTargetKF, int idx) {
	cv::Mat PC = p3D - pTargetKF->GetCameraCenter();
	float dist = cv::norm(PC);
	int level = pTargetKF->mvKeyPoints[idx].octave;
	float levelScaleFactor = pTargetKF->mvScaleFactors[level];
	int nLevels = pTargetKF->mnScaleLevels;
	mfMaxDistance = dist*levelScaleFactor;
	mfMinDistance = mfMaxDistance / pTargetKF->mvScaleFactors[nLevels - 1];
}

int UVR_SLAM::MapPoint::GetMapPointID() {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	return mnMapPointID;
}

void UVR_SLAM::MapPoint::SetPlaneID(int nid) {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	mnPlaneID = nid;
}

int UVR_SLAM::MapPoint::GetPlaneID() {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	return mnPlaneID;
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
	std::unique_lock<std::mutex>(mMutexObjectType);
	mObjectType = nType;
}
UVR_SLAM::ObjectType  UVR_SLAM::MapPoint::GetObjectType(){
	std::unique_lock<std::mutex>(mMutexObjectType);
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
	for (auto iter = mmpFrames.begin(); iter != mmpFrames.end(); iter++) {
		UVR_SLAM::Frame* pF = iter->first;
		int idx = iter->second;
		pF->RemoveMP(idx);
	}
	//...std::cout << "Delete=" << mmpF;
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

void UVR_SLAM::MapPoint::SetFrameWindowIndex(int nIdx){
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	mnFrameWindowIndex = nIdx;
}

int UVR_SLAM::MapPoint::GetFrameWindowIndex(){
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	return mnFrameWindowIndex;
}

bool UVR_SLAM::MapPoint::Projection(cv::Point2f& _P2D, cv::Mat& _Pcam, cv::Mat R, cv::Mat t, cv::Mat K, int w, int h) {
	std::unique_lock<std::mutex> lockMP(mMutexMP);
	_Pcam = R*p3D + t;
	cv::Mat temp = K*_Pcam;
	_P2D = cv::Point2f(temp.at<float>(0) / temp.at<float>(2), temp.at<float>(1) / temp.at<float>(2));
	mfDepth = temp.at<float>(2);
	bool bres = false;
	if (mfDepth > 0.0f && (_P2D.x >= 0 && _P2D.x < w && _P2D.y >= 0 && _P2D.y < h)) {
		bres = true;
	}
	else {
		mbSeen = false;
		return false;
	}
	
	// Check distance is in the scale invariance region of the MapPoint
	float minDistance = 0.8*mfMinDistance;
	float maxDistance = 1.2*mfMaxDistance;

	cv::Mat mOw = -R.t()*t;
	cv::Mat PO = p3D - mOw;
	float dist = cv::norm(PO);
	if (dist<minDistance || dist>maxDistance)
		bres = false;
	mbSeen = bres;
	return bres;
}