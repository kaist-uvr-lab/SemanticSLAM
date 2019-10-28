#include <MapPoint.h>
#include <Frame.h>
#include <FrameWindow.h>
#include <SegmentationData.h>

static int nMapPointID = 0;

UVR_SLAM::MapPoint::MapPoint()
	:p3D(cv::Mat::zeros(3, 1, CV_32FC1)), mbNewMP(true), mbSeen(false), mnVisibleCount(0), mnMatchingCount(0), mnConnectedFrames(0), mfDepth(0.0), mbDelete(false), mObjectType(OBJECT_NONE)
{}
UVR_SLAM::MapPoint::MapPoint(cv::Mat _p3D, cv::Mat _desc)
:p3D(_p3D), desc(_desc), mbNewMP(true), mbSeen(false), mnVisibleCount(0), mnMatchingCount(0), mnConnectedFrames(0), mfDepth(0.0), mnMapPointID(++nMapPointID), mbDelete(false), mObjectType(OBJECT_NONE)
{}
UVR_SLAM::MapPoint::~MapPoint(){}

int UVR_SLAM::MapPoint::GetMapPointID() {
	std::unique_lock<std::mutex> lockMP(mMutexMPID);
	return mnMapPointID;
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
	desc = _desc.clone();
}
cv::Mat UVR_SLAM::MapPoint::GetDescriptor(){
	return desc;
}
bool UVR_SLAM::MapPoint::isSeen(){
	return mbSeen;
}

void UVR_SLAM::MapPoint::SetFrameWindowIndex(int nIdx){
	std::unique_lock<std::mutex> lockMP(mMutexFrameWindowIndex);
	mnFrameWindowIndex = nIdx;
}

int UVR_SLAM::MapPoint::GetFrameWindowIndex(){
	std::unique_lock<std::mutex> lockMP(mMutexFrameWindowIndex);
	return mnFrameWindowIndex;
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