#include <Plane.h>
#include <Frame.h>

static int nWallPlaneID = 0;

UVR_SLAM::Line::Line(Frame* pF, cv::Point2f f, cv::Point2f t):from(f), to(t), mnPlaneID(-1){
	mpFrame = pF;
}
UVR_SLAM::Line::~Line(){}

UVR_SLAM::WallPlane::WallPlane():mnPlaneID(-1), mnRecentKeyFrameID(-1){

}
UVR_SLAM::WallPlane::~WallPlane(){

}

//update param
void UVR_SLAM::WallPlane::SetParam(cv::Mat p) {
	std::unique_lock<std::mutex> lock(mMutexParam);
	param = p.clone();
}
cv::Mat UVR_SLAM::WallPlane::GetParam() {
	std::unique_lock<std::mutex> lock(mMutexParam);
	return param;
}
void UVR_SLAM::WallPlane::AddLine(Line* line) {
	std::unique_lock<std::mutex> lock(mMutexLiens);
	/*if (line->mnPlaneID > 0)
		return;*/
	if (mnPlaneID > 0)
		line->mnPlaneID = mnPlaneID;
	mvLines.push_back(line);
	//AddFrame(line->mpFrame);
	//line->mnPlaneID = mnPlaneID;
}
size_t UVR_SLAM::WallPlane::GetSize(){
	std::unique_lock<std::mutex> lock(mMutexLiens);
	return mvLines.size();
}
std::vector<UVR_SLAM::Line*> UVR_SLAM::WallPlane::GetLines(){
	std::unique_lock<std::mutex> lock(mMutexLiens);
	return std::vector<UVR_SLAM::Line*>(mvLines.begin(), mvLines.end());
}
void UVR_SLAM::WallPlane::CreateWall(){
	mnPlaneID = ++nWallPlaneID;
	for (int i = 0; i < mvLines.size(); i++) {
		mvLines[i]->mnPlaneID = mnPlaneID;
	}
}

bool UVR_SLAM::WallPlane::isContain(Frame* pF) {
	std::unique_lock<std::mutex> lock(mMutexFrames);
	return mspFrames.find(pF) != mspFrames.end();
}
int  UVR_SLAM::WallPlane::GetNumFrames() {
	std::unique_lock<std::mutex> lock(mMutexFrames);
	return mspFrames.size();
}
void UVR_SLAM::WallPlane::AddFrame(Frame* pF){
	if (!isContain(pF)){
		std::unique_lock<std::mutex> lock(mMutexFrames);
		mspFrames.insert(pF);
	}
}

int UVR_SLAM::WallPlane::GetRecentKeyFrameID() {
	std::unique_lock<std::mutex> lock(mMutexRecentKeyFrameID);
	return mnRecentKeyFrameID;
}
void UVR_SLAM::WallPlane::SetRecentKeyFrameID(int id) {
	std::unique_lock<std::mutex> lock(mMutexRecentKeyFrameID);
	mnRecentKeyFrameID = id;
}