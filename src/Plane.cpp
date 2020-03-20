#include <Plane.h>
#include <Frame.h>
#include <PlaneEstimator.h>

static int nWallPlaneID = 0;




/////////////////////////////////////////////////////////////
//PlaneProcessInformation
UVR_SLAM::PlaneProcessInformation::PlaneProcessInformation() {

}
UVR_SLAM::PlaneProcessInformation::PlaneProcessInformation(Frame* pF, PlaneInformation* pPlane) {
	mpFrame = pF;
	mpFloor = pPlane;
}
UVR_SLAM::PlaneProcessInformation::~PlaneProcessInformation(){}

void UVR_SLAM::PlaneProcessInformation::Calculate(){
	std::unique_lock<std::mutex> lock(mMutexProessor);
	
	cv::Mat planeParam = mpFloor->GetParam();
	
	cv::Mat R, t;
	mpFrame->GetPose(R, t);
	cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
	R.copyTo(T.rowRange(0, 3).colRange(0, 3));
	t.copyTo(T.col(3).rowRange(0, 3));

	invT = T.inv();
	invP = invT.t()*planeParam;
	invK = mpFrame->mK.inv();
}
void UVR_SLAM::PlaneProcessInformation::GetInformation(cv::Mat& pInvP, cv::Mat& pInvT, cv::Mat& pInvK) {
	std::unique_lock<std::mutex> lock(mMutexProessor);
	pInvP = invP.clone();
	pInvT = invT.clone();
	pInvK = invK.clone();
}
UVR_SLAM::PlaneInformation* UVR_SLAM::PlaneProcessInformation::GetFloorPlane() {
	return mpFloor;
}
//PlaneProcessInformation
/////////////////////////////////////////////////////////////




UVR_SLAM::Line::Line(Frame* pF, cv::Point2f f, cv::Point2f t):from(f), to(t), mnPlaneID(-1){
	mpFrame = pF;
}
UVR_SLAM::Line::~Line(){}

//3차원으로 가상의 포인트를 생성.y값은 0으로 해야 2차원의 선을 구할 수 있다.
void UVR_SLAM::Line::SetLinePts() {
	cv::Mat invP; cv::Mat invT; cv::Mat invK;
	mpFrame->mpPlaneInformation->GetInformation(invP, invT, invK);

	mvLinePts = cv::Mat::zeros(0, 3, CV_32FC1);

	std::unique_lock<std::mutex> lock(mMutexLinePts);
	cv::Point2f diffPt = to - from;
	
	int nPts = 30;
	float scale = 0.7;
	diffPt.x *= (scale / nPts);
	diffPt.y *= (scale / nPts);
	//std::cout <<"to::"<< to << ", " << from << std::endl;
	for (int i = 0; i < nPts; i++) {
		cv::Point2f pt(to.x - diffPt.x*i, to.y - diffPt.y*i);
		cv::Mat s = UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(pt, invP, invT, invK);
		//std::cout << i << " :: " << pt <<" "<<s.t()<< std::endl;
		cv::Mat temp = cv::Mat::zeros(1, 3, CV_32FC1);
		temp.at<float>(0) = s.at<float>(0);
		temp.at<float>(1) = s.at<float>(2);
		temp.at<float>(2) = 1.0;
		mvLinePts.push_back(temp);
	}

}
cv::Mat UVR_SLAM::Line::GetLinePts() {
	std::unique_lock<std::mutex> lock(mMutexLinePts);
	return mvLinePts.clone();//std::vector<cv::Point2f>(mvLinePts.begin(), mvLinePts.end());
}















/////////////////////////////////////////////////////////////////////////////////////////


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