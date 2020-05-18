#include <Plane.h>
#include <Frame.h>
#include <PlaneEstimator.h>

static int nWallPlaneID = 0;




/////////////////////////////////////////////////////////////
//PlaneProcessInformation
UVR_SLAM::PlaneProcessInformation::PlaneProcessInformation() : mpFrame(nullptr){
	//mpCeil(nullptr), mpFloor(nullptr)
}
UVR_SLAM::PlaneProcessInformation::PlaneProcessInformation(Frame* pF, PlaneInformation* pPlane):mpFrame(pF){
	//mpCeil(nullptr), mpFloor(nullptr)
	mpFrame = pF;
}
UVR_SLAM::PlaneProcessInformation::~PlaneProcessInformation(){}

void UVR_SLAM::PlaneProcessInformation::Calculate(){
	std::unique_lock<std::mutex> lock(mMutexProessor);
	
	auto pFloor = mmpPlanes[1];
	cv::Mat planeParam = pFloor->GetParam();
	
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
void UVR_SLAM::PlaneProcessInformation::SetReferenceFrame(Frame* pF) {
	mpFrame = pF;
}
UVR_SLAM::Frame* UVR_SLAM::PlaneProcessInformation::GetReferenceFrame() {
	return mpFrame;
}
void UVR_SLAM::PlaneProcessInformation::AddPlane(PlaneInformation* p, int type) {
	std::unique_lock<std::mutex> lock(mMutexPlanes);
	/*if (type == 1) {
		mpFloor = p;
	}
	else if (type == 2) {
		mpCeil = p;
	}*/
	mmpPlanes[type] = p;
}
UVR_SLAM::PlaneInformation* UVR_SLAM::PlaneProcessInformation::GetPlane(int type){
	std::unique_lock<std::mutex> lock(mMutexPlanes);
	auto findres = mmpPlanes.find(type);
	if (findres == mmpPlanes.end())
		return nullptr;
	else
		return findres->second;
	/*UVR_SLAM::PlaneInformation* res = nullptr;
	if (type == 1) {
		res = mpFloor;
	}
	else if (type == 2) {
		res = mpCeil;
	}
	return res;*/
}
std::map<int, UVR_SLAM::PlaneInformation*> UVR_SLAM::PlaneProcessInformation::GetPlanes() {
	std::unique_lock<std::mutex> lock(mMutexPlanes);
	/*std::vector<PlaneInformation*> temp;
	for (auto iter = mmpPlanes.begin(); iter != mmpPlanes.end(); iter++) {
		temp.push_back(iter->second);
	}*/
	return std::map<int, PlaneInformation*>(mmpPlanes.begin(), mmpPlanes.end());
}

//PlaneProcessInformation
/////////////////////////////////////////////////////////////

UVR_SLAM::Line::Line(Frame* pF, int w, cv::Point2f f, cv::Point2f t):from(f), to(t), mnPlaneID(-1){
	mpFrame = pF;
	
	//line equ : from : pt2, to : pt1
	cv::Point2f diff = from - to;
	float a = diff.y;
	float b = -diff.x;
	float c = diff.x*to.y - diff.y*to.x;
	mLineEqu = (cv::Mat_<float>(3, 1) << a,b,c);
	if (a < 0)
		mLineEqu *= -1.0;
	//unit vector 계산
	float dist = sqrt(a*a + b*b);
	mLineEqu /= dist;

	fromExt = CalcLinePoint(0.0);
	toExt = CalcLinePoint(w);

	//기울기
	if (b == 0) {
		mfSlope = 9999.0f;
	}
	else {
		mfSlope = a / b;
	}
}
UVR_SLAM::Line::~Line(){}
cv::Point2f UVR_SLAM::Line::CalcLinePoint(float y){
	float x = 0.0;
	if (mLineEqu.at<float>(0) != 0)
		x = (-mLineEqu.at<float>(2) - mLineEqu.at<float>(1)*y) / mLineEqu.at<float>(0);
	return cv::Point2f(x, y);
}
float UVR_SLAM::Line::CalcLineDistance(cv::Point2f pt) {
	cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	return abs(mLineEqu.dot(temp));
}

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
void UVR_SLAM::WallPlane::AddLine(Line* line, Frame* pF) {
	std::unique_lock<std::mutex> lock(mMutexLines);
	/*if (line->mnPlaneID > 0)
		return;*/
	if (mnPlaneID > 0)
		line->mnPlaneID = mnPlaneID;
	mvLines.push_back(line);
	mmpLines.insert(std::pair<Frame*, Line*>(pF, line));
}
size_t UVR_SLAM::WallPlane::GetSize(){
	std::unique_lock<std::mutex> lock(mMutexLines);
	return mvLines.size();
}
std::vector<UVR_SLAM::Line*> UVR_SLAM::WallPlane::GetLines(){
	std::unique_lock<std::mutex> lock(mMutexLines);
	return std::vector<UVR_SLAM::Line*>(mvLines.begin(), mvLines.end());
}

std::pair<std::multimap<UVR_SLAM::Frame*, UVR_SLAM::Line*>::iterator, std::multimap<UVR_SLAM::Frame*, UVR_SLAM::Line*>::iterator> UVR_SLAM::WallPlane::GetLines(UVR_SLAM::Frame* pF) {
	std::unique_lock<std::mutex> lock(mMutexLines);
	return mmpLines.equal_range(pF);
}

void UVR_SLAM::WallPlane::CreateWall(){
	mnPlaneID = ++nWallPlaneID;
	for (int i = 0; i < mvLines.size(); i++) {
		mvLines[i]->mnPlaneID = mnPlaneID;
	}
}

bool UVR_SLAM::WallPlane::isContain(Frame* pF) {
	std::unique_lock<std::mutex> lock(mMutexLines);
	return mmpLines.find(pF) != mmpLines.end();
}

//int  UVR_SLAM::WallPlane::GetNumFrames() {
//	std::unique_lock<std::mutex> lock(mMutexFrames);
//	return mspFrames.size();
//}
//void UVR_SLAM::WallPlane::AddFrame(Frame* pF){
//	if (!isContain(pF)){
//		std::unique_lock<std::mutex> lock(mMutexFrames);
//		mspFrames.insert(pF);
//	}
//}

int UVR_SLAM::WallPlane::GetRecentKeyFrameID() {
	std::unique_lock<std::mutex> lock(mMutexRecentKeyFrameID);
	return mnRecentKeyFrameID;
}
void UVR_SLAM::WallPlane::SetRecentKeyFrameID(int id) {
	std::unique_lock<std::mutex> lock(mMutexRecentKeyFrameID);
	mnRecentKeyFrameID = id;
}