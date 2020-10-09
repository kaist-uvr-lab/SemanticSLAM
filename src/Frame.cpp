//
// Created by UVR-KAIST on 2019-02-01.
//

#include <Frame.h>
#include <CandidatePoint.h>
#include <MapGrid.h>
#include <MatrixOperator.h>
#include <System.h>
#include <ORBextractor.h>
#include <Plane.h>
int UVR_SLAM::MatchInfo::nMaxMP = 500;
int UVR_SLAM::Frame::mnRadius = 7;
bool UVR_SLAM::Frame::mbInitialComputations = true;
float UVR_SLAM::Frame::cx, UVR_SLAM::Frame::cy, UVR_SLAM::Frame::fx, UVR_SLAM::Frame::fy, UVR_SLAM::Frame::invfx, UVR_SLAM::Frame::invfy;
float UVR_SLAM::Frame::mnMinX, UVR_SLAM::Frame::mnMinY, UVR_SLAM::Frame::mnMaxX, UVR_SLAM::Frame::mnMaxY;
float UVR_SLAM::Frame::mfGridElementWidthInv, UVR_SLAM::Frame::mfGridElementHeightInv;

static int nFrameID = 0;


UVR_SLAM::Frame::Frame(cv::Mat _src, int w, int h, cv::Mat K, double ts):mnWidth(w), mnHeight(h), mK(K), mnType(0), mnInliers(0), mnKeyFrameID(0), mnFuseFrameID(0), mnLocalBAID(0), mnFixedBAID(0), mnLocalMapFrameID(0), mnRecentTrackedFrameId(0),
mpPlaneInformation(nullptr),mvpPlanes(), bSegmented(false), mbMapping(false), mdTimestamp(ts)
{
	matOri = _src.clone();
	cv::cvtColor(matOri, matFrame, CV_RGBA2GRAY);
	matFrame.convertTo(matFrame, CV_8UC1);
	R = cv::Mat::eye(3, 3, CV_32FC1);
	t = cv::Mat::zeros(3, 1, CV_32FC1);
	////////////canny
	//cv::Mat filtered;
	//GaussianBlur(matFrame, filtered, cv::Size(5, 5), 0.0);
	//cv::Canny(filtered, mEdgeImg, 50, 200);
	//for (int y = 0; y < matFrame.rows; y++) {
	//	for (int x = 0; x < matFrame.cols; x++) {
	//		if (mEdgeImg.at<uchar>(y, x) > 0)
	//			mvEdgePts.push_back(cv::Point2f(x, y));
	//	}
	//}
	////////////canny
	SetFrameID();
}
UVR_SLAM::Frame::Frame(void *ptr, int id, int w, int h, cv::Mat K) :mnWidth(w), mnHeight(h), mK(K), mnType(0), mnInliers(0), mnKeyFrameID(0), mnFuseFrameID(0), mnLocalBAID(0), mnFixedBAID(0), mnLocalMapFrameID(0), mnRecentTrackedFrameId(0)
, mpPlaneInformation(nullptr), mvpPlanes(), bSegmented(false), mbMapping(false), mdTimestamp(0.0)
{
	cv::Mat tempImg = cv::Mat(h, w, CV_8UC4, ptr);
	matOri = tempImg.clone();
	cv::cvtColor(matOri, matFrame, CV_RGBA2GRAY);
	matFrame.convertTo(matFrame, CV_8UC1);
	R = cv::Mat::eye(3, 3, CV_32FC1);
	t = cv::Mat::zeros(3, 1, CV_32FC1);
	////////////canny
	//cv::Mat filtered;
	//GaussianBlur(matFrame, filtered, cv::Size(5, 5), 0.0);
	//cv::Canny(filtered, mEdgeImg, 50, 200);
	//for (int y = 0; y < matFrame.rows; y++) {
	//	for (int x = 0; x < matFrame.cols; x++) {
	//		if (mEdgeImg.at<uchar>(y, x) > 0)
	//			mvEdgePts.push_back(cv::Point2f(x, y));
	//	}
	//}
	////////////canny
	SetFrameID();
}

UVR_SLAM::Frame::Frame(void* ptr, int id, int w, int h, cv::Mat _R, cv::Mat _t, cv::Mat K) :mnWidth(w), mnHeight(h), mK(K), mnType(0), mnInliers(0), mnKeyFrameID(0), mnFuseFrameID(0), mnLocalBAID(0), mnFixedBAID(0), mnLocalMapFrameID(0), mnRecentTrackedFrameId(0)
, mpPlaneInformation(nullptr), mvpPlanes(), bSegmented(false), mbMapping(false), mdTimestamp(0.0)
{
	cv::Mat tempImg = cv::Mat(h, w, CV_8UC4, ptr);
	matOri = tempImg.clone();
	cv::cvtColor(matOri, matFrame, CV_RGBA2GRAY);
	matFrame.convertTo(matFrame, CV_8UC1);
	R = cv::Mat::eye(3, 3, CV_32FC1);
	t = cv::Mat::zeros(3, 1, CV_32FC1);
	SetFrameID();
}

UVR_SLAM::Frame::~Frame() {
	close();
}

void UVR_SLAM::Frame::SetRecentTrackedFrameID(int id) {
	std::unique_lock<std::mutex>(mMutexTrackedFrame);
	mnRecentTrackedFrameId = id;
}
int UVR_SLAM::Frame::GetRecentTrackedFrameID() {
	std::unique_lock<std::mutex>(mMutexTrackedFrame);
	return mnRecentTrackedFrameId;
}

void UVR_SLAM::Frame::SetFrameType(int n) {
	mnType = n;
}

void UVR_SLAM::Frame::SetFrameID() {
	mnFrameID = ++nFrameID;
}

void UVR_SLAM::Frame::SetKeyFrameID() {
	mnKeyFrameID = UVR_SLAM::System::nKeyFrameID++;
}
void UVR_SLAM::Frame::SetKeyFrameID(int n) {
	UVR_SLAM::System::nKeyFrameID = n;
	mnKeyFrameID = UVR_SLAM::System::nKeyFrameID++;
}
int UVR_SLAM::Frame::GetKeyFrameID() {
	return mnKeyFrameID;
}

void UVR_SLAM::Frame::SetInliers(int nInliers){
	std::unique_lock<std::mutex>(mMutexNumInliers);
	mnInliers = nInliers;
}
int UVR_SLAM::Frame::GetInliers() {
	std::unique_lock<std::mutex>(mMutexNumInliers);
	return mnInliers;
}

void UVR_SLAM::Frame::close() {
	/*
	deltaR.release();
	matFrame.release();
	matOri.release();
	matInlierDescriptor.release();
	matDescriptor.release();
	matMachedImage.release();
	mvKeyPoints.clear();
	mvkInliers.clear();

	std::vector<KeyPoint>().swap(mvKeyPoints);
	std::vector<KeyPoint>().swap(mvkInliers);

	mvbMPInliers.clear();
	std::vector<bool>().swap(mvbMPInliers);
	mvnMPMatchingIdx.clear();
	std::vector<int>().swap(mvnMPMatchingIdx);

	mvbCPInliers.clear();
	std::vector<bool>().swap(mvbCPInliers);
	mvnCPMatchingIdx.clear();
	std::vector<int>().swap(mvnCPMatchingIdx);

	mvbMPMatchingInlier.clear();
	std::vector<bool>().swap(mvbMPMatchingInlier);
	mvbCPMatchingInlier.clear();
	std::vector<bool>().swap(mvbCPMatchingInlier);

	mvScaleFactors.clear();
	std::vector<float>().swap(mvScaleFactors);
	mvInvScaleFactors.clear();
	std::vector<float>().swap(mvInvScaleFactors);
	mvLevelSigma2.clear();
	std::vector<float>().swap(mvLevelSigma2);
	mvInvLevelSigma2.clear();
	std::vector<float>().swap(mvInvLevelSigma2);

	mvTempKPs.clear();
	mspMPs.clear();
	mspKFs.clear();
	std::vector<cv::KeyPoint>().swap(mvTempKPs);
	std::set<UVR::MapPoint*>().swap(mspMPs);
	std::set<UVR::KeyFrame*>().swap(mspKFs);
	*/
}

void UVR_SLAM::Frame::SetPose(cv::Mat _R, cv::Mat _t) {
	std::unique_lock<std::mutex>(mMutexPose);
	R = _R.clone();
	t = _t.clone();
}

void UVR_SLAM::Frame::GetPose(cv::Mat&_R, cv::Mat& _t) {
	std::unique_lock<std::mutex>(mMutexPose);
	_R = R.clone();
	_t = t.clone();
}
cv::Mat UVR_SLAM::Frame::GetRotation() {
	std::unique_lock<std::mutex>(mMutexPose);
	return R.clone();
}
cv::Mat UVR_SLAM::Frame::GetTranslation() {
	std::unique_lock<std::mutex>(mMutexPose);
	return t.clone();
}

void UVR_SLAM::Frame::process(cv::Ptr<cv::Feature2D> detector) {
	detector->detectAndCompute(matFrame, cv::noArray(), mvKeyPoints, matDescriptor);
}

bool CheckKeyPointOverlap(cv::Mat& overlap, cv::Point2f pt, int r) {
	if (overlap.at<uchar>(pt) > 0) {
		return false;
	}
	circle(overlap, pt, r, cv::Scalar(255), -1);
	return true;
}
/////////////////////////////////

int UVR_SLAM::Frame::GetFrameID() {
	std::unique_lock<std::mutex> lockMP(mMutexFrame);
	return mnFrameID;
}
cv::Mat UVR_SLAM::Frame::GetFrame() {
	std::unique_lock<std::mutex> lockMP(mMutexFrame);
	return matFrame.clone();
}
cv::Mat UVR_SLAM::Frame::GetOriginalImage() {
	std::unique_lock<std::mutex> lockMP(mMutexFrame);
	return matOri.clone();
}



std::vector<UVR_SLAM::ObjectType> UVR_SLAM::Frame::GetObjectVector() {
	std::unique_lock<std::mutex> lockMP(mMutexObjectTypes);
	return std::vector<UVR_SLAM::ObjectType>(mvObjectTypes.begin(), mvObjectTypes.end());
}
void UVR_SLAM::Frame::SetObjectVector(std::vector<UVR_SLAM::ObjectType> vObjTypes){
	std::unique_lock<std::mutex> lockMP(mMutexObjectTypes);
	mvObjectTypes = std::vector<UVR_SLAM::ObjectType>(vObjTypes.begin(), vObjTypes.end());
}

void UVR_SLAM::Frame::SetObjectType(UVR_SLAM::ObjectType type, int idx){
	std::unique_lock<std::mutex> lockMP(mMutexObjectTypes);
	mvObjectTypes[idx] = type;
}
UVR_SLAM::ObjectType UVR_SLAM::Frame::GetObjectType(int idx){
	std::unique_lock<std::mutex> lockMP(mMutexObjectTypes);
	return mvObjectTypes[idx];
}

void UVR_SLAM::Frame::SetBoolSegmented(bool b) {
	std::unique_lock<std::mutex> lockMP(mMutexSegmented);
	bSegmented = b;
}
bool UVR_SLAM::Frame::isSegmented() {
	std::unique_lock<std::mutex> lockMP(mMutexSegmented);
	return bSegmented;
}

//UVR_SLAM::MapPoint* UVR_SLAM::Frame::GetMapPoint(int idx) {
//	std::unique_lock<std::mutex> lockMP(mMutexFrame);
//	return mvpMPs[idx];
//}
//void UVR_SLAM::Frame::SetMapPoint(UVR_SLAM::MapPoint* pMP, int idx) {
//	std::unique_lock<std::mutex> lockMP(mMutexFrame);
//	mvpMPs[idx] = pMP;
//}
//bool UVR_SLAM::Frame::GetBoolInlier(int idx) {
//	std::unique_lock<std::mutex> lockMP(mMutexFrame);
//	return mvbMPInliers[idx];
//}
//void UVR_SLAM::Frame::SetBoolInlier(bool flag, int idx) {
//	std::unique_lock<std::mutex> lockMP(mMutexFrame);
//	mvbMPInliers[idx] = flag;
//}


//void UVR_SLAM::Frame::Increase(){
//	std::unique_lock<std::mutex> lockMP(mMutexFrame);
//	mnInliers++;
//}
//void UVR_SLAM::Frame::Decrease(){
//	std::unique_lock<std::mutex> lockMP(mMutexFrame);
//	mnInliers--;
//}

int UVR_SLAM::Frame::GetNumInliers() {
	std::unique_lock<std::mutex> lockMP(mMutexNumInliers);
	return mnInliers;
}

bool UVR_SLAM::Frame::isInImage(float x, float y, float w)
{
	return (x >= w && x<=mnWidth-w-1 && y >= w && y<=mnHeight-w-1);
	//return (x >= mnMinX && x<mnMaxX && y >= mnMinY && y<mnMaxY);
}

cv::Point2f UVR_SLAM::Frame::Projection(cv::Mat w3D) {
	cv::Mat tempR, tempT;
	{
		std::unique_lock<std::mutex> lock(mMutexPose);
		tempR = R.clone();
		tempT = t.clone();
	}
	cv::Mat pCam = tempR*w3D + tempT;
	cv::Mat temp = mK*pCam;
	cv::Point2f p2D = cv::Point2f(temp.at<float>(0) / temp.at<float>(2), temp.at<float>(1) / temp.at<float>(2));
	return p2D;
}

void UVR_SLAM::Frame::SetDepthRange(float min, float max){
	std::unique_lock<std::mutex> lock(mMutexDepthRange);
	mfMaxDepth = max;
	mfMinDepth = min;
}
void UVR_SLAM::Frame::GetDepthRange(float& min, float& max) {
	std::unique_lock<std::mutex> lock(mMutexDepthRange);
	min = mfMinDepth;
	max = mfMaxDepth;
}

unsigned char UVR_SLAM::Frame::GetFrameType() {
	std::unique_lock<std::mutex>(mMutexType);
	return mnType;
}
void UVR_SLAM::Frame::TurnOnFlag(unsigned char opt) {
	std::unique_lock<std::mutex>(mMutexType);
	if (opt == UVR_SLAM::FLAG_KEY_FRAME) {
		SetKeyFrameID();
	}
	mnType |= opt;
}
void UVR_SLAM::Frame::TurnOnFlag(unsigned char opt, int n){
	std::unique_lock<std::mutex>(mMutexType);
	if (opt == UVR_SLAM::FLAG_KEY_FRAME) {
		SetKeyFrameID(n);
	}
	mnType |= opt;
}
void UVR_SLAM::Frame::TurnOffFlag(unsigned char opt){
	std::unique_lock<std::mutex>(mMutexType);
	mnType &= ~opt;
}

bool UVR_SLAM::Frame::CheckFrameType(unsigned char opt) {
	std::unique_lock<std::mutex>(mMutexType);
	unsigned char flag = mnType & opt;
	//std::cout << "flag=" <<(int)flag <<", "<<(int)mnType<< std::endl<<std::endl<<std::endl<<std::endl;
	return flag == opt;
}

int UVR_SLAM::Frame::TrackedMapPoints(int minObservation) {
	std::unique_lock<std::mutex> lock(mMutexFrame);
	int nPoints = 0;
	bool bCheckObs = minObservation>0;

	/*auto mvpDenseMPs = GetDenseVectors();

	for (int i = 0; i < mvpDenseMPs.size(); i++) {
		MapPoint* pMP = mvpDenseMPs[i];
		if (pMP) {
			if (pMP->isDeleted())
				continue;
			if (bCheckObs) {
				if (pMP->GetNumDensedFrames() >= minObservation)
					nPoints++;
			}
			else {
				nPoints++;
			}
		}
	}*/

	/*for (int i = 0; i < mvpMPs.size(); i++) {
		MapPoint* pMP = mvpMPs[i];
		if (pMP) {
			if (pMP->isDeleted())
				continue;
			if (bCheckObs) {
				if (pMP->GetNumConnectedFrames() >= minObservation)
					nPoints++;
			}else{
				nPoints++;
			}
		}
	}*/
	return nPoints;
}

//Ow2가 현재 함수를 콜한 쪽이며, neighbor keyframe, Ow1이 현재 키프레임
//메디안 뎁스는 neighbor에서 계산함. 이 함수는 변경 가능함.
//현재 키프레임에서 이전 프레임을 불러오는 형태.
//즉 이걸 키프레임 추가 과정에서 쓸려면
//this가 curr frame
//target = prev keyframe
bool UVR_SLAM::Frame::CheckBaseLine(UVR_SLAM::Frame* pTargetKF) {
	
	cv::Mat Ow1 = this->GetCameraCenter();
	cv::Mat Ow2 = pTargetKF->GetCameraCenter();

	cv::Mat vBaseline = Ow2 - Ow1;
	float baseline = cv::norm(vBaseline);

	float medianDepthKF2 = pTargetKF->GetSceneMedianDepth();
	if(medianDepthKF2 < 0.0)
		return false;

	float ratioBaselineDepth = baseline / medianDepthKF2;

	if (ratioBaselineDepth<0.01)
		return false;

	return true;
}

//두 키프레임의 베이스라인을 계산할 때 이용 됨.
bool UVR_SLAM::Frame::ComputeSceneMedianDepth(std::vector<UVR_SLAM::MapPoint*> vpMPs, cv::Mat R, cv::Mat t, float& fMedianDepth)
{
	std::vector<float> vDepths;
	cv::Mat Rcw2 = R.row(2);
	Rcw2 = Rcw2.t();
	float zcw = t.at<float>(2);

	for (int i = 0; i < vpMPs.size(); i++)
	{
		UVR_SLAM::MapPoint* pMP = vpMPs[i];
		if (!pMP) {
			continue;
		}
		if (pMP->isDeleted())
			continue;
		cv::Mat x3Dw = pMP->GetWorldPos();
		float z = (float)Rcw2.dot(x3Dw) + zcw;
		vDepths.push_back(z);
	}
	
	if (vDepths.size() == 0)
		return false;
	int nidx = vDepths.size() / 2;
	std::nth_element(vDepths.begin(), vDepths.begin() + nidx, vDepths.end());
	fMedianDepth = vDepths[(nidx)];
	return true;
}

////20.09.05 수정 필요.
void UVR_SLAM::Frame::ComputeSceneMedianDepth()
{
	/*std::vector<float> vDepths;
	cv::Mat Rcw2 = R.row(2);
	Rcw2 = Rcw2.t();
	float zcw = t.at<float>(2);
	auto vpMPs = mpMatchInfo->GetMatchingMPs();
	for (int i = 0; i < vpMPs.size(); i++)
	{
		UVR_SLAM::MapPoint* pMP = vpMPs[i];
		if (!pMP || pMP->isDeleted()) {
			continue;
		}
		cv::Mat x3Dw = pMP->GetWorldPos();
		float z = (float)Rcw2.dot(x3Dw) + zcw;
		vDepths.push_back(z);
	}
	if (vDepths.size() == 0){
		mfMedianDepth = -1.0;
		return;
	}
	int nidx = vDepths.size() / 2;
	std::nth_element(vDepths.begin(), vDepths.begin() + nidx, vDepths.end());
	{
		std::unique_lock<std::mutex> lockMP(mMutexMedianDepth);
		mfMedianDepth = vDepths[nidx];
	}*/
}

float UVR_SLAM::Frame::GetSceneMedianDepth() {
	std::unique_lock<std::mutex> lockMP(mMutexMedianDepth);
	return mfMedianDepth;
}

cv::Mat UVR_SLAM::Frame::GetCameraCenter() {
	std::unique_lock<std::mutex> lockMP(mMutexPose);
	return -R.t()*t;
}

void UVR_SLAM::Frame::SetBoolMapping(bool b) {
	std::unique_lock<std::mutex> lockMP(mMutexMapping);
	mbMapping = b;
}
bool UVR_SLAM::Frame::GetBoolMapping(){
	std::unique_lock<std::mutex> lockMP(mMutexMapping);
	return mbMapping;
}

void UVR_SLAM::Frame::AddKF(UVR_SLAM::Frame* pKF, int weight){
	mmpConnectedKFs.insert(std::make_pair(weight,pKF));
}
void UVR_SLAM::Frame::RemoveKF(UVR_SLAM::Frame* pKF, int weight){
	//mmpConnectedKFs.erase(pKF);
	auto range = mmpConnectedKFs.equal_range(weight);
	for (auto iter = range.first; iter != range.second; iter++) {
		UVR_SLAM::Frame* pKFi = iter->second;
		if (pKFi == pKF) {
			mmpConnectedKFs.erase(iter);
			return;
		}
	}
}
std::vector<UVR_SLAM::Frame*> UVR_SLAM::Frame::GetConnectedKFs(int n){
	//return std::vector<UVR_SLAM::Frame*>(mmpConnectedKFs.begin(), mmpConnectedKFs.end());
	std::vector<UVR_SLAM::Frame*> tempKFs;
	for (std::multimap<int, UVR_SLAM::Frame*, std::greater<int>>::iterator iter = mmpConnectedKFs.begin(); iter != mmpConnectedKFs.end(); iter++) {
		UVR_SLAM::Frame* pKFi = iter->second;
		tempKFs.push_back(pKFi);
	}
	if (n == 0 || tempKFs.size() < n) {
		return std::vector<UVR_SLAM::Frame*>(tempKFs.begin(), tempKFs.end());
	}
	return std::vector<UVR_SLAM::Frame*>(tempKFs.begin(), tempKFs.begin() + n);
}

std::multimap<int, UVR_SLAM::Frame*, std::greater<int>> UVR_SLAM::Frame::GetConnectedKFsWithWeight() {
	/*std::multimap<int, UVR_SLAM::Frame*> tempKFs;
	for (std::multimap<int, UVR_SLAM::Frame*, std::greater<int>>::iterator iter = mmpConnectedKFs.begin(); iter != mmpConnectedKFs.end(); iter++) {
		UVR_SLAM::Frame* pKFi = iter->second;
		tempKFs.insert(std::make_pair(iter->first, iter->second));
	}*/
	return std::multimap<int, UVR_SLAM::Frame*, std::greater<int>>(mmpConnectedKFs.begin(), mmpConnectedKFs.end());
}

void UVR_SLAM::Frame::SetLines(std::vector<Line*> lines) {
	std::unique_lock<std::mutex> lock(mMutexLines);
	mvLines = std::vector<Line*>(lines.begin(), lines.end());
}
std::vector<UVR_SLAM::Line*> UVR_SLAM::Frame::Getlines() {
	std::unique_lock<std::mutex> lock(mMutexLines);
	return std::vector<Line*>(mvLines.begin(), mvLines.end());
}

//std::vector<UVR_SLAM::Frame*> UVR_SLAM::Frame::GetConnectedKFs(int n) {
//	auto mvpKFs = GetConnectedKFs();
//	if (mvpKFs.size() < n)
//		return mvpKFs;
//	return std::vector<UVR_SLAM::Frame*>(mvpKFs.begin(), mvpKFs.begin()+n);
//}
/////////////////////////////////
fbow::fBow UVR_SLAM::Frame::GetBowVec() {
	return mBowVec;
}
void UVR_SLAM::Frame::SetBowVec(fbow::Vocabulary* pfvoc) {
	mBowVec = pfvoc->transform(matDescriptor);
}

double UVR_SLAM::Frame::Score(UVR_SLAM::Frame* pF) {
	return fbow::fBow::score(mBowVec, pF->GetBowVec());
}


///////////////////////////////
void UVR_SLAM::Frame::Init(ORBextractor* _e, cv::Mat _k, cv::Mat _d)
{

	mpORBextractor = _e;
	mnScaleLevels = mpORBextractor->GetLevels();
	mfScaleFactor = mpORBextractor->GetScaleFactor();
	mfLogScaleFactor = log(mfScaleFactor);
	mvScaleFactors = mpORBextractor->GetScaleFactors();
	mvInvScaleFactors = mpORBextractor->GetInverseScaleFactors();
	mvLevelSigma2 = mpORBextractor->GetScaleSigmaSquares();
	mvInvLevelSigma2 = mpORBextractor->GetInverseScaleSigmaSquares();

	mK = _k.clone();
	mDistCoef = _d.clone();

	//tempDesc와 tempKPs는 이미지에서 겹치는 키포인트를 제거하기 위함.
	//ExtractORB(matFrame, mvKeyPoints, matDescriptor);
	//////여기에서 중복되는 키포인트들 제거하기
	cv::Mat tempDesc;
	{
		ExtractORB(matFrame, mvTempKPs, tempDesc);
		matDescriptor = cv::Mat::zeros(0, tempDesc.cols, tempDesc.type());
	}

	cv::Mat overlap = cv::Mat::zeros(matFrame.size(), CV_8UC1);
	for (int i = 0; i < mvTempKPs.size(); i++) {

		if (!CheckKeyPointOverlap(overlap, mvTempKPs[i].pt, 3)) {
			continue;
		}

		mvKeyPoints.push_back(mvTempKPs[i]);
		matDescriptor.push_back(tempDesc.row(i));
		//200410 추가
		mvnOctaves.push_back(mvTempKPs[i].octave);
		mvPts.push_back(mvTempKPs[i].pt);
		//200410 추가
	}
	////여기에서 중복되는 키포인트들 제거하기

	if (mvKeyPoints.empty())
		return;

	if (mbInitialComputations)
	{
		ComputeImageBounds(matFrame);

		mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
		mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

		fx = mK.at<float>(0, 0);
		fy = mK.at<float>(1, 1);
		cx = mK.at<float>(0, 2);
		cy = mK.at<float>(1, 2);
		invfx = 1.0f / fx;
		invfy = 1.0f / fy;

		mbInitialComputations = false;
	}
	UndistortKeyPoints();

	//에러나면 풀어야 함
	//AssignFeaturesToGrid();

	//임시로 키포인트 복사
	mvKeyPoints = mvKeyPointsUn;
	//mvpMPs 초기화
	//cv::undistort(matOri, undistorted, mK, mDistCoef);
	
	//////////canny
	//edge는 setkeyframe에서 추가.
	//canny는 이전에 돌리고 엣지 포인트만 여기서 추가하기.(0704)
	//cv::Mat filtered;
	//GaussianBlur(matFrame, filtered, cv::Size(5, 5), 0.0);
	//cv::Canny(filtered, mEdgeImg, 50, 200);//150
	DetectEdge();
	//////////canny

	/*mvpMPs = std::vector<UVR_SLAM::MapPoint*>(mvKeyPoints.size(), nullptr);
	mvbMPInliers = std::vector<bool>(mvKeyPoints.size(), false);
	mvObjectTypes = std::vector<ObjectType>(mvKeyPoints.size(), OBJECT_NONE);*/

	//mvMapObjects = std::vector<std::multimap<ObjectType, int, std::greater<int>>>(mvKeyPoints.size());
	//파트별 매칭을 위한 것.
	/*mWallDescriptor = cv::Mat::zeros(0, matDescriptor.cols, matDescriptor.type());
	mObjectDescriptor = cv::Mat::zeros(0, matDescriptor.cols, matDescriptor.type());
	mPlaneDescriptor = cv::Mat::zeros(0, matDescriptor.cols, matDescriptor.type());
	mLabelStatus = cv::Mat::zeros(mvKeyPoints.size(), 1, CV_8UC1);*/
}

void UVR_SLAM::Frame::DetectEdge() {
	cv::Mat filtered;
	GaussianBlur(matFrame, filtered, cv::Size(5, 5), 0.0);
	cv::Canny(filtered, mEdgeImg, 50, 200);//150
	
	for (int y = 0; y < matFrame.rows; y++) {
		for (int x = 0; x < matFrame.cols; x++) {
			if (mEdgeImg.at<uchar>(y, x) > 0)
				mvEdgePts.push_back(cv::Point2f(x, y));
		}
	}
}

void UVR_SLAM::Frame::ExtractORB(const cv::Mat &im, std::vector<cv::KeyPoint>& vKPs, cv::Mat& desc)
{
	(*mpORBextractor)(im, cv::Mat(), vKPs, desc);
}
void UVR_SLAM::Frame::UndistortKeyPoints()
{

	int N = mvKeyPoints.size();

	if (mDistCoef.at<float>(0) == 0.0)
	{
		mvKeyPointsUn = mvKeyPoints;
		return;
	}

	// Fill matrix with points
	cv::Mat mat(N, 2, CV_32F);
	for (int i = 0; i<N; i++)
	{
		mat.at<float>(i, 0) = mvKeyPoints[i].pt.x;
		mat.at<float>(i, 1) = mvKeyPoints[i].pt.y;
	}

	// Undistort points
	mat = mat.reshape(2);
	cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
	mat = mat.reshape(1);

	// Fill undistorted keypoint vector
	mvKeyPointsUn.resize(N);
	for (int i = 0; i<N; i++)
	{
		cv::KeyPoint kp = mvKeyPoints[i];
		kp.pt.x = mat.at<float>(i, 0);
		kp.pt.y = mat.at<float>(i, 1);
		mvKeyPointsUn[i] = kp;
	}
}
void UVR_SLAM::Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
	if (mDistCoef.at<float>(0) != 0.0)
	{
		cv::Mat mat(4, 2, CV_32F);
		mat.at<float>(0, 0) = 0.0; mat.at<float>(0, 1) = 0.0;
		mat.at<float>(1, 0) = imLeft.cols; mat.at<float>(1, 1) = 0.0;
		mat.at<float>(2, 0) = 0.0; mat.at<float>(2, 1) = imLeft.rows;
		mat.at<float>(3, 0) = imLeft.cols; mat.at<float>(3, 1) = imLeft.rows;

		// Undistort corners
		mat = mat.reshape(2);
		cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
		mat = mat.reshape(1);

		mnMinX = cv::min(mat.at<float>(0, 0), mat.at<float>(2, 0));
		mnMaxX = cv::max(mat.at<float>(1, 0), mat.at<float>(3, 0));
		mnMinY = cv::min(mat.at<float>(0, 1), mat.at<float>(1, 1));
		mnMaxY = cv::max(mat.at<float>(2, 1), mat.at<float>(3, 1));

	}
	else
	{
		mnMinX = 0.0f;
		mnMaxX = imLeft.cols;
		mnMinY = 0.0f;
		mnMaxY = imLeft.rows;
	}
}
void UVR_SLAM::Frame::AssignFeaturesToGrid()
{
	int N = mvKeyPoints.size();
	int nReserve = 0.5f*N / (FRAME_GRID_COLS*FRAME_GRID_ROWS);

	for (unsigned int i = 0; i<FRAME_GRID_COLS; i++)
		for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++) {
			mGrid[i][j].clear();
			mGrid[i][j].reserve(nReserve);
		}

	//for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
	//for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
	//mGrid[i][j].reserve(nReserve);

	for (int i = 0; i<N; i++)
	{
		const cv::KeyPoint &kp = mvKeyPointsUn[i];

		int nGridPosX, nGridPosY;
		if (PosInGrid(kp, nGridPosX, nGridPosY))
			mGrid[nGridPosX][nGridPosY].push_back(i);
	}
}
bool UVR_SLAM::Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
	posX = round((kp.pt.x - mnMinX)*mfGridElementWidthInv);
	posY = round((kp.pt.y - mnMinY)*mfGridElementHeightInv);

	//Keypoint's coordinates are undistorted, which could cause to go out of the image
	if (posX<0 || posX >= FRAME_GRID_COLS || posY<0 || posY >= FRAME_GRID_ROWS)
		return false;

	return true;
}
std::vector<size_t> UVR_SLAM::Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel)
{
	std::vector<size_t> vIndices;
	vIndices.reserve(mvKeyPointsUn.size());

	const int nMinCellX = cv::max(0, (int)floor((x - mnMinX - r)*mfGridElementWidthInv));
	if (nMinCellX >= FRAME_GRID_COLS)
		return vIndices;

	const int nMaxCellX = cv::min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + r)*mfGridElementWidthInv));
	if (nMaxCellX<0)
		return vIndices;

	const int nMinCellY = cv::max(0, (int)floor((y - mnMinY - r)*mfGridElementHeightInv));
	if (nMinCellY >= FRAME_GRID_ROWS)
		return vIndices;

	const int nMaxCellY = cv::min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + r)*mfGridElementHeightInv));
	if (nMaxCellY<0)
		return vIndices;

	const bool bCheckLevels = (minLevel>0) || (maxLevel >= 0);

	for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
	{
		for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
		{
			const std::vector<size_t> vCell = mGrid[ix][iy];
			if (vCell.empty())
				continue;

			for (size_t j = 0, jend = vCell.size(); j<jend; j++)
			{
				const cv::KeyPoint &kpUn = mvKeyPointsUn[vCell[j]];
				if (bCheckLevels)
				{
					if (kpUn.octave<minLevel)
						continue;
					if (maxLevel >= 0)
						if (kpUn.octave>maxLevel)
							continue;
				}

				const float distx = kpUn.pt.x - x;
				const float disty = kpUn.pt.y - y;

				if (fabs(distx)<r && fabs(disty)<r)
					vIndices.push_back(vCell[j]);
			}
		}
	}

	return vIndices;
}

bool UVR_SLAM::Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
	//pMP->mbTrackInView = false;

	// 3D in absolute coordinates
	cv::Mat P = pMP->GetWorldPos();

	// 3D in camera coordinates
	cv::Mat R, t;
	GetPose(R, t);
	cv::Mat Ow = GetCameraCenter();
	const cv::Mat Pc = R*P + t;
	const float &PcX = Pc.at<float>(0);
	const float &PcY = Pc.at<float>(1);
	const float &PcZ = Pc.at<float>(2);

	// Check positive depth
	if (PcZ<0.0f)
		return false;

	// Project in image and check it is not outside
	const float invz = 1.0f / PcZ;
	const float u = fx*PcX*invz + cx;
	const float v = fy*PcY*invz + cy;

	if (u<mnMinX || u>mnMaxX)
		return false;
	if (v<mnMinY || v>mnMaxY)
		return false;

	// Check distance is in the scale invariance region of the MapPoint
	/*const float maxDistance = pMP->GetMaxDistance();
	const float minDistance = pMP->GetMinDistance();
	if (dist<minDistance || dist>maxDistance)
		return false;*/

	/////////////////Viewing angle
	//const cv::Mat PO = P - Ow;
	//const float dist = cv::norm(PO);

	//// Check viewing angle
	//cv::Mat Pn = pMP->GetNormal();

	//const float viewCos = PO.dot(Pn) / dist;

	//if (viewCos<viewingCosLimit)
	//	return false;
	/////////////////Viewing angle

	//// Predict scale in the image
	//const int nPredictedLevel = pMP->PredictScale(dist, this);
	//// Data used by the tracking
	//pMP->mbTrackInView = true;
	//pMP->mTrackProjX = u;
	//pMP->mTrackProjXR = u - mbf*invz;
	//pMP->mTrackProjY = v;
	//pMP->mnTrackScaleLevel = nPredictedLevel;
	//pMP->mTrackViewCos = viewCos;

	return true;
}

bool UVR_SLAM::Frame::isInFrustum(MapGrid *pMG, float viewingCosLimit) {

	// 3D in absolute coordinates
	cv::Mat P = pMG->Xw.clone();

	// 3D in camera coordinates
	cv::Mat R, t;
	GetPose(R, t);
	cv::Mat Ow = GetCameraCenter();
	const cv::Mat Pc = R*P + t;
	const float &PcX = Pc.at<float>(0);
	const float &PcY = Pc.at<float>(1);
	const float &PcZ = Pc.at<float>(2);

	// Check positive depth
	if (PcZ<0.0f)
		return false;

	// Project in image and check it is not outside
	const float invz = 1.0f / PcZ;
	const float u = fx*PcX*invz + cx;
	const float v = fy*PcY*invz + cy;

	if (u<mnMinX || u>mnMaxX)
		return false;
	if (v<mnMinY || v>mnMaxY)
		return false;

	
	/////////////////Viewing angle
	//const cv::Mat PO = P - Ow;
	//const float dist = cv::norm(PO);

	//// Check viewing angle
	//cv::Mat Pn = pMP->GetNormal();

	//const float viewCos = PO.dot(Pn) / dist;

	//if (viewCos<viewingCosLimit)
	//	return false;
	/////////////////Viewing angle
	
	return true;
}

std::vector<cv::Mat> UVR_SLAM::Frame::GetWallParams() {
	std::unique_lock<std::mutex>(mMutexWallParams);
	return std::vector<cv::Mat>(mvWallParams.begin(), mvWallParams.end());
}
void UVR_SLAM::Frame::SetWallParams(std::vector<cv::Mat> vParams){
	std::unique_lock<std::mutex>(mMutexWallParams);
	mvWallParams = std::vector<cv::Mat>(vParams.begin(), vParams.end());
}

void UVR_SLAM::Frame::Reset() {
	//mvObjectTypes = std::vector<ObjectType>(mvKeyPoints.size(), OBJECT_NONE);
	mnInliers = 0;
	mmpConnectedKFs.clear();
}

float UVR_SLAM::Frame::CalcDiffAngleAxis(UVR_SLAM::Frame* pF) {
	/*cv::Mat DirZ1 = R.row(2);
	cv::Mat DirZ2 = pF->GetRotation().row(2);
	float dist1 = sqrt(DirZ1.dot(DirZ1));
	float dist2 = sqrt(DirZ2.dot(DirZ2));
	float val = DirZ1.dot(DirZ2);
	val = acos(val / (dist1*dist2))*UVR_SLAM::MatrixOperator::rad2deg;*/
	cv::Mat R2 = pF->GetRotation();
	cv::Mat temp =  UVR_SLAM::MatrixOperator::LOG(R.t()*R2);
	return sqrt(temp.dot(temp))*UVR_SLAM::MatrixOperator::rad2deg;
}

//////////////matchinfo
UVR_SLAM::MatchInfo::MatchInfo():mnMatch(0), mfLowQualityRatio(0.0){}
UVR_SLAM::MatchInfo::MatchInfo(Frame* pRef, Frame* pTarget, int w, int h):mnHeight(h), mnWidth(w), mnMatch(0), mfLowQualityRatio(0.0){
	mpTargetFrame = pTarget;
	mpRefFrame = pRef;
	mMapCP = cv::Mat::zeros(h, w, CV_16SC1);
}
UVR_SLAM::MatchInfo::~MatchInfo(){}

//트래킹
int UVR_SLAM::MatchInfo::CheckOpticalPointOverlap(int radius, int margin, cv::Point2f pt) {
	//range option도 필요할 듯
	if (pt.x < margin || pt.x >= mnWidth - margin || pt.y < margin || pt.y >= mnHeight - margin) {
		return -1;
	}
	int res = mMapCP.at<ushort>(pt)-1;
	return res;
	/*if (mMapCP.at<ushort>(pt) > 0) {
		return ;
	}
	return true;*/
}
bool UVR_SLAM::MatchInfo::CheckOpticalPointOverlap(cv::Mat& overlap, int radius, int margin, cv::Point2f pt) {
	//range option도 필요할 듯
	if (pt.x < margin || pt.x >= mnWidth - margin || pt.y < margin || pt.y >= mnHeight - margin) {
		return false;
	}
	if (overlap.at<uchar>(pt) > 0) {
		return false;
	}
	//overlap.at<uchar>(pt) = 255;
	//circle(overlap, pt, radius, cv::Scalar(255), -1);
	return true;
}

void UVR_SLAM::MatchInfo::SetLabel() {
	auto labelMat = mpRefFrame->matLabeled.clone();
	std::vector<CandidatePoint*> vpCPs;
	auto vPTs = mpRefFrame->mpMatchInfo->GetMatchingPtsMapping(vpCPs);
	for (int i = 0; i < vPTs.size(); i++) {
		auto pCPi = vpCPs[i];
		auto pt = vPTs[i];
		int label = labelMat.at<uchar>(pt.y / 2, pt.x / 2);
		//mvObjectLabels[i] = label1;
		pCPi->SetLabel(label);
		auto pMPi = pCPi->GetMP();
		if (pMPi)
			pMPi->SetLabel(label);
	}
}

//새로운 맵포인트를 생성하기 위한 키포인트를 생성.
void UVR_SLAM::MatchInfo::SetMatchingPoints() {

	/*int N = this->GetNumCPs.size();
	if (N > 1000)
		return;*/
	int radius = 5;
	int nMax = 100; //둘다 하면 500
	int nIncEdge = mpRefFrame->mvEdgePts.size() / nMax;
	int nIncORB = mpRefFrame->mvPts.size() / nMax;

	if (nIncEdge == 0)
		nIncEdge = 1;
	if (nIncORB == 0)
		nIncORB = 1;

	for (int i = 0; i < mpRefFrame->mvEdgePts.size(); i += nIncEdge) {
		auto pt = mpRefFrame->mvEdgePts[i];
		if (CheckOpticalPointOverlap(1, 10, pt)>-1) {
			continue;
		}
		auto pCP = new UVR_SLAM::CandidatePoint(this);
		int idx = this->AddCP(pCP, pt);
		pCP->ConnectFrame(this, idx);
	}
	for (int i = 0; i < mpRefFrame->mvPts.size(); i+= nIncORB) {
		auto pt = mpRefFrame->mvPts[i];
		if (CheckOpticalPointOverlap(1, 10, pt)>-1) {
			continue;
		}
		auto pCP = new UVR_SLAM::CandidatePoint(this, mpRefFrame->mvnOctaves[i]);
		int idx = this->AddCP(pCP, pt);
		pCP->ConnectFrame(this, idx);
	}
	
}

void UVR_SLAM::MatchInfo::AddMP() {
	std::unique_lock<std::mutex>(mMutexData);
	mnMatch++;
}
void UVR_SLAM::MatchInfo::RemoveMP() {
	std::unique_lock<std::mutex>(mMutexData);
	mnMatch--;
}

////////20.09.05 수정 필요
int UVR_SLAM::MatchInfo::GetNumMapPoints() {
	std::unique_lock<std::mutex>(mMutexData);
	return mnMatch;
}
int UVR_SLAM::MatchInfo::GetNumCPs() {
	std::unique_lock<std::mutex>(mMutexCPs);
	return mvpMatchingCPs.size();
}

int UVR_SLAM::MatchInfo::AddCP(CandidatePoint* pCP, cv::Point2f pt){
	std::unique_lock<std::mutex>(mMutexCPs);
	int res = mvpMatchingCPs.size();
	mvpMatchingCPs.push_back(pCP);
	mvMatchingPts.push_back(pt);
	cv::circle(mMapCP, pt, Frame::mnRadius, cv::Scalar(res+1), -1);
	return res;
}
////이것은 사용이 안될 수도 있음.
void UVR_SLAM::MatchInfo::RemoveCP(int idx){
	std::unique_lock<std::mutex>(mMutexCPs);
	cv::circle(mMapCP, mvMatchingPts[idx], Frame::mnRadius, cv::Scalar(-1), -1);
	mvpMatchingCPs[idx] = nullptr;	
}
void UVR_SLAM::MatchInfo::ReplaceCP(CandidatePoint* pCP, int idx) {
	std::unique_lock<std::mutex>(mMutexCPs);
	mvpMatchingCPs[idx] = pCP;
}
UVR_SLAM::CandidatePoint* UVR_SLAM::MatchInfo::GetCP(int idx){
	std::unique_lock<std::mutex>(mMutexCPs);
	return mvpMatchingCPs[idx];
}
cv::Point2f UVR_SLAM::MatchInfo::GetPt(int idx) {
	std::unique_lock<std::mutex>(mMutexCPs);
	return mvMatchingPts[idx];
}

void UVR_SLAM::MatchInfo::ConnectAll() {
	int N;
	{
		std::unique_lock<std::mutex>(mMutexCPs);
		N = mvpMatchingCPs.size();
	}
	int nCurrID = this->mpRefFrame->GetKeyFrameID();
	for (int i = 0; i < N; i++) {
		
		auto pCPi = mvpMatchingCPs[i];
		int idx = pCPi->GetPointIndexInFrame(this);
		if (idx != -1)
			continue;
		pCPi->ConnectFrame(this, i);
		auto pMPi = pCPi->GetMP();
		if (!pMPi || !pMPi->GetQuality() || pMPi->isDeleted())
			continue;
		this->AddMP();
		pMPi->ConnectFrame(this, i);
		pMPi->IncreaseVisible();
		pMPi->IncreaseFound();
		pMPi->SetLastVisibleFrame(std::move(nCurrID));
		pMPi->SetLastSuccessFrame(std::move(nCurrID));
	}
}
void UVR_SLAM::MatchInfo::DisconnectAll() {
	int N;
	{
		std::unique_lock<std::mutex>(mMutexCPs);
		N = mvpMatchingCPs.size();
	}
	for (int i = 0; i < N; i++) {
		auto pCPi = mvpMatchingCPs[i];
		if (!pCPi)
			continue;
		//pCPi->DisconnectFrame(this);
		auto pMPi = pCPi->GetMP();
		if (!pMPi || pMPi->isDeleted())
			continue;
		this->RemoveMP();
		pMPi->DisconnectFrame(this);
	}
}
bool UVR_SLAM::MatchInfo::UpdateFrameQuality() {

	int N;
	{
		std::unique_lock<std::mutex>(mMutexCPs);
		N = mvpMatchingCPs.size();
	}
	int nMP = 0;
	int nLow = 0;
	for (int i = 0; i < N; i++) {
		auto pCPi = mvpMatchingCPs[i];
		
		auto pMPi = pCPi->GetMP();
		if (pMPi && !pMPi->isDeleted()){
			pMPi->ComputeQuality();
			if (!pMPi->GetQuality()){
				pMPi->Delete();
				nLow++;
			}else
				nMP++;
		}
	}
	bool b1 = false;//mfLowQualityRatio > 0.4;
	bool b2 = nMP < 200;
	/*if(nMP < 100){
		std::cout <<"ID = "<<this->mpRefFrame->GetKeyFrameID()<< "FrameQuality = " << nMP << "+" << nLow << "=" << N << std::endl;
		cv::Mat img = this->mpRefFrame->GetOriginalImage();
		cv::imshow("low quality", img);
		waitKey(1);
	}*/
	//b3은 이전 프레임과 비교시 차이가 갑자기 클 때로
	//전체 MP 수, quality가 안좋은 애들은 이미 여기에 존재하지를 못함. 
	//std::cout << "FrameQuality = " << nMP << "+" <<nLow<<"="<< N << std::endl;
	return b1 || b2;
}

int UVR_SLAM::MatchInfo::GetMatchingPtsTracking(std::vector<UVR_SLAM::CandidatePoint*>& vpCPs, std::vector<UVR_SLAM::MapPoint*>& vpMPs, std::vector<cv::Point2f>& vPTs) {
	
	//std::vector<cv::Point2f> res;
	//std::cout << "MatchInfo::CP::" << mvpMatchingCPs.size() << std::endl;
	int nres;
	{
		std::unique_lock<std::mutex>(mMutexCPs);
		nres = mvpMatchingCPs.size();

		for (int i = 0; i < nres; i++) {
			auto pCPi = mvpMatchingCPs[i];
			if (vPTs.size() == nMaxMP)
				break;
			auto pMPi = pCPi->GetMP();
			if (pMPi && !pMPi->isDeleted() && pMPi->GetQuality() && pMPi->isOptimized()) {
				vPTs.push_back(mvMatchingPts[i]);
				vpCPs.push_back(pCPi);
				vpMPs.push_back(pMPi);
			}
		}

	}
	
	
	return nres;
}

std::vector<cv::Point2f> UVR_SLAM::MatchInfo::GetMatchingPts(std::vector<UVR_SLAM::MapPoint*>& vpMPs) {
	int N;
	{
		
	}

	std::unique_lock<std::mutex>(mMutexCPs);
	N = mvpMatchingCPs.size();

	std::vector<cv::Point2f> res;
	for (int i = 0; i < N; i++) {
		auto pCPi = mvpMatchingCPs[i];
		
		auto pMPi = pCPi->GetMP();
		if (pMPi && !pMPi->isDeleted() && pMPi->GetQuality()) {
			res.push_back(mvMatchingPts[i]);
			vpMPs.push_back(pMPi);
		}
	}
	return res;
}

std::vector<cv::Point2f> UVR_SLAM::MatchInfo::GetMatchingPtsMapping(std::vector<UVR_SLAM::CandidatePoint*>& vpCPs){
	int N;
	{
		
	}

	std::unique_lock<std::mutex>(mMutexCPs);
	N = mvpMatchingCPs.size();

	std::vector<cv::Point2f> res;
	for (int i = 0; i < N; i++) {
		res.push_back(mvMatchingPts[i]);
		vpCPs.push_back(mvpMatchingCPs[i]);
	}
	return res;
}

std::vector<cv::Point2f> UVR_SLAM::MatchInfo::GetMatchingPts() {
	int N;
	{
		
	}

	std::unique_lock<std::mutex>(mMutexCPs);
	N = mvpMatchingCPs.size();

	std::vector<cv::Point2f> res;
	for (int i = 0; i < N; i++) {
		res.push_back(mvMatchingPts[i]);
	}
	return res;
}
//////////////matchinfo
