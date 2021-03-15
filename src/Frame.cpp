//
// Created by UVR-KAIST on 2019-02-01.
//

#include <Frame.h>
#include <CandidatePoint.h>
#include <MatrixOperator.h>
#include <ORBextractor.h>
#include <MapPoint.h>
#include <FrameGrid.h>
#include <Plane.h>
#include <DepthFilter.h>
#include <Database.h>
#include <System.h>
#include <Map.h>
#include <MapGrid.h>
#include <Vocabulary.h>
#include <Converter.h>

//bool UVR_SLAM::Frame::mbInitialComputations = true;
//float UVR_SLAM::Frame::cx, UVR_SLAM::Frame::cy, UVR_SLAM::Frame::fx, UVR_SLAM::Frame::fy, UVR_SLAM::Frame::invfx, UVR_SLAM::Frame::invfy;
//float UVR_SLAM::Frame::mnMinX, UVR_SLAM::Frame::mnMinY, UVR_SLAM::Frame::mnMaxX, UVR_SLAM::Frame::mnMaxY;
//float UVR_SLAM::Frame::mfGridElementWidthInv, UVR_SLAM::Frame::mfGridElementHeightInv;


////매핑 서버 로드맵용

UVR_SLAM::Frame::Frame(System* pSys, int id, int w, int h, float _fx, float _fy, float _cx, float _cy, double ts) :mpSystem(pSys), mnWidth(w), mnHeight(h), mnInliers(0), mnKeyFrameID(0), mnFuseFrameID(0), mnLocalBAID(0), mnFixedBAID(0), mnLocalMapFrameID(0), mnTrackingID(-1), mbDeleted(false),
mfMeanDepth(0.0), mfMinDepth(FLT_MAX), mfMaxDepth(0.0), mfMedianDepth(0.0), fx(_fx), fy(_fy), cx(_cx), cy(_cy),
mpPlaneInformation(nullptr), mvpPlanes(), bSegmented(false), mbMapping(false), mdTimestamp(ts) {
	R = cv::Mat::eye(3, 3, CV_32FC1);
	t = cv::Mat::zeros(3, 1, CV_32FC1);
	mnFrameID = id;
	mK = cv::Mat::eye(3, 3, CV_32FC1);
	mK.at<float>(0, 0) = fx;
	mK.at<float>(1, 1) = fy;
	mK.at<float>(0, 2) = cx;
	mK.at<float>(1, 2) = cy;
	mInvK = mK.inv();
}

////매핑 서버에서 생성
UVR_SLAM::Frame::Frame(System* pSys, int id, int w, int h, cv::Mat K, cv::Mat invK, double ts) :mpSystem(pSys), mnWidth(w), mnHeight(h), mK(K), mInvK(invK), mnInliers(0), mnKeyFrameID(0), mnFuseFrameID(0), mnLocalBAID(0), mnFixedBAID(0), mnLocalMapFrameID(0), mnTrackingID(-1), mbDeleted(false),
mfMeanDepth(0.0), mfMinDepth(FLT_MAX), mfMaxDepth(0.0), mfMedianDepth(0.0),
mpPlaneInformation(nullptr), mvpPlanes(), bSegmented(false), mbMapping(false), mdTimestamp(ts) {
	R = cv::Mat::eye(3, 3, CV_32FC1);
	t = cv::Mat::zeros(3, 1, CV_32FC1);
	mnFrameID = id;
	fx = K.at<float>(0, 0);
	fy = K.at<float>(1, 1);
	cx = K.at<float>(0, 2);
	cy = K.at<float>(1, 2);
}

UVR_SLAM::Frame::~Frame() {
	close();
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

namespace UVR_SLAM {
	void Frame::AddMapPoint(MapPoint* pMP,int idx){
		std::unique_lock<std::mutex>(mMutexMPs);
		mvpMPs[idx] = pMP;
	}
	void Frame::EraseMapPoint(int idx){
		std::unique_lock<std::mutex>(mMutexMPs);
		mvpMPs[idx] = nullptr;
	}
	MapPoint* Frame::GetMapPoint(int idx){
		std::unique_lock<std::mutex>(mMutexMPs);
		return mvpMPs[idx];
	}
	void Frame::SetMapPoints(int n) {
		std::unique_lock<std::mutex>(mMutexMPs);
		mvpMPs = std::vector<MapPoint*>(n, nullptr);
	}
	std::vector<MapPoint*> Frame::GetMapPoints(){
		std::unique_lock<std::mutex>(mMutexMPs);
		return mvpMPs;
	}
}

////////////////////////////////////////////////////////////////////////////////////////
float UVR_SLAM::Frame::GetDepth(cv::Mat X3D) {
	cv::Mat Rcw2;
	float zcw;
	{
		std::unique_lock<std::mutex> lockMP(mMutexPose);
		Rcw2 = R.row(2).t();
		zcw = t.at<float>(2);
	}
	return (float)Rcw2.dot(X3D) + zcw;
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
void UVR_SLAM::Frame::GetInversePose(cv::Mat&_Rinv, cv::Mat& _Tinv) {
	std::unique_lock<std::mutex>(mMutexPose);
	_Rinv = R.t();
	_Tinv = -_Rinv*t;
}
void UVR_SLAM::Frame::GetRelativePoseFromTargetFrame(Frame* pTargetFrame, cv::Mat& Rft, cv::Mat& Tft){
	cv::Mat Rinv, Tinv;
	pTargetFrame->GetInversePose(Rinv, Tinv);
	std::unique_lock<std::mutex>(mMutexPose);
	Rft = R*Rinv;
	Tft = R*Tinv + t;
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
cv::Mat UVR_SLAM::Frame::GetFrame() {
	std::unique_lock<std::mutex> lockMP(mMutexFrame);
	return matFrame.clone();
}
cv::Mat UVR_SLAM::Frame::GetOriginalImage() {
	//std::unique_lock<std::mutex> lockMP(mMutexFrame);
	return matOri;
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
	//pTargetKF->ComputeSceneMedianDepth();
	float medianDepthKF2 = pTargetKF->mfMedianDepth;
	if(medianDepthKF2 < 0.0){
		std::cout << "Not enough baseline!!" << std::endl;
		return false;
	}
	
	float ratioBaselineDepth = baseline / medianDepthKF2;

	if (ratioBaselineDepth<0.01){
		std::cout << "Not enough baseline!!" << std::endl;
		return false;
	}
	return true;
}

void UVR_SLAM::Frame::ComputeSceneDepth() {

	cv::Mat Rcw2;
	float zcw;
	{
		std::unique_lock<std::mutex> lockMP(mMutexPose);
		Rcw2 = R.row(2);
		zcw = t.at<float>(2);
	}
	std::vector<MapPoint*> vpMPs;
	{
		std::unique_lock<std::mutex> lockMP(mMutexMPs);
		vpMPs = mvpMPs;
	}
	std::vector<float> vDepths;
	Rcw2 = Rcw2.t();

	for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
	{
		auto pMPi = vpMPs[i];
		if (!pMPi || pMPi->isDeleted()) {
			continue;
		}
		cv::Mat x3Dw = pMPi->GetWorldPos();
		float z = (float)Rcw2.dot(x3Dw) + zcw;
		mfMinDepth = fmin(z, mfMinDepth);
		vDepths.push_back(z);
	}
	if (vDepths.size() == 0) {
		return;
	}
	int nidx = vDepths.size() / 2;
	std::nth_element(vDepths.begin(), vDepths.begin() + nidx, vDepths.end());

	//median
	mfMedianDepth = vDepths[nidx];
	//mean & stddev
	cv::Mat mMean, mDev;
	meanStdDev(vDepths, mMean, mDev);
	mfMeanDepth = (float)mMean.at<double>(0);
	mfStdDev = (float)mDev.at<double>(0);
	
	////max depth 설정
	//1) mfrange
	mfRange = sqrt(mfMinDepth*mfMinDepth*36.0);//36
	mfMaxDepth = mfMedianDepth + mfRange;
	//2) stddev 이용하기
	//mfMaxDepth = mfMedianDepth + 1.2*mfStdDev;

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

void UVR_SLAM::Frame::Frame::ComputeBoW()
{
	if (mBowVec.empty())
	{
		std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(matDescriptor);
		mpSystem->mpDBoWVoc->transform(vCurrentDesc, mBowVec, mFeatVec, 0);  // 5 is better
	}
}

///////////////////////////////
//void UVR_SLAM::Frame::Init(ORBextractor* _e, cv::Mat _k, cv::Mat _d)
//{
//
//	mpORBextractor = _e;
//	mnScaleLevels = mpORBextractor->GetLevels();
//	mfScaleFactor = mpORBextractor->GetScaleFactor();
//	mfLogScaleFactor = log(mfScaleFactor);
//	mvScaleFactors = mpORBextractor->GetScaleFactors();
//	mvInvScaleFactors = mpORBextractor->GetInverseScaleFactors();
//	mvLevelSigma2 = mpORBextractor->GetScaleSigmaSquares();
//	mvInvLevelSigma2 = mpORBextractor->GetInverseScaleSigmaSquares();
//
//	mK = _k.clone();
//	mDistCoef = _d.clone();
//
//	//에러나면 풀어야 함
//	//AssignFeaturesToGrid();
//
//	//임시로 키포인트 복사
//	
//	//mvpMPs 초기화
//	//cv::undistort(matOri, undistorted, mK, mDistCoef);
//	
//	//////////canny
//	//edge는 setkeyframe에서 추가.
//	//canny는 이전에 돌리고 엣지 포인트만 여기서 추가하기.(0704)
//	//cv::Mat filtered;
//	//GaussianBlur(matFrame, filtered, cv::Size(5, 5), 0.0);
//	//cv::Canny(filtered, mEdgeImg, 50, 200);//150
//	
//	//////////canny
//
//	/*mvpMPs = std::vector<UVR_SLAM::MapPoint*>(mvKeyPoints.size(), nullptr);
//	mvbMPInliers = std::vector<bool>(mvKeyPoints.size(), false);
//	mvObjectTypes = std::vector<ObjectType>(mvKeyPoints.size(), OBJECT_NONE);*/
//
//	//mvMapObjects = std::vector<std::multimap<ObjectType, int, std::greater<int>>>(mvKeyPoints.size());
//	//파트별 매칭을 위한 것.
//	/*mWallDescriptor = cv::Mat::zeros(0, matDescriptor.cols, matDescriptor.type());
//	mObjectDescriptor = cv::Mat::zeros(0, matDescriptor.cols, matDescriptor.type());
//	mPlaneDescriptor = cv::Mat::zeros(0, matDescriptor.cols, matDescriptor.type());
//	mLabelStatus = cv::Mat::zeros(mvKeyPoints.size(), 1, CV_8UC1);*/
//}

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

	if (u<5.0 || u>mnWidth-5.0)
		return false;
	if (v<5.0 || v>mnHeight-5.0)
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
	std::unique_lock<std::mutex> lockMP(mMutexConnection);
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

////////////////FrameGrid
void UVR_SLAM::Frame::ComputeGradientImage(cv::Mat src, cv::Mat& dst, int ksize) {
	cv::Mat edge;
	cv::cvtColor(src, edge, CV_BGR2GRAY);
	edge.convertTo(edge, CV_8UC1);
	cv::Mat matDY, matDX;
	cv::Sobel(edge, matDX, CV_64FC1, 1, 0, ksize);
	cv::Sobel(edge, matDY, CV_64FC1, 0, 1, ksize);
	matDX = abs(matDX);
	matDY = abs(matDY);
	//matDX.convertTo(matDX, CV_8UC1);
	//matDY.convertTo(matDY, CV_8UC1);
	dst = (matDX + matDY) / 2.0;
	dst.convertTo(dst, CV_8UC1);
}

cv::Point2f UVR_SLAM::Frame::GetExtendedRect(cv::Point2f pt, int size) {
	auto basePt = GetGridBasePt(pt, size);
	auto diffPt = pt - basePt;
	int nHalf = size / 2;
	if (diffPt.x < nHalf)
		basePt.x -= size;
	if (diffPt.y < nHalf)
		basePt.y -= size;
	return basePt;
}

cv::Point2f UVR_SLAM::Frame::GetGridBasePt(cv::Point2f pt, int size) {
	int a = pt.x / size;
	int b = pt.y / size;
	/*int aa =(int)pt.x%size;
	int bb = (int)pt.y%size;
	if (aa == 0)
		a--;
	if (bb == 0)
		b--;*/
	return std::move(cv::Point2f(a*size, b*size));
}

cv::Mat UVR_SLAM::Frame::ComputeFundamentalMatrix(Frame* pTarget) {
	/*cv::Mat Rcw, Tcw;
	{
		std::unique_lock<std::mutex> lockMP(mMutexPose);
		Rcw = R.clone();
		Tcw = t.clone();
	}
	cv::Mat Rtw, Ttw;
	pTarget->GetPose(Rtw, Ttw);

	cv::Mat R12 = Rcw*Rtw.t();
	cv::Mat t12 = -Rcw*Rtw.t()*Ttw + Tcw;*/

	cv::Mat Rrel, Trel;
	GetRelativePoseFromTargetFrame(pTarget,Rrel, Trel);

	Trel.convertTo(Trel, CV_64FC1);
	cv::Mat t12x = UVR_SLAM::MatrixOperator::GetSkewSymetricMatrix(Trel);
	t12x.convertTo(Trel, CV_32FC1);
	return mK.t().inv()*Trel*Rrel*mK.inv();
}

bool UVR_SLAM::Frame::isDeleted() {
	std::unique_lock<std::mutex> lockMP(mMutexConnection);
	return mbDeleted;
}

//수정 필요
void UVR_SLAM::Frame::Delete() {
	////커넥션 관련된 뮤텍스 추가해야 할 듯. KF제거시
	//if (mnKeyFrameID == 0)
	//	return;

	//std::map<Frame*, int> tempKeyFrameCount;
	//std::multimap<int, UVR_SLAM::Frame*, std::greater<int>> tempConnectedKFs;
	//{
	//	std::unique_lock<std::mutex> lockMP(mMutexConnection);
	//	tempKeyFrameCount = mmKeyFrameCount;
	//	tempConnectedKFs = mmpConnectedKFs;
	//	mbDeleted = true;
	//	mmKeyFrameCount.clear();
	//	mmpConnectedKFs.clear();
	//}

	////MP제거
	//
	////KF제거
	//for (auto iter = tempKeyFrameCount.begin(), iend = tempKeyFrameCount.end(); iter != iend; iter++) {
	//	auto pKFi = iter->first;
	//	pKFi->RemoveKF(this);
	//}
	////mmpConnectedKFs.clear();
	//////DB 제거
	//mpSystem->mpMap->RemoveFrame(this);
}

void UVR_SLAM::Frame::AddKF(UVR_SLAM::Frame* pKF, int weight) {
	std::unique_lock<std::mutex> lockMP(mMutexConnection);
	mmpConnectedKFs.insert(std::make_pair(weight, pKF));
}
void UVR_SLAM::Frame::RemoveKF(Frame* pKF) {
	std::unique_lock<std::mutex> lockMP(mMutexConnection);
	if (mmKeyFrameCount.count(pKF)) {
		int c = mmKeyFrameCount[pKF];
		RemoveKF(pKF, c);
	}
}
void UVR_SLAM::Frame::RemoveKF(UVR_SLAM::Frame* pKF, int weight) {
	std::unique_lock<std::mutex> lockMP(mMutexConnection);
	auto range = mmpConnectedKFs.equal_range(weight);
	for (auto iter = range.first; iter != range.second; iter++) {
		UVR_SLAM::Frame* pKFi = iter->second;
		if (pKFi == pKF) {
			mmpConnectedKFs.erase(iter);
			return;
		}
	}
}
std::vector<UVR_SLAM::Frame*> UVR_SLAM::Frame::GetConnectedKFs(int n) {
	std::unique_lock<std::mutex> lockMP(mMutexConnection);
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
std::set<UVR_SLAM::Frame*> UVR_SLAM::Frame::GetConnectedKeyFrameSet(int n) {
	std::unique_lock<std::mutex> lockMP(mMutexConnection);
	std::set<UVR_SLAM::Frame*> tempKFs;
	for (std::multimap<int, UVR_SLAM::Frame*, std::greater<int>>::iterator iter = mmpConnectedKFs.begin(); iter != mmpConnectedKFs.end(); iter++) {
		if (n != 0 && tempKFs.size() == n)
			break;
		UVR_SLAM::Frame* pKFi = iter->second;
		tempKFs.insert(pKFi);
	}
	return std::set<UVR_SLAM::Frame*>(tempKFs.begin(), tempKFs.end());
}

std::multimap<int, UVR_SLAM::Frame*, std::greater<int>> UVR_SLAM::Frame::GetConnectedKFsWithWeight() {
	/*std::multimap<int, UVR_SLAM::Frame*> tempKFs;
	for (std::multimap<int, UVR_SLAM::Frame*, std::greater<int>>::iterator iter = mmpConnectedKFs.begin(); iter != mmpConnectedKFs.end(); iter++) {
	UVR_SLAM::Frame* pKFi = iter->second;
	tempKFs.insert(std::make_pair(iter->first, iter->second));
	}*/
	std::unique_lock<std::mutex> lockMP(mMutexConnection);
	return std::multimap<int, UVR_SLAM::Frame*, std::greater<int>>(mmpConnectedKFs.begin(), mmpConnectedKFs.end());
}