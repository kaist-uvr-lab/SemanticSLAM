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
#include <LocalBinaryPatternProcessor.h>
#include <Database.h>
#include <System.h>
#include <Map.h>

bool UVR_SLAM::Frame::mbInitialComputations = true;
float UVR_SLAM::Frame::cx, UVR_SLAM::Frame::cy, UVR_SLAM::Frame::fx, UVR_SLAM::Frame::fy, UVR_SLAM::Frame::invfx, UVR_SLAM::Frame::invfy;
float UVR_SLAM::Frame::mnMinX, UVR_SLAM::Frame::mnMinY, UVR_SLAM::Frame::mnMaxX, UVR_SLAM::Frame::mnMaxY;
float UVR_SLAM::Frame::mfGridElementWidthInv, UVR_SLAM::Frame::mfGridElementHeightInv;

UVR_SLAM::Frame::Frame(System* pSys, cv::Mat _src, int w, int h, cv::Mat K, double ts):mpSystem(pSys), mnWidth(w), mnHeight(h), mK(K), mnInliers(0), mnKeyFrameID(0), mnFuseFrameID(0), mnLocalBAID(0), mnFixedBAID(0), mnLocalMapFrameID(0), mnTrackingID(-1), mbDeleted(false),
mfMeanDepth(0.0), mfMinDepth(FLT_MAX), mfMedianDepth(0.0),
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
	mnFrameID = UVR_SLAM::System::nFrameID++;

	////�Ƕ�̵� ����
	/*mvPyramidImages.push_back(matOri);
	int level = 3;
	for (int i = 1; i < level; i++) {
		int a = i * 2;
		cv::Mat resized1, resized2;
		cv::resize(matOri, resized1, cv::Size(w / a, h / a));
		mvPyramidImages.push_back(resized1);
	}*/
	////�Ƕ�̵� ����
}

UVR_SLAM::Frame::Frame(void *ptr, int id, int w, int h, cv::Mat K) :mnWidth(w), mnHeight(h), mK(K), mnInliers(0), mnKeyFrameID(0), mnFuseFrameID(0), mnLocalBAID(0), mnFixedBAID(0), mnLocalMapFrameID(0), mnTrackingID(-1), mbDeleted(false),
mfMeanDepth(0.0), mfMinDepth(FLT_MAX), mfMedianDepth(0.0),
mpPlaneInformation(nullptr), mvpPlanes(), bSegmented(false), mbMapping(false), mdTimestamp(0.0)
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
	mnFrameID = UVR_SLAM::System::nFrameID++;
}

UVR_SLAM::Frame::Frame(void* ptr, int id, int w, int h, cv::Mat _R, cv::Mat _t, cv::Mat K) :mnWidth(w), mnHeight(h), mK(K), mnInliers(0), mnKeyFrameID(0), mnFuseFrameID(0), mnLocalBAID(0), mnFixedBAID(0), mnLocalMapFrameID(0), mnTrackingID(-1), mbDeleted(false),
mfMeanDepth(0.0), mfMinDepth(FLT_MAX), mfMedianDepth(0.0),
mpPlaneInformation(nullptr), mvpPlanes(), bSegmented(false), mbMapping(false), mdTimestamp(0.0)
{
	cv::Mat tempImg = cv::Mat(h, w, CV_8UC4, ptr);
	matOri = tempImg.clone();
	cv::cvtColor(matOri, matFrame, CV_RGBA2GRAY);
	matFrame.convertTo(matFrame, CV_8UC1);
	R = cv::Mat::eye(3, 3, CV_32FC1);
	t = cv::Mat::zeros(3, 1, CV_32FC1);
	mnFrameID = UVR_SLAM::System::nFrameID++;
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

//Ow2�� ���� �Լ��� ���� ���̸�, neighbor keyframe, Ow1�� ���� Ű������
//�޵�� ������ neighbor���� �����. �� �Լ��� ���� ������.
//���� Ű�����ӿ��� ���� �������� �ҷ����� ����.
//�� �̰� Ű������ �߰� �������� ������
//this�� curr frame
//target = prev keyframe
bool UVR_SLAM::Frame::CheckBaseLine(UVR_SLAM::Frame* pTargetKF) {
	
	cv::Mat Ow1 = this->GetCameraCenter();
	cv::Mat Ow2 = pTargetKF->GetCameraCenter();

	cv::Mat vBaseline = Ow2 - Ow1;
	float baseline = cv::norm(vBaseline);
	pTargetKF->ComputeSceneMedianDepth();
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

//�� Ű�������� ���̽������� ����� �� �̿� ��.
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

////20.09.05 ���� �ʿ�.
void UVR_SLAM::Frame::ComputeSceneDepth() {

	float fMaxDepth = 0.0;

	cv::Mat Rcw2;
	float zcw;
	{
		std::unique_lock<std::mutex> lockMP(mMutexPose);
		Rcw2 = R.row(2);
		zcw = t.at<float>(2);
	}
	std::vector<float> vDepths;
	Rcw2 = Rcw2.t();
	for (size_t i = 0, iend = mpMatchInfo->mvpMatchingCPs.size(); i < iend; i++)
	{
		auto pCPi = mpMatchInfo->mvpMatchingCPs[i];
		auto pMPi = pCPi->GetMP();
		if (!pMPi || pMPi->isDeleted()) {
			continue;
		}
		cv::Mat x3Dw = pMPi->GetWorldPos();
		float z = (float)Rcw2.dot(x3Dw) + zcw;
		mfMinDepth = fmin(z, mfMinDepth);
		fMaxDepth = fmax(z, fMaxDepth);
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
	
	//range
	mfRange = sqrt(mfMinDepth*mfMinDepth*16.0);//36
	//std::cout <<"range test::"<< mfRange + mfMedianDepth <<"::"<< fMaxDepth << std::endl;;
	
	////min
	//double minVal, maxVal;
	//cv::minMaxIdx(vDepths, &minVal);
	//mfMinDepth = (float)minVal;
	//std::cout << mfMinDepth << ", " << mfMeanDepth << ", " << mfMedianDepth << std::endl;
}
void UVR_SLAM::Frame::ComputeSceneMedianDepth()
{
	cv::Mat Rcw2;
	float zcw;
	{
		std::unique_lock<std::mutex> lockMP(mMutexPose);
		Rcw2 = R.row(2);
		zcw = t.at<float>(2);
	}
	std::vector<float> vDepths;
	Rcw2 = Rcw2.t();
	for (size_t i = 0, iend = mpMatchInfo->mvpMatchingCPs.size(); i < iend; i++)
	{
		auto pCPi = mpMatchInfo->mvpMatchingCPs[i];
		auto pMPi = pCPi->GetMP();
		if (!pMPi || pMPi->isDeleted()) {
			continue;
		}
		cv::Mat x3Dw = pMPi->GetWorldPos();
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
		mfMedianDepth = vDepths[nidx];
	}
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

	//�������� Ǯ��� ��
	//AssignFeaturesToGrid();

	//�ӽ÷� Ű����Ʈ ����
	
	//mvpMPs �ʱ�ȭ
	//cv::undistort(matOri, undistorted, mK, mDistCoef);
	
	//////////canny
	//edge�� setkeyframe���� �߰�.
	//canny�� ������ ������ ���� ����Ʈ�� ���⼭ �߰��ϱ�.(0704)
	//cv::Mat filtered;
	//GaussianBlur(matFrame, filtered, cv::Size(5, 5), 0.0);
	//cv::Canny(filtered, mEdgeImg, 50, 200);//150
	
	//////////canny

	/*mvpMPs = std::vector<UVR_SLAM::MapPoint*>(mvKeyPoints.size(), nullptr);
	mvbMPInliers = std::vector<bool>(mvKeyPoints.size(), false);
	mvObjectTypes = std::vector<ObjectType>(mvKeyPoints.size(), OBJECT_NONE);*/

	//mvMapObjects = std::vector<std::multimap<ObjectType, int, std::greater<int>>>(mvKeyPoints.size());
	//��Ʈ�� ��Ī�� ���� ��.
	/*mWallDescriptor = cv::Mat::zeros(0, matDescriptor.cols, matDescriptor.type());
	mObjectDescriptor = cv::Mat::zeros(0, matDescriptor.cols, matDescriptor.type());
	mPlaneDescriptor = cv::Mat::zeros(0, matDescriptor.cols, matDescriptor.type());
	mLabelStatus = cv::Mat::zeros(mvKeyPoints.size(), 1, CV_8UC1);*/
}

void UVR_SLAM::Frame::DetectFeature() {
	//tempDesc�� tempKPs�� �̹������� ��ġ�� Ű����Ʈ�� �����ϱ� ����.
	//ExtractORB(matFrame, mvKeyPoints, matDescriptor);
	//////���⿡�� �ߺ��Ǵ� Ű����Ʈ�� �����ϱ�
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
		//200410 �߰�
		mvnOctaves.push_back(mvTempKPs[i].octave);
		mvPts.push_back(mvTempKPs[i].pt);
		//200410 �߰�
	}
	////���⿡�� �ߺ��Ǵ� Ű����Ʈ�� �����ϱ�

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
	mvKeyPoints = mvKeyPointsUn;
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

//////////////matchinfo
UVR_SLAM::MatchInfo::MatchInfo(): mfLowQualityRatio(0.0){}
UVR_SLAM::MatchInfo::MatchInfo(System* pSys, Frame* pRef, Frame* pTarget, int w, int h):mnHeight(h), mnWidth(w), mfLowQualityRatio(0.0){
	mpPrevFrame = pTarget;
	mpRefFrame = pRef;
	mMapCP = cv::Mat::zeros(h, w, CV_16SC1);
	mpSystem = pSys;
}
UVR_SLAM::MatchInfo::~MatchInfo(){}

void UVR_SLAM::MatchInfo::UpdateKeyFrame() {
	int nCurrID = this->mpRefFrame->mnKeyFrameID;
	for (size_t i = 0, iend = mvpMatchingCPs.size(); i < iend; i++) {
		if (!mvbMapPointInliers[i])
			continue;
		auto pCPi = mvpMatchingCPs[i];
		pCPi->ConnectFrame(this, i);
		auto pMPi = pCPi->GetMP();
		
		if (!pMPi || !pMPi->GetQuality() || pMPi->isDeleted())
			continue;
		//this->AddMP(pMPi, i);
		pMPi->ConnectFrame(this, i);
	}
}

//Ʈ��ŷ
int UVR_SLAM::MatchInfo::CheckOpticalPointOverlap(cv::Point2f pt, int radius, int margin) {
	//range option�� �ʿ��� ��
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
bool UVR_SLAM::MatchInfo::CheckOpticalPointOverlap(cv::Mat& overlap, cv::Point2f pt, int radius, int margin) {
	//range option�� �ʿ��� ��
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
	auto vpCPs = mpRefFrame->mpMatchInfo->mvpMatchingCPs;
	auto vPTs = mpRefFrame->mpMatchInfo->mvMatchingPts;
	for (size_t i = 0, iend = vpCPs.size(); i < iend; i++){
		auto pCPi = vpCPs[i];
		auto pt = vPTs[i];
		int label = labelMat.at<uchar>(pt.y / 2, pt.x / 2);
		pCPi->SetLabel(label);
		auto pMPi = pCPi->GetMP();
		if (pMPi)
			pMPi->SetLabel(pCPi->GetLabel());
		////object ��Ƽ�ʿ� �߰�
		for(auto iter = this->mmLabelRectCPs.equal_range(label).first, eiter = this->mmLabelRectCPs.equal_range(label).second; iter != eiter; iter++){
			auto rect  = iter->second.first;
			//auto lpCPs = iter->second.second;
			if (rect.contains(pt/2)) {
				//std::cout << "add" << std::endl;
				iter->second.second.push_back(pCPi);
				break;
			}
			//iter->second->second = lpCPs;
		}
		////object ��Ƽ�ʿ� �߰�
	}

	//������Ʈ ��ó�� �ʿ�
	
}

//���ο� ������Ʈ�� �����ϱ� ���� Ű����Ʈ�� ����.
//Ŀ��Ʈ ������X
void UVR_SLAM::MatchInfo::SetMatchingPoints() {

	//auto mmpFrameGrids = mpRefFrame->mmpFrameGrids;
	int nGridSize = mpSystem->mnRadius * 2;

	//int nMax = (mpSystem->mnMaxMP+100-nCP)/2;//150; //�Ѵ� �ϸ� 500
	int nMax = 150;
	int nIncEdge = mpRefFrame->mvEdgePts.size() / nMax;
	int nIncORB = mpRefFrame->mvPts.size() / nMax;

	if (nIncEdge == 0)
		nIncEdge = 1;
	if (nIncORB == 0)
		nIncORB = 1;
	
	cv::Mat currMap = cv::Mat::zeros(mnHeight, mnWidth, CV_8UC1);
	
	for (int i = 0; i < mpRefFrame->mvEdgePts.size(); i += nIncEdge) {
		auto pt = mpRefFrame->mvEdgePts[i];
		bool b1 = CheckOpticalPointOverlap(pt, mpSystem->mnRadius) > -1;
		bool b2 = !CheckOpticalPointOverlap(currMap, pt, mpSystem->mnRadius);
		if (b1 || b2) {
			continue;
		}
		auto gridPt = mpRefFrame->GetGridBasePt(pt, nGridSize);
		if (mpRefFrame->mmbFrameGrids[gridPt])
			continue;
		cv::rectangle(currMap, pt - mpSystem->mRectPt, pt + mpSystem->mRectPt, cv::Scalar(255, 0, 0), -1);
		auto pCP = new UVR_SLAM::CandidatePoint(this->mpRefFrame);
		int idx = this->AddCP(pCP, pt);
		//pCP->ConnectFrame(this, idx);
		
		////grid
		mpRefFrame->mmbFrameGrids[gridPt] = true;
		mpRefFrame->mmpFrameGrids[gridPt]->pt = pt;
		mpRefFrame->mmpFrameGrids[gridPt]->mpCP = pCP;
		////grid
	}
	for (int i = 0; i < mpRefFrame->mvPts.size(); i+= nIncORB) {
		auto pt = mpRefFrame->mvPts[i];
		bool b1 = CheckOpticalPointOverlap(pt, mpSystem->mnRadius) > -1;
		bool b2 = !CheckOpticalPointOverlap(currMap, pt, mpSystem->mnRadius);
		if (b1 || b2) {
			continue;
		}
		auto gridPt = mpRefFrame->GetGridBasePt(pt, nGridSize);
		if (mpRefFrame->mmbFrameGrids[gridPt])
			continue;
		cv::rectangle(currMap, pt - mpSystem->mRectPt, pt + mpSystem->mRectPt, cv::Scalar(255, 0, 0), -1);
		auto pCP = new UVR_SLAM::CandidatePoint(this->mpRefFrame, mpRefFrame->mvnOctaves[i]);
		int idx = this->AddCP(pCP, pt);
		//pCP->ConnectFrame(this, idx);

		////grid
		mpRefFrame->mmbFrameGrids[gridPt] = true;
		mpRefFrame->mmpFrameGrids[gridPt]->pt = pt;
		mpRefFrame->mmpFrameGrids[gridPt]->mpCP = pCP;
		////grid
	}
}

//void UVR_SLAM::MatchInfo::AddMP(MapPoint* pMP, int idx) {
//	std::unique_lock<std::mutex>(mMutexMPs);
//	mvpMatchingMPs[idx] = pMP;
//	mnNumMapPoint++;
//}
//void UVR_SLAM::MatchInfo::RemoveMP(int idx) {
//	std::unique_lock<std::mutex>(mMutexMPs);
//	mvpMatchingMPs[idx] = nullptr;
//	mnNumMapPoint--;
//}
//UVR_SLAM::MapPoint*  UVR_SLAM::MatchInfo::GetMP(int idx) {
//	std::unique_lock<std::mutex>(mMutexMPs);
//	return mvpMatchingMPs[idx];
//}
int UVR_SLAM::MatchInfo::GetNumMPs() {
	std::unique_lock<std::mutex>(mMutexMPs);
	return mnNumMapPoint;
}
////////20.09.05 ���� �ʿ�

int UVR_SLAM::MatchInfo::AddCP(CandidatePoint* pCP, cv::Point2f pt, int idx){
	//std::unique_lock<std::mutex>(mMutexCPs);
	int res = mvpMatchingCPs.size();
	mvpMatchingCPs.push_back(pCP);
	mvMatchingPts.push_back(pt);
	mvbMapPointInliers.push_back(true);
	mvMatchingIdxs.push_back(idx);
	cv::rectangle(mMapCP, pt- mpSystem->mRectPt, pt+ mpSystem->mRectPt, cv::Scalar(res + 1), -1);
	//cv::circle(mMapCP, pt, Frame::mnRadius, cv::Scalar(res+1), -1);
	return res;
}
////�̰��� ����� �ȵ� ���� ����.
void UVR_SLAM::MatchInfo::RemoveCP(int idx){
	//std::unique_lock<std::mutex>(mMutexCPs);
	auto pt = mvMatchingPts[idx];
	cv::rectangle(mMapCP, pt - mpSystem->mRectPt, pt + mpSystem->mRectPt, cv::Scalar(-1), -1);
	//cv::circle(mMapCP, mvMatchingPts[idx], Frame::mnRadius, cv::Scalar(-1), -1);
	mvpMatchingCPs[idx] = nullptr;	
}
void UVR_SLAM::MatchInfo::ConnectAll() {
	
	int nCurrID = this->mpRefFrame->mnKeyFrameID;
	for (size_t i = 0, iend = mvpMatchingCPs.size(); i < iend; i++) {
		
		auto pCPi = mvpMatchingCPs[i];
		int idx = pCPi->GetPointIndexInFrame(this);
		if (idx != -1)
			continue;
		pCPi->ConnectFrame(this, i);
		auto pMPi = pCPi->GetMP();
		if (!pMPi || !pMPi->GetQuality() || pMPi->isDeleted())
			continue;
		//this->AddMP(pMPi, i);
		pMPi->ConnectFrame(this, i);
		pMPi->IncreaseVisible();
		pMPi->IncreaseFound();
		pMPi->SetLastVisibleFrame(std::move(nCurrID));
		pMPi->SetLastSuccessFrame(std::move(nCurrID));
	}
}
void UVR_SLAM::MatchInfo::DisconnectAll() {
	for (int i = 0, iend = mvpMatchingCPs.size(); i < iend; i++) {
		auto pCPi = mvpMatchingCPs[i];
		if (!pCPi)
			continue;
		pCPi->DisconnectFrame(this);
		auto pMPi = pCPi->GetMP();
		if (!pMPi || pMPi->isDeleted())
			continue;
		//this->RemoveMP(i);
		pMPi->DisconnectFrame(this);
	}
}
bool UVR_SLAM::MatchInfo::UpdateFrameQuality() {
	int nMP = 0;
	int nLow = 0;
	for (int i = 0, iend = mvpMatchingCPs.size(); i < iend; i++) {
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
	
	//b3�� ���� �����Ӱ� �񱳽� ���̰� ���ڱ� Ŭ ����
	//��ü MP ��, quality�� ������ �ֵ��� �̹� ���⿡ ���������� ����. 
	//std::cout << "FrameQuality = " << nMP << "+" <<nLow<<"="<< N << std::endl;
	return b1 || b2;
}

std::vector<cv::Point2f> UVR_SLAM::MatchInfo::GetMatchingPtsMapping(std::vector<UVR_SLAM::CandidatePoint*>& vpCPs){
	
	std::vector<cv::Point2f> res;
	for (int i = 0, iend = mvpMatchingCPs.size(); i < iend; i++) {
		res.push_back(mvMatchingPts[i]);
		vpCPs.push_back(mvpMatchingCPs[i]);
	}
	return res;
}

//////////////matchinfo

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

void UVR_SLAM::Frame::SetGrids() {
	
	////LBP
	/*cv::Mat currFrame = this->matFrame.clone();
	cv::Mat testImg = this->GetOriginalImage().clone();
	cv::Mat blurred;
	cv::GaussianBlur(currFrame, blurred, cv::Size(7, 7), 5, 3, cv::BORDER_CONSTANT);
	cv::Mat lbpImg = mpSystem->mpLBPProcessor->ConvertDescriptor(blurred);*/
	////LBP

	int nHalf = mpMatchInfo->mpSystem->mnRadius;
	int nSize = nHalf * 2;
	int ksize = 1;
	int thresh = ksize*10;
	cv::Mat matGradient;
	ComputeGradientImage(GetOriginalImage(), matGradient);
	
	//////�Ƕ�̵� ���� �׽�Ʈ
	//{
	//	int level = 3;
	//	for (int l = 1; l < level; l++) {
	//		cv::Mat matLevelGra;
	//		ComputeGradientImage(this->mvPyramidImages[l], matLevelGra);

	//		int a = pow(2, l);
	//		int w = mnWidth / a;
	//		int h = mnHeight / a;
	//		
	//		for (int x = 0; x < w; x += nSize) {
	//			for (int y = 0; y < h; y += nSize) {
	//				cv::Point2f ptLeft(x, y);
	//				cv::Point2f ptRight(x + nSize, y + nSize);
	//				if (ptRight.x > w || ptRight.y > h) {
	//					continue;
	//				}
	//				
	//				cv::Rect rect(ptLeft, ptRight);
	//				auto pGrid = new FrameGrid(std::move(ptLeft), std::move(rect), l);
	//				bool bGrid = false;
	//				cv::Mat mGra = matLevelGra(rect);// .clone();// .clone();
	//				cv::Point2f pt;
	//				int localthresh;
	//				if (pGrid->CalcActivePoints(mGra, thresh, localthresh, pt)) {
	//					if (l == 2) {
	//						FrameGridLevelKey a(ptLeft, l);
	//						mmpFrameLevelGrids.insert(std::make_pair(a, pGrid));
	//						mvPyramidPts.push_back(pt);
	//					}
	//				}//if active
	//			}
	//		}//�̹���
	//	}//����
	//}
	//////�Ƕ�̵� ���� �׽�Ʈ


	////����Ʈ �ߺ� �� �׸��� �� �߰� ����Ʈ ����
	cv::Mat occupied = cv::Mat::zeros(mnWidth, mnHeight, CV_8UC1);
	cv::Point2f gridTempRect(3,3);//nHalf/2, nHalf/2
	////����Ʈ �ߺ� �� �׸��� �� �߰� ����Ʈ ����
	
	for (int x = 0; x < mnWidth; x += nSize) {
		for (int y = 0; y < mnHeight; y += nSize) {
			cv::Point2f ptLeft(x, y);
			if (mmbFrameGrids[ptLeft]){
				/*if (!mmpFrameGrids[ptLeft]) {
					cv::circle(testImg, ptLeft, 3, cv::Scalar(255,0,255), -1);
				}
				else {
					cv::circle(testImg, ptLeft, 3, cv::Scalar(0,0,255), -1);
				}*/
				continue;
			}
			
			cv::Point2f ptRight(x + nSize, y + nSize);

			/*auto prevGridPt = GetGridBasePt(ptLeft, nSize);
			if (ptLeft.x != prevGridPt.x || ptLeft.y != prevGridPt.y) {
				std::cout << "setgrids::error" << std::endl;
			}*/
			if (ptRight.x > mnWidth || ptRight.y > mnHeight){
				//cv::circle(testImg, ptLeft, 3, cv::Scalar(255, 255, 0), -1);
				continue;
			}

			cv::Rect rect(ptLeft, ptRight);
			auto pGrid = new FrameGrid(std::move(ptLeft), std::move(rect), 0);
			bool bGrid = false;
			cv::Mat mGra = matGradient(rect);// .clone();
			cv::Point2f pt;
			int localthresh;
			////LBP ���
			
			////active point ���
			if (pGrid->CalcActivePoints(mGra, thresh, localthresh,pt)) {
				bool bOccupied = this->mpMatchInfo->CheckOpticalPointOverlap(pt, mpSystem->mnRadius) > -1;
				if (bOccupied)
					continue;

				bGrid = true;
				auto pCP = new UVR_SLAM::CandidatePoint(mpMatchInfo->mpRefFrame);
				int idx = mpMatchInfo->AddCP(pCP, pt);
				pGrid->pt = pt;
				pGrid->mpCP = pCP;
				cv::rectangle(occupied, pt - gridTempRect, pt + gridTempRect, cv::Scalar(255, 0, 0), -1);
			
				////seed����
				cv::Mat a = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
				pCP->mpSeed = new Seed(std::move(mpSystem->mInvK*a), mfMedianDepth, mfMinDepth);
				////seed����

				//////LBP code
				////�ȼ� LBP
				//pGrid->mCharCode = lbp->run(currFrame, pt);
				////��ġ ���� LBP
				/*cv::Point2f ptLeft1( pt.x - 10, pt.y - 10);
				cv::Point2f ptRight1(pt.x + 10, pt.y + 10);
				bool bLeft = ptLeft1.x >= 0 && ptLeft1.y >= 0;
				bool bRight = ptRight1.x < mnWidth && ptRight1.y < mnHeight;
				bool bCode = bLeft && bRight;
				if (bCode) {
					cv::Rect rect1(ptLeft1, ptRight1);
					cv::Mat hist = mpSystem->mpLBPProcessor->ConvertHistogram(lbpImg, rect1);
					auto id = mpSystem->mpLBPProcessor->GetID(hist);
					int label = mpSystem->mpDatabase->GetData(id);
					if(label > 0)
						cv::circle(testImg, pGrid->pt, 4, ObjectColors::mvObjectLabelColors[label], -1);
					else
						cv::circle(testImg, pGrid->pt, 4, cv::Scalar(0,0,0), -1);
				}*/
				/*else {
					pGrid->mHistLBP = cv::Mat::zeros(1, lbp->numPatterns, CV_8UC1);
				}*/
				//////LBP code

				////�׸��� ���� �߰� ����Ʈ ó��
				//for (int gy = 0; gy < mGra.rows; gy++) {
				//	for (int gx = 0; gx < mGra.cols; gx++) {
				//		int val = mGra.at<uchar>(gy, gx);
				//		if(val > localthresh){
				//			cv::Point2f tpt(gx+pGrid->basePt.x, gy+pGrid->basePt.y);
				//			if (!mpMatchInfo->CheckOpticalPointOverlap(occupied, tpt, mpSystem->mnRadius)) {
				//				continue;
				//			}
				//			//pGrid->vecPTs.push_back(tpt);//pGrid->basePt
				//			auto pCP2 = new UVR_SLAM::CandidatePoint(mpMatchInfo);
				//			int idx2 = mpMatchInfo->AddCP(pCP2, tpt);
				//			cv::rectangle(occupied, tpt - gridTempRect, tpt + gridTempRect, cv::Scalar(255, 0, 0), -1);
				//		}//thresh
				//	}//for x
				//}//fory

				/*mmpFrameGrids.insert(std::make_pair(ptLeft, pGrid));
				mmbFrameGrids.insert(std::make_pair(ptLeft, bGrid));*/
			}
			//imshow("gra ", mGra); waitKey();
			mmpFrameGrids.insert(std::make_pair(ptLeft, pGrid));
			mmbFrameGrids.insert(std::make_pair(ptLeft, bGrid));
			
		}
	}
	////���� �������� �ε��� �ִ� ����
	mpMatchInfo->mvPrevMatchingIdxs = std::vector<int>(mpMatchInfo->mvMatchingIdxs.begin(), mpMatchInfo->mvMatchingIdxs.end());
	mpMatchInfo->mvMatchingIdxs.resize(mpMatchInfo->mvpMatchingCPs.size());// = std::vector<int>(mpMatchInfo->mvpMatchingCPs.size());
	for (size_t i = 0, iend = mpMatchInfo->mvpMatchingCPs.size(); i < iend; i++) {
		mpMatchInfo->mvMatchingIdxs[i] = i;
	}
	
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

void UVR_SLAM::Frame::Delete() {

	//Ŀ�ؼ� ���õ� ���ؽ� �߰��ؾ� �� ��. KF���Ž�
	if (mnKeyFrameID == 0)
		return;

	std::map<Frame*, int> tempKeyFrameCount;
	std::multimap<int, UVR_SLAM::Frame*, std::greater<int>> tempConnectedKFs;
	{
		std::unique_lock<std::mutex> lockMP(mMutexConnection);
		tempKeyFrameCount = mmKeyFrameCount;
		tempConnectedKFs = mmpConnectedKFs;
		mbDeleted = true;
		mmKeyFrameCount.clear();
		mmpConnectedKFs.clear();
	}

	//MP����
	auto matchInfo = this->mpMatchInfo;
	auto vpCPs = matchInfo->mvpMatchingCPs;
	auto vPTs = matchInfo->mvMatchingPts;
	for (size_t i = 0, iend = vpCPs.size(); i < iend; i++) {
		auto pCPi = vpCPs[i];
		pCPi->DisconnectFrame(matchInfo);
		auto pMPi = pCPi->GetMP();
		if (!pMPi || pMPi->isDeleted())
			continue;
		pMPi->DisconnectFrame(matchInfo);
	}
	//KF����
	for (auto iter = tempKeyFrameCount.begin(), iend = tempKeyFrameCount.end(); iter != iend; iter++) {
		auto pKFi = iter->first;
		pKFi->RemoveKF(this);
	}
	//mmpConnectedKFs.clear();
	////DB ����
	mpSystem->mpMap->RemoveFrame(this);
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

std::multimap<int, UVR_SLAM::Frame*, std::greater<int>> UVR_SLAM::Frame::GetConnectedKFsWithWeight() {
	/*std::multimap<int, UVR_SLAM::Frame*> tempKFs;
	for (std::multimap<int, UVR_SLAM::Frame*, std::greater<int>>::iterator iter = mmpConnectedKFs.begin(); iter != mmpConnectedKFs.end(); iter++) {
	UVR_SLAM::Frame* pKFi = iter->second;
	tempKFs.insert(std::make_pair(iter->first, iter->second));
	}*/
	std::unique_lock<std::mutex> lockMP(mMutexConnection);
	return std::multimap<int, UVR_SLAM::Frame*, std::greater<int>>(mmpConnectedKFs.begin(), mmpConnectedKFs.end());
}