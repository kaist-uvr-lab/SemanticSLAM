#include <Matcher.h>
#include <omp.h>
#include <random>
#include <Frame.h>
#include <FrameGrid.h>
#include <System.h>
#include <MapPoint.h>
#include <CandidatePoint.h>
#include <MatrixOperator.h>
#include <gms_matcher.h>
#include <PlaneEstimator.h>
#include <Plane.h>
#include <Visualizer.h>
#include <Map.h>

UVR_SLAM::Matcher::Matcher(){}
UVR_SLAM::Matcher::Matcher(System* pSys, cv::Ptr < cv::DescriptorMatcher> _matcher, int w, int h)
	:mpSystem(pSys), mWidth(w), mHeight(h), TH_HIGH(100), TH_LOW(50), HISTO_LENGTH(30), mfNNratio(0.7), mbCheckOrientation(true), matcher(_matcher)
{
	//cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1);
	//cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);
	//matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
	
	//matcher = cv::makePtr<cv::BFMatcher>(cv::NORM_HAMMING, true);
	//matcher = DescriptorMatcher::create("FlannBased");
		
	//cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2), cv::makePtr<cv::flann::SearchParams>(50));
}
UVR_SLAM::Matcher::~Matcher(){}
void UVR_SLAM::Matcher::Init() {
	mpVisualizer = mpSystem->mpVisualizer;
}
const double nn_match_ratio = 0.7f; // Nearest-neighbour matching ratio

//////////////////////////////////////////////////////////////////
////////////일단 남겨놓긴 함. 근데 사용 안함
////////Fundamental Matrix를 위해 이용
bool UVR_SLAM::Matcher::CheckEpiConstraints(cv::Mat F12, cv::Point2f pt1, cv::Point2f pt2, float sigma, cv::Mat & epiLine, float& res, bool& bLine) {
	// Epipolar line in second image l = x1'F12 = [a b c]
	// Epipolar line in second image l = x1'F12 = [a b c]
	const float a = pt1.x*F12.at<float>(0, 0) + pt1.y*F12.at<float>(0, 1) + F12.at<float>(0, 2);
	const float b = pt1.x*F12.at<float>(1, 0) + pt1.y*F12.at<float>(1, 1) + F12.at<float>(1, 2);
	const float c = pt1.x*F12.at<float>(2, 0) + pt1.y*F12.at<float>(2, 1) + F12.at<float>(2, 2);
	const float den = a*a + b*b;
	if (den == 0){
		bLine = false;
		return false;
	}
	bLine = true;
	epiLine = (cv::Mat_<float>(3, 1) << a, b, c);
	const float num = a*pt2.x + b*pt2.y + c;
	const float dsqr = num*num / den;
	res = abs(num) / sqrt(den);
	return dsqr<3.84*sigma;
}

bool UVR_SLAM::Matcher::FeatureMatchingWithEpipolarConstraints(int& matchIDX, UVR_SLAM::Frame* pTargetKF, cv::Mat F12, cv::KeyPoint kp, cv::Mat desc, float sigma, int thresh){

	int nMinDist = thresh;
	int bestIdx = -1;
	for (int j = 0; j < pTargetKF->mvKeyPoints.size(); j++) {
		//if(pCurrKF->mvpMPs[j])
		//    continue;
		cv::KeyPoint prevKP = pTargetKF->mvKeyPoints[j];

		float epiDist;
		cv::Mat epiLine;
		bool bEpiLine;
		if (!CheckEpiConstraints(F12, prevKP.pt, kp.pt, sigma, epiLine, epiDist, bEpiLine))
			continue;

		cv::Mat descPrev = pTargetKF->matDescriptor.row(j);
		int descDist = DescriptorDistance(desc, descPrev);
		if (nMinDist > descDist && descDist < thresh) {
			nMinDist = descDist;
			bestIdx = j;
		}
	}
	matchIDX = bestIdx;
	if (bestIdx == -1)
		return false;
	return true;
}

//에센셜 매트릭스임
//이건 그냥 프레임으로 옮기는 것도 좋을 듯.
//앞이 현재, 뒤가 타겟
cv::Mat UVR_SLAM::Matcher::CalcFundamentalMatrix(cv::Mat R1, cv::Mat t1, cv::Mat R2, cv::Mat t2, cv::Mat K) {

	cv::Mat R12 = R1*R2.t();
	cv::Mat t12 = -R1*R2.t()*t2 + t1;
	t12.convertTo(t12, CV_64FC1);
	cv::Mat t12x = UVR_SLAM::MatrixOperator::GetSkewSymetricMatrix(t12);
	t12x.convertTo(t12, CV_32FC1);
	return K.t().inv()*t12*R12*K.inv();
}

void UVR_SLAM::Matcher::FindFundamental(UVR_SLAM::Frame* pInit, UVR_SLAM::Frame* pCurr, std::vector<cv::DMatch> vMatches, std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
	// Number of putative matches
	const int N = vMatches.size();

	// Normalize coordinates
	std::vector<cv::Point2f> vPn1, vPn2;
	cv::Mat T1, T2;
	Normalize(pInit->mvKeyPoints, vPn1, T1);
	Normalize(pCurr->mvKeyPoints, vPn2, T2);
	cv::Mat T2t = T2.t();

	// Best Results variables
	score = 0.0;
	vbMatchesInliers = std::vector<bool>(N, false);

	int mMaxIterations = 1000;

#pragma  omp parallel for
	for (int it = 0; it<mMaxIterations; it++)
	{
		// Iteration variables
		std::vector<cv::Point2f> vPn1i(8);
		std::vector<cv::Point2f> vPn2i(8);
		std::vector<bool> vbCurrentInliers(N, false);
		float currentScore;
		std::vector<size_t> vAllIndices;

		vAllIndices.reserve(vMatches.size());
		for (int i = 0; i<vMatches.size(); i++)
		{
			vAllIndices.push_back(i);
		}

		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(vAllIndices.begin(), vAllIndices.end(), g);

		// Select a minimum set
		for (int j = 0; j<8; j++)
		{
			int idx1 = vMatches[vAllIndices[j]].queryIdx;
			int idx2 = vMatches[vAllIndices[j]].trainIdx;

			vPn1i[j] = vPn1[idx1];
			vPn2i[j] = vPn2[idx2];
		}

		cv::Mat Fn = ComputeF21(vPn1i, vPn2i);

		cv::Mat F21i = T2t*Fn*T1;

		//homography check
		currentScore = CheckFundamental(pInit, pCurr, F21i, vMatches, vbCurrentInliers, 1.0);

		if (currentScore>score)
		{
			F21 = F21i.clone();
			vbMatchesInliers = vbCurrentInliers;
			score = currentScore;
		}//if
	}//for

}

void UVR_SLAM::Matcher::FindFundamental(UVR_SLAM::Frame* pInit, UVR_SLAM::Frame* pCurr, std::vector<std::pair<cv::Point2f, cv::Point2f>> vMatches, std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
	// Number of putative matches
	const int N = vMatches.size();

	// Normalize coordinates
	std::vector<cv::Point2f> vPn1, vPn2;
	cv::Mat T1, T2;
	Normalize(pInit->mvPts, vPn1, T1);
	Normalize(pCurr->mvPts, vPn2, T2);
	cv::Mat T2t = T2.t();

	// Best Results variables
	score = 0.0;
	vbMatchesInliers = std::vector<bool>(N, false);

	int mMaxIterations = 1000;

#pragma  omp parallel for
	for (int it = 0; it<mMaxIterations; it++)
	{
		// Iteration variables
		std::vector<cv::Point2f> vPn1i(8);
		std::vector<cv::Point2f> vPn2i(8);
		std::vector<bool> vbCurrentInliers(N, false);
		float currentScore;
		std::vector<size_t> vAllIndices;

		vAllIndices.reserve(vMatches.size());
		for (int i = 0; i<vMatches.size(); i++)
		{
			vAllIndices.push_back(i);
		}

		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(vAllIndices.begin(), vAllIndices.end(), g);

		// Select a minimum set
		for (int j = 0; j<8; j++)
		{
			vPn1i[j] = vMatches[vAllIndices[j]].first;
			vPn2i[j] = vMatches[vAllIndices[j]].second;
		}

		cv::Mat Fn = ComputeF21(vPn1i, vPn2i);

		cv::Mat F21i = T2t*Fn*T1;

		//homography check
		currentScore = CheckFundamental(pInit, pCurr, F21i, vMatches, vbCurrentInliers, 1.0);

		if (currentScore>score)
		{
			F21 = F21i.clone();
			vbMatchesInliers = vbCurrentInliers;
			score = currentScore;
		}//if
	}//for

}

void UVR_SLAM::Matcher::Normalize(const std::vector<cv::Point2f> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
	float meanX = 0;
	float meanY = 0;
	const int N = vKeys.size();

	vNormalizedPoints.resize(N);

	for (int i = 0; i<N; i++)
	{
		meanX += vKeys[i].x;
		meanY += vKeys[i].y;
	}

	meanX = meanX / N;
	meanY = meanY / N;

	float meanDevX = 0;
	float meanDevY = 0;

	for (int i = 0; i<N; i++)
	{
		vNormalizedPoints[i].x = vKeys[i].x - meanX;
		vNormalizedPoints[i].y = vKeys[i].y - meanY;

		meanDevX += fabs(vNormalizedPoints[i].x);
		meanDevY += fabs(vNormalizedPoints[i].y);
	}

	meanDevX = meanDevX / N;
	meanDevY = meanDevY / N;

	float sX = 1.0 / meanDevX;
	float sY = 1.0 / meanDevY;

	for (int i = 0; i<N; i++)
	{
		vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
		vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
	}

	T = cv::Mat::eye(3, 3, CV_32F);
	T.at<float>(0, 0) = sX;
	T.at<float>(1, 1) = sY;
	T.at<float>(0, 2) = -meanX*sX;
	T.at<float>(1, 2) = -meanY*sY;
}

void UVR_SLAM::Matcher::Normalize(const std::vector<cv::KeyPoint> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
	float meanX = 0;
	float meanY = 0;
	const int N = vKeys.size();

	vNormalizedPoints.resize(N);

	for (int i = 0; i<N; i++)
	{
		meanX += vKeys[i].pt.x;
		meanY += vKeys[i].pt.y;
	}

	meanX = meanX / N;
	meanY = meanY / N;

	float meanDevX = 0;
	float meanDevY = 0;

	for (int i = 0; i<N; i++)
	{
		vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
		vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

		meanDevX += fabs(vNormalizedPoints[i].x);
		meanDevY += fabs(vNormalizedPoints[i].y);
	}

	meanDevX = meanDevX / N;
	meanDevY = meanDevY / N;

	float sX = 1.0 / meanDevX;
	float sY = 1.0 / meanDevY;

	for (int i = 0; i<N; i++)
	{
		vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
		vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
	}

	T = cv::Mat::eye(3, 3, CV_32F);
	T.at<float>(0, 0) = sX;
	T.at<float>(1, 1) = sY;
	T.at<float>(0, 2) = -meanX*sX;
	T.at<float>(1, 2) = -meanY*sY;
}

cv::Mat UVR_SLAM::Matcher::ComputeF21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2)
{
	const int N = vP1.size();

	cv::Mat A(N, 9, CV_32F);

	for (int i = 0; i<N; i++)
	{
		const float u1 = vP1[i].x;
		const float v1 = vP1[i].y;
		const float u2 = vP2[i].x;
		const float v2 = vP2[i].y;

		A.at<float>(i, 0) = u2*u1;
		A.at<float>(i, 1) = u2*v1;
		A.at<float>(i, 2) = u2;
		A.at<float>(i, 3) = v2*u1;
		A.at<float>(i, 4) = v2*v1;
		A.at<float>(i, 5) = v2;
		A.at<float>(i, 6) = u1;
		A.at<float>(i, 7) = v1;
		A.at<float>(i, 8) = 1;
	}

	cv::Mat u, w, vt;

	cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	cv::Mat Fpre = vt.row(8).reshape(0, 3);

	cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	w.at<float>(2) = 0;

	return  u*cv::Mat::diag(w)*vt;
}

float UVR_SLAM::Matcher::CheckFundamental(UVR_SLAM::Frame* pInit, UVR_SLAM::Frame* pCurr, const cv::Mat &F21, std::vector<cv::DMatch> vMatches, std::vector<bool> &vbMatchesInliers, float sigma)
{
	const int N = vMatches.size();

	const float f11 = F21.at<float>(0, 0);
	const float f12 = F21.at<float>(0, 1);
	const float f13 = F21.at<float>(0, 2);
	const float f21 = F21.at<float>(1, 0);
	const float f22 = F21.at<float>(1, 1);
	const float f23 = F21.at<float>(1, 2);
	const float f31 = F21.at<float>(2, 0);
	const float f32 = F21.at<float>(2, 1);
	const float f33 = F21.at<float>(2, 2);

	vbMatchesInliers.resize(N);

	float score = 0;

	const float th = 3.841;
	const float thScore = 5.991;

	const float invSigmaSquare = 1.0 / (sigma*sigma);

	for (int i = 0; i<N; i++)
	{
		bool bIn = true;

		const cv::Point2f &p1 = pInit->mvKeyPoints[vMatches[i].queryIdx].pt;
		const cv::Point2f &p2 = pCurr->mvKeyPoints[vMatches[i].trainIdx].pt;

		const float u1 = p1.x;
		const float v1 = p1.y;
		const float u2 = p2.x;
		const float v2 = p2.y;

		// Reprojection error in second image
		// l2=F21x1=(a2,b2,c2)

		const float a2 = f11*u1 + f12*v1 + f13;
		const float b2 = f21*u1 + f22*v1 + f23;
		const float c2 = f31*u1 + f32*v1 + f33;

		const float num2 = a2*u2 + b2*v2 + c2;

		const float squareDist1 = num2*num2 / (a2*a2 + b2*b2);

		const float chiSquare1 = squareDist1*invSigmaSquare;

		if (chiSquare1>th)
			bIn = false;
		else
			score += thScore - chiSquare1;

		// Reprojection error in second image
		// l1 =x2tF21=(a1,b1,c1)

		const float a1 = f11*u2 + f21*v2 + f31;
		const float b1 = f12*u2 + f22*v2 + f32;
		const float c1 = f13*u2 + f23*v2 + f33;

		const float num1 = a1*u1 + b1*v1 + c1;

		const float squareDist2 = num1*num1 / (a1*a1 + b1*b1);

		const float chiSquare2 = squareDist2*invSigmaSquare;

		if (chiSquare2>th)
			bIn = false;
		else
			score += thScore - chiSquare2;

		if (bIn)
			vbMatchesInliers[i] = true;
		else
			vbMatchesInliers[i] = false;
	}

	return score;
}

float UVR_SLAM::Matcher::CheckFundamental(UVR_SLAM::Frame* pInit, UVR_SLAM::Frame* pCurr, const cv::Mat &F21, std::vector<std::pair<cv::Point2f, cv::Point2f>> vMatches, std::vector<bool> &vbMatchesInliers, float sigma)
{
	const int N = vMatches.size();

	const float f11 = F21.at<float>(0, 0);
	const float f12 = F21.at<float>(0, 1);
	const float f13 = F21.at<float>(0, 2);
	const float f21 = F21.at<float>(1, 0);
	const float f22 = F21.at<float>(1, 1);
	const float f23 = F21.at<float>(1, 2);
	const float f31 = F21.at<float>(2, 0);
	const float f32 = F21.at<float>(2, 1);
	const float f33 = F21.at<float>(2, 2);

	vbMatchesInliers.resize(N);

	float score = 0;

	const float th = 3.841;
	const float thScore = 5.991;

	const float invSigmaSquare = 1.0 / (sigma*sigma);

	for (int i = 0; i<N; i++)
	{
		bool bIn = true;

		const cv::Point2f &p1 = vMatches[i].first;
		const cv::Point2f &p2 = vMatches[i].second;

		const float u1 = p1.x;
		const float v1 = p1.y;
		const float u2 = p2.x;
		const float v2 = p2.y;

		// Reprojection error in second image
		// l2=F21x1=(a2,b2,c2)

		const float a2 = f11*u1 + f12*v1 + f13;
		const float b2 = f21*u1 + f22*v1 + f23;
		const float c2 = f31*u1 + f32*v1 + f33;

		const float num2 = a2*u2 + b2*v2 + c2;

		const float squareDist1 = num2*num2 / (a2*a2 + b2*b2);

		const float chiSquare1 = squareDist1*invSigmaSquare;

		if (chiSquare1>th)
			bIn = false;
		else
			score += thScore - chiSquare1;

		// Reprojection error in second image
		// l1 =x2tF21=(a1,b1,c1)

		const float a1 = f11*u2 + f21*v2 + f31;
		const float b1 = f12*u2 + f22*v2 + f32;
		const float c1 = f13*u2 + f23*v2 + f33;

		const float num1 = a1*u1 + b1*v1 + c1;

		const float squareDist2 = num1*num1 / (a1*a1 + b1*b1);

		const float chiSquare2 = squareDist2*invSigmaSquare;

		if (chiSquare2>th)
			bIn = false;
		else
			score += thScore - chiSquare2;

		if (bIn)
			vbMatchesInliers[i] = true;
		else
			vbMatchesInliers[i] = false;
	}

	return score;
}

int UVR_SLAM::Matcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
	const int *pa = a.ptr<int32_t>();
	const int *pb = b.ptr<int32_t>();

	int dist = 0;

	for (int i = 0; i<8; i++, pa++, pb++)
	{
		unsigned  int v = *pa ^ *pb;
		v = v - ((v >> 1) & 0x55555555);
		v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
		dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	}

	return dist;
}

void UVR_SLAM::Matcher::ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
	int max1 = 0;
	int max2 = 0;
	int max3 = 0;

	for (int i = 0; i<L; i++)
	{
		const int s = histo[i].size();
		if (s>max1)
		{
			max3 = max2;
			max2 = max1;
			max1 = s;
			ind3 = ind2;
			ind2 = ind1;
			ind1 = i;
		}
		else if (s>max2)
		{
			max3 = max2;
			max2 = s;
			ind3 = ind2;
			ind2 = i;
		}
		else if (s>max3)
		{
			max3 = s;
			ind3 = i;
		}
	}

	if (max2<0.1f*(float)max1)
	{
		ind2 = -1;
		ind3 = -1;
	}
	else if (max3<0.1f*(float)max1)
	{
		ind3 = -1;
	}
}
////////////일단 남겨놓긴 함. 근데 사용 안함
//////////////////////////////////////////////////////////////////


//true이면 y값 기준으로, false이면 x값 기준으로 복원
cv::Point2f CalcLinePoint(float val, Point3f mLine, bool opt) {
	float x, y;
	if (opt) {
		x = 0.0;
		y = val;
		if (mLine.x != 0)
			x = (-mLine.z - mLine.y*y) / mLine.x;
	}
	else {
		y = 0.0;
		x = val;
		if (mLine.y != 0)
			y = (-mLine.z - mLine.x*x) / mLine.y;
	}
	
	return cv::Point2f(x, y);
}

float CalcNCC(cv::Mat src1, cv::Mat src2) {
	cv::Mat vec1 = src1.reshape(1, src1.rows*src1.cols*src1.channels());
	cv::Mat vec2 = src2.reshape(1, src1.rows*src1.cols*src1.channels());

	float len1 = (vec1.dot(vec1));
	float len2 = (vec2.dot(vec2));
	if (len1 < 0.001 || len2 < 0.001)
		return 0.0;
	float len = sqrt(len1)*sqrt(len2);
	return abs(vec1.dot(vec2))/ len;
}

float CalcSSD(cv::Mat src1, cv::Mat src2) {
	cv::Mat diff = abs(src1 - src2);
	return diff.dot(diff);
	/*
	float sum = 0.0;
	int num = diff.cols*diff.rows;
	sum = sqrt(diff.dot(diff));*/
	cv::Mat a = diff.reshape(1, diff.rows*diff.cols*diff.channels());
	//std::cout << "aaaa:" << a.dot(a) << std::endl << a << std::endl;

	float res2 = diff.dot(diff);
	float res1 = a.dot(a);

	
	
	float sum = 0.0;
	int num = diff.cols*diff.rows;
	sum = sqrt(diff.dot(diff));
	return sum / num;
	
}

cv::Mat CreateWorldPoint(cv::Point2f pt, cv::Mat invT, cv::Mat invK, float depth){
	cv::Mat temp = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
	temp = invK*temp;
	temp *= depth;
	temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));

	cv::Mat estimated = invT*temp;
	return estimated.rowRange(0, 3);
}

bool CheckBoundary(float x, float y, int rows, int cols) {
	if (x < 0 || y < 0 || y >= rows || x >= cols) {
		return false;
	}
	return true;
}

bool Projection(cv::Point2f& pt, float& depth, cv::Mat R, cv::Mat T, cv::Mat K, cv::Mat X3D) {
	cv::Mat prev = R*X3D + T;
	prev = K*prev;
	depth = prev.at<float>(2);
	pt = cv::Point2f(prev.at<float>(0) / prev.at<float>(2), prev.at<float>(1) / prev.at<float>(2));
	if (depth < 0.0) {
		return false;
	}
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
////200410 Optical flow

int UVR_SLAM::Matcher::OpticalMatchingForInitialization(Frame* init, Frame* curr, std::vector<std::pair<cv::Point2f, cv::Point2f>>& resMatches) {

	cv::Mat overlap = cv::Mat::zeros(init->mnHeight, init->mnWidth, CV_8UC1);
	//////////////////////////
	////Optical flow
	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	std::vector<cv::Mat> currPyr, prevPyr;
	std::vector<uchar> status;
	std::vector<float> err;
	
	cv::Mat prevImg = init->GetOriginalImage();
	cv::Mat currImg = curr->GetOriginalImage();

	cv::Mat prevGray, currGray;
	cvtColor(prevImg, prevGray, CV_BGR2GRAY);
	cvtColor(currImg, currGray, CV_BGR2GRAY);
	///////debug
	cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
	cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
	cv::Mat debugging = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
	prevImg.copyTo(debugging(mergeRect1));
	currImg.copyTo(debugging(mergeRect2));
	///////debug

	///////////
	//matKPs, mvKPs
	//init -> curr로 매칭
	////////
	std::vector<cv::Point2f> prevPts, currPts;
	prevPts = init->mvPts;
	int maxLvl = 3;
	int searchSize = 21;
	cv::buildOpticalFlowPyramid(currImg, currPyr, cv::Size(searchSize, searchSize), maxLvl);
	maxLvl = cv::buildOpticalFlowPyramid(prevImg, prevPyr, cv::Size(searchSize, searchSize), maxLvl);
	cv::calcOpticalFlowPyrLK(prevPyr, currPyr, prevPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);
	//바운더리 에러도 고려해야 함.
	
	int res = 0;
	int nTotal = 0;
	int nKeypoint = 0;
	int nBad = 0;
	int nEpi = 0;
	int n3D = 0;

	std::cout << "opti::"<<init->mnFrameID<<"::" << prevPts.size() << std::endl;

	for (int i = 0; i < prevPts.size(); i++) {
		if (status[i] == 0) {
			nBad++;
			continue;
		}

		if (!curr->mpMatchInfo->CheckOpticalPointOverlap(overlap, currPts[i], mpSystem->mnRadius)) {
			nBad++;
			continue;
		}

		/////
		//매칭 결과
		float diffX = abs(prevPts[i].x - currPts[i].x);
		bool bMatch = false;
		if (diffX < 15) {
			bMatch = true;
			res++;
			cv::line(debugging, prevPts[i], currPts[i] + ptBottom, cv::Scalar(255, 0, 255));
		}
		else if (diffX >= 15 && diffX < 90) {
			res++;
			cv::line(debugging, prevPts[i], currPts[i] + ptBottom, cv::Scalar(0, 255, 255));
			bMatch = true;
		}
		else {
			cv::line(debugging, prevPts[i], currPts[i] + ptBottom, cv::Scalar(255, 255, 0));
		}

		if (bMatch){
			cv::rectangle(overlap, currPts[i] - mpSystem->mRectPt, currPts[i] + mpSystem->mRectPt, cv::Scalar(255, 0, 0), -1);
			resMatches.push_back(std::pair<cv::Point2f, cv::Point2f>(prevPts[i], currPts[i]));
		}
		//매칭 결과
		////

	}
	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;

	//fuse time text 
	std::stringstream ss;
	ss << "Optical flow init= " << res<<", "<<tttt;
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	//mpVisualizer->SetOutputImage(debugging, 0);
	/////////////////////////

	return res;
}

int UVR_SLAM::Matcher::OpticalMatchingForInitialization(Frame* init, Frame* curr, std::vector<cv::Point2f>& vpPts1, std::vector<cv::Point2f>& vpPts2, std::vector<bool>& vbInliers, std::vector<int>& vnIDXs, cv::Mat& debugging) {

	cv::Mat overlap = cv::Mat::zeros(init->mnHeight, init->mnWidth, CV_8UC1);
	//////////////////////////
	////Optical flow
	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	std::vector<cv::Mat> currPyr, prevPyr;
	std::vector<uchar> status;
	std::vector<float> err;
	cv::Mat prevImg = init->GetOriginalImage();
	cv::Mat currImg = curr->GetOriginalImage();
	cv::Mat prevGray, currGray;
	cvtColor(prevImg, prevGray, CV_BGR2GRAY);
	cvtColor(currImg, currGray, CV_BGR2GRAY);
	////Optical flow
	
	///////debug
	cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
	cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
	debugging = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
	prevImg.copyTo(debugging(mergeRect1));
	currImg.copyTo(debugging(mergeRect2));
	///////debug
	
	///////////
	//matKPs, mvKPs
	//init -> curr로 매칭
	////////
	std::vector<cv::Point2f> prevPts, currPts;
	std::vector<CandidatePoint*> vpCPs;
	prevPts = init->mpMatchInfo->GetMatchingPtsMapping(vpCPs);//init->mpMatchInfo->mvTempPts;

	int maxLvl = 3;
	int searchSize = 21;
	int tlv1 = cv::buildOpticalFlowPyramid(currImg, currPyr, cv::Size(searchSize, searchSize), maxLvl);
	maxLvl = cv::buildOpticalFlowPyramid(prevImg, prevPyr, cv::Size(searchSize, searchSize), maxLvl);
	cv::calcOpticalFlowPyrLK(prevPyr, currPyr, prevPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);
	//바운더리 에러도 고려해야 함.

	int res = 0;
	int nTotal = 0;
	int nKeypoint = 0;
	int nBad = 0;
	int nEpi = 0;
	int n3D = 0;

	for (int i = 0; i < prevPts.size(); i++) {
		if (status[i] == 0) {
			nBad++;
			continue;
		}
		if (!curr->isInImage(currPts[i].x, currPts[i].y, 10)) {
			continue;
		}
		
		if (!curr->mpMatchInfo->CheckOpticalPointOverlap(overlap, currPts[i], mpSystem->mnRadius)) {
			nBad++;
			continue;
		}

		/////
		//매칭 결과
		float diffX = abs(prevPts[i].x - currPts[i].x);
		if (diffX > 90) {
			continue;
		}
		cv::rectangle(overlap, currPts[i] - mpSystem->mRectPt, currPts[i] + mpSystem->mRectPt, cv::Scalar(255, 0, 0), -1);

		/*if (!curr->mpMatchInfo->CheckOpticalPointOverlap(curr->mpMatchInfo->used, 1, 10, currPts[i]))
			continue;*/

		vpPts1.push_back(prevPts[i]);
		vpPts2.push_back(currPts[i]);
		vbInliers.push_back(true);
		vnIDXs.push_back(i);
		/*if (i > init->mpMatchInfo->mvnMatchingPtIDXs.size())
			std::cout << "match::" << i << ", " << init->mpMatchInfo->mvnMatchingPtIDXs[i] << ", " << init->mpMatchInfo->mvnMatchingPtIDXs.size() << std::endl;
		vnIDXs.push_back(init->mpMatchInfo->mvnMatchingPtIDXs[i]);*/
		cv::circle(debugging, prevPts[i], 2, cv::Scalar(255, 255, 0));
		cv::circle(debugging, currPts[i]+ptBottom, 2, cv::Scalar(255, 255, 0));
	}

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;

	////fuse time text 
	std::stringstream ss;
	ss << "Optical flow init= " << vpPts1.size() << ", " << tttt;
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	
	cv::Mat resized;
	cv::resize(debugging, resized, cv::Size(debugging.cols / 2, debugging.rows / 2));
	cv::Rect rect1(0, 0, resized.cols, resized.rows/2);
	cv::Rect rect2(0, resized.rows / 2, resized.cols, resized.rows / 2);
	mpVisualizer->SetOutputImage(resized(rect1), 0);
	mpVisualizer->SetOutputImage(resized(rect2), 1);
	///////////////////////////

	return vpPts2.size();
}

bool  UVR_SLAM::Matcher::OpticalGridMatching(FrameGrid* grid1, cv::Mat src1, cv::Mat src2, std::vector<cv::Point2f>& vPrevPTs, std::vector<cv::Point2f>& vCurrPTs) {
	int searchSize = mpSystem->mnRadius;
	int maxLvl = 3;
	std::vector<uchar> status;
	std::vector<float> err;
	std::vector<cv::Point2f> pts1, pts2;

	if (grid1->vecPTs.size() < 1){
		std:cout << "????????????????" << std::endl;
		return false;
	}
	cv::calcOpticalFlowPyrLK(src1, src2, grid1->vecPTs, pts2, status, err, cv::Size(searchSize, searchSize), maxLvl);
	for (size_t i = 0, iend = grid1->vecPTs.size(); i < iend; i++) {
		if (!status[i])
			continue;
		vPrevPTs.push_back(grid1->vecPTs[i]);
		vCurrPTs.push_back(pts2[i]);
	}
	if (vCurrPTs.size() == 0){
		return false;
	}
	return true;
}
int UVR_SLAM::Matcher::OpticalGridsMatching(Frame* pFrame1, Frame* pFrame2, std::vector<cv::Point2f>& vpPTs) {

	cv::Mat img1 = pFrame1->GetOriginalImage().clone();
	cv::Mat img2 = pFrame2->GetOriginalImage().clone();

	cv::Rect mergeRect1 = cv::Rect(0, 0, img1.cols, img1.rows);
	cv::Rect mergeRect2 = cv::Rect(0, img1.rows, img1.cols, img1.rows);
	
	int nCurrFrameID = pFrame2->mnFrameID;
	cv::Mat overlap = cv::Mat::zeros(mHeight, mWidth, CV_8UC1);
	int nGridSize = mpSystem->mnRadius * 2;
	int searchSize = mpSystem->mnRadius;
	int maxLvl = 3;
	std::vector<cv::Point2f> pts1, pts2;
	std::vector<uchar> status;
	std::vector<float> err;
	for (auto iter = pFrame1->mmpFrameGrids.begin(), iend = pFrame1->mmpFrameGrids.end(); iter != iend; iter++) {
		auto rectPt = iter->first;
		auto pGrid = iter->second;
		auto bGrid = pFrame1->mmbFrameGrids[rectPt];
		if (!bGrid)
			continue;
		auto rect = pGrid->rect;
		
		/*std::vector<uchar> status;
		std::vector<float> err;*/
		
		std::vector<cv::Mat> pyr1, pyr2;
		/*cv::buildOpticalFlowPyramid(img1(rect), pyr1, cv::Size(searchSize, searchSize), maxLvl);
		maxLvl = cv::buildOpticalFlowPyramid(img2(rect), pyr2, cv::Size(searchSize, searchSize), maxLvl);*/
		
		//if (pGrid->vecPTs.size()<3){
		//	continue;
		//}
		//////여기는 테스트 과정에서 일단 추가한 것. 원래는 없으면 생성하고 만들어야 함.
		//auto pGrid2 = pFrame2->mmpFrameGrids[rectPt];
		//if (!pGrid2)
		//	continue;
		//cv::calcOpticalFlowPyrLK(img1(rect), img2(rect), pGrid->vecPTs, pts2, status, err, cv::Size(searchSize, searchSize), maxLvl);
		//cv::calcOpticalFlowPyrLK(img1, img2, pGrid->vecPTs, pts2, status, err, cv::Size(searchSize, searchSize), maxLvl);
		for (size_t j = 0, jend = pGrid->vecPTs.size(); j < jend; j++) {
			pts1.push_back(pGrid->vecPTs[j]);
			//if (!status[j])
			//	continue;
			//auto pt1 = pGrid->vecPTs[j];// +pGrid->basePt;
			//auto pt2 = pts2[j];// +pGrid->basePt;
			//
			//pGrid2->vecPTs.push_back(pts2[j]);

			//cv::circle(img1, pt1, 1, cv::Scalar(255, 0, 255), -1);
			//cv::circle(img2, pt2, 1, cv::Scalar(255, 0, 255), -1);
			//cv::line(img2, pt1, pt2, cv::Scalar(255, 255, 0), 1);
		}

		//auto pt1 = pGrid->pt;
		//pt1.x -= rect.x;
		//pt1.y -= rect.y;
		//pts1.push_back(pt1);
		//cv::calcOpticalFlowPyrLK(img1(rect), img2(rect), pts1, pts2, status, err, cv::Size(searchSize, searchSize), maxLvl);
		//if (!status[0]) {
		//	continue;
		//}
		////////포인트 위치 변환
		//cv::Point2f matchPt1, matchPt2;
		//matchPt1 = pt1;
		//matchPt2 = pts2[0];
		//matchPt1.x += rect.x;
		//matchPt1.y += rect.y;
		//matchPt2.x += rect.x;
		//matchPt2.y += rect.y;

		//cv::rectangle(overlap, matchPt2 - mpSystem->mRectPt, matchPt2 + mpSystem->mRectPt, cv::Scalar(255, 0, 0), -1);
		//cv::circle(img1, matchPt1, 2, cv::Scalar(255, 0, 255), -1);
		//cv::circle(img2, matchPt2, 2, cv::Scalar(255, 0, 255), -1);
		//cv::line(img2, matchPt1, matchPt2, cv::Scalar(255, 255, 0), 1);

		//////포인트 위치 변환
		//if (!pFrame1->mpMatchInfo->CheckOpticalPointOverlap(overlap, matchPt2, mpSystem->mnRadius)) {
		//	continue;
		//}
		////정식 매칭에서 수행 예정
		//auto gridPt = pFrame1->GetGridBasePt(matchPt2, nGridSize);
		//if (pFrame2->mmbFrameGrids[gridPt]) {
		//	continue;
		//}
		//auto pCPi = pGrid->mpCP;
		//if (pCPi->mnTrackingFrameID == nCurrFrameID) {
		//	std::cout << "tracking::" << pCPi->mnCandidatePointID << ", " << nCurrFrameID << std::endl;
		//	continue;
		//}

		//////grid
		//pFrame2->mmbFrameGrids[gridPt] = true;
		//pFrame2->mmpFrameGrids[gridPt] = new FrameGrid(gridPt, nGridSize);
		//pFrame2->mmpFrameGrids[gridPt]->mpCP = pCPi;
		//pFrame2->mmpFrameGrids[gridPt]->pt = matchPt2;
		//////grid
		//pCPi->mnTrackingFrameID = nCurrFrameID;
		///*vpCPs.push_back(pCPi);
		//vpPts.push_back(currPts[i]);*/
		////정식 매칭에서 수행 예정
		
		
	}
	cv::calcOpticalFlowPyrLK(img1, img2, pts1, pts2, status, err, cv::Size(searchSize, searchSize), maxLvl);
	for (size_t j = 0, jend = pts1.size(); j < jend; j++) {
		if (!status[j])
			continue;
		auto pt1 = pts1[j];// +pGrid->basePt;
		auto pt2 = pts2[j];// +pGrid->basePt;
		
		cv::circle(img1, pt1, 1, cv::Scalar(255, 0, 255), -1);
		cv::circle(img2, pt2, 1, cv::Scalar(255, 0, 255), -1);
		cv::line(img2, pt1, pt2, cv::Scalar(255, 255, 0), 1);
	}
	cv::Mat debugMatch = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	img1.copyTo(debugMatch(mergeRect1));
	img2.copyTo(debugMatch(mergeRect2));

	cv::moveWindow("Output::MatchTest", mpSystem->mnDisplayX, mpSystem->mnDisplayY);
	cv::imshow("Output::MatchTest", debugMatch);
	cv::waitKey(1);
}

int UVR_SLAM::Matcher::OpticalMatchingForTracking(Frame* prev, Frame* curr, std::vector<UVR_SLAM::CandidatePoint*>& vpCPs, std::vector<cv::Point2f>& vPrevPts, std::vector<cv::Point2f>& vCurrPts, std::vector<bool>& vbInliers) {
	
	std::vector<cv::Mat> currPyr, prevPyr;
	std::vector<uchar> status;
	std::vector<float> err;
	cv::Mat prevImg = prev->GetOriginalImage();
	cv::Mat currImg = curr->GetOriginalImage();

	int maxLvl = 3;
	int searchSize = 15;
	std::vector<cv::Point2f> prevPts, currPts;
	prevPts = prev->mpMatchInfo->mvMatchingPts;
	cv::buildOpticalFlowPyramid(currImg, currPyr, cv::Size(searchSize, searchSize), maxLvl);
	maxLvl = cv::buildOpticalFlowPyramid(prevImg, prevPyr, cv::Size(searchSize, searchSize), maxLvl);
	cv::waitKey(1);
	cv::calcOpticalFlowPyrLK(prevPyr, currPyr, prevPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);
	cv::Mat overlap = cv::Mat::zeros(mHeight, mWidth, CV_8UC1);

	int nCurrFrameID = curr->mnFrameID;
	int res = 0;
	int nBad = 0;
	int nGridSize = mpSystem->mnRadius * 2;
	////그리드와 점유 체크
	for (size_t i = 0, iend = prevPts.size(); i < iend; i++){
		if (status[i] == 0) {
			continue;
		}
		/*if (!prev->mpMatchInfo->CheckOpticalPointOverlap(overlap, currPts[i], mpSystem->mnRadius)) {
			nBad++;
			continue;
		}*/
		
		/*auto gridPt = prev->GetGridBasePt(currPts[i], nGridSize);
		if (curr->mmbFrameGrids[gridPt]) {
			continue;
		}*/

		auto pCPi = prev->mpMatchInfo->mvpMatchingCPs[i];
		if (pCPi->mnTrackingFrameID == nCurrFrameID) {
			std::cout << "tracking::" << pCPi->mnCandidatePointID << ", " << nCurrFrameID << std::endl;
			continue;
		}
		
		/*auto prevGridPt = prev->GetGridBasePt(prevPts[i], nGridSize);
		auto diffx = abs(prevGridPt.x - gridPt.x);
		auto diffy = abs(prevGridPt.y - gridPt.y);
		
		if (diffx > 2 * nGridSize || diffy > 2 * nGridSize) {
			cv::line(currImg, currPts[i], prevPts[i], cv::Scalar(0, 255, 255), 1);
			continue;
		}*/
		
		//////grid matching
		//auto rect = cv::Rect(gridPt, std::move(cv::Point2f(gridPt.x + nGridSize, gridPt.y + nGridSize)));
		//auto prevGrid = prev->mmpFrameGrids[prevGridPt];
		//if (!prevGrid) {
		//	std::cout << "tracking::error" << std::endl;
		//	continue;
		//}
		//auto prevRect = prevImg(prevGrid->rect);
		//auto currRect = currImg(rect);
		//std::vector<cv::Point2f> vPrevGridPTs, vGridPTs;
		///*bool bGridMatch = this->OpticalGridMatching(prevGrid, prevRect, currRect, vPrevGridPTs, vGridPTs);
		//if (!bGridMatch)
		//	continue;*/
		//////grid matching
		//////grid 추가
		//curr->mmbFrameGrids[gridPt] = true;
		//auto currGrid = new FrameGrid(gridPt, rect);
		////currGrid->vecPTs = vGridPTs;
		//curr->mmpFrameGrids[gridPt] = currGrid;
		//curr->mmpFrameGrids[gridPt]->mpCP = pCPi;
		//curr->mmpFrameGrids[gridPt]->pt = currPts[i];
		//////grid 추가

		pCPi->mnTrackingFrameID = nCurrFrameID;
		vpCPs.push_back(pCPi);
		vPrevPts.push_back(prevPts[i]);
		vCurrPts.push_back(currPts[i]);
		vbInliers.push_back(true);
		//cv::rectangle(overlap, currPts[i] - mpSystem->mRectPt, currPts[i] + mpSystem->mRectPt, cv::Scalar(255, 0, 0), -1);

		//auto pMPi = pCPi->GetMP();
		//
		//if (pMPi && !pMPi->isDeleted() && pMPi->GetRecentTrackingFrameID() != nCurrFrameID && curr->isInFrustum(pMPi, 0.6f)) {
		//	pMPi->SetRecentTrackingFrameID(nCurrFrameID);
		//	vpMPs.push_back(pMPi); 의 index
		//	vpPts.push_back(currPts[i]);
		//	vbInliers.push_back(true);
		//}
		
		////그리드 내의 추가 포인트 시각화
		//for (size_t j = 0, jend = vGridPTs.size(); j < jend; j++) {
		//	cv::circle(prevImg, vPrevGridPTs[j]+prevGridPt, 2, cv::Scalar(0, 0, 255), -1);
		//	//cv::circle(currImg, vGridPTs[j]+gridPt, 2, cv::Scalar(0, 0, 255), -1);
		//	cv::line(currImg, vGridPTs[j]+ gridPt, vPrevGridPTs[j]+ prevGridPt, cv::Scalar(255, 0, 255), 1);
		//}
		////그리드 내의 추가 포인트 시각화

		/*cv::circle(prevImg, prevPts[i], 2, cv::Scalar(255, 0, 0), -1);
		cv::circle(currImg, currPts[i], 2, cv::Scalar(255, 0, 0), -1);
		cv::line(currImg, currPts[i], prevPts[i], cv::Scalar(255, 255, 0), 1);*/
		res++;
	}

	/*cv::Mat debugMatch = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
	prevImg.copyTo(debugMatch(mergeRect1));
	currImg.copyTo(debugMatch(mergeRect2));
	cv::moveWindow("Output::MatchTest2", mpSystem->mnDisplayX+prevImg.cols, mpSystem->mnDisplayY);
	cv::imshow("Output::MatchTest2", debugMatch);*/
	return res;
}

int UVR_SLAM::Matcher::OpticalMatchingForTracking(Frame* prev, Frame* curr, std::vector<UVR_SLAM::CandidatePoint*>& vpCPs, std::vector<UVR_SLAM::MapPoint*>& vpMPs, std::vector<cv::Point2f>& vpPts,std::vector<bool>& vbInliers, std::vector<int>& vnIDXs, cv::Mat& overlap) {
	
	//////////////////////////
	////Optical flow
	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	std::vector<cv::Mat> currPyr, prevPyr;
	std::vector<uchar> status;
	std::vector<float> err;
	cv::Mat prevImg = prev->GetOriginalImage();
	cv::Mat currImg = curr->GetOriginalImage();
	
	///////////
	//matKPs, mvKPs
	//init -> curr로 매칭
	////////
	std::vector<cv::Point2f> prevPts, currPts;
	std::vector<MapPoint*> vpTempMPs;
	std::vector<UVR_SLAM::CandidatePoint*> vpTempCPs;

	for (size_t i = 0, iend = prev->mpMatchInfo->mvpMatchingCPs.size(); i < iend; i++){
		auto pCPi = prev->mpMatchInfo->mvpMatchingCPs[i];
		if (prevPts.size() == 500)
			break;
		auto pMPi = pCPi->GetMP();
		if (pMPi && !pMPi->isDeleted() && pMPi->GetQuality() && pMPi->isOptimized()) {
			prevPts.push_back(prev->mpMatchInfo->mvMatchingPts[i]);
			vpTempCPs.push_back(pCPi);
			vpTempMPs.push_back(pMPi);
		}
	}
	int maxLvl = 3;
	int searchSize = 21;
	//int searchSize = 21 + 10*(curr->GetFrameID() - prev->GetFrameID()-1);
	cv::buildOpticalFlowPyramid(currImg, currPyr, cv::Size(searchSize, searchSize), maxLvl);
	maxLvl = cv::buildOpticalFlowPyramid(prevImg, prevPyr, cv::Size(searchSize, searchSize), maxLvl);
	cv::calcOpticalFlowPyrLK(prevPyr, currPyr, prevPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);
	//바운더리 에러도 고려해야 함.
	
	int nCurrFrameID = curr->mnFrameID;
	int res = 0;
	int nBad = 0;

	for (int i = 0; i < prevPts.size(); i++) {

		if (status[i] == 0) {
			continue;
		}

		if (!curr->isInImage(currPts[i].x, currPts[i].y,10)) {
			continue;
		}
		/*if (curr->mEdgeImg.at<uchar>(currPts[i]) == 0)
			continue;*/
		if (!curr->mpMatchInfo->CheckOpticalPointOverlap(overlap, currPts[i], mpSystem->mnRadius)) {
			nBad++;
			continue;
		}

		////매칭 결과
		//float diffX = abs(prevPts[i].x - currPts[i].x);
		//if (diffX > 25) {
		//	continue;
		//}
		if (vpTempCPs[i]->mnTrackingFrameID == nCurrFrameID){
			std::cout << "tracking::" <<vpTempCPs[i]->mnCandidatePointID<<", "<<nCurrFrameID<< std::endl;
			continue;
		}
		cv::rectangle(overlap, currPts[i] - mpSystem->mRectPt, currPts[i] + mpSystem->mRectPt, cv::Scalar(255, 0, 0), -1);
		vpTempCPs[i]->mnTrackingFrameID = nCurrFrameID;
		UVR_SLAM::MapPoint* pMPi = vpTempMPs[i];
		if (pMPi && !pMPi->isDeleted() && pMPi->GetRecentTrackingFrameID() != nCurrFrameID && curr->isInFrustum(pMPi, 0.6f)) {
			pMPi->SetRecentTrackingFrameID(nCurrFrameID);
			vpMPs.push_back(pMPi);
			vpCPs.push_back(vpTempCPs[i]);
			vnIDXs.push_back(i);
			vpPts.push_back(currPts[i]);
			vbInliers.push_back(true);
		}
		/*else {
			cv::circle(debugging, prevPts[i], 1, cv::Scalar(255, 0, 255), -1);
			cv::circle(debugging, currPts[i] + ptBottom, 1, cv::Scalar(255, 0, 255), -1);
		}*/
		res++;
	}
	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;
	return res;
}

int UVR_SLAM::Matcher::OpticalMatchingForMapping(Map* pMap, Frame* pCurrKF, Frame* pPrevKF, Frame* pPPrevKF, std::vector<cv::Point2f>& vMatchedPPrevPts, std::vector<cv::Point2f>& vMatchedPrevPts, std::vector<cv::Point2f>& vMatchedCurrPts, std::vector<CandidatePoint*>& vMatchedCPs, cv::Mat K, cv::Mat InvK, double& dtime, cv::Mat& debugging) {
	//////////////////////////
	////Optical flow
	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	std::vector<cv::Mat> currPyr, prevPyr, pprevPyr;
	std::vector<uchar> statusCurr, statusPPrev;
	std::vector<float> errCurr, errPPrev;
	////Optical flow
	////debug
	////이미지 생성
	
	////이미지 생성
	cv::Mat prevImg = pPrevKF->GetOriginalImage();
	cv::Mat currImg = pCurrKF->GetOriginalImage();
	cv::Mat pprevImg = pPPrevKF->GetOriginalImage();
	//cv::Mat testImg = debugging.clone();
	
	cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
	cv::Mat testImg = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
	cv::Point2f ptBottom(0, testImg.rows / 2);
	prevImg.copyTo(testImg(mergeRect1));
	pprevImg.copyTo(testImg(mergeRect2));

	///////////

	std::vector<cv::Point2f> pprevPts, prevPts, currPts;
	std::vector<int> vnOctaves;
	std::vector<CandidatePoint*> vpCPs;

	int nCurrKeyFrameID = pCurrKF->mnKeyFrameID;
	vpCPs = pPrevKF->mpMatchInfo->mvpMatchingCPs;
	prevPts = pPrevKF->mpMatchInfo->mvMatchingPts;//pPrevKF->mpMatchInfo->GetMatchingPtsMapping(vpCPs);
	auto pPPrevMatchInfo = pPPrevKF->mpMatchInfo;
	auto pPrevMatchInfo = pPrevKF->mpMatchInfo;
	auto pCurrMatchInfo = pCurrKF->mpMatchInfo;

	int maxLvl = 3;
	int searchSize = 21;//21
	cv::buildOpticalFlowPyramid(currImg, currPyr, cv::Size(searchSize, searchSize), maxLvl);
	maxLvl = cv::buildOpticalFlowPyramid(prevImg, prevPyr, cv::Size(searchSize, searchSize), maxLvl);
	//maxLvl = cv::buildOpticalFlowPyramid(pprevImg, pprevPyr, cv::Size(searchSize, searchSize), maxLvl);
	cv::calcOpticalFlowPyrLK(prevPyr, currPyr, prevPts, currPts, statusCurr, errCurr, cv::Size(searchSize, searchSize), maxLvl);
	//cv::calcOpticalFlowPyrLK(prevPyr, pprevPyr, prevPts, pprevPts, statusPPrev, errPPrev, cv::Size(searchSize, searchSize), maxLvl);

	////옵티컬 플로우 매칭 저장
	std::vector<cv::Point2f> vTempPrevPts, vTempCurrPts;
	std::vector<int> vTempIDXs;

	cv::Mat usedCurr = cv::Mat::zeros(prevImg.size(), CV_8UC1);
	cv::Mat usedPrev = cv::Mat::zeros(prevImg.size(), CV_8UC1);
	cv::Mat usedPPrev = cv::Mat::zeros(prevImg.size(), CV_8UC1);

	for (int i = 0; i < prevPts.size(); i++) {
		//if (statusCurr[i] == 0 || statusPPrev[i] == 0) {
		if (statusCurr[i] == 0){
			continue;
		}

		auto pCPi = vpCPs[i];
		if (pCPi->GetLastVisibleFrame() == nCurrKeyFrameID) {
			std::cout << "???????????::" <<pCPi->mnCandidatePointID<< std::endl;
			continue;
		}

		bool b3 = pPrevMatchInfo->CheckOpticalPointOverlap(usedPrev, prevPts[i], mpSystem->mnRadius);
		bool b4 = pPrevMatchInfo->CheckOpticalPointOverlap(usedCurr, currPts[i], mpSystem->mnRadius);
		bool b5 = true;//pPrevMatchInfo->CheckOpticalPointOverlap(usedPPrev, Frame::mnRadius, 10, pprevPts[i]);
		
		if (!b3 || !b4 || !b5) {
			continue;
		}
		pCPi->SetLastVisibleFrame(nCurrKeyFrameID);

		cv::rectangle(usedPrev, prevPts[i] - mpSystem->mRectPt, prevPts[i] + mpSystem->mRectPt, cv::Scalar(255, 0, 0), -1);
		cv::rectangle(usedCurr, currPts[i] - mpSystem->mRectPt, currPts[i] + mpSystem->mRectPt, cv::Scalar(255, 0, 0), -1);
		//cv::circle(usedPrev, prevPts[i], Frame::mnRadius, cv::Scalar(255, 0, 0), -1);
		//cv::circle(usedCurr, currPts[i], Frame::mnRadius, cv::Scalar(255, 0, 0), -1);

		//vMatchedPPrevPts.push_back(pprevPts[i]);
		vMatchedPrevPts.push_back(prevPts[i]);
		vMatchedCurrPts.push_back(currPts[i]);
		vMatchedCPs.push_back(pCPi);
		cv::circle(debugging, prevPts[i], 4, cv::Scalar(0, 255, 255));
		cv::circle(debugging, currPts[i]+ptBottom, 4, cv::Scalar(0, 255, 255));
	}

	/////MP교환 테스트
	for (int i = 0; i < vMatchedCPs.size(); i++) {
		auto pCPi = vMatchedCPs[i];
		int cidx = pCPi->GetPointIndexInFrame(pCurrMatchInfo);
		int currIDX = pCurrMatchInfo->CheckOpticalPointOverlap(vMatchedCurrPts[i], mpSystem->mnRadius);
		if (currIDX >= 0) {
			if (cidx >= 0)
				std::cout << "a;sldjf;alskdfa" << std::endl;
			auto pCP2 = pCurrMatchInfo->mvpMatchingCPs[currIDX];
			if(pCP2->mnCandidatePointID != pCPi->mnCandidatePointID){
				pCurrMatchInfo->mvpMatchingCPs[currIDX] = pCPi;
			}
		}
		//////prev & pprev
		//bool bConnectPPrevKF = false;
		//int pidx = pCPi->GetPointIndexInFrame(pPPrevMatchInfo);
		//int pprevIDX = pPPrevMatchInfo->CheckOpticalPointOverlap(Frame::mnRadius, 10, vMatchedPPrevPts[i]);
		//if (pprevIDX >= 0) {
		//	auto pCP2 = pPPrevMatchInfo->mvpMatchingCPs[pprevIDX];
		//	if (pCP2->mnCandidatePointID != pCPi->mnCandidatePointID) {
		//		if (pidx > -1) {
		//			pCPi->DisconnectFrame(pPPrevMatchInfo);
		//			auto pMPi = pCPi->GetMP();
		//			if (pMPi && pMPi->GetQuality() && !pMPi->isDeleted()) {
		//				pMPi->DisconnectFrame(pPPrevMatchInfo);
		//			}
		//		}
		//		pCP2->DisconnectFrame(pPPrevMatchInfo);
		//		auto pMPi = pCP2->GetMP();
		//		if (pMPi && pMPi->GetQuality() && !pMPi->isDeleted()) {
		//			pMPi->DisconnectFrame(pPPrevMatchInfo);
		//		}
		//		bConnectPPrevKF = true;
		//		pPPrevMatchInfo->mvpMatchingCPs[pprevIDX] = pCPi;
		//	}
		//}
		//else if (pprevIDX == -1) {
		//	if (pidx > -1) {
		//		pCPi->DisconnectFrame(pPPrevMatchInfo);
		//		auto pMPi = pCPi->GetMP();
		//		if (pMPi && pMPi->GetQuality() && !pMPi->isDeleted()) {
		//			pMPi->DisconnectFrame(pPPrevMatchInfo);
		//		}
		//	}
		//	bConnectPPrevKF = true;
		//	pprevIDX = pPPrevMatchInfo->AddCP(pCPi, vMatchedPPrevPts[i]);
		//}
		//if (bConnectPPrevKF) {
		//	pCPi->ConnectFrame(pPPrevMatchInfo, pprevIDX);
		//	auto pMPi = pCPi->GetMP();
		//	if (pMPi && pMPi->GetQuality() && !pMPi->isDeleted()) {
		//		pMPi->ConnectFrame(pPPrevMatchInfo, pprevIDX);
		//	}
		//}
		//////prev & pprev
	}

	//for (int i = 0; i < prevPts.size(); i++) {
	//	if (statusCurr[i] == 0 || statusPPrev[i] == 0) {
	//		continue;
	//	}

	//	auto pCPi = vpCPs[i];
	//	if (pCPi->GetLastVisibleFrame() == nCurrKeyFrameID){
	//		continue;
	//	}
	//	int currIDX = pCurrMatchInfo->CheckOpticalPointOverlap(Frame::mnRadius, 10, currPts[i]);
	//	int pprevIDX = pPPrevMatchInfo->CheckOpticalPointOverlap(Frame::mnRadius, 10, pprevPts[i]);
	//	auto pCPcurr = pCurrMatchInfo->mvpMatchingCPs[currIDX];
	//	auto pCPpprev = pPPrevMatchInfo->mvpMatchingCPs[pprevIDX];

	//	bool b1 = currIDX >= 0 && pCPcurr->mnCandidatePointID == pCPi->mnCandidatePointID;;
	//	bool b2 = pprevIDX >= 0 && pCPpprev->mnCandidatePointID == pCPi->mnCandidatePointID;
	//	
	//	bool b3 = pPrevMatchInfo->CheckOpticalPointOverlap(usedPrev, Frame::mnRadius, 10, prevPts[i]);
	//	bool b4 = pPrevMatchInfo->CheckOpticalPointOverlap(usedCurr, Frame::mnRadius, 10, currPts[i]);
	//	bool b5 = pPrevMatchInfo->CheckOpticalPointOverlap(usedPPrev, Frame::mnRadius, 10, pprevPts[i]);
	//	if (b1) {
	//		//update frame 참조
	//		//pCPi->SetLastSuccessFrame(nCurrKeyFrameID);
	//		continue;
	//	}
	//	else if (currIDX >= 0) {
	//		pCurrMatchInfo->mvpMatchingCPs[currIDX] = pCPi;
	//		//std::cout << pCurrMatchInfo->mvpMatchingCPs[currIDX]->mnCandidatePointID << ", " << pCPi->mnCandidatePointID << std::endl;

	//		/*auto pCP2 = pCurrMatchInfo->mvpMatchingCPs[currIDX];
	//		int tempIDx = pCP2->GetPointIndexInFrame(pPrevMatchInfo);
	//		auto pt222 = pPrevMatchInfo->mvMatchingPts[tempIDx];
	//		int tempIDx2 = pCPi->GetPointIndexInFrame(pCurrMatchInfo);
	//		if (tempIDx2 > -1) {
	//			auto pt333 = pCurrMatchInfo->mvMatchingPts[tempIDx2];
	//			cv::circle(testImg, pt333, 2, cv::Scalar(0, 255, 255));
	//		}
	//		cv::circle(testImg, prevPts[i], 3, cv::Scalar(255, 0, 255));
	//		cv::circle(testImg, pt222, 2, cv::Scalar(255, 255, 0));
	//		cv::circle(testImg, currPts[i]+ptBottom, 3, cv::Scalar(255, 0, 255));*/
	//		continue;
	//	}
	//	if (pprevIDX >= 0 && !b2) {
	//		//replace
	//		//disconnect & connect
	//		auto pCP2 = pPPrevMatchInfo->mvpMatchingCPs[pprevIDX];
	//		int tempIDx = pCP2->GetPointIndexInFrame(pPPrevMatchInfo);
	//		auto pt222 = pPPrevMatchInfo->mvMatchingPts[tempIDx];
	//		int tempIDx2 = pCPi->GetPointIndexInFrame(pPPrevMatchInfo);
	//		if (tempIDx2 > -1) {
	//			auto pt333 = pPPrevMatchInfo->mvMatchingPts[tempIDx2];
	//			//cv::circle(testImg, pt333+ptBottom, 2, cv::Scalar(0, 255, 255));
	//		}
	//		/*cv::circle(testImg, prevPts[i], 3, cv::Scalar(255, 0, 255));
	//		cv::circle(testImg, pt222, 2, cv::Scalar(255, 255, 0));
	//		cv::circle(testImg, pprevPts[i]+ptBottom, 3, cv::Scalar(255, 0, 255));*/
	//		//std::cout << pPPrevMatchInfo->mvpMatchingCPs[pprevIDX]->mnCandidatePointID << ", " << pCPi->mnCandidatePointID << std::endl;
	//		continue;
	//	}else if (pprevIDX == -1) {
	//		int tempIDx2 = pCPi->GetPointIndexInFrame(pPPrevMatchInfo);
	//		if (tempIDx2 > -1) {
	//			//test
	//			auto pt333 = pPPrevMatchInfo->mvMatchingPts[tempIDx2];

	//			/*cv::circle(testImg, prevPts[i], 3, cv::Scalar(0, 255, 0));
	//			cv::circle(testImg, pt333 + ptBottom, 3, cv::Scalar(125, 125, 125));
	//			cv::circle(testImg, pprevPts[i] + ptBottom, 3, cv::Scalar(0, 255, 0));*/
	//		}
	//		else {
	//			//add, prev && pprev
	//			/*cv::circle(testImg, prevPts[i], 3, cv::Scalar(255, 0, 0));
	//			cv::circle(testImg, pprevPts[i] + ptBottom, 3, cv::Scalar(255, 0, 0));*/
	//		}
	//	}
	//	if (!b3 || !b4 || !b5) {
	//		continue;
	//	}
	//	//pCPi->SetLastVisibleFrame(nCurrKeyFrameID);

	//	cv::circle(usedPrev, prevPts[i], Frame::mnRadius, cv::Scalar(255, 0, 0), -1);
	//	cv::circle(usedPPrev, pprevPts[i], Frame::mnRadius, cv::Scalar(255, 0, 0), -1);
	//	cv::circle(usedCurr, currPts[i], Frame::mnRadius, cv::Scalar(255, 0, 0), -1);

	//	vMatchedPPrevPts.push_back(pprevPts[i]);
	//	vMatchedPrevPts.push_back(prevPts[i]);
	//	vMatchedCurrPts.push_back(currPts[i]);
	//	vMatchedCPs.push_back(pCPi);

	//}
	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	dtime = duration / 1000.0;

	//imshow("test img", testImg); cv::waitKey(1);

	return vMatchedCurrPts.size();
}

int UVR_SLAM::Matcher::OpticalMatchingForMapping(Map* pMap, Frame* pCurrKF, Frame* pPrevKF, std::vector<cv::Point2f>& vMatchedPrevPts, std::vector<cv::Point2f>& vMatchedCurrPts, std::vector<CandidatePoint*>& vMatchedCPs, cv::Mat K, cv::Mat InvK, double& dtime, cv::Mat& debugging) {
	//////////////////////////
	////Optical flow
	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	std::vector<cv::Mat> currPyr, prevPyr;
	std::vector<uchar> status;
	std::vector<float> err;
	////Optical flow
	////debug
	cv::Mat prevImg = pPrevKF->GetOriginalImage();
	cv::Mat currImg = pCurrKF->GetOriginalImage();
	///////////

	std::vector<cv::Point2f> prevPts,currPts;
	std::vector<int> vnOctaves;
	std::vector<CandidatePoint*> vpCPs;

	int nCurrKeyFrameID = pCurrKF->mnKeyFrameID;
	vpCPs = pPrevKF->mpMatchInfo->mvpMatchingCPs;
	prevPts = pPrevKF->mpMatchInfo->mvMatchingPts;//pPrevKF->mpMatchInfo->GetMatchingPtsMapping(vpCPs);
	auto pPrevMatchInfo = pPrevKF->mpMatchInfo;
	auto pCurrMatchInfo = pCurrKF->mpMatchInfo;
	
	int maxLvl = 3;
	int searchSize = 21;//21
	cv::buildOpticalFlowPyramid(currImg, currPyr, cv::Size(searchSize, searchSize), maxLvl);
	maxLvl = cv::buildOpticalFlowPyramid(prevImg, prevPyr, cv::Size(searchSize, searchSize), maxLvl);
	cv::calcOpticalFlowPyrLK(prevPyr, currPyr, prevPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);
	
	////옵티컬 플로우 매칭 저장
	std::vector<cv::Point2f> vTempPrevPts, vTempCurrPts;
	std::vector<int> vTempIDXs;

	cv::Mat usedPrev = cv::Mat::zeros(prevImg.size(), CV_8UC1);
	cv::Mat usedCurr = cv::Mat::zeros(prevImg.size(), CV_8UC1);
	for (int i = 0; i < prevPts.size(); i++) {
		if (status[i] == 0) {
			continue;
		}
	
		auto pCPi = vpCPs[i];
		if (pCPi->GetLastVisibleFrame() == nCurrKeyFrameID)
			continue;
		bool b1 = pCurrMatchInfo->CheckOpticalPointOverlap(currPts[i], mpSystem->mnRadius) >= 0;
		bool b3 = pPrevMatchInfo->CheckOpticalPointOverlap(usedPrev, prevPts[i], mpSystem->mnRadius);
		bool b4 = pPrevMatchInfo->CheckOpticalPointOverlap(usedCurr, currPts[i], mpSystem->mnRadius);
		if (b1) {
			pCPi->SetLastSuccessFrame(nCurrKeyFrameID);
			continue;
		}
		if (!b3 || !b4) {
			continue;
		}
		pCPi->SetLastVisibleFrame(nCurrKeyFrameID);
		
		cv::rectangle(usedPrev, prevPts[i] - mpSystem->mRectPt, prevPts[i] + mpSystem->mRectPt, cv::Scalar(255, 0, 0), -1);
		cv::rectangle(usedCurr, currPts[i] - mpSystem->mRectPt, currPts[i] + mpSystem->mRectPt, cv::Scalar(255, 0, 0), -1);
		/*cv::circle(usedPrev, prevPts[i], Frame::mnRadius, cv::Scalar(255, 0, 0), -1);
		cv::circle(usedCurr, currPts[i], Frame::mnRadius, cv::Scalar(255, 0, 0), -1);*/

		vMatchedPrevPts.push_back(prevPts[i]);
		vMatchedCurrPts.push_back(currPts[i]);
		vMatchedCPs.push_back(pCPi);

	}
	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	dtime = duration / 1000.0;
	return vMatchedCurrPts.size();
}

int UVR_SLAM::Matcher::OpticalMatchingForMapping2(Map* pMap, Frame* pCurrKF, Frame* pPrevKF, std::vector<cv::Point2f>& vMatchedPrevPts, std::vector<cv::Point2f>& vMatchedCurrPts, std::vector<CandidatePoint*>& vMatchedCPs, cv::Mat K, cv::Mat InvK, double& dtime, cv::Mat& debugging) {
	//////////////////////////
	////Optical flow
	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	std::vector<cv::Mat> currPyr, prevPyr;
	std::vector<uchar> status;
	std::vector<float> err;
	////Optical flow
	////debug
	cv::Mat prevImg = pPrevKF->GetOriginalImage();
	cv::Mat currImg = pCurrKF->GetOriginalImage();
	cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
	cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
	debugging = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
	prevImg.copyTo(debugging(mergeRect1));
	currImg.copyTo(debugging(mergeRect2));
	///////////

	std::vector<cv::Point2f> prevPts, currPts;
	std::vector<int> vnOctaves;
	std::vector<CandidatePoint*> vpCPs;

	vpCPs = pPrevKF->mpMatchInfo->mvpMatchingCPs;
	prevPts = pPrevKF->mpMatchInfo->mvMatchingPts;//pPrevKF->mpMatchInfo->GetMatchingPtsMapping(vpCPs);
	auto pPrevMatchInfo = pPrevKF->mpMatchInfo;
	auto pCurrMatchInfo = pCurrKF->mpMatchInfo;

	int maxLvl = 3;
	int searchSize = 21;//21
	cv::buildOpticalFlowPyramid(currImg, currPyr, cv::Size(searchSize, searchSize), maxLvl);
	maxLvl = cv::buildOpticalFlowPyramid(prevImg, prevPyr, cv::Size(searchSize, searchSize), maxLvl);
	cv::calcOpticalFlowPyrLK(prevPyr, currPyr, prevPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);

	////옵티컬 플로우 매칭 저장
	std::vector<cv::Point2f> vTempPrevPts, vTempCurrPts;
	std::vector<int> vTempIDXs;

	cv::Mat usedPrev = cv::Mat::zeros(prevImg.size(), CV_8UC1);
	cv::Mat usedCurr = cv::Mat::zeros(prevImg.size(), CV_8UC1);
	for (int i = 0; i < prevPts.size(); i++) {
		if (status[i] == 0) {
			continue;
		}
		auto pCPi = vpCPs[i];
		bool b1 = pCurrMatchInfo->CheckOpticalPointOverlap(currPts[i], mpSystem->mnRadius) >= 0;
		bool b3 = pPrevMatchInfo->CheckOpticalPointOverlap(usedPrev, prevPts[i], mpSystem->mnRadius);
		bool b4 = pPrevMatchInfo->CheckOpticalPointOverlap(usedCurr, currPts[i], mpSystem->mnRadius);
		if (b1 || !b3 || !b4) {
			continue;
		}
		cv::rectangle(usedPrev, prevPts[i] - mpSystem->mRectPt, prevPts[i] + mpSystem->mRectPt, cv::Scalar(255, 0, 0), -1);
		cv::rectangle(usedCurr, currPts[i] - mpSystem->mRectPt, currPts[i] + mpSystem->mRectPt, cv::Scalar(255, 0, 0), -1);
		/*cv::circle(usedPrev, prevPts[i], Frame::mnRadius, cv::Scalar(255, 0, 0), -1);
		cv::circle(usedCurr, currPts[i], Frame::mnRadius, cv::Scalar(255, 0, 0), -1);*/

		vTempPrevPts.push_back(prevPts[i]);
		vTempCurrPts.push_back(currPts[i]);
		vTempIDXs.push_back(i);

	}
	if (vTempPrevPts.size() < 10) {
		std::cout << "LM::Matching::error" << std::endl;
	}

	std::vector<uchar> vFInliers;
	cv::Mat E12 = cv::findEssentialMat(vTempPrevPts, vTempCurrPts, K, cv::FM_RANSAC, 0.999, 1.0, vFInliers);

	int nRes = 0;
	int nTargetID = pPrevKF->mnFrameID;
	for (unsigned long i = 0; i < vFInliers.size(); i++) {
		if (vFInliers[i]) {
			auto currPt = vTempCurrPts[i];
			auto prevPt = vTempPrevPts[i];
			vMatchedCurrPts.push_back(currPt);
			vMatchedPrevPts.push_back(prevPt);
			int nCPidx = vTempIDXs[i];
			auto pCPi = vpCPs[nCPidx];

			////시각화
			if (pCPi->mnFirstID == nTargetID) {
				cv::circle(debugging, prevPt, 2, cv::Scalar(0, 0, 255), -1);
				cv::circle(debugging, currPt + ptBottom, 2, cv::Scalar(0, 0, 255), -1);
			}
			else {
				cv::circle(debugging, prevPt, 2, cv::Scalar(255, 255, 0), -1);
				cv::circle(debugging, currPt + ptBottom, 2, cv::Scalar(255, 255, 0), -1);
			}
			if (pCPi->GetMP()) {
				cv::circle(debugging, prevPt, 3, cv::Scalar(0, 255, 255));
				cv::circle(debugging, currPt + ptBottom, 3, cv::Scalar(0, 255, 255));
			}
			//else {
			//	////MP 생성 확인
			//	int label = pCPi->GetLabel();
			//	auto pMP = new UVR_SLAM::MapPoint(pMap, pCurrKF, pCPi, X3D, cv::Mat(), label, pCPi->octave);
			//	////MP 생성 확인
			//}
			pCurrKF->mpMatchInfo->AddCP(pCPi, currPt);
			vMatchedCPs.push_back(pCPi);

			nRes++;
		}
	}

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	dtime = duration / 1000.0;
}

////200410 Optical flow
///////////////////////////////////////////////////////////////////////////////////////////////////////