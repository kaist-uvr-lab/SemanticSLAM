#include <Matcher.h>
#include <omp.h>
#include <random>
#include <Frame.h>
#include <MapPoint.h>
#include <FrameWindow.h>
#include <MatrixOperator.h>
#include <gms_matcher.h>
#include <PlaneEstimator.h>
#include <Plane.h>
#include <Map.h>

UVR_SLAM::Matcher::Matcher(){}
UVR_SLAM::Matcher::Matcher(cv::Ptr < cv::DescriptorMatcher> _matcher, int w, int h)
	:mWidth(w), mHeight(h), TH_HIGH(100), TH_LOW(50), HISTO_LENGTH(30), mfNNratio(0.7), mbCheckOrientation(true), matcher(_matcher)
{
	//cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1);
	//cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);
	//matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
	
	//matcher = cv::makePtr<cv::BFMatcher>(cv::NORM_HAMMING, true);
	//matcher = DescriptorMatcher::create("FlannBased");
		
	//cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2), cv::makePtr<cv::flann::SearchParams>(50));
}
UVR_SLAM::Matcher::~Matcher(){}

const double nn_match_ratio = 0.7f; // Nearest-neighbour matching ratio

//////////////////////////////////////////////////////////////////
////////////일단 남겨놓긴 함. 근데 사용 안함
////////Fundamental Matrix를 위해 이용
bool UVR_SLAM::Matcher::CheckEpiConstraints(cv::Mat F12, cv::Point2f pt1, cv::Point2f pt2, float sigma, float& res) {
	// Epipolar line in second image l = x1'F12 = [a b c]
	// Epipolar line in second image l = x1'F12 = [a b c]
	const float a = pt1.x*F12.at<float>(0, 0) + pt1.y*F12.at<float>(0, 1) + F12.at<float>(0, 2);
	const float b = pt1.x*F12.at<float>(1, 0) + pt1.y*F12.at<float>(1, 1) + F12.at<float>(1, 2);
	const float c = pt1.x*F12.at<float>(2, 0) + pt1.y*F12.at<float>(2, 1) + F12.at<float>(2, 2);
	const float den = a*a + b*b;
	if (den == 0)
		return false;
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
		if (!CheckEpiConstraints(F12, prevKP.pt, kp.pt, sigma, epiDist))
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
bool CheckOpticalPointOverlap(cv::Mat& overlap, int radius, cv::Point2f pt) {
	if (overlap.at<uchar>(pt) > 0) {
		return false;
	}
	circle(overlap, pt, radius, cv::Scalar(255), -1);
	return true;
}

int UVR_SLAM::Matcher::OpticalMatchingForInitialization(Frame* init, Frame* curr, std::vector<std::pair<cv::Point2f, cv::Point2f>>& resMatches) {

	cv::Mat overlap = cv::Mat::zeros(init->GetOriginalImage().size(), CV_8UC1);
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

	std::cout << "opti::"<<init->GetFrameID()<<"::" << prevPts.size() << std::endl;

	for (int i = 0; i < prevPts.size(); i++) {
		if (status[i] == 0) {
			nBad++;
			continue;
		}

		if (!CheckOpticalPointOverlap(overlap, 2, currPts[i])) {
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

		if (bMatch)
			resMatches.push_back(std::pair<cv::Point2f, cv::Point2f>(prevPts[i], currPts[i]));
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
	cv::imshow("Init::OpticalFlow ", debugging);
	/////////////////////////

	return res;
}

int UVR_SLAM::Matcher::OpticalMatchingForInitialization(Frame* init, Frame* curr, std::vector<cv::Point2f>& vpPts1, std::vector<cv::Point2f>& vpPts2, std::vector<bool>& vbInliers, std::vector<int>& vnIDXs, cv::Mat& debugging) {

	cv::Mat overlap = cv::Mat::zeros(init->GetOriginalImage().size(), CV_8UC1);
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
	std::vector<int> vOctaves;
	prevPts = init->mpMatchInfo->GetMatchingPts(vOctaves);//init->mpMatchInfo->mvTempPts;

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
		
		/*if (!CheckOpticalPointOverlap(overlap, 2, currPts[i])) {
			nBad++;
			continue;
		}*/

		/////
		//매칭 결과
		float diffX = abs(prevPts[i].x - currPts[i].x);
		if (diffX > 90) {
			continue;
		}

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
	imshow("Init::OpticalFlow ", debugging);
	///////////////////////////

	return vpPts2.size();
}

int UVR_SLAM::Matcher::OpticalMatchingForTracking(Frame* prev, Frame* curr, std::vector<UVR_SLAM::MapPoint*>& vpMPs, std::vector<cv::Point2f>& vpPts,std::vector<bool>& vbInliers, std::vector<int>& vnIDXs, cv::Mat& overlap, cv::Mat& debugging) {
	
	//////////////////////////
	////Optical flow
	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	std::vector<cv::Mat> currPyr, prevPyr;
	std::vector<uchar> status;
	std::vector<float> err;
	
	///////debug
	cv::Mat prevImg = prev->GetOriginalImage();
	cv::Mat currImg = curr->GetOriginalImage();
	cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
	cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
	debugging = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
	prevImg.copyTo(debugging(mergeRect1));
	currImg.copyTo(debugging(mergeRect2));
	
	///////////
	//matKPs, mvKPs
	//init -> curr로 매칭
	////////
	std::vector<cv::Point2f> prevPts, currPts;
	prevPts = prev->mpMatchInfo->GetMatchingPts();//prev->mvMatchingPts;
	auto mvpMatchingMPs = prev->mpMatchInfo->GetMatchingMPs();
	int maxLvl = 3;
	int searchSize = 21;
	cv::buildOpticalFlowPyramid(currImg, currPyr, cv::Size(searchSize, searchSize), maxLvl);
	maxLvl = cv::buildOpticalFlowPyramid(prevImg, prevPyr, cv::Size(searchSize, searchSize), maxLvl);
	cv::calcOpticalFlowPyrLK(prevPyr, currPyr, prevPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);
	//바운더리 에러도 고려해야 함.
	
	int nCurrFrameID = curr->GetFrameID();
	int res = 0;
	int nBad = 0;

	for (int i = 0; i < prevPts.size(); i++) {

		cv::circle(debugging, prevPts[i], 3, cv::Scalar(255, 0, 0));

		if (status[i] == 0) {
			continue;
		}

		if (!curr->isInImage(currPts[i].x, currPts[i].y,10)) {
			continue;
		}
		/*if (curr->mEdgeImg.at<uchar>(currPts[i]) == 0)
			continue;*/
		if (!CheckOpticalPointOverlap(overlap, 2, currPts[i])) {
			nBad++;
			continue;
		}

		////매칭 결과
		//float diffX = abs(prevPts[i].x - currPts[i].x);
		//if (diffX > 25) {
		//	continue;
		//}
		
		UVR_SLAM::MapPoint* pMPi = mvpMatchingMPs[i]; 

		if (pMPi && !pMPi->isDeleted() && pMPi->GetRecentTrackingFrameID() != nCurrFrameID && curr->isInFrustum(pMPi, 0.6f)) {
			pMPi->SetRecentTrackingFrameID(nCurrFrameID);
			vpMPs.push_back(pMPi);
			vnIDXs.push_back(i);
			vpPts.push_back(currPts[i]);
			vbInliers.push_back(true);
		}
		/*else {
			cv::circle(debugging, prevPts[i], 1, cv::Scalar(255, 0, 255), -1);
			cv::circle(debugging, currPts[i] + ptBottom, 1, cv::Scalar(255, 0, 255), -1);
		}*/

		//트래킹 결과 출력
		
		cv::circle(debugging, prevPts[i], 1, cv::Scalar(255, 0, 255),-1);
		cv::circle(debugging, currPts[i] + ptBottom, 1, cv::Scalar(255, 0, 255), -1);
		cv::line(debugging, prevPts[i]+ ptBottom, currPts[i] + ptBottom, cv::Scalar(255, 255, 0));
		//cv::line(debugging2, curr->mpMatchInfo->mpTargetFrame->mpMatchInfo->mvMatchingPts[prev->mpMatchInfo->mvnMatchingPtIDXs[i]], currPts[i] + ptBottom, cv::Scalar(255, 255, 0));
		res++;
	}

	//////////////////////////////////
	//////////Edge Test
	//std::chrono::high_resolution_clock::time_point edge_start = std::chrono::high_resolution_clock::now();
	//std::vector<cv::Point2f> tempPts;
	//cv::Mat tempEdgeMap = cv::Mat::zeros(prevImg.size(), CV_8UC1);
	//int nEdge = 0;
	//prevPts = prev->mpMatchInfo->mvEdgePts;//prev->mvMatchingPts;
	//maxLvl = 3;
	//searchSize = 21;//21
	//cv::buildOpticalFlowPyramid(currImg, currPyr, cv::Size(searchSize, searchSize), maxLvl);
	//maxLvl = cv::buildOpticalFlowPyramid(prevImg, prevPyr, cv::Size(searchSize, searchSize), maxLvl);
	//cv::calcOpticalFlowPyrLK(prevPyr, currPyr, prevPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);
	////cv::calcOpticalFlowPyrLK(prevImg, currImg, prevPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);
	//for (int i = 0; i < prevPts.size(); i+=1) {
	//	if (status[i] == 0) {
	//		continue;
	//	}
	//	if (!curr->isInImage(currPts[i].x, currPts[i].y, 10)) {
	//		continue;
	//	}
	//	//매칭 결과
	//	float diffX = abs(prevPts[i].x - currPts[i].x);
	//	if (diffX > 25) {
	//		continue;
	//	}
	//	////바로 추가
	//	if (!curr->mpMatchInfo->CheckOpticalPointOverlap(curr->mpMatchInfo->edgeMap, 1,10, currPts[i])) {
	//		continue;
	//	}
	//	curr->mpMatchInfo->mvEdgePts.push_back(currPts[i]);
	//	/*int prevIdx = prev->mpMatchInfo->mvnEdgePtIDXs[i];
	//	curr->mpMatchInfo->mvnEdgePtIDXs.push_back(prevIdx);*/
	//	//cv::line(debugging, prevPts[i], currPts[i] + ptBottom, cv::Scalar(255, 255, 0), 1);
	//	cv::circle(debugging, prevPts[i], 1, cv::Scalar(0, 0, 255), -1);
	//	cv::circle(debugging, currPts[i] + ptBottom, 1, cv::Scalar(0, 0, 255), -1);
	//	////바로 추가
	//	////컨티뉴이티 체크
	//	/*tempEdgeMap.at<uchar>(currPts[i]) = 255;
	//	tempPts.push_back(currPts[i]);*/
	//	////컨티뉴이티 체크
	//	nEdge++;
	//}
	//imshow("edge edge", curr->mpMatchInfo->edgeMap); cv::waitKey(1);
	//////Check continuity
	///*int k = 2;
	//int nk = 2 * k + 1;
	//for (int i = 0; i < tempPts.size(); i++) {
	//	auto pt = tempPts[i];
	//	cv::Rect rect = cv::Rect(pt.x - k, pt.y - k, nk, nk);
	//	auto val = countNonZero(tempEdgeMap(rect))-1;
	//	if(val > 0){
	//		if (!curr->mpMatchInfo->CheckOpticalPointOverlap(curr->mpMatchInfo->edgeMap, 1, 10, currPts[i]))
	//			continue;
	//		curr->mpMatchInfo->mvEdgePts.push_back(tempPts[i]);
	//		cv::circle(debugging, prevPts[i], 1, cv::Scalar(0, 0, 255), -1);
	//		cv::circle(debugging, currPts[i] + ptBottom, 1, cv::Scalar(0, 0, 255), -1);
	//	}
	//}*/
	//////Check continuity
	//////////Edge Test
	//////////////////////////////////

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;

	////fuse time text 
	std::stringstream ss;
	//ss << "Optical flow tracking= " << res <<", "<<vpMPs.size()<<", "<<nBad<< "::" << tttt;
	ss << "Optical flow tracking= " << res << "::" << tttt;
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	//cv::imshow("edge+optical", debugging);
	return res;
}

int UVR_SLAM::Matcher::DenseOpticalMatchingForTracking(Frame* pCurrKF, Frame* pPrevKF, cv::Mat& flow, double& ttime, cv::Mat& debugging){
	
	std::chrono::high_resolution_clock::time_point matching_start = std::chrono::high_resolution_clock::now();
	cv::Mat prevGra = pPrevKF->GetFrame();//pPrevKF->mEdgeImg.clone();
	cv::Mat currGra = pCurrKF->GetFrame();//pCurrKF->mEdgeImg.clone();
	cv::Size size(prevGra.cols / 2, prevGra.rows / 2);
	cv::resize(prevGra, prevGra, size);
	cv::resize(currGra, currGra, size);
	cv::calcOpticalFlowFarneback(prevGra, currGra, flow, 0.5, 3, 11, 3, 5, 1.1, cv::OPTFLOW_FARNEBACK_GAUSSIAN);//OPTFLOW_FARNEBACK_GAUSSIAN
	std::chrono::high_resolution_clock::time_point matching_end = std::chrono::high_resolution_clock::now();
	auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(matching_end - matching_start).count();
	ttime = duration1 / 1000.0;

	////////////////////////////
	//////Optical flow
	//std::chrono::high_resolution_clock::time_point matching_start = std::chrono::high_resolution_clock::now();
	//std::vector<cv::Mat> currPyr, prevPyr;
	//std::vector<uchar> status1, status2;
	//std::vector<float> err1, err2;
	////cv::Mat pprevImg = pPPrevKF->GetOriginalImage();
	//cv::Mat prevImg = pPrevKF->GetOriginalImage();
	//cv::Mat currImg = pCurrKF->GetOriginalImage();
	//									  ///////debug
	//cv::Point2f ptBottom1 = cv::Point2f(0, prevImg.rows);
	//cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	//cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
	//debugging = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
	////pprevImg.copyTo(debugging(mergeRect1));
	//prevImg.copyTo(debugging(mergeRect1));
	//currImg.copyTo(debugging(mergeRect2));
	/////////////
	return 0;
	//cv::Mat K = pCurrKF->mK.clone();

	//std::vector<cv::Point2f> pprevPts, prevPts, currPts;
	//std::vector<cv::Point2f> tempPrevPts, tempCurrPts;
	//prevPts = pPrevKF->mvEdgePts;
	//for (int i = 0; i < pPrevKF->mvKeyPoints.size(); i++) {
	//	prevPts.push_back(pPrevKF->mvKeyPoints[i].pt);
	//}
	////mpMatchInfo->mvTempPts;
	///*
	//auto prevMatchingPTs = pPrevKF->mpMatchInfo->GetMatchingPts();
	//int nTemp = prevPts.size();
	//for (int i = 0; i < prevMatchingPTs.size(); i++) {
	//	prevPts.push_back(prevMatchingPTs[i]);
	//}*/
	//	
	//for (int i = 0; i < prevPts.size(); i++) {
	//	auto prevPt = prevPts[i];
	//	Vec2f val = flow.at<Vec2f>(prevPt.y / 2, prevPt.x / 2);
	//	int dx = val.val[0];
	//	int dy = val.val[1];
	//	if (dx == 0 && dy == 0)
	//		continue;
	//	cv::Point2f currPt = prevPt + cv::Point2f(dx * 2, dy * 2);
	//	cv::circle(debugging, prevPt, 1, cv::Scalar(255, 255, 0), -1);
	//	cv::circle(debugging, currPt + ptBottom1, 1, cv::Scalar(255, 255, 0), -1);
	//	cv::line(debugging, prevPt + ptBottom1, currPt + ptBottom1, cv::Scalar(255, 0, 255));

	//	tempPrevPts.push_back(prevPt);
	//	tempCurrPts.push_back(currPt);
	//}
	//std::chrono::high_resolution_clock::time_point matching_end = std::chrono::high_resolution_clock::now();

	//auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(matching_end - matching_start).count();
	//double tttt1 = duration1 / 1000.0;

	////fuse time text 
	//std::stringstream ss;
	//ss << "Dense Tracking= " << pCurrKF->GetFrameID() << ", " << pPrevKF->GetFrameID() << ", " << "::" << tttt1;
	//cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	//cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
}

int UVR_SLAM::Matcher::TestOpticalMatchingForMapping(Frame* pCurrKF, Frame* pPrevKF, Frame* pPPrevKF, cv::Mat& debugging) {
	//////////////////////////
	////Optical flow
	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	std::vector<cv::Mat> currPyr, prevPyr, pprevPyr;
	std::vector<uchar> status1, status2;
	std::vector<float> err1, err2;
	//cv::Mat pprevImg = pPPrevKF->GetOriginalImage();
	cv::Mat prevImg = pPrevKF->GetOriginalImage();
	cv::Mat currImg = pCurrKF->GetOriginalImage();

	///////debug
	cv::Point2f ptBottom1 = cv::Point2f(0, prevImg.rows);
	cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
	debugging = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
	//pprevImg.copyTo(debugging(mergeRect1));
	prevImg.copyTo(debugging(mergeRect1));
	currImg.copyTo(debugging(mergeRect2));
	///////////

	std::vector<cv::Point2f> pprevPts, prevPts, currPts;
	prevPts = pPrevKF->mpMatchInfo->mvTempPts;
	auto prevMatchingPTs = pPrevKF->mpMatchInfo->GetMatchingPts();
	int nTemp = prevPts.size();
	for (int i = 0; i < prevMatchingPTs.size(); i++) {
		prevPts.push_back(prevMatchingPTs[i]);
	}
	auto pPPrevMatchInfo = pPPrevKF->mpMatchInfo;
	auto pPrevMatchInfo = pPrevKF->mpMatchInfo;

	int maxLvl = 3;
	int searchSize = 21;
	cv::buildOpticalFlowPyramid(currImg, currPyr, cv::Size(searchSize, searchSize), maxLvl);
	maxLvl = cv::buildOpticalFlowPyramid(prevImg, prevPyr, cv::Size(searchSize, searchSize), maxLvl);
	
	cv::calcOpticalFlowPyrLK(prevPyr, currPyr, prevPts, currPts, status1, err1, cv::Size(searchSize, searchSize), maxLvl);

	auto prevMatchingMPs = pPrevKF->mpMatchInfo->GetMatchingMPs();
	cv::Mat R, t;
	pCurrKF->GetPose(R, t);
	cv::Mat K = pCurrKF->mK.clone();

	for (int i = 0; i < prevMatchingPTs.size(); i++) {
		cv::circle(debugging, prevMatchingPTs[i], 3, cv::Scalar(0, 0, 255));
	}
	auto currMatchingPTs = pCurrKF->mpMatchInfo->GetMatchingPts();
	for (int i = 0; i < currMatchingPTs.size(); i++) {
		cv::circle(debugging, currMatchingPTs[i] + ptBottom1, 5, cv::Scalar(0, 0, 255));
	}

	int radius = 5;
	cv::Mat used = cv::Mat::zeros(prevImg.size(), CV_8UC1);
	for (int i = 0; i < prevPts.size(); i++) {
		if (status1[i] == 0) {
			continue;
		}
		////MP 영역과 매칭이 안되도록 함. 그리고 자기가 매칭 된 곳 근처가 안되도록 함. 자기 자신은 이미 포인트 생성할 때 걸러짐.
		//bool b1 = pPPrevMatchInfo->CheckOpticalPointOverlap(pPPrevMatchInfo->used, radius, 10, pprevPts[i]);//pPPrevMatchInfo->usedCPMap
		bool b2 = pPPrevMatchInfo->CheckOpticalPointOverlap(pCurrKF->mpMatchInfo->used, radius, 10, currPts[i]);
		bool b3 = pPPrevMatchInfo->CheckOpticalPointOverlap(used, radius, 10, prevPts[i]); //used //얘는 왜 used 따로 만듬???

		//if ( !b2 || !b3) {//|| b4 || b5
		//						//std::cout << "OpticalMatchingForMapping::???????????" << std::endl;
		//	continue;
		//}
		//cv::circle(used, prevPts[i], radius, cv::Scalar(255, 0, 0), -1);
		
		//cv::circle(debugging, pprevPts[i], 2, cv::Scalar(255, 255, 0), -1);
		cv::circle(debugging, prevPts[i], 2, cv::Scalar(255, 255, 0), -1);
		cv::circle(debugging, currPts[i] + ptBottom1, 2, cv::Scalar(255, 255, 0), -1);
		if (i >= nTemp) {
			cv::circle(debugging, currPts[i] + ptBottom1, 4, cv::Scalar(255, 0, 0));
			
			int nMPIDX = i - nTemp;
			auto pMPi = prevMatchingMPs[nMPIDX];
			if (pMPi && !pMPi->isDeleted()) {
				cv::Point2f pPt;
				cv::Mat Xcam;
				if (pMPi->Projection(pPt, Xcam, R, t, K, mWidth, mHeight)) {
					cv::line(debugging, pPt + ptBottom1, currPts[i] + ptBottom1, cv::Scalar(255, 0, 0));
				}
			}
		}
	}
	
}

int UVR_SLAM::Matcher::TestOpticalMatchingForMapping2(Frame* pCurrKF, Frame* pPrevKF, Frame* pPPrevKF, cv::Mat& debugging) {
	//////////////////////////
	////Optical flow
	std::chrono::high_resolution_clock::time_point matching_start = std::chrono::high_resolution_clock::now();
	std::vector<cv::Mat> currPyr, prevPyr, pprevPyr;
	std::vector<uchar> status1, status2;
	std::vector<float> err1, err2;
	//cv::Mat pprevImg = pPPrevKF->GetOriginalImage();
	cv::Mat prevImg = pPrevKF->GetOriginalImage();
	cv::Mat currImg = pCurrKF->GetOriginalImage();
	cv::Mat prevGra = pPrevKF->GetFrame();//pPrevKF->mEdgeImg.clone();
	cv::Mat currGra = pCurrKF->GetFrame();//pCurrKF->mEdgeImg.clone();
	///////debug
	cv::Point2f ptBottom1 = cv::Point2f(0, prevImg.rows);
	cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
	debugging = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
	//pprevImg.copyTo(debugging(mergeRect1));
	prevImg.copyTo(debugging(mergeRect1));
	currImg.copyTo(debugging(mergeRect2));
	///////////

	cv::Mat K = pCurrKF->mK.clone();

	std::vector<cv::Point2f> pprevPts, prevPts, currPts;
	std::vector<cv::Point2f> tempPrevPts, tempCurrPts;
	prevPts = pPrevKF->mvEdgePts;//mpMatchInfo->mvTempPts;
	/*auto prevMatchingPTs = pPrevKF->mpMatchInfo->GetMatchingPts();
	int nTemp = prevPts.size();
	for (int i = 0; i < prevMatchingPTs.size(); i++) {
		prevPts.push_back(prevMatchingPTs[i]);
	}*/

	auto pPPrevMatchInfo = pPPrevKF->mpMatchInfo;
	auto pPrevMatchInfo = pPrevKF->mpMatchInfo;
	cv::Size size(prevGra.cols / 2, prevGra.rows / 2);
	
	cv::resize(prevGra, prevGra, size);
	cv::resize(currGra, currGra, size);
	cv::Mat flow;
	cv::calcOpticalFlowFarneback(prevGra, currGra, flow, 0.5, 3, 11, 3, 5,1.1, cv::OPTFLOW_FARNEBACK_GAUSSIAN);//OPTFLOW_FARNEBACK_GAUSSIAN
	for (int i = 0; i < prevPts.size(); i++) {
		auto prevPt = prevPts[i];
		Vec2f val = flow.at<Vec2f>(prevPt.y/2, prevPt.x/2);
		int dx = val.val[0];
		int dy = val.val[1];
		if (dx == 0 && dy == 0)
			continue;
		cv::Point2f currPt = prevPt + cv::Point2f(dx*2,dy*2);
		cv::circle(debugging, prevPt, 2, cv::Scalar(255, 255, 0), -1);
		cv::circle(debugging, currPt + ptBottom1, 2, cv::Scalar(255, 255, 0), -1);
		cv::line(debugging, prevPt + ptBottom1, currPt + ptBottom1, cv::Scalar(255, 0, 255));

		tempPrevPts.push_back(prevPt);
		tempCurrPts.push_back(currPt);
	}

	//int maxLvl = 3;
	//int searchSize = 21;
	//cv::buildOpticalFlowPyramid(currImg, currPyr, cv::Size(searchSize, searchSize), maxLvl);
	//maxLvl = cv::buildOpticalFlowPyramid(prevImg, prevPyr, cv::Size(searchSize, searchSize), maxLvl);

	//cv::calcOpticalFlowPyrLK(prevPyr, currPyr, prevPts, currPts, status1, err1, cv::Size(searchSize, searchSize), maxLvl);

	//auto prevMatchingMPs = pPrevKF->mpMatchInfo->GetMatchingMPs();
	//cv::Mat R, t;
	//pCurrKF->GetPose(R, t);
	

	//for (int i = 0; i < prevMatchingPTs.size(); i++) {
	//	cv::circle(debugging, prevMatchingPTs[i], 3, cv::Scalar(0, 0, 255));
	//}
	//auto currMatchingPTs = pCurrKF->mpMatchInfo->GetMatchingPts();
	//for (int i = 0; i < currMatchingPTs.size(); i++) {
	//	cv::circle(debugging, currMatchingPTs[i] + ptBottom1, 5, cv::Scalar(0, 0, 255));
	//}

	//int radius = 5;
	//cv::Mat used = cv::Mat::zeros(prevImg.size(), CV_8UC1);
	//for (int i = 0; i < prevPts.size(); i++) {
	//	if (status1[i] == 0) {
	//		continue;
	//	}
	//	////MP 영역과 매칭이 안되도록 함. 그리고 자기가 매칭 된 곳 근처가 안되도록 함. 자기 자신은 이미 포인트 생성할 때 걸러짐.
	//	//bool b1 = pPPrevMatchInfo->CheckOpticalPointOverlap(pPPrevMatchInfo->used, radius, 10, pprevPts[i]);//pPPrevMatchInfo->usedCPMap
	//	bool b2 = pPPrevMatchInfo->CheckOpticalPointOverlap(pCurrKF->mpMatchInfo->used, radius, 10, currPts[i]);
	//	bool b3 = pPPrevMatchInfo->CheckOpticalPointOverlap(used, radius, 10, prevPts[i]); //used //얘는 왜 used 따로 만듬???

	//	tempPrevPts.push_back(prevPts[i]);
	//	tempCurrPts.push_back(currPts[i]);

	//	cv::circle(debugging, prevPts[i], 2, cv::Scalar(255, 255, 0), -1);
	//	cv::circle(debugging, currPts[i] + ptBottom1, 2, cv::Scalar(255, 255, 0), -1);
	//	if (i >= nTemp) {
	//		cv::circle(debugging, currPts[i] + ptBottom1, 4, cv::Scalar(255, 0, 0));

	//		int nMPIDX = i - nTemp;
	//		auto pMPi = prevMatchingMPs[nMPIDX];
	//		if (pMPi && !pMPi->isDeleted()) {
	//			cv::Point2f pPt;
	//			cv::Mat Xcam;
	//			if (pMPi->Projection(pPt, Xcam, R, t, K, mWidth, mHeight)) {
	//				cv::line(debugging, pPt + ptBottom1, currPts[i] + ptBottom1, cv::Scalar(255, 0, 0));
	//			}
	//		}
	//	}
	//}
	
	std::chrono::high_resolution_clock::time_point time_temp = std::chrono::high_resolution_clock::now();
	std::vector<uchar> vFInliers;
	std::vector<cv::Point2f> vTempMatchPts1, vTempMatchPts2;
	/*cv::Mat E12 = cv::findEssentialMat(tempPrevPts, tempCurrPts, K, cv::FM_RANSAC, 0.999, 1.0, vFInliers);
	for (unsigned long i = 0; i < vFInliers.size(); i++) {
		if (vFInliers[i]) {
			vTempMatchPts1.push_back(tempPrevPts[i]);
			vTempMatchPts2.push_back(tempCurrPts[i]);
			cv::circle(debugging, tempPrevPts[i], 2, cv::Scalar(255, 0, 255), -1);
			cv::circle(debugging, tempCurrPts[i] + ptBottom1, 2, cv::Scalar(255, 0, 255), -1);
		}
	}*/
	std::chrono::high_resolution_clock::time_point matching_end = std::chrono::high_resolution_clock::now();

	auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(time_temp - matching_start).count();
	double tttt1 = duration1 / 1000.0;

	auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(matching_end - time_temp).count();
	double tttt2 = duration2 / 1000.0;

	//fuse time text 
	std::stringstream ss;
	ss << "Optical flow Mapping2= " << pCurrKF->GetFrameID() << ", " << pPrevKF->GetFrameID() << ", " << pPPrevKF->GetFrameID() << "::" << tttt1<<", "<<tttt2;
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
}

int UVR_SLAM::Matcher::OpticalMatchingForMapping(Frame* pCurrKF, Frame* pPrevKF, std::vector<cv::Point2f> vMatchPts, std::vector<cv::Point2f>& vMatchedPrevPts, std::vector<cv::Point2f>& vMatchedCurrPts, std::vector<int>& vnIDXs, std::vector<bool>& vbInliers, cv::Mat& debugging) {
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
	debugging = cv::Mat::zeros(prevImg.rows*2, prevImg.cols, prevImg.type());
	prevImg.copyTo(debugging(mergeRect1));
	currImg.copyTo(debugging(mergeRect2));
	///////////

	std::vector<cv::Point2f> currPts;
	std::vector<int> vnOctaves;
	//prevPts = pPrevKF->mpMatchInfo->mvTempPts;
	auto pPrevMatchInfo = pPrevKF->mpMatchInfo;

	int maxLvl = 3;
	int searchSize = 21;
	cv::buildOpticalFlowPyramid(currImg, currPyr, cv::Size(searchSize, searchSize), maxLvl);
	maxLvl = cv::buildOpticalFlowPyramid(prevImg, prevPyr, cv::Size(searchSize, searchSize), maxLvl);
	cv::calcOpticalFlowPyrLK(prevPyr, currPyr, vMatchPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);

	cv::Mat used = cv::Mat::zeros(prevImg.size(), CV_8UC1);
	for (int i = 0; i < vMatchPts.size(); i++) {
		if (status[i] == 0) {
			continue;
		}
		////MP 영역과 매칭이 안되도록 함. 그리고 자기가 매칭 된 곳 근처가 안되도록 함. 자기 자신은 이미 포인트 생성할 때 걸러짐.
		//bool b2 = pPrevMatchInfo->CheckOpticalPointOverlap(pCurrKF->mpMatchInfo->used, Frame::mnRadius, 10, currPts[i]);
		//bool b3 = pPrevMatchInfo->CheckOpticalPointOverlap(used, Frame::mnRadius, 10, prevPts[i]); //used //얘는 왜 used 따로 만듬???

		//if (!b1 || !b2 || !b3) {//|| b4 || b5
		//						//std::cout << "OpticalMatchingForMapping::???????????" << std::endl;
		//	continue;
		//}
		bool b3 = pPrevMatchInfo->CheckOpticalPointOverlap(used, Frame::mnRadius, 10, vMatchPts[i]); //used //얘는 왜 used 따로 만듬???
		if (!b3) {//|| b4 || b5
			continue;
		}
		cv::circle(used, vMatchPts[i], Frame::mnRadius, cv::Scalar(255, 0, 0), -1);
		
		vMatchedPrevPts.push_back(vMatchPts[i]);
		vMatchedCurrPts.push_back(currPts[i]);
		vnIDXs.push_back(i);
		vbInliers.push_back(true);

		cv::circle(debugging, vMatchPts[i], 2, cv::Scalar(255, 255, 0), -1);
		cv::circle(debugging, currPts[i] + ptBottom, 2, cv::Scalar(255, 255, 0), -1);
	}

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;

	//fuse time text 
	std::stringstream ss;
	ss << "Optical flow Mapping2= " << pCurrKF->GetFrameID() << ", " << pPrevKF->GetFrameID() << ", " << "::" << tttt << "::" << vMatchedPrevPts.size();
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
}

int UVR_SLAM::Matcher::OpticalMatchingForMapping(Frame* pCurrKF, Frame* pPrevKF, Frame* pPPrevKF, std::vector<cv::Point2f>& vMatchedPPrevPts, std::vector<cv::Point2f>& vMatchedPrevPts, std::vector<cv::Point2f>& vMatchedCurrPts, std::vector<int>& vnIDXs, std::vector<bool>& vbInliers, cv::Mat& debugging) {
	//////////////////////////
	////Optical flow
	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	std::vector<cv::Mat> currPyr, prevPyr, pprevPyr;
	std::vector<uchar> status1, status2;
	std::vector<float> err1, err2;
	cv::Mat pprevImg = pPPrevKF->GetOriginalImage();
	cv::Mat prevImg = pPrevKF->GetOriginalImage();
	cv::Mat currImg = pCurrKF->GetOriginalImage();

	///////debug
	cv::Point2f ptBottom1 = cv::Point2f(pprevImg.cols, 0);
	cv::Point2f ptBottom2 = cv::Point2f(pprevImg.cols*2, 0);
	cv::Rect mergeRect1 = cv::Rect(0, 0, pprevImg.cols, pprevImg.rows);
	cv::Rect mergeRect2 = cv::Rect(pprevImg.cols, 0, pprevImg.cols, pprevImg.rows);
	cv::Rect mergeRect3 = cv::Rect(pprevImg.cols*2, 0, pprevImg.cols, pprevImg.rows);
	debugging = cv::Mat::zeros(pprevImg.rows, pprevImg.cols*3, pprevImg.type());
	pprevImg.copyTo(debugging(mergeRect1));
	prevImg.copyTo(debugging(mergeRect2));
	currImg.copyTo(debugging(mergeRect3));
	///////////
	//matKPs, mvKPs
	//init -> curr로 매칭
	////////
	std::vector<cv::Point2f> pprevPts, prevPts, currPts;
	prevPts = pPrevKF->mpMatchInfo->mvTempPts;
	auto pPPrevMatchInfo = pPPrevKF->mpMatchInfo;
	auto pPrevMatchInfo = pPrevKF->mpMatchInfo;

	int maxLvl = 3;
	int searchSize = 21;
	cv::buildOpticalFlowPyramid(currImg, currPyr, cv::Size(searchSize, searchSize), maxLvl);
	maxLvl = cv::buildOpticalFlowPyramid(prevImg, prevPyr, cv::Size(searchSize, searchSize), maxLvl);
	maxLvl = cv::buildOpticalFlowPyramid(pprevImg, pprevPyr, cv::Size(searchSize, searchSize), maxLvl);
	
	cv::calcOpticalFlowPyrLK(prevPyr, pprevPyr, prevPts, pprevPts, status2, err2, cv::Size(searchSize, searchSize), maxLvl);
	cv::calcOpticalFlowPyrLK(prevPyr, currPyr, prevPts, currPts, status1, err1, cv::Size(searchSize, searchSize), maxLvl);

	int radius = 5;
	cv::Mat used = cv::Mat::zeros(prevImg.size(), CV_8UC1);
	for (int i = 0; i < prevPts.size(); i++) {
		if (status2[i] == 0 || status1[i] == 0) {
			continue;
		}
		////MP 영역과 매칭이 안되도록 함. 그리고 자기가 매칭 된 곳 근처가 안되도록 함. 자기 자신은 이미 포인트 생성할 때 걸러짐.
		bool b1 = pPPrevMatchInfo->CheckOpticalPointOverlap(pPPrevMatchInfo->used, radius, 10, pprevPts[i]);//pPPrevMatchInfo->usedCPMap
		bool b2 = pPPrevMatchInfo->CheckOpticalPointOverlap(pCurrKF->mpMatchInfo->used, radius, 10, currPts[i]);
		bool b3 = pPPrevMatchInfo->CheckOpticalPointOverlap(used, radius, 10, prevPts[i]); //used //얘는 왜 used 따로 만듬???

		//bool b4 = pPPrevKF->mEdgeImg.at<uchar>(pprevPts[i]) == 0;
		//bool b5 = pCurrKF->mEdgeImg.at<uchar>(currPts[i]) == 0;
		
		if (!b1 || !b2 || !b3 ){//|| b4 || b5
			//std::cout << "OpticalMatchingForMapping::???????????" << std::endl;
			continue;
		}
		cv::circle(used, prevPts[i], radius, cv::Scalar(255, 0, 0), -1);
		//vTempIDXs.push_back(i);
		vMatchedPPrevPts.push_back(pprevPts[i]);
		vMatchedPrevPts.push_back(prevPts[i]);
		vMatchedCurrPts.push_back(currPts[i]);
		vnIDXs.push_back(i);
		vbInliers.push_back(true);

		cv::circle(debugging, pprevPts[i], 2, cv::Scalar(255, 255, 0), -1);
		cv::circle(debugging, prevPts[i] + ptBottom1, 2, cv::Scalar(255, 255, 0), -1);
		cv::circle(debugging, currPts[i] + ptBottom2, 2, cv::Scalar(255, 255, 0), -1);
	}

	//std::vector<cv::Point3f> lines[2];
	//cv::Mat Rpprev, Tpprev, Rprev, Tprev, Rcurr, Tcurr;
	//pCurrKF->GetPose(Rcurr, Tcurr);
	//pPrevKF->GetPose(Rprev, Tprev);
	//pPPrevKF->GetPose(Rpprev, Tpprev);
	//cv::Mat mK = pCurrKF->mK.clone();
	//cv::Mat Fpptop = CalcFundamentalMatrix(Rpprev, Tpprev, Rprev, Tprev, mK);
	//cv::Mat Fctop = CalcFundamentalMatrix(Rcurr, Tcurr, Rprev, Tprev, mK);

	//cv::computeCorrespondEpilines(vMatchedPPrevPts, 2, Fpptop, lines[0]);
	//cv::computeCorrespondEpilines(vMatchedCurrPts, 2, Fctop, lines[1]);

	//////////create

	////조건들
	////abs(b1) > 0.00001
	//for (int i = 0; i < vMatchedPPrevPts.size(); i++) {
	//	float a1 = lines[0][i].x;
	//	float b1 = lines[0][i].y;
	//	float c1 = lines[0][i].z;

	//	float a2 = lines[1][i].x;
	//	float b2 = lines[1][i].y;
	//	float c2 = lines[1][i].z;

	//	float a = a1*b2 - a2*b1;
	//	if (abs(a) < 0.00001)
	//		continue;
	//	float b = b1*c2 - b2*c1;
	//	float x = b / a;
	//	float y = -a1 / b1*x - c1 / b1;

	//	cv::Point2f pt(x, y);
	//	if (!pPrevKF->isInImage(x, y, 10.0)) {
	//		continue;
	//	}
	//	if (i % 50 == 0) {
	//		cv::Point2f spt1, ept1;
	//		spt1 = CalcLinePoint(0.0, lines[0][i], true);
	//		ept1 = CalcLinePoint(prevImg.rows, lines[0][i], true);
	//		cv::Point2f spt2, ept2;
	//		spt2 = CalcLinePoint(0.0, lines[1][i], true);
	//		ept2 = CalcLinePoint(prevImg.rows, lines[1][i], true);
	//		/*cv::line(debug, spt1+ptBottom, ept1+ ptBottom, cv::Scalar(255,0,0) , 1);
	//		cv::line(debug, spt2+ ptBottom, ept2+ ptBottom, cv::Scalar(0,255,0), 1);*/
	//		cv::circle(debugging, pt + ptBottom, 3, cv::Scalar(255, 0, 255), -1);
	//		cv::circle(currImg, vMatchedCurrPts[i], 2, cv::Scalar(255, 255, 0), -1);
	//		cv::circle(debugging, vMatchedPrevPts[i]+ ptBottom, 2, cv::Scalar(255, 255, 0), -1);
	//		cv::circle(debugging, vMatchedPPrevPts[i], 3, cv::Scalar(255, 0, 255), -1);
	//	}
	//}

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;

	//fuse time text 
	std::stringstream ss;
	ss << "Optical flow Mapping2= " <<pCurrKF->GetFrameID()<<", "<<pPrevKF->GetFrameID()<<", "<<pPPrevKF->GetFrameID()<<"::"<< tttt << "::" << vMatchedPPrevPts.size();
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	/*imshow("Mapping::2::curr", currImg);
	imshow("Mapping::2", debugging); waitKey(1);*/
}

//int UVR_SLAM::Matcher::OpticalMatchingForTracking3(Frame* pCurrF, Frame* pKF, Frame* pF1, Frame* pF2, cv::Mat& debug) {
//
//	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
//
//	///////debug
//	cv::Mat prevImg = pF1->GetOriginalImage();
//	cv::Mat currImg = pCurrF->GetOriginalImage();
//	cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
//	cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
//	cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
//	debug = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
//	prevImg.copyTo(debug(mergeRect1));
//	currImg.copyTo(debug(mergeRect2));
//	///////debug
//
//	//시간 체크 & 포인트 선택 & 라인 출력
//
//	cv::Mat Rcurr, Tcurr, R1, T1, R2, T2;
//	pCurrF->GetPose(Rcurr, Tcurr);
//	pF1->GetPose(R1, T1);
//	pF2->GetPose(R2, T2);
//	cv::Mat mK = pCurrF->mK.clone();
//	cv::Mat F1tocurr = CalcFundamentalMatrix(R1, T1, Rcurr, Tcurr, mK);
//	cv::Mat F2tocurr = CalcFundamentalMatrix(R2, T2, Rcurr, Tcurr, mK);
//
//	//공통된 포인트 찾기
//	std::vector<cv::Point3f> lines[2];
//	cv::Mat matLines;
//	std::vector<cv::Point2f> pts1, pts2, ptsInFuse1;
//	std::vector<UVR_SLAM::MapPoint*> vpMPs;
//	std::vector<bool> vb1(pKF->mpMatchInfo->mvpMatchingMPs.size(), false);
//	std::vector<bool> vb2(pKF->mpMatchInfo->mvpMatchingMPs.size(), false);
//	std::vector<int> vn1(pKF->mpMatchInfo->mvpMatchingMPs.size());
//	std::vector<int> vn2(pKF->mpMatchInfo->mvpMatchingMPs.size());
//	
//	//내용 수정 필요
//	/*for (int i = 0; i < pF1->mpMatchInfo->mvnMatchingPtIDXs.size(); i++) {
//		int idx = pF1->mpMatchInfo->mvnMatchingPtIDXs[i];
//		vb1[idx] = true;
//		vn1[idx] = i;
//	}
//
//	for (int i = 0; i < pF2->mpMatchInfo->mvnMatchingPtIDXs.size(); i++) {
//		int idx = pF2->mpMatchInfo->mvnMatchingPtIDXs[i];
//		vb2[idx] = true;
//		vn2[idx] = i;
//	}*/
//	//내용 수정 필요
//
//	for (int i = 0; i < vb1.size(); i++) {
//		if (!vb1[i] || !vb2[i])
//			continue;
//		auto pt1 = pF1->mpMatchInfo->mvMatchingPts[vn1[i]];
//		auto pt2 = pF2->mpMatchInfo->mvMatchingPts[vn2[i]];
//
//		pts1.push_back(pt1);
//		pts2.push_back(pt2);
//
//	}
//	
//	if (pts1.size() < 10)
//		return 0;
//	/*for (int i = 0; i < pFuseKF1->mpMatchInfo->mnTargetMatch; i++) {
//		auto pMPi = pFuseKF1->mpMatchInfo->mvpMatchingMPs[i];
//		if (!pMPi || pMPi->isDeleted())
//			continue;
//		auto X3D = pMPi->GetWorldPos();
//		auto currPt = mpTargetFrame->Projection(X3D);
//		if (!mpTargetFrame->isInImage(currPt.x, currPt.y, 10.0)) {
//			continue;
//		}
//		vpMPs.push_back(pMPi);
//		pts1.push_back(pFuseKF1->mpMatchInfo->mvMatchingPts[i]);
//	}*/
//
//	////포인트를 선택해야 함
//
//	cv::computeCorrespondEpilines(pts1, 2, F1tocurr, lines[0]);
//	cv::computeCorrespondEpilines(pts2, 2, F2tocurr, lines[1]);
//
//	//조건들
//	//abs(b1) > 0.00001
//
//	for (int i = 0; i < pts1.size(); i++) {
//		float a1 = lines[0][i].x;
//		float b1 = lines[0][i].y;
//		float c1 = lines[0][i].z;
//
//		float a2 = lines[1][i].x;
//		float b2 = lines[1][i].y;
//		float c2 = lines[1][i].z;
//
//		float a = a1*b2 - a2*b1;
//		if (abs(a) < 0.00001)
//			continue;
//		float b = b1*c2 - b2*c1;
//		float x = b / a;
//		float y = -a1 / b1*x - c1 / b1;
//
//		cv::Point2f pt(x, y);
//		if (!pCurrF->isInImage(x, y, 10.0)) {
//			continue;
//		}
//		if(i % 50 == 0){
//			cv::Point2f spt1, ept1;
//			spt1 = CalcLinePoint(0.0, lines[0][i], true);
//			ept1 = CalcLinePoint(prevImg.rows, lines[0][i], true);
//			cv::Point2f spt2, ept2;
//			spt2 = CalcLinePoint(0.0, lines[1][i], true);
//			ept2 = CalcLinePoint(prevImg.rows, lines[1][i], true);
//			/*cv::line(debug, spt1+ptBottom, ept1+ ptBottom, cv::Scalar(255,0,0) , 1);
//			cv::line(debug, spt2+ ptBottom, ept2+ ptBottom, cv::Scalar(0,255,0), 1);*/
//			cv::circle(debug, pt+ ptBottom, 3, cv::Scalar(255, 0, 255), -1);
//			cv::circle(debug, pts1[i], 3, cv::Scalar(255, 0, 255), -1);
//		}
//	}
//
//	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
//	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
//	double tttt = duration / 1000.0;
//
//	//fuse time text 
//	std::stringstream ss;
//	ss << "Optical flow init= " << tttt<<"::"<<pts1.size();
//	cv::rectangle(debug, cv::Point2f(0, 0), cv::Point2f(debug.cols, 30), cv::Scalar::all(0), -1);
//	cv::putText(debug, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
//	imshow("matching::3", debug);
//
//}

int UVR_SLAM::Matcher::OpticalMatchingForTracking2(Frame* pKF, Frame* pF, cv::Mat& debug) {

	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	////////optical flow parameter
	std::vector<cv::Mat> currPyr, prevPyr;
	std::vector<uchar> status;
	std::vector<float> err;
	cv::Mat prevImg = pKF->GetOriginalImage();
	cv::Mat currImg = pF->GetOriginalImage();
	////////optical flow parameter

	///////debug
	cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
	cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
	debug = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
	prevImg.copyTo(debug(mergeRect1));
	currImg.copyTo(debug(mergeRect2));
	///////debug

	////옵티컬 플로우 매칭을 위한 포인트 벡터 생성
	std::vector<cv::Point2f> prevPts, currPts;
	//prevPts = pKF->mvEdgePts;//pKF->mpMatchInfo->mvMatchingPts;//prev->mvMatchingPts;
	for (int i = 0; i < pKF->mvEdgePts.size(); i += 10) {
		prevPts.push_back(pKF->mvEdgePts[i]);
	}

	//////optical flow
	int maxLvl = 3;
	int searchSize = 21;
	cv::Mat resized_curr;
	cv::buildOpticalFlowPyramid(currImg, currPyr, cv::Size(searchSize, searchSize), maxLvl);
	maxLvl = cv::buildOpticalFlowPyramid(prevImg, prevPyr, cv::Size(searchSize, searchSize), maxLvl);
	cv::calcOpticalFlowPyrLK(prevPyr, currPyr, prevPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);
	//////optical flow

	/////////매칭 결과 확인
	int n = 0;
	for (int i = 0; i < prevPts.size(); i+=1) {
		if (status[i] == 0) {
			continue;
		}

		if (!pF->isInImage(currPts[i].x, currPts[i].y, 10)) {
			continue;
		}

		//////오버랩은 꼭 확인해야 함
		/*if (!CheckOpticalPointOverlap(overlap, 2, currPts[i])) {
			nBad++;
			continue;
		}*/
		if (pF->mEdgeImg.at<uchar>(currPts[i]) == 0)
			continue;

		//매칭 결과
		/*float diffX = abs(prevPts[i].x - currPts[i].x);
		if (diffX > 25) {
			continue;
		}*/
		//매칭 결과

		n++;
		cv::circle(debug, prevPts[i], 2, cv::Scalar(255, 0, 255), -1);
		cv::circle(debug, currPts[i] + ptBottom, 2, cv::Scalar(255, 0, 255), -1);
	}
	/////////매칭 결과 확인


	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;

	//fuse time text 
	std::stringstream ss;
	ss << "Optical flow init= " <<n<<", "<<pF->mnFrameID - pKF->mnFrameID<< ", " << tttt<<"::"<<prevPts.size();
	cv::rectangle(debug, cv::Point2f(0, 0), cv::Point2f(debug.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debug, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	imshow("matchign :: edge+optical", debug);

}

int UVR_SLAM::Matcher::OpticalKeyframeAndFrameMatchingForTracking(Frame* prev, Frame* curr, cv::Mat& debugging) {
	//////////////////////////
	////Optical flow
	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	std::vector<cv::Mat> currPyr, prevPyr;
	std::vector<uchar> status;
	std::vector<float> err;
	cv::Mat prevImg = prev->GetOriginalImage();
	//cv::Mat prevImg = curr->mpMatchInfo->mpTargetFrame->GetOriginalImage();
	cv::Mat currImg = curr->GetOriginalImage();

	///////debug
	cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
	cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
	debugging = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
	prevImg.copyTo(debugging(mergeRect1));
	currImg.copyTo(debugging(mergeRect2));

	///////////
	//matKPs, mvKPs
	//init -> curr로 매칭
	////////
	std::vector<cv::Point2f> prevPts, currPts;
	int nIncPt = 10;
	for (int i = 0; i < prev->mvEdgePts.size(); i += nIncPt) {
		prevPts.push_back(prev->mvEdgePts[i]);
	}
	for (int i = 0; i < prev->mvPts.size(); i += nIncPt) {
		prevPts.push_back(prev->mvPts[i]);
	}

	int maxLvl = 3;
	int searchSize = 21;
	cv::buildOpticalFlowPyramid(currImg, currPyr, cv::Size(searchSize, searchSize), maxLvl);
	maxLvl = cv::buildOpticalFlowPyramid(prevImg, prevPyr, cv::Size(searchSize, searchSize), maxLvl);
	//cv::calcOpticalFlowPyrLK(prevPyr, currPyr, prevPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);
	cv::calcOpticalFlowPyrLK(prevImg, currImg, prevPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);
	//바운더리 에러도 고려해야 함.
	
	int nRes = 0;
	//std::vector<cv::Scalar> colors;
	//cv::RNG rng(12345);
	//for (int i = 0; i < prevPts.size(); i++) {
	//	cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	//	colors.push_back(color);
	//	if (status[i] == 0) {
	//		continue;
	//	}
	//	if (!curr->isInImage(currPts[i].x, currPts[i].y, 10)) {
	//		continue;
	//	}

	//	/////
	//	//매칭 결과
	//	float diffX = abs(prevPts[i].x - currPts[i].x);
	//	if (diffX > 90) {
	//		continue;
	//	}
	//	
	//	cv::line(debugging, prevPts[i], currPts[i] + ptBottom, color);
	//	nRes++;
	//}

	std::vector<cv::Point2f> tPrevPts, tCurrPts;

	for (int i = 0; i < prevPts.size(); i++) {
		if (status[i] == 0) {
			continue;
		}
		if (!curr->isInImage(currPts[i].x, currPts[i].y, 10)) {
			continue;
		}

		/////
		//매칭 결과
		float diffX = abs(prevPts[i].x - currPts[i].x);
		if (diffX > 90) {
			continue;
		}

		tPrevPts.push_back(prevPts[i]);
		tCurrPts.push_back(currPts[i]);

		/*cv::circle(debugging, prevPts[i], 1, cv::Scalar(255, 0, 255),-1);
		cv::circle(debugging, currPts[i] + ptBottom, 1, cv::Scalar(255, 0, 255), -1);*/
		nRes++;
	}

	//////////////////////////////////////////
	////Epipolar constraints
	cv::Mat Rcurr, Tcurr, Rprev, Tprev;
	prev->GetPose(Rprev, Tprev);
	curr->GetPose(Rcurr, Tcurr);
	cv::Mat mK = prev->mK.clone();
	cv::Mat mD = prev->mDistCoef.clone();
	cv::Mat mK2 = prev->mK.clone();
	cv::Mat mD2 = prev->mDistCoef.clone();
	cv::Mat F12 = CalcFundamentalMatrix(Rprev, Tprev, Rcurr, Tcurr, mK);

	/////////////////////////////Rectification
	/////포인트 매칭을 위해서
	cv::Mat used = cv::Mat::zeros(prevImg.size(), CV_16UC1);
	for (int i = 0; i < tPrevPts.size(); i+=20) {
		used.at<ushort>(tPrevPts[i]) = i + 1;
	}
	/////포인트 매칭을 위해서

	cv::Mat R, t;
	/*R = Rprev*Rcurr.t();
	t = -Rprev*Rcurr.t()*Tcurr + Tprev;*/
	R = Rprev.t()*Rcurr;
	t = -Rprev.t()*Tprev + Rprev.t()*Tcurr;

	R.convertTo(R, CV_64FC1);
	t.convertTo(t, CV_64FC1);
	cv::Mat Q;
	cv::Mat R1, R2, P1, P2;
	
	stereoRectify(mK, mD, mK, mD, prevImg.size(), R, t, R1, R2, P1, P2, Q);

	cv::Mat mapPrev1, mapPrev2;
	cv::Mat mapCurr1, mapCurr2;
	initUndistortRectifyMap(mK, mD, R1, P1.colRange(0,3).rowRange(0,3), prevImg.size(), CV_32FC1, mapPrev1, mapPrev2);
	initUndistortRectifyMap(mK, mD, R2, P2.colRange(0,3).rowRange(0,3), prevImg.size(), CV_32FC1, mapCurr1, mapCurr2);

	cv::Mat prevRectified, currRectified;
	remap(prevImg, prevRectified, mapPrev1, mapPrev2, cv::INTER_LINEAR);
	remap(currImg, currRectified, mapCurr1, mapCurr2, cv::INTER_LINEAR);

	cv::Point2f ptRight = cv::Point2f(prevImg.cols,0);
	cv::Rect rmergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	cv::Rect rmergeRect2 = cv::Rect(prevImg.cols, 0, prevImg.cols, prevImg.rows);
	cv::Mat rectified = cv::Mat::zeros(prevImg.rows, prevImg.cols * 2, prevImg.type());
	prevRectified.copyTo(rectified(rmergeRect1));
	currRectified.copyTo(rectified(rmergeRect2));

	////SSD 설정
	int nHalfWindowSize = 5;
	int nFullWindow = nHalfWindowSize * 2 + 1;
	////SSD 설정

	for (int y = 0; y < prevRectified.rows; y++) {
		for (int x = 0; x < prevRectified.cols; x++) {
			
			cv::Point2f prevPt(mapPrev1.at<float>(y,x), mapPrev2.at<float>(y,x)); //이게 원래 이미지에서의 위치임.
			
			if (!prev->isInImage(prevPt.x, prevPt.y, 10.0))
				continue;

			if (used.at<ushort>(prevPt) > 0) {
				cv::Point2d pt(x, y);
				cv::line(rectified, cv::Point2f(0, pt.y), cv::Point2f(rectified.cols, pt.y), cv::Scalar(255, 0, 0));
				cv::circle(rectified, pt, 2, cv::Scalar(0, 0, 255));
				/*cv::line(rectified, cv::Point2f(0, prevPt.y), cv::Point2f(rectified.cols, prevPt.y), cv::Scalar(255, 0, 0));
				cv::circle(rectified, prevPt, 2, cv::Scalar(0, 0, 255));*/

				////ssd를 위한 rect 체크 및 획득
				bool b1 = CheckBoundary(pt.x - nHalfWindowSize, pt.y - nHalfWindowSize, prevImg.rows, prevImg.cols);
				bool b2 = CheckBoundary(pt.x + nHalfWindowSize, pt.y + nHalfWindowSize, prevImg.rows, prevImg.cols);
				cv::Rect rect1 = cv::Rect(pt.x - nHalfWindowSize, pt.y - nHalfWindowSize, nFullWindow, nFullWindow);
				if (!b1 || !b2)
					continue;
				cv::Mat prevPatch = prevRectified(rect1);
				float minValue = 0;//
				cv::Point2f minPt(0,0);
				bool bMin = false;
				for (int nx = 15; nx < prevImg.cols - 15; nx++) {
					cv::Point2f pt2(nx, pt.y);
					bool b3 = CheckBoundary(pt2.x - nHalfWindowSize, pt2.y - nHalfWindowSize, prevImg.rows, prevImg.cols);
					bool b4 = CheckBoundary(pt2.x + nHalfWindowSize, pt2.y + nHalfWindowSize, prevImg.rows, prevImg.cols);
					cv::Rect rect2 = cv::Rect(pt2.x - nHalfWindowSize, pt2.y - nHalfWindowSize, nFullWindow, nFullWindow);
					
					if (b3 && b4) {
						cv::Mat currPatch = currRectified(rect2);
						/*float val = CalcSSD(prevPatch, currPatch);
						if (val < 5.0) {
							bMin = true;
							cv::circle(rectified, pt2 + ptRight, 2, cv::Scalar(0, 0, 255));
							if (val < minValue) {
								minPt = pt2;
								minValue = val;
							}
						}*/
						std::cout << "??" << std::endl;
						float val = CalcNCC(prevPatch, currPatch);
						std::cout << "val::" << val << std::endl;
						if (val > 0.9) {
							bMin = true;
							cv::circle(rectified, pt2 + ptRight, 2, cv::Scalar(0, 0, 255));
							if (val > minValue) {
								minPt = pt2;
								minValue = val;
							}
						}
					}
				}
				if (bMin) {
					cv::circle(rectified, minPt + ptRight, 2, cv::Scalar(255, 255, 0));
				}
				
				
				////ssd를 위한 rect 체크 및 획득

			}
		}
	}

	//float nfx = (float)P1.at<double>(0, 0);
	//float ncx = (float)P1.at<double>(0, 2);
	//float nfy = (float)P1.at<double>(1, 1);
	//float ncy = (float)P1.at<double>(1, 2);
	//std::cout << mapPrev1.type() <<", "<<mapPrev1.channels()<<"???????????????????" << std::endl;
	//for (int i = 0; i < tPrevPts.size(); i += 20) {
	//	auto prevPt = tPrevPts[i];
	//	cv::Point2f rPrevPt(prevPt.x*nfx + ncx, prevPt.y*nfy + ncy);
	//	//cv::Point2f rPrevPt(mapPrev1.at<float>(prevPt), mapPrev2.at<float>(prevPt));
	//	cv::circle(rectified, rPrevPt, 2, cv::Scalar(0, 0, 255));
	//	cv::line(rectified, cv::Point2f(0, rPrevPt.y), cv::Point2f(rectified.cols, rPrevPt.y), cv::Scalar(255, 0, 0));
	//}

	imshow("rectified", rectified);
	//imshow("rectified::curr", currRectified);
	cv::waitKey(1);
	/////////////////////////////Rectification

	vector<cv::Point3f> lines[2];
	cv::computeCorrespondEpilines(tPrevPts, 2, F12, lines[0]);
	RNG rng(12345);
	for (int i = 0; i < tPrevPts.size(); i+=20) {
		
		float m = 9999.0;
		if (lines[0][i].x != 0)
			m = abs(lines[0][i].x / lines[0][i].y);
		bool opt = false;
		if (m > 1.0)
			opt = true;
		cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		////////에피 라인 
		cv::Point2f spt, ept, lpt;
		if (opt) {
			spt = CalcLinePoint(0.0, lines[0][i], opt);
			lpt = CalcLinePoint(tCurrPts[i].y, lines[0][i], opt);
			ept = CalcLinePoint(prevImg.rows, lines[0][i], opt);
		}
		else {
			spt = CalcLinePoint(0.0, lines[0][i], opt);
			lpt = CalcLinePoint(tCurrPts[i].x, lines[0][i], opt);
			ept = CalcLinePoint(prevImg.cols, lines[0][i], opt);
		}
		spt += ptBottom;
		ept += ptBottom;
		lpt += ptBottom;
		cv::line(debugging, spt, ept, color, 1);
		cv::line(debugging, tCurrPts[i]+ptBottom, lpt, color, 1);
		cv::circle(debugging, tPrevPts[i], 1, color, -1);
		cv::circle(debugging, tCurrPts[i] + ptBottom, 1, color, -1);
		////////에피 라인 
	}
	////Epipolar constraints
	//////////////////////////////////////////
	

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;

	//fuse time text 
	std::stringstream ss;
	ss << "Optical flow init= " << nRes << ", " << tttt;
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	imshow("edge+optical", debugging);
	///////////////////////////

}

int UVR_SLAM::Matcher::OpticalMatchingForMapping(Frame* prev, Frame* curr, std::vector<std::pair<cv::Point2f, cv::Point2f>>& resMatches, cv::Mat& debugging) {
	//////////////////////////
	////Optical flow
	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	std::vector<cv::Mat> currPyr, prevPyr;
	std::vector<uchar> status;
	std::vector<float> err;
	cv::Mat prevImg = prev->GetOriginalImage();
	cv::Mat currImg = curr->GetOriginalImage();

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
	prevPts = prev->mpMatchInfo->GetMatchingPts();
	auto mvpMatchingMPs = prev->mpMatchInfo->GetMatchingMPs();
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

	for (int i = 0; i < prevPts.size(); i++) {
		if (status[i] == 0) {
			nBad++;
			continue;
		}

		////추가적인 에러처리
		////레이블드에서 255 150 100 벽 바닥 천장
		//int prevLabel = init->matLabeled.at<uchar>(prevPts[i].y / 2, prevPts[i].x / 2);
		//if (prevLabel != 255 && prevLabel != 150 && prevLabel != 100) {
		//	nBad++;
		//	continue;
		//}
		//int currLabel = curr->matLabeled.at<uchar>(currPts[i].y / 2, currPts[i].x / 2);
		//if (prevLabel != currLabel) {
		//	nBad++;
		//	continue;
		//}

		/////
		//매칭 결과
		float diffX = abs(prevPts[i].x - currPts[i].x);
		bool bMatch = false;
		if (diffX < 90) {
			bMatch = true;
			res++;
		}

		if (bMatch){
			resMatches.push_back(std::pair<cv::Point2f, cv::Point2f>(prevPts[i], currPts[i]));

			UVR_SLAM::MapPoint* pMPi = mvpMatchingMPs[i];
			if (pMPi && !pMPi->isDeleted()) {
				cv::circle(debugging, prevPts[i], 1, cv::Scalar(255, 255, 0), -1);
				cv::circle(debugging, currPts[i] + ptBottom, 1, cv::Scalar(255, 255, 0), -1);
			}
			else {
				cv::circle(debugging, prevPts[i], 1, cv::Scalar(255, 0, 0), -1);
				cv::circle(debugging, currPts[i] + ptBottom, 1, cv::Scalar(255, 0, 0), -1);
			}

		}
		//매칭 결과
		////

	}
	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;

	//fuse time text 
	std::stringstream ss;
	ss << "Optical flow mapping= " << res << ", " << tttt;
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	imshow("Mapping::OpticalFlow ", debugging);
	/////////////////////////
	return res;
}

int UVR_SLAM::Matcher::Fuse(Frame* pKF1, Frame* pKF2, Frame* pKF3, cv::Mat& debug) {
	////////////////////////////
	//////Optical flow
	//std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	//cv::Mat imgKF1 = pKF1->GetOriginalImage();
	//cv::Mat imgKF2 = pKF2->GetOriginalImage();
	//cv::Mat imgKF3 = pKF3->GetOriginalImage();

	/////////debug
	//cv::Point2f ptBottom = cv::Point2f(0, imgKF1.rows);
	//cv::Rect mergeRect1 = cv::Rect(0, 0, imgKF1.cols, imgKF1.rows);
	//cv::Rect mergeRect2 = cv::Rect(0, imgKF1.rows, imgKF1.cols, imgKF1.rows);
	//debug = cv::Mat::zeros(imgKF1.rows * 2, imgKF1.cols, imgKF1.type());
	//imgKF1.copyTo(debug(mergeRect1));
	//imgKF2.copyTo(debug(mergeRect2));

	//////Epipolar constraints
	//cv::Mat Rkf1, Rkf2, Tkf1, Tkf2, Rkf3, Tkf3;
	//pKF1->GetPose(Rkf1, Tkf1);
	//pKF2->GetPose(Rkf2, Tkf2);
	//pKF3->GetPose(Rkf3, Tkf3);
	//cv::Mat mK = pKF1->mK.clone();
	//cv::Mat F12 = CalcFundamentalMatrix(Rkf1, Tkf1, Rkf2, Tkf2, mK);
	//cv::Mat F23 = CalcFundamentalMatrix(Rkf2, Tkf2, Rkf3, Tkf3, mK);
	//cv::Mat F13 = CalcFundamentalMatrix(Rkf1, Tkf1, Rkf3, Tkf3, mK);

	//////collect mps
	//std::vector<cv::Point2f> vPtsKF1, vPtsKF2, vPtsKF3;
	//std::vector<int> vIdxs;
	////두 KF에 포함되는 포인트만 일단 모으기.
	////pKF2의 MP가 pKF3에 포함되어야 함.
	//for (int i = 0; i < pKF2->mpMatchInfo->mnTargetMatch; i++) {
	//	//projection test
	//	auto pMPi = pKF2->mpMatchInfo->mvpMatchingMPs[i];
	//	if (!pMPi || pMPi->isDeleted())
	//		continue;
	//	if (!pMPi->isInFrame(pKF3->mpMatchInfo)) {
	//		continue;
	//	}
	//	int idx = pMPi->GetPointIndexInFrame(pKF3->mpMatchInfo);
	//	cv::Point2f ptKF3 = pKF3->mpMatchInfo->mvMatchingPts[idx];

	//	//해당 MP가 현재 프레임에서 이미지 밖에 존재하지 않는지 체크
	//	auto X3D = pMPi->GetWorldPos();
	//	auto prevPt = pKF2->mpMatchInfo->mvMatchingPts[i];
	//	auto currPt = pKF1->Projection(X3D);

	//	if (!pKF1->isInImage(currPt.x, currPt.y, 10.0)) {
	//		continue;
	//	}
	//	//해당 MP가 현재 프레임에서 이미지 밖에 존재하지 않는지 체크

	//	vPtsKF1.push_back(currPt);
	//	vPtsKF2.push_back(prevPt);
	//	vPtsKF3.push_back(ptKF3);

	//	vIdxs.push_back(idx);
	//}

	////////epipolar line
	//vector<cv::Point3f> lines[2];
	//cv::computeCorrespondEpilines(vPtsKF2, 1, F12, lines[0]);
	//cv::computeCorrespondEpilines(vPtsKF3, 1, F13, lines[1]);
	//RNG rng(12345);
	//for (int i = 0; i < vPtsKF2.size(); i += 10){

	//	float m = 9999.0;
	//	if (lines[0][i].x != 0)
	//		m = abs(lines[0][i].x / lines[0][i].y);
	//	bool opt = false;
	//	if (m > 1.0)
	//		opt = true;
	//	cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	//	////////에피 라인 
	//	cv::Point2f spt, ept, lpt;
	//	spt = CalcLinePoint(0.0, lines[0][i], true);
	//	ept = CalcLinePoint(imgKF1.rows, lines[0][i], true);

	//	//pKF1에서 pKF2의 에피라인 상의 모든 점을 탐색할 때의 코드
	//	/*std::vector<cv::Point2f> pts2;
	//	std::vector<cv::Point3f> tempLines, tempLines23;
	//	int a, b;

	//	if (spt.x < 0) {
	//		spt = CalcLinePoint(0.0, lines[0][i], false);
	//	}
	//	if(ept.x >= imgKF1.cols) {
	//		ept = CalcLinePoint(imgKF1.cols, lines[0][i], false);
	//	}
	//	if (opt) {
	//		a = spt.y;
	//		b = ept.y;
	//	}
	//	else {
	//		if (spt.x < ept.x) {
	//			a = spt.x;
	//			b = ept.x;
	//		}
	//		else {
	//			b = spt.x;
	//			a = ept.x;
	//		}
	//	}
	//	for (int j = a; j < b; j++) {
	//		cv::Point2f tempPt = CalcLinePoint(j, lines[0][i], opt);
	//		pts2.push_back(tempPt);
	//		//cv::circle(debug, tempPt + ptBottom, 2, cv::Scalar(255,0,255), -1);
	//	}
	//	*/
	//	//pKF1에서 pKF2의 에피라인

	//	//pKF1에서 pKF3의 에피라인
	//	cv::Point2f spt2, ept2;
	//	spt2 = CalcLinePoint(0.0, lines[1][i], true);
	//	ept2 = CalcLinePoint(imgKF1.rows, lines[1][i], true);
	//	//pKF1에서 pKF3의 에피라인
	//	
	//	//두 라인의 교점
	//	if (i%60==0) {
	//		cv::line(debug, spt, ept, color, 1);
	//		cv::line(debug, spt2, ept2, color, 1);
	//	}
	//		float a1 = lines[0][i].x;
	//		float b1 = lines[0][i].y;
	//		float c1 = lines[0][i].z;

	//		float a2 = lines[1][i].x;
	//		float b2 = lines[1][i].y;
	//		float c2 = lines[1][i].z;

	//		float a = a1*b2 - a2*b1;
	//		
	//		if (abs(b1) < 0.0001 && abs(b2) < 0.0001)
	//			continue;
	//		if (abs(a) < 0.0001)
	//			continue;

	//		float b = b1*c2 - b2*c1;
	//		float x = b / a;
	//		float y = -a1/b1*x - c1 / b1;
	//			
	//		cv::Point2f pt(x, y);
	//		if (!pKF1->isInImage(x, y, 10.0)) {
	//			continue;
	//		}
	//		cv::circle(debug, vPtsKF2[i] + ptBottom, 3, color, -1);
	//		cv::circle(imgKF3, vPtsKF3[i], 3, color, -1);
	//		cv::circle(debug, pt, 3, color, -1);
	//	//}
	//	//두 라인의 교점

	//	
	//	//float minDist = FLT_MAX;
	//	//int minIDX = 0;
	//	//cv::computeCorrespondEpilines(pts2, 1, F12, tempLines); //이들은 모두 값이 동일함.
	//	//cv::computeCorrespondEpilines(pts2, 2, F23, tempLines23); //이들은 모두 값이 동일함.
	//	//for (int j = 0; j < tempLines.size(); j++) {
	//	//	float a1 = tempLines[j].x;
	//	//	float b1 = tempLines[j].y;
	//	//	float c1 = tempLines[j].z;

	//	//	//	auto pMPi = vpMPs[i];
	//	//	//	auto X3D = pMPi->GetWorldPos();
	//	//	//	auto currPt = mpTargetFrame->Projection(X3D);
	//	//	float dist = abs(vPtsKF1[i].x*a1 + vPtsKF1[i].y*b1 + c1);
	//	//	if (dist < minDist) {
	//	//		minIDX = j;
	//	//		minDist = dist;
	//	//	}
	//	//	if (i == 0) {
	//	//		cv::Point2f spt2, ept2;
	//	//		spt2 = CalcLinePoint(0.0, tempLines[j], true);
	//	//		ept2 = CalcLinePoint(imgKF1.rows, tempLines[j], true);
	//	//		cv::line(debug, spt2, ept2, color, 1);
	//	//		cv::line(debug, spt+ptBottom, ept+ptBottom, color, 1);
	//	//	}
	//	//}
	//	//////pKF2, pKF3 사이의 에피폴라 라인 확인
	//	////for (int j = 0; j < tempLines23.size(); j++) {
	//	////	cv::Point2f spt2, ept2;
	//	////	spt2 = CalcLinePoint(0.0, tempLines23[j], true);
	//	////	ept2 = CalcLinePoint(imgKF1.rows, tempLines23[j], true);
	//	////	if (i == 0)
	//	////		cv::line(imgKF3, spt2, ept2, color, 1);
	//	////}

	//	//spt += ptBottom;
	//	//ept += ptBottom;
	//	lpt += ptBottom;
	//	
	//	//cv::line(debug, tCurrPts[i] + ptBottom, lpt, color, 1);
	//	//cv::circle(debug, pts2[minIDX] + ptBottom, 3, color, -1);
	//	////////에피 라인 
	//}

	//std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	//double tttt = duration / 1000.0;

	////fuse time text 
	//std::stringstream ss;
	//ss << "Fuse = " << vPtsKF1.size() << ", " << tttt;
	//cv::rectangle(debug, cv::Point2f(0, 0), cv::Point2f(debug.cols, 30), cv::Scalar::all(0), -1);
	//cv::putText(debug, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	////imshow("edge+optical", debugging);
	//imshow("FUSE", debug);
	//imshow("pKF2-pKF3", imgKF3);
	/////////////////////////////
	//waitKey(1);
	return 0;

}

int UVR_SLAM::Matcher::OpticalMatchingForFuseWithEpipolarGeometry(Frame* prev, Frame* curr, cv::Mat& debugging) {
	////////////////////////////
	//////Optical flow
	//std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	//std::vector<cv::Mat> currPyr, prevPyr;
	//std::vector<uchar> status;
	//std::vector<float> err;
	//cv::Mat prevImg = prev->GetOriginalImage();
	////cv::Mat prevImg = curr->mpMatchInfo->mpTargetFrame->GetOriginalImage();
	//cv::Mat currImg = curr->GetOriginalImage();

	/////////debug
	//cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
	//cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	//cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
	//debugging = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
	//prevImg.copyTo(debugging(mergeRect1));
	//currImg.copyTo(debugging(mergeRect2));

	/////////////
	////matKPs, mvKPs
	////init -> curr로 매칭
	//////////
	//std::vector<cv::Point2f> prevPts, currPts;
	//std::vector<int> vIdxs;
	//for (int i = 0; i < prev->mpMatchInfo->mnTargetMatch; i++) {
	//	//projection test
	//	auto pMPi = prev->mpMatchInfo->mvpMatchingMPs[i];
	//	if (!pMPi || pMPi->isDeleted())
	//		continue;
	//	auto X3D = pMPi->GetWorldPos();
	//	auto prevPt = prev->mpMatchInfo->mvMatchingPts[i];
	//	auto currPt = curr->Projection(X3D);
	//	if (!curr->isInImage(currPt.x, currPt.y, 10.0)) {
	//		continue;
	//	}
	//	prevPts.push_back(prevPt);
	//	currPts.push_back(currPt);
	//	vIdxs.push_back(i);
	//}

	////////////////////////////////////////////
	//////Epipolar constraints
	//cv::Mat Rcurr, Tcurr, Rprev, Tprev;
	//prev->GetPose(Rprev, Tprev);
	//curr->GetPose(Rcurr, Tcurr);
	//cv::Mat mK = prev->mK.clone();
	//cv::Mat mD = prev->mDistCoef.clone();
	//cv::Mat mK2 = prev->mK.clone();
	//cv::Mat mD2 = prev->mDistCoef.clone();
	//cv::Mat F12 = CalcFundamentalMatrix(Rprev, Tprev, Rcurr, Tcurr, mK);

	///////////////////////////////Rectification
	///////포인트 매칭을 위해서
	//cv::Mat used = cv::Mat::zeros(prevImg.size(), CV_16UC1);
	//for (int i = 0; i < prevPts.size(); i += 5) {
	//	used.at<ushort>(prevPts[i]) = i + 1;
	//}
	//cv::Mat used2 = cv::Mat::zeros(prevImg.size(), CV_16UC1);
	//for (int i = 0; i < currPts.size(); i += 5) {
	//	used2.at<ushort>(currPts[i]) = i + 1;
	//}
	///////포인트 매칭을 위해서

	//cv::Mat R, t;
	///*R = Rprev*Rcurr.t();
	//t = -Rprev*Rcurr.t()*Tcurr + Tprev;*/
	//R = Rprev.t()*Rcurr;
	//t = -Rprev.t()*Tprev + Rprev.t()*Tcurr;

	//R.convertTo(R, CV_64FC1);
	//t.convertTo(t, CV_64FC1);
	//cv::Mat Q;
	//cv::Mat R1, R2, P1, P2;

	//stereoRectify(mK, mD, mK, mD, prevImg.size(), R, t, R1, R2, P1, P2, Q);

	//cv::Mat mapPrev1, mapPrev2;
	//cv::Mat mapCurr1, mapCurr2;
	//initUndistortRectifyMap(mK, mD, R1, P1.colRange(0, 3).rowRange(0, 3), prevImg.size(), CV_32FC1, mapPrev1, mapPrev2);
	//initUndistortRectifyMap(mK, mD, R2, P2.colRange(0, 3).rowRange(0, 3), prevImg.size(), CV_32FC1, mapCurr1, mapCurr2);

	//cv::Mat prevRectified, currRectified;
	//remap(prevImg, prevRectified, mapPrev1, mapPrev2, cv::INTER_LINEAR);
	//remap(currImg, currRectified, mapCurr1, mapCurr2, cv::INTER_LINEAR);

	////흑백 변환
	//cv::Mat prevGrayImg, currGrayImg;
	//cvtColor(prevRectified, prevGrayImg, CV_BGR2GRAY);
	//cvtColor(currRectified, currGrayImg, CV_BGR2GRAY);
	//prevGrayImg.convertTo(prevGrayImg, CV_8UC1);
	//currGrayImg.convertTo(currGrayImg, CV_8UC1);

	////이미지 복사
	//cv::Point2f ptRight = cv::Point2f(prevImg.cols, 0);
	//cv::Rect rmergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	//cv::Rect rmergeRect2 = cv::Rect(prevImg.cols, 0, prevImg.cols, prevImg.rows);
	//cv::Mat rectified = cv::Mat::zeros(prevImg.rows, prevImg.cols * 2, prevImg.type());
	//prevRectified.copyTo(rectified(rmergeRect1));
	//currRectified.copyTo(rectified(rmergeRect2));
	//////SSD 설정
	//int nHalfWindowSize = 5;
	//int nFullWindow = nHalfWindowSize * 2 + 1;
	//////SSD 설정

	////for (int y = 0; y < prevRectified.rows; y++) {
	////	for (int x = 0; x < prevRectified.cols; x++) {
	////		cv::Point2f pt(x, y);
	////		cv::Point2f currPt(mapCurr1.at<float>(y, x), mapCurr2.at<float>(y, x)); //이게 원래 이미지에서의 위치임.
	////		cv::Point2f prevPt(mapPrev1.at<float>(y, x), mapPrev2.at<float>(y, x));

	////		if (prev->isInImage(prevPt.x, prevPt.y, 10.0) && used.at<ushort>(prevPt) > 0) {
	////			int idx = used.at<ushort>(prevPt) - 1;
	////			bool b = prev->mpMatchInfo->mvpMatchingMPs[vIdxs[idx]]->isInFrame(curr->mpMatchInfo);
	////			if (b)
	////				cv::circle(rectified, pt, 3, cv::Scalar(255, 0, 255));
	////			else
	////				cv::circle(rectified, pt, 3, cv::Scalar(0, 0, 255));
	////		}
	////		if (prev->isInImage(currPt.x, currPt.y, 10.0) && used2.at<ushort>(currPt) > 0) {
	////			int idx = used2.at<ushort>(currPt) - 1;
	////			bool b = prev->mpMatchInfo->mvpMatchingMPs[vIdxs[idx]]->isInFrame(curr->mpMatchInfo);
	////			if(b)
	////				cv::circle(rectified, pt + ptRight, 3, cv::Scalar(255, 0, 255),-1);
	////			else
	////				cv::circle(rectified, pt + ptRight, 3, cv::Scalar(0, 0, 255),-1);
	////		}
	////	}
	////}

	//cv::Mat temp = prevImg.clone();
	//for (int i = 0; i < prevPts.size(); i++) {
	//	cv::circle(temp, prevPts[i], 3, cv::Scalar(0, 0, 255), 1);
	//}
	//imshow("test::prev::", temp);

	//for (int y = 0; y < prevRectified.rows; y++) {
	//	for (int x = 0; x < prevRectified.cols; x++) {

	//		cv::Point2f prevPt(mapPrev1.at<float>(y, x), mapPrev2.at<float>(y, x)); //이게 원래 이미지에서의 위치임.

	//		if (!prev->isInImage(prevPt.x, prevPt.y, 10.0))
	//			continue;

	//		if (used.at<ushort>(prevPt) > 0) {
	//			
	//			int idx = used.at<ushort>(prevPt) - 1;
	//			bool b = prev->mpMatchInfo->mvpMatchingMPs[vIdxs[idx]]->isInFrame(curr->mpMatchInfo);
	//			cv::Point2d pt(x, y);
	//			if (b) {
	//				cv::circle(rectified, pt, 2, cv::Scalar(0, 0, 255),-1);
	//			}
	//			else {
	//				cv::circle(rectified, pt, 2, cv::Scalar(0,255,0),-1);
	//			}
	//			//cv::line(rectified, cv::Point2f(0, pt.y), cv::Point2f(rectified.cols, pt.y), cv::Scalar(255, 0, 0));
	//			/*cv::line(rectified, cv::Point2f(0, prevPt.y), cv::Point2f(rectified.cols, prevPt.y), cv::Scalar(255, 0, 0));
	//			cv::circle(rectified, prevPt, 2, cv::Scalar(0, 0, 255));*/

	//			////ssd를 위한 rect 체크 및 획득
	//			bool b1 = CheckBoundary(pt.x - nHalfWindowSize, pt.y - nHalfWindowSize, prevImg.rows, prevImg.cols);
	//			bool b2 = CheckBoundary(pt.x + nHalfWindowSize, pt.y + nHalfWindowSize, prevImg.rows, prevImg.cols);
	//			cv::Rect rect1 = cv::Rect(pt.x - nHalfWindowSize, pt.y - nHalfWindowSize, nFullWindow, nFullWindow);
	//			if (!b1 || !b2)
	//				continue;
	//			cv::Mat prevPatch = prevGrayImg(rect1).clone();
	//			float minValue = 0; //0;FLT_MAX;
	//			cv::Point2f minPt(0, 0);
	//			bool bMin = false;
	//			
	//			for (int nx = 15; nx < prevImg.cols - 15; nx++) {
	//				cv::Point2f pt2(nx, pt.y);
	//				bool b3 = CheckBoundary(pt2.x - nHalfWindowSize, pt2.y - nHalfWindowSize, prevImg.rows, prevImg.cols);
	//				bool b4 = CheckBoundary(pt2.x + nHalfWindowSize, pt2.y + nHalfWindowSize, prevImg.rows, prevImg.cols);
	//				cv::Rect rect2 = cv::Rect(pt2.x - nHalfWindowSize, pt2.y - nHalfWindowSize, nFullWindow, nFullWindow);
	//				
	//				if (b3 && b4) {
	//					cv::Mat currPatch = currGrayImg(rect2).clone();
	//					
	//					/*float val = CalcSSD(prevPatch, currPatch); 
	//					if (val < 10000.0) {
	//						bMin = true;
	//						cv::circle(rectified, pt2 + ptRight, 2, cv::Scalar(0, 0, 255));
	//						if (val < minValue) {
	//							minPt = pt2;
	//							minValue = val;
	//						}
	//					}*/
	//					
	//					float val = CalcNCC(prevPatch, currPatch);
	//					if (val > 0.99) {
	//						bMin = true;
	//						//cv::circle(rectified, pt2 + ptRight, 2, cv::Scalar(0, 0, 255));
	//						if (val > minValue) {
	//							minPt = pt2;
	//							minValue = val;
	//						}
	//					}
	//				}
	//			}
	//			
	//			if (bMin) {
	//				//std::cout << minValue << std::endl;
	//				cv::circle(rectified, minPt + ptRight, 2, cv::Scalar(0, 255, 0));
	//			}
	//			////ssd를 위한 rect 체크 및 획득

	//		}
	//	}
	//}

	////float nfx = (float)P1.at<double>(0, 0);
	////float ncx = (float)P1.at<double>(0, 2);
	////float nfy = (float)P1.at<double>(1, 1);
	////float ncy = (float)P1.at<double>(1, 2);
	////std::cout << mapPrev1.type() <<", "<<mapPrev1.channels()<<"???????????????????" << std::endl;
	////for (int i = 0; i < tPrevPts.size(); i += 20) {
	////	auto prevPt = tPrevPts[i];
	////	cv::Point2f rPrevPt(prevPt.x*nfx + ncx, prevPt.y*nfy + ncy);
	////	//cv::Point2f rPrevPt(mapPrev1.at<float>(prevPt), mapPrev2.at<float>(prevPt));
	////	cv::circle(rectified, rPrevPt, 2, cv::Scalar(0, 0, 255));
	////	cv::line(rectified, cv::Point2f(0, rPrevPt.y), cv::Point2f(rectified.cols, rPrevPt.y), cv::Scalar(255, 0, 0));
	////}

	//
	////imshow("rectified::curr", currRectified);
	//cv::waitKey(1);
	///////////////////////////////Rectification

	////vector<cv::Point3f> lines[2];
	////cv::computeCorrespondEpilines(tPrevPts, 2, F12, lines[0]);
	////RNG rng(12345);
	////for (int i = 0; i < tPrevPts.size(); i += 20) {

	////	float m = 9999.0;
	////	if (lines[0][i].x != 0)
	////		m = abs(lines[0][i].x / lines[0][i].y);
	////	bool opt = false;
	////	if (m > 1.0)
	////		opt = true;
	////	cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	////	////////에피 라인 
	////	cv::Point2f spt, ept, lpt;
	////	if (opt) {
	////		spt = CalcLinePoint(0.0, lines[0][i], opt);
	////		lpt = CalcLinePoint(tCurrPts[i].y, lines[0][i], opt);
	////		ept = CalcLinePoint(prevImg.rows, lines[0][i], opt);
	////	}
	////	else {
	////		spt = CalcLinePoint(0.0, lines[0][i], opt);
	////		lpt = CalcLinePoint(tCurrPts[i].x, lines[0][i], opt);
	////		ept = CalcLinePoint(prevImg.cols, lines[0][i], opt);
	////	}
	////	spt += ptBottom;
	////	ept += ptBottom;
	////	lpt += ptBottom;
	////	cv::line(debugging, spt, ept, color, 1);
	////	cv::line(debugging, tCurrPts[i] + ptBottom, lpt, color, 1);
	////	cv::circle(debugging, tPrevPts[i], 1, color, -1);
	////	cv::circle(debugging, tCurrPts[i] + ptBottom, 1, color, -1);
	////	////////에피 라인 
	////}
	//////Epipolar constraints
	////////////////////////////////////////////


	//std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	//double tttt = duration / 1000.0;

	////fuse time text 
	//std::stringstream ss;
	//ss << "Optical flow init= " << prevPts.size() << ", " << tttt;
	//cv::rectangle(rectified, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	//cv::putText(rectified, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	////imshow("edge+optical", debugging);
	//imshow("rectified", rectified);
	/////////////////////////////
	//waitKey(1);
}
////200410 Optical flow
///////////////////////////////////////////////////////////////////////////////////////////////////////