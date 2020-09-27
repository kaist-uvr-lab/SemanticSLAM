#include <Matcher.h>
#include <omp.h>
#include <random>
#include <Frame.h>
#include <MapPoint.h>
#include <CandidatePoint.h>
#include <FrameWindow.h>
#include <MatrixOperator.h>
#include <gms_matcher.h>
#include <PlaneEstimator.h>
#include <Plane.h>
#include <Visualizer.h>
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
void UVR_SLAM::Matcher::SetVisualizer(Visualizer* pVis) {
	mpVisualizer = pVis;
}
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
	//mpVisualizer->SetOutputImage(debugging, 0);
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
	cv::Mat resized;
	cv::resize(debugging, resized, cv::Size(debugging.cols / 2, debugging.rows / 2));
	mpVisualizer->SetOutputImage(resized,0);
	///////////////////////////

	return vpPts2.size();
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

	int Ncp = prev->mpMatchInfo->GetNumCPs();
	for (int i = 0; i < Ncp; i++) {
		auto pCPi = prev->mpMatchInfo->mvpMatchingCPs[i];
		if (prevPts.size() == 500)
			break;
		auto pMPi = pCPi->GetMP();
		if (pMPi && pCPi->GetQuality() && pCPi->isOptimized()) {
			if (pMPi->isDeleted())
				continue;
			prevPts.push_back(prev->mpMatchInfo->mvMatchingPts[i]);
			vpTempCPs.push_back(pCPi);
			vpTempMPs.push_back(pMPi);
		}
	}

	//prev->mpMatchInfo->GetMatchingPtsTracking(vpTempCPs, vpTempMPs, prevPts);//prev->mvMatchingPts;
	//std::cout << "Matcher::Tracking::" << vpTempCPs.size() << std::endl;
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
	cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
	cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
	debugging = cv::Mat::zeros(prevImg.rows*2, prevImg.cols, prevImg.type());
	prevImg.copyTo(debugging(mergeRect1));
	currImg.copyTo(debugging(mergeRect2));
	///////////

	std::vector<cv::Point2f> prevPts,currPts;
	std::vector<int> vnOctaves;
	std::vector<CandidatePoint*> vpCPs;
	
	int nCP = pPrevKF->mpMatchInfo->GetNumCPs();
	vpCPs = pPrevKF->mpMatchInfo->mvpMatchingCPs;
	prevPts = pPrevKF->mpMatchInfo->mvMatchingPts;//pPrevKF->mpMatchInfo->GetMatchingPtsMapping(vpCPs);
	auto pPrevMatchInfo = pPrevKF->mpMatchInfo;
	auto pCurrMatchInfo = pCurrKF->mpMatchInfo;
	
	int maxLvl = 3;
	int searchSize = 21;
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
		bool b1 = pCurrMatchInfo->CheckOpticalPointOverlap(Frame::mnRadius, 10, currPts[i]) >= 0;
		bool b3 = pPrevMatchInfo->CheckOpticalPointOverlap(usedPrev, Frame::mnRadius, 10, prevPts[i]);
		bool b4 = pPrevMatchInfo->CheckOpticalPointOverlap(usedCurr, Frame::mnRadius, 10, currPts[i]);
		if (b1 || !b3 || !b4) {
			continue;
		}
		cv::circle(usedPrev, prevPts[i], Frame::mnRadius, cv::Scalar(255, 0, 0), -1);
		cv::circle(usedCurr, currPts[i], Frame::mnRadius, cv::Scalar(255, 0, 0), -1);
		
		vTempPrevPts.push_back(prevPts[i]);
		vTempCurrPts.push_back(currPts[i]);
		vTempIDXs.push_back(i);

	}
	if (vTempPrevPts.size() < 10) {
		std::cout << "LM::Matching::error" << std::endl;
	}

	///////////////////projection based matching
	cv::Mat Rprev, Tprev, Rcurr, Tcurr;
	pCurrKF->GetPose(Rcurr, Tcurr);
	pPrevKF->GetPose(Rprev, Tprev);

	cv::Mat Pcurr, Pprev;
	cv::hconcat(Rcurr, Tcurr, Pcurr);
	cv::hconcat(Rprev, Tprev, Pprev);
	cv::Mat Map;
	cv::triangulatePoints(K*Pprev, K*Pcurr, vTempPrevPts, vTempCurrPts, Map);

	///////데이터 전처리
	cv::Mat Rcfromc = Rcurr.t();
	cv::Mat Rpfromc = Rprev.t();
	
	int nRes = 0;
	int nTargetID = pPrevKF->GetFrameID();
	for (int i = 0; i < Map.cols; i++) {

		cv::Mat X3D = Map.col(i);
		auto currPt = vTempCurrPts[i];
		auto prevPt = vTempPrevPts[i];

		if (abs(X3D.at<float>(3)) < 0.0001) {
			/*std::cout << "test::" << X3D.at<float>(3) << std::endl;
			cv::circle(debugging, currPt + ptBottom, 2, cv::Scalar(0, 0, 0), -1);
			cv::circle(debugging, prevPt, 2, cv::Scalar(0, 0, 0), -1);*/
			continue;
		}

		X3D /= X3D.at<float>(3);
		X3D = X3D.rowRange(0, 3);

		cv::Mat proj1 = Rcurr*X3D + Tcurr;
		cv::Mat proj2 = Rprev*X3D + Tprev;

		////depth test
		//if (proj1.at<float>(2) < 0.0 || proj2.at<float>(2) < 0.0) {
		//	cv::circle(debugging, currPt + ptBottom, 2, cv::Scalar(0, 255, 0), -1);
		//	cv::circle(debugging, prevPt, 2, cv::Scalar(0, 255, 0), -1);
		//	/*if (proj1.at<float>(0) < 0 && proj1.at<float>(1) < 0 && proj1.at<float>(2) < 0) {
		//	cv::circle(debugMatch, pt1 + ptBottom, 2, cv::Scalar(255, 0, 0), -1);
		//	cv::circle(debugMatch, pt2, 2, cv::Scalar(255, 0, 0), -1);
		//	}*/
		//	continue;
		//}
		////depth test

		////reprojection error
		proj1 = K*proj1;
		proj2 = K*proj2;
		cv::Point2f projected1(proj1.at<float>(0) / proj1.at<float>(2), proj1.at<float>(1) / proj1.at<float>(2));
		cv::Point2f projected2(proj2.at<float>(0) / proj2.at<float>(2), proj2.at<float>(1) / proj2.at<float>(2));

		auto diffPt1 = projected1 - currPt;
		auto diffPt2 = projected2 - prevPt;
		float err1 = (diffPt1.dot(diffPt1));
		float err2 = (diffPt2.dot(diffPt2));
		if (err1 > 9.0 || err2 > 9.0) {
			
			cv::circle(debugging, currPt + ptBottom, 2, cv::Scalar(255, 0, 0), -1);
			cv::circle(debugging, prevPt, 2, cv::Scalar(255, 0, 0), -1);
			continue;
		}
		////reprojection error

		////CP 연결하기
		vMatchedCurrPts.push_back(vTempCurrPts[i]);
		vMatchedPrevPts.push_back(vTempPrevPts[i]);
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

			cv::circle(debugging, projected1 + ptBottom, 2, cv::Scalar(0, 255, 0), -1);
			cv::circle(debugging, projected2, 2, cv::Scalar(0, 255, 0), -1);
			cv::line(debugging, currPt + ptBottom, projected1 + ptBottom, cv::Scalar(0, 255, 255), 2);
			cv::line(debugging, prevPt, projected2, cv::Scalar(0, 255, 255), 2);

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

	///////////////////projection based matching

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	dtime = duration / 1000.0;

	//fuse time text 
	/*std::stringstream ss;
	ss << "Optical flow Mapping2= " << pCurrKF->GetFrameID() << ", " << pPrevKF->GetFrameID() << ", " << "::" << tttt << "::" << nRes;
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));*/
}

int UVR_SLAM::Matcher::OpticalMatchingForMapping2(Frame* pCurrKF, Frame* pPrevKF, std::vector<cv::Point2f>& vMatchedPrevPts, std::vector<cv::Point2f>& vMatchedCurrPts, std::vector<CandidatePoint*>& vMatchedCPs, cv::Mat K, cv::Mat InvK, double& dtime, cv::Mat& debugging) {
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
	std::vector<CandidatePoint*> vPrevCPs;
	prevPts = pPrevKF->mpMatchInfo->GetMatchingPtsMapping(vPrevCPs);
	if (prevPts.size() < 5)
		return 0;
	auto pPrevMatchInfo = pPrevKF->mpMatchInfo;
	auto pCurrMatchInfo = pCurrKF->mpMatchInfo;

	int maxLvl = 3;
	int searchSize = 21;
	cv::buildOpticalFlowPyramid(currImg, currPyr, cv::Size(searchSize, searchSize), maxLvl);
	maxLvl = cv::buildOpticalFlowPyramid(prevImg, prevPyr, cv::Size(searchSize, searchSize), maxLvl);
	cv::calcOpticalFlowPyrLK(prevPyr, currPyr, prevPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);

	////옵티컬 플로우 매칭 저장
	std::vector<cv::Point2f> vTempPrevPts, vTempCurrPts;
	std::vector<int> vTempIDXs;

	cv::Mat usedPrev = cv::Mat::zeros(prevImg.size(), CV_8UC1);
	cv::Mat usedCurr = cv::Mat::zeros(prevImg.size(), CV_8UC1);
	for (int i = 0; i < prevPts.size(); i++) {
		cv::circle(debugging, prevPts[i], 2, cv::Scalar(0, 0, 0), -1);
		if (status[i] == 0) {
			continue;
		}
		auto pCPi = vPrevCPs[i];
		bool b1 = pCurrMatchInfo->CheckOpticalPointOverlap(Frame::mnRadius, 10, currPts[i]) >= 0; //used //얘는 왜 used 따로 만듬???
		bool b3 = pPrevMatchInfo->CheckOpticalPointOverlap(usedPrev, Frame::mnRadius, 10, prevPts[i]); //used //얘는 왜 used 따로 만듬???
		bool b4 = pPrevMatchInfo->CheckOpticalPointOverlap(usedCurr, Frame::mnRadius, 10, currPts[i]); //used //얘는 왜 used 따로 만듬???
		if (b1 || !b3 || !b4) {//|| b4 || b5
			continue;
		}
		cv::circle(usedPrev, prevPts[i], Frame::mnRadius, cv::Scalar(255, 0, 0), -1);
		cv::circle(usedCurr, currPts[i], Frame::mnRadius, cv::Scalar(255, 0, 0), -1);
		cv::circle(debugging, prevPts[i], 2, cv::Scalar(255, 0, 0), -1);
		cv::circle(debugging, currPts[i] + ptBottom, 2, cv::Scalar(255, 0, 0), -1);
		vTempPrevPts.push_back(prevPts[i]);
		vTempCurrPts.push_back(currPts[i]);
		vTempIDXs.push_back(i);

	}
	
	//Find fundamental matrix & matching
	std::vector<uchar> vFInliers;
	std::vector<cv::Point2f> vTempFundPrevPts, vTempFundCurrPts;
	std::vector<int> vTempMatchIDXs; 
	cv::Mat E12 = cv::findEssentialMat(vTempPrevPts, vTempCurrPts, K, cv::FM_RANSAC, 0.999, 1.0, vFInliers);
	
	int nRes = 0;
	int nTargetID = pPrevKF->GetFrameID();
	for (unsigned long i = 0; i < vFInliers.size(); i++) {
		if (vFInliers[i]) {
			
			auto prevPt = vTempPrevPts[i];
			auto currPt = vTempCurrPts[i];
			int idx = vTempIDXs[i];
			vMatchedPrevPts.push_back(prevPt);
			vMatchedCurrPts.push_back(currPt);
			auto pCPi = vPrevCPs[idx];
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

			pCurrKF->mpMatchInfo->AddCP(pCPi, currPt);
			vMatchedCPs.push_back(pCPi);
			nRes++;
		}
	}
	//for (unsigned long i = 0; i < vFInliers.size(); i++) {
	//	if (vFInliers[i]) {
	//		vTempFundPrevPts.push_back(vTempPrevPts[i]);
	//		vTempFundCurrPts.push_back(vTempCurrPts[i]);
	//		vTempMatchIDXs.push_back(vTempIDXs[i]);//vTempIndexs[i]
	//	}
	//}
	//
	////////F, E를 통한 매칭 결과 반영
	/////////삼각화 : OpenCV
	//cv::Mat R1, t1;
	//cv::Mat matTriangulateInliers;
	//cv::Mat Map3D;
	//int res2 = cv::recoverPose(E12, vTempFundPrevPts, vTempFundCurrPts, K, R1, t1, 50.0, matTriangulateInliers, Map3D);
	//R1.convertTo(R1, CV_32FC1);
	//t1.convertTo(t1, CV_32FC1);
	//Map3D.convertTo(Map3D, CV_32FC1);
	//
	//int nRes = 0;
	//for (int i = 0; i < matTriangulateInliers.rows; i++) {
	//	int val = matTriangulateInliers.at<uchar>(i);
	//	if (val == 0)
	//		continue;

	//	cv::Mat X3D = Map3D.col(i);
	//	//if (abs(X3D.at<float>(3)) < 0.0001) {
	//	//	/*std::cout << "test::" << X3D.at<float>(3) << std::endl;
	//	//	cv::circle(debugging, currPt + ptBottom, 2, cv::Scalar(0, 0, 0), -1);
	//	//	cv::circle(debugging, prevPt, 2, cv::Scalar(0, 0, 0), -1);*/
	//	//	continue;
	//	//}
	//	X3D /= X3D.at<float>(3);
	//	X3D = X3D.rowRange(0, 3);
	//	
	//	auto currPt = vTempFundCurrPts[i];
	//	auto prevPt = vTempFundPrevPts[i];
	//	int idx = vTempMatchIDXs[i]; //cp idx

	//	////reprojection error
	//	cv::Mat proj1 = X3D;
	//	cv::Mat proj2 = R1*X3D + t1;
	//	proj1 = K*proj1;
	//	proj2 = K*proj2;
	//	cv::Point2f projected1(proj1.at<float>(0) / proj1.at<float>(2), proj1.at<float>(1) / proj1.at<float>(2));
	//	cv::Point2f projected2(proj2.at<float>(0) / proj2.at<float>(2), proj2.at<float>(1) / proj2.at<float>(2));

	//	auto diffPt1 = projected1 - currPt;
	//	auto diffPt2 = projected2 - prevPt;
	//	float err1 = (diffPt1.dot(diffPt1));
	//	float err2 = (diffPt2.dot(diffPt2));
	//	if (err1 > 9.0 || err2 > 9.0) {
	//		cv::circle(debugging, currPt + ptBottom, 2, cv::Scalar(255, 0, 0), -1);
	//		cv::circle(debugging, prevPt, 2, cv::Scalar(255, 0, 0), -1);
	//		continue;
	//	}
	//	////reprojection error

	//	///CP 연결하기
	//	vMatchedCurrPts.push_back(currPt);
	//	vMatchedPrevPts.push_back(prevPt);
	//	auto pCPi = vPrevCPs[idx];
	//	if (pCPi->GetNumSize() == 1) {
	//		cv::circle(debugging, prevPt, 2, cv::Scalar(0, 0, 255), -1);
	//		cv::circle(debugging, currPt + ptBottom, 2, cv::Scalar(0, 0, 255), -1);
	//	}
	//	else {
	//		cv::circle(debugging, prevPt, 2, cv::Scalar(255, 255, 0), -1);
	//		cv::circle(debugging, currPt + ptBottom, 2, cv::Scalar(255, 255, 0), -1);
	//	}
	//	pCPi->AddFrame(pCurrKF->mpMatchInfo, currPt);
	//	vMatchedCPs.push_back(pCPi);

	//	nRes++;
	//}
	//std::cout << "res::" << vTempFundPrevPts.size() << ", " << res2 <<", "<< nRes << std::endl;
	///////////////////projection based matching

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	dtime = duration / 1000.0;

	return nRes;
	//fuse time text 
	/*std::stringstream ss;
	ss << "Optical flow Mapping2= " << pCurrKF->GetFrameID() << ", " << pPrevKF->GetFrameID() << ", " << "::" << tttt << "::" << nRes;
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));*/
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

////200410 Optical flow
///////////////////////////////////////////////////////////////////////////////////////////////////////