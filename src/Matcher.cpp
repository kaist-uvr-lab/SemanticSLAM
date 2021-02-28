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
#include <DBoW3.h>

UVR_SLAM::Matcher::Matcher():TH_HIGH(100), TH_LOW(50), HISTO_LENGTH(30) {}
UVR_SLAM::Matcher::Matcher(System* pSys, cv::Ptr < cv::DescriptorMatcher> _matcher)
	:mpSystem(pSys), TH_HIGH(100), TH_LOW(50), HISTO_LENGTH(30), mfNNratio(0.7), mbCheckOrientation(true), matcher(_matcher)
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

namespace UVR_SLAM {

	int Matcher::OpticalFlowMatching(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> vecPoints, std::vector<cv::Point2f>& vecMatchPoints1, std::vector<cv::Point2f>& vecMatchPoints2, std::vector<int>& vecIndexes) {
		int maxLvl = 3;
		int searchSize = 21;
		std::vector<uchar> status;
		std::vector<float> err;
		std::vector<cv::Point2f> tempPts;
		cv::calcOpticalFlowPyrLK(img1, img2, vecPoints, tempPts, status, err, cv::Size(searchSize, searchSize), maxLvl);

		int nRes = 0;
		for (int i = 0; i < vecPoints.size(); i++) {
			if (status[i] == 0) {
				continue;
			}
			vecMatchPoints1.push_back(vecPoints[i]);
			vecMatchPoints2.push_back(tempPts[i]);
			vecIndexes.push_back(i);
			nRes++;
		}
		return nRes;
	}

	int Matcher::OpticalFlowMatching(int nFrameID, cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> vecPoints, std::vector<MapPoint*> vecMPs, std::vector<cv::Point2f>& vecMatchPoints1, std::vector<cv::Point2f>& vecMatchPoints2, std::vector<MapPoint*>& vecMatchMPs, std::vector<bool>& vecInliers, cv::Mat& overlap){
		int maxLvl = 3;
		int searchSize = 21;
		std::vector<uchar> status;
		std::vector<float> err;
		std::vector<cv::Point2f> tempPts;
		cv::calcOpticalFlowPyrLK(img1, img2, vecPoints, tempPts, status, err, cv::Size(searchSize, searchSize), maxLvl);

		int nRes = 0;
		cv::Point2f ptCheckOverlap(1, 1);
		for (int i = 0; i < vecPoints.size(); i++) {
			if (status[i] == 0) {
				continue;
			}
			
			if (tempPts[i].x < 5.0 || tempPts[i].x >= img1.cols - 5.0 || tempPts[i].y < 5.0 || tempPts[i].y >= img1.rows - 5.0) {
				continue;
			}
			if (overlap.at<uchar>(tempPts[i]) > 0) {
				continue;
			}
			cv::rectangle(overlap, tempPts[i] - ptCheckOverlap, tempPts[i] + ptCheckOverlap, cv::Scalar(255, 0, 0), -1);
			auto pMPi = vecMPs[i];
			if (!pMPi || pMPi->isDeleted())
				continue;
			
			vecMatchPoints1.push_back(vecPoints[i]);
			vecMatchPoints2.push_back(tempPts[i]);
			vecInliers.push_back(true);
			vecMatchMPs.push_back(vecMPs[i]);
			nRes++;
		}
		return nRes;
	}

	float Matcher::SuperPointDescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
		float dist = (float)cv::norm(a, b, cv::NORM_L2);
		return dist;
	}
	////BoW를 이용한 매칭.
	//mbCheckOrientation는 미구현
	int Matcher::BagOfWordsMatching(Frame* pF1, Frame* pF2, std::vector<MapPoint*>& vpMatches12) {
		const vector<cv::Point2f> &vKeysUn1 = pF1->mvPts;
		const DBoW3::FeatureVector &vFeatVec1 = pF1->mFeatVec;
		const vector<MapPoint*> vpMapPoints1 = pF1->GetMapPoints();
		const cv::Mat &Descriptors1 = pF1->matDescriptor;

		const vector<cv::Point2f> &vKeysUn2 = pF2->mvPts;
		const DBoW3::FeatureVector &vFeatVec2 = pF2->mFeatVec;
		const vector<MapPoint*> vpMapPoints2 = pF2->GetMapPoints();
		const cv::Mat &Descriptors2 = pF2->matDescriptor;

		vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(), static_cast<MapPoint*>(NULL));
		vector<bool> vbMatched2(vpMapPoints2.size(), false);

		/*vector<int> rot
		Hist[HISTO_LENGTH];
		for (int i = 0; i<HISTO_LENGTH; i++)
			rotHist[i].reserve(500);*/

		const float factor = 1.0f / HISTO_LENGTH;

		int nmatches = 0;

		DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
		DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
		DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
		DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

		////두 프레임 사이의 일치하는 BoW끼리만 매칭하도록 함.
		////이건 세그멘테이션 아이디가 같은 것끼리만 매칭하도록 하는 것과 동일.
		while (f1it != f1end && f2it != f2end)
		{
			if (f1it->first == f2it->first)
			{
				for (size_t i1 = 0, iend1 = f1it->second.size(); i1<iend1; i1++)
				{
					const size_t idx1 = f1it->second[i1];

					MapPoint* pMP1 = vpMapPoints1[idx1];
					if (!pMP1)
						continue;
					if (pMP1->isDeleted())
						continue;

					const cv::Mat &d1 = Descriptors1.row(idx1);

					float bestDist1 = 256.0;
					int bestIdx2 = -1;
					float bestDist2 = 256.0;

					for (size_t i2 = 0, iend2 = f2it->second.size(); i2<iend2; i2++)
					{
						const size_t idx2 = f2it->second[i2];

						MapPoint* pMP2 = vpMapPoints2[idx2];

						if (vbMatched2[idx2] || !pMP2)
							continue;

						if (pMP2->isDeleted())
							continue;

						const cv::Mat &d2 = Descriptors2.row(idx2);

						float dist = SuperPointDescriptorDistance(d1, d2);

						if (dist<bestDist1)
						{
							bestDist2 = bestDist1;
							bestDist1 = dist;
							bestIdx2 = idx2;
						}
						else if (dist<bestDist2)
						{
							bestDist2 = dist;
						}
					}

					if (bestDist1<(float)TH_LOW)
					{
						if (static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
						{
							vpMatches12[idx1] = vpMapPoints2[bestIdx2];
							vbMatched2[bestIdx2] = true;

							/*if (mbCheckOrientation)
							{
								float rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle;
								if (rot<0.0)
									rot += 360.0f;
								int bin = round(rot*factor);
								if (bin == HISTO_LENGTH)
									bin = 0;
								assert(bin >= 0 && bin<HISTO_LENGTH);
								rotHist[bin].push_back(idx1);
							}*/
							nmatches++;
						}
					}
				}

				f1it++;
				f2it++;
			}
			else if (f1it->first < f2it->first)
			{
				f1it = vFeatVec1.lower_bound(f2it->first);
			}
			else
			{
				f2it = vFeatVec2.lower_bound(f1it->first);
			}
		}

		/*if (mbCheckOrientation)
		{
			int ind1 = -1;
			int ind2 = -1;
			int ind3 = -1;

			ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

			for (int i = 0; i<HISTO_LENGTH; i++)
			{
				if (i == ind1 || i == ind2 || i == ind3)
					continue;
				for (size_t j = 0, jend = rotHist[i].size(); j<jend; j++)
				{
					vpMatches12[rotHist[i][j]] = static_cast<MapPoint*>(NULL);
					nmatches--;
				}
			}
		}*/

		return nmatches;
	}
}





























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

bool  UVR_SLAM::Matcher::OpticalGridMatching(FrameGrid* grid1, cv::Mat src1, cv::Mat src2, std::vector<cv::Point2f>& vPrevPTs, std::vector<cv::Point2f>& vCurrPTs) {
	
	int searchSize = mpSystem->mnRadius;
	int maxLvl = 3;
	std::vector<uchar> status;
	std::vector<float> err;
	std::vector<cv::Point2f> pts1, pts2;

	if (grid1->mvPTs.size() < 1){
		std:cout << "????????????????" << std::endl;
		return false;
	}
	cv::calcOpticalFlowPyrLK(src1, src2, grid1->mvPTs, pts2, status, err, cv::Size(searchSize, searchSize), maxLvl);
	for (size_t i = 0, iend = grid1->mvPTs.size(); i < iend; i++) {
		if (!status[i])
			continue;
		vPrevPTs.push_back(grid1->mvPTs[i]);
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

	//for (auto iter = pFrame1->mmpFrameGrids.begin(), iend = pFrame1->mmpFrameGrids.end(); iter != iend; iter++) {
	//	auto rectPt = iter->first;
	//	auto pGrid = iter->second;
	//	auto bGrid = pFrame1->mmbFrameGrids[rectPt];
	//	if (!bGrid)
	//		continue;
	//	auto rect = pGrid->rect;
	//	
	//	/*std::vector<uchar> status;
	//	std::vector<float> err;*/
	//	
	//	std::vector<cv::Mat> pyr1, pyr2;
	//	/*cv::buildOpticalFlowPyramid(img1(rect), pyr1, cv::Size(searchSize, searchSize), maxLvl);
	//	maxLvl = cv::buildOpticalFlowPyramid(img2(rect), pyr2, cv::Size(searchSize, searchSize), maxLvl);*/
	//	
	//	//if (pGrid->vecPTs.size()<3){
	//	//	continue;
	//	//}
	//	//////여기는 테스트 과정에서 일단 추가한 것. 원래는 없으면 생성하고 만들어야 함.
	//	//auto pGrid2 = pFrame2->mmpFrameGrids[rectPt];
	//	//if (!pGrid2)
	//	//	continue;
	//	//cv::calcOpticalFlowPyrLK(img1(rect), img2(rect), pGrid->vecPTs, pts2, status, err, cv::Size(searchSize, searchSize), maxLvl);
	//	//cv::calcOpticalFlowPyrLK(img1, img2, pGrid->vecPTs, pts2, status, err, cv::Size(searchSize, searchSize), maxLvl);
	//	for (size_t j = 0, jend = pGrid->vecPTs.size(); j < jend; j++) {
	//		pts1.push_back(pGrid->vecPTs[j]);
	//		//if (!status[j])
	//		//	continue;
	//		//auto pt1 = pGrid->vecPTs[j];// +pGrid->basePt;
	//		//auto pt2 = pts2[j];// +pGrid->basePt;
	//		//
	//		//pGrid2->vecPTs.push_back(pts2[j]);

	//		//cv::circle(img1, pt1, 1, cv::Scalar(255, 0, 255), -1);
	//		//cv::circle(img2, pt2, 1, cv::Scalar(255, 0, 255), -1);
	//		//cv::line(img2, pt1, pt2, cv::Scalar(255, 255, 0), 1);
	//	}

	//	//auto pt1 = pGrid->pt;
	//	//pt1.x -= rect.x;
	//	//pt1.y -= rect.y;
	//	//pts1.push_back(pt1);
	//	//cv::calcOpticalFlowPyrLK(img1(rect), img2(rect), pts1, pts2, status, err, cv::Size(searchSize, searchSize), maxLvl);
	//	//if (!status[0]) {
	//	//	continue;
	//	//}
	//	////////포인트 위치 변환
	//	//cv::Point2f matchPt1, matchPt2;
	//	//matchPt1 = pt1;
	//	//matchPt2 = pts2[0];
	//	//matchPt1.x += rect.x;
	//	//matchPt1.y += rect.y;
	//	//matchPt2.x += rect.x;
	//	//matchPt2.y += rect.y;

	//	//cv::rectangle(overlap, matchPt2 - mpSystem->mRectPt, matchPt2 + mpSystem->mRectPt, cv::Scalar(255, 0, 0), -1);
	//	//cv::circle(img1, matchPt1, 2, cv::Scalar(255, 0, 255), -1);
	//	//cv::circle(img2, matchPt2, 2, cv::Scalar(255, 0, 255), -1);
	//	//cv::line(img2, matchPt1, matchPt2, cv::Scalar(255, 255, 0), 1);

	//	//////포인트 위치 변환
	//	//if (!pFrame1->mpMatchInfo->CheckOpticalPointOverlap(overlap, matchPt2, mpSystem->mnRadius)) {
	//	//	continue;
	//	//}
	//	////정식 매칭에서 수행 예정
	//	//auto gridPt = pFrame1->GetGridBasePt(matchPt2, nGridSize);
	//	//if (pFrame2->mmbFrameGrids[gridPt]) {
	//	//	continue;
	//	//}
	//	//auto pCPi = pGrid->mpCP;
	//	//if (pCPi->mnTrackingFrameID == nCurrFrameID) {
	//	//	std::cout << "tracking::" << pCPi->mnCandidatePointID << ", " << nCurrFrameID << std::endl;
	//	//	continue;
	//	//}

	//	//////grid
	//	//pFrame2->mmbFrameGrids[gridPt] = true;
	//	//pFrame2->mmpFrameGrids[gridPt] = new FrameGrid(gridPt, nGridSize);
	//	//pFrame2->mmpFrameGrids[gridPt]->mpCP = pCPi;
	//	//pFrame2->mmpFrameGrids[gridPt]->pt = matchPt2;
	//	//////grid
	//	//pCPi->mnTrackingFrameID = nCurrFrameID;
	//	///*vpCPs.push_back(pCPi);
	//	//vpPts.push_back(currPts[i]);*/
	//	////정식 매칭에서 수행 예정
	//	
	//	
	//}
	//cv::calcOpticalFlowPyrLK(img1, img2, pts1, pts2, status, err, cv::Size(searchSize, searchSize), maxLvl);
	//for (size_t j = 0, jend = pts1.size(); j < jend; j++) {
	//	if (!status[j])
	//		continue;
	//	auto pt1 = pts1[j];// +pGrid->basePt;
	//	auto pt2 = pts2[j];// +pGrid->basePt;
	//	
	//	cv::circle(img1, pt1, 1, cv::Scalar(255, 0, 255), -1);
	//	cv::circle(img2, pt2, 1, cv::Scalar(255, 0, 255), -1);
	//	cv::line(img2, pt1, pt2, cv::Scalar(255, 255, 0), 1);
	//}
	//cv::Mat debugMatch = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	//img1.copyTo(debugMatch(mergeRect1));
	//img2.copyTo(debugMatch(mergeRect2));

	//cv::moveWindow("Output::MatchTest", mpSystem->mnDisplayX, mpSystem->mnDisplayY);
	//cv::imshow("Output::MatchTest", debugMatch);
	//cv::waitKey(1);
}

////200410 Optical flow
///////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////
////201114 Epipolar constraints
cv::Mat UVR_SLAM::Matcher::ComputeLineEquation(cv::Point2f pt1, cv::Point2f pt2) {
	float a = pt2.x - pt1.x;
	float b = pt2.y - pt1.y;

	//bx - ay - bx1 + ay1 = 0;
	float x = b;
	float y = -a;
	float z = -b*pt1.x + a*pt1.y;
	if (b == 0.0)
	{
		x = 1.0;
		y = 0.0;
		z = -pt1.x;
	}
	return (cv::Mat_<float>(3, 1) << x, y, z);
}

bool UVR_SLAM::Matcher::CheckLineDistance(cv::Mat line, cv::Point2f pt, float sigma) {
	float a = line.at<float>(0);
	float b = line.at<float>(1);
	float c = line.at<float>(2);
	const float den = a*a + b*b;
	if (den == 0) {
		return false;
	}
	const float num = a*pt.x + b*pt.y + c;
	const float dsqr = num*num / den;
	//res = abs(num) / sqrt(den);
	return dsqr<3.84*sigma;
}

void UVR_SLAM::Matcher::ComputeEpiLinePoint(cv::Point2f& sPt, cv::Point2f& ePt, cv::Mat ray, float minDepth, float maxDepth, cv::Mat Rrel, cv::Mat Trel, cv::Mat K) {
	
	cv::Mat Xcmin = Rrel*ray*minDepth + Trel;
	cv::Mat Xcmax = Rrel*ray*maxDepth + Trel;
	
	cv::Mat temp1 = K*Xcmin;
	cv::Mat temp2 = K*Xcmax;

	float d1 = temp1.at<float>(2);
	float d2 = temp2.at<float>(2);

	sPt = cv::Point2f(temp1.at<float>(0) / d1, temp1.at<float>(1) / d1);
	ePt = cv::Point2f(temp2.at<float>(0) / d2, temp2.at<float>(1) / d2);
}

cv::Point2f UVR_SLAM::Matcher::CalcLinePoint(float val, cv::Mat mLine, bool opt) {
	float x, y;
	if (opt) {
		x = 0.0;
		y = val;
		if (mLine.at<float>(0) != 0)
			x = (-mLine.at<float>(2) - mLine.at<float>(1)*y) / mLine.at<float>(0);
	}
	else {
		y = 0.0;
		x = val;
		if (mLine.at<float>(1) != 0)
			y = (-mLine.at<float>(2) - mLine.at<float>(0)*x) / mLine.at<float>(1);
	}

	return cv::Point2f(x, y);
}
////201114 Epipolar constraints
///////////////////////////////////////////////////////////