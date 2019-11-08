#include <Matcher.h>
#include <omp.h>
#include <random>
#include <Frame.h>
#include <FrameWindow.h>
#include <MatrixOperator.h>

UVR_SLAM::Matcher::Matcher(){}
UVR_SLAM::Matcher::Matcher(cv::Ptr < cv::DescriptorMatcher> _matcher, int w, int h)
	:mWidth(w), mHeight(h), TH_HIGH(100), mfNNratio(0.8), matcher(_matcher)
{}
UVR_SLAM::Matcher::~Matcher(){}

const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio

int UVR_SLAM::Matcher::FeatureMatchingWithSemanticFrames(UVR_SLAM::Frame* pSemantic, UVR_SLAM::Frame* pFrame) {
	std::vector<bool> vbTemp(pFrame->mvKeyPoints.size(), true);
	std::vector< std::vector<cv::DMatch> > matches;
	matcher->knnMatch(pSemantic->matDescriptor, pFrame->matDescriptor, matches, 2);

	int Nf1 = 0;
	int Nf2 = 0;
	int count = 0;

	std::vector<cv::DMatch> vecMatches;
	for (unsigned long i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
			
			if (vbTemp[matches[i][0].trainIdx]) {
				vbTemp[matches[i][0].trainIdx] = false;
			}
			else {
				Nf1++;
				continue;
			}
			vecMatches.push_back(matches[i][0]);
			auto otype =  pSemantic->GetObjectType(matches[i][0].queryIdx);
			pFrame->SetObjectType(otype, matches[i][0].trainIdx);
			count++;
		}
	}
	//std::cout << "Matching::" << count << ", " << Nf1 << ", " << Nf2 << std::endl;

	

	return count;
}

int UVR_SLAM::Matcher::FeatureMatchingForPoseTrackingByProjection(UVR_SLAM::FrameWindow* pWindow, UVR_SLAM::Frame* pF, float rr) {
	int nmatches = 0;
	int nf = 0;

	//pWindow->mvMatchingInfo.clear();
	//pWindow->SetVectorInlier(pWindow->LocalMapSize, false);

	for (int i = 0; i < pWindow->GetLocalMapSize(); i++) {
		UVR_SLAM::MapPoint* pMP = pWindow->GetMapPoint(i);
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		if (pWindow->GetBoolInlier(i)) {
			continue;
		}

		cv::Mat pCam;
		cv::Point2f p2D;
		bool bProjection = pMP->Projection(p2D, pCam, pWindow->GetRotation(), pWindow->GetTranslation(), pF->mK, mWidth, mHeight);
		if (!bProjection)
			continue;
		//if (!pMP->Projection(p2D, pCam, pF->GetRotation(), pF->GetTranslation(), pF->mK, mWidth, mHeight))
		//	continue;

		//중복 맵포인트 체크
		//if(!CheckOverlap(overlap1, p2D)){
		//    continue;
		//}
		
		//중복 맵포인트 체크
		std::vector<size_t> vIndices = pF->GetFeaturesInArea(p2D.x, p2D.y, rr, 0, pF->mnScaleLevels);
		if (vIndices.empty())
			continue;

		const cv::Mat MPdescriptor = pMP->GetDescriptor();

		int bestDist = 256;
		int bestLevel = -1;
		int bestDist2 = 256;
		int bestLevel2 = -1;
		int bestIdx = -1;

		// Get best and second matches with near keypoints
		for (std::vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
		{
			const size_t idx = *vit;
			
			if (pF->GetBoolInlier(idx))
				continue;

			const cv::Mat &d = pF->matDescriptor.row(idx);

			const int dist = UVR_SLAM::MatrixOperator::DescriptorDistance(MPdescriptor, d);

			if (dist<bestDist)
			{
				bestDist2 = bestDist;
				bestDist = dist;
				bestLevel2 = bestLevel;
				bestLevel = pF->mvKeyPoints[idx].octave;
				bestIdx = idx;
			}
			else if (dist<bestDist2)
			{
				bestLevel2 = pF->mvKeyPoints[idx].octave;
				bestDist2 = dist;
			}
		}//for vindices

		if (bestDist <= TH_HIGH)
		{
			if (bestLevel == bestLevel2 && bestDist>mfNNratio*bestDist2)
				continue;

			cv::Point2f pt = pF->mvKeyPoints[bestIdx].pt;
			//if(!CheckOverlap(overlap2, pt)){
			//    continue;
			//}

			pF->SetBoolInlier(true, bestIdx);
			pF->SetMapPoint(pMP, bestIdx);
			pWindow->SetBoolInlier(true, i);

			cv::DMatch tempMatch;
			tempMatch.queryIdx = i;
			tempMatch.trainIdx = bestIdx;
			pWindow->mvPairMatchingInfo.push_back(std::make_pair(tempMatch, true));
			nmatches++;

			auto otype = pMP->GetObjectType();
			pF->SetObjectType(otype, bestIdx);
		}
	}//pMP
	//std::cout << "Tracker::MatchingByProjection::" << nmatches << std::endl;
	return nmatches;
}

//포즈  찾을 때 초기 매칭
int UVR_SLAM::Matcher::FeatureMatchingForInitialPoseTracking(UVR_SLAM::FrameWindow* pWindow, UVR_SLAM::Frame* pF) {
	
	std::vector<bool> vbTemp(pWindow->GetLocalMapSize(), true);
	std::vector< std::vector<cv::DMatch> > matches;
	matcher->knnMatch(pF->matDescriptor, pWindow->descLocalMap, matches, 2);

	int Nf1 = 0;
	int Nf2 = 0;
	int count = 0;
	//pWindow->mvMatchingInfo.clear();
	//pWindow->SetVectorInlier(pWindow->LocalMapSize, false);
	for (unsigned long i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
			UVR_SLAM::MapPoint* pMP = pWindow->GetMapPoint(matches[i][0].trainIdx);
			if (!pMP)
				continue;
			
			if (vbTemp[matches[i][0].trainIdx]) {
				vbTemp[matches[i][0].trainIdx] = false;
			}
			else {
				Nf1++;
				continue;
			}
			cv::Mat pCam;
			cv::Point2f p2D;
			if (!pMP->Projection(p2D, pCam, pWindow->GetRotation(), pWindow->GetTranslation(), pF->mK, mWidth, mHeight)) {
				Nf2++;
				continue;
			}
			pF->SetMapPoint(pMP, matches[i][0].queryIdx);
			pWindow->mvPairMatchingInfo.push_back(std::make_pair(matches[i][0], true));
			pF->SetBoolInlier(true, matches[i][0].queryIdx);
			pWindow->SetBoolInlier(true, matches[i][0].trainIdx);
			count++;
		}
	}
	std::cout << "Matching::" << count << ", " << Nf1 << ", " << Nf2 << std::endl;
	return count;
}

int UVR_SLAM::Matcher::FeatureMatchingForInitialPoseTracking(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr, UVR_SLAM::FrameWindow* pWindow, std::vector<cv::DMatch>& vMatchInfos) {

	std::vector<bool> vbTemp(pCurr->mvKeyPoints.size(), true);
	std::vector< std::vector<cv::DMatch> > matches;
	matcher->knnMatch(pPrev->matDescriptor, pCurr->matDescriptor, matches, 2);

	int Nf1 = 0;
	int Nf2 = 0;
	int count = 0;
	
	for (unsigned long i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
			if (!pPrev->GetBoolInlier(matches[i][0].queryIdx)) {
				//vMatchInfos.push_back(matches[i][0]);
				continue;
			}
			UVR_SLAM::MapPoint* pMP = pPrev->GetMapPoint(matches[i][0].queryIdx);
			if (!pMP)
				continue;
			if (pMP->isDeleted()){
				pPrev->SetBoolInlier(false, matches[i][0].queryIdx);
				continue;
			}
			if (vbTemp[matches[i][0].trainIdx]) {
				vbTemp[matches[i][0].trainIdx] = false;
			}
			else {
				Nf1++;
				continue;
			}
			cv::Mat pCam;
			cv::Point2f p2D;
			if (!pMP->Projection(p2D, pCam, pWindow->GetRotation(), pWindow->GetTranslation(), pCurr->mK, mWidth, mHeight)) {
				Nf2++;
				continue;
			}
			pCurr->SetMapPoint(pMP, matches[i][0].trainIdx);
			pCurr->SetBoolInlier(true, matches[i][0].trainIdx);

			cv::DMatch tempMatch;
			tempMatch.queryIdx = pMP->GetFrameWindowIndex();
			tempMatch.trainIdx = matches[i][0].trainIdx;

			//pWindow->mvMatchingInfo.push_back(tempMatch);
			pWindow->mvPairMatchingInfo.push_back(std::make_pair(tempMatch, true));
			pWindow->SetBoolInlier(true, tempMatch.queryIdx);

			//labeling
			auto otype = pPrev->GetObjectType(matches[i][0].queryIdx);
			pCurr->SetObjectType(otype, matches[i][0].trainIdx);

			//매칭 성능 확인용
			vMatchInfos.push_back(matches[i][0]);

			count++;
		}
	}
	//std::cout << "Matching::" << count << ", " << Nf1 << ", " << Nf2 << std::endl;
	return count;
}


//초기화에서 F를 계산하고 매칭
int UVR_SLAM::Matcher::MatchingProcessForInitialization(UVR_SLAM::Frame* init, UVR_SLAM::Frame* curr, cv::Mat& F, std::vector<cv::DMatch>& resMatches) {

	//////Debuging
	//SetDir("/debug/keyframe");
	//std::stringstream dirss;
	//dirss << "/debug/keyframe/keyframe_" << 0;
	//std::string dirStr = SetDir(dirss.str());

	//cv::Rect mergeRect1 = cv::Rect(0, 0, curr->matOri.cols, curr->matOri.rows);
	//cv::Rect mergeRect2 = cv::Rect(curr->matOri.cols, 0, curr->matOri.cols, curr->matOri.rows);
	//cv::Mat featureImg = cv::Mat::zeros(curr->matOri.rows, curr->matOri.cols * 2, curr->matOri.type());

	//std::stringstream sfile;
	//sfile << "/keyframe_" << 0;

	//curr->matOri.copyTo(featureImg(mergeRect1));
	//init->matOri.copyTo(featureImg(mergeRect2));
	//cvtColor(featureImg, featureImg, CV_RGBA2BGR);
	//featureImg.convertTo(featureImg, CV_8UC3);
	//////Debuging

	//중복 제거용
	//cv::Mat overlap1 = cv::Mat::zeros(curr->matOri.size(), CV_8UC1);
	//cv::Mat overlap2 = cv::Mat::zeros(curr->matOri.size(), CV_8UC1);
	int nf1 = 0;
	int nf2 = 0;
	int Nfalse = 0;

	std::vector<bool> vbTemp(curr->mvKeyPoints.size(), true);

	std::vector< std::vector<cv::DMatch> > matches;
	std::vector<cv::DMatch> vMatches;

	matcher->knnMatch(init->matDescriptor, curr->matDescriptor, matches, 2);
	for (unsigned long i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
			
			if (vbTemp[matches[i][0].trainIdx]) {
				vbTemp[matches[i][0].trainIdx] = false;
			}
			else {
				nf2++;
				continue;
			}
			vMatches.push_back(matches[i][0]);
		}
	}
	
	std::vector<bool> mvInliers;
	float score;

	if ((int)vMatches.size() >= 8) {
		FindFundamental(init, curr, vMatches, mvInliers, score, F);
		F.convertTo(F, CV_32FC1);
	}
	if ((int)vMatches.size() < 8 || F.empty()) {
		F.release();
		F = cv::Mat::zeros(0, 0, CV_32FC1);
		return 0;
	}

	int count = 0;
	

	for (unsigned long i = 0; i < vMatches.size(); i++) {
		//if(inlier_mask.at<uchar>((int)i)) {
		if (mvInliers[i]) {

			cv::Point2f pt1 = init->mvKeyPoints[vMatches[i].queryIdx].pt;
			cv::Point2f pt2 = curr->mvKeyPoints[vMatches[i].trainIdx].pt;
			//init->mvnCPMatchingIdx.push_back(vMatches[i].queryIdx);
			//curr->mvnCPMatchingIdx.push_back(vMatches[i].trainIdx);
			resMatches.push_back(vMatches[i]);
			count++;
		}
	}
	return count; //190116 //inliers.size();
}


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
		int descDist = UVR_SLAM::MatrixOperator::DescriptorDistance(desc, descPrev);
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
		cv::Mat F21i;
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

		F21i = T2t*Fn*T1;

		currentScore = CheckFundamental(pInit, pCurr, F21i, vMatches, vbCurrentInliers, 1.0);

		if (currentScore>score)
		{
			F21 = F21i.clone();
			vbMatchesInliers = vbCurrentInliers;
			score = currentScore;
		}//if
	}//for
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
