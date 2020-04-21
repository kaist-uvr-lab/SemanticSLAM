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

int UVR_SLAM::Matcher::MatchingWithLabeling(std::vector<cv::KeyPoint> kps1, std::vector<cv::KeyPoint> kps2,cv::Mat desc1, cv::Mat desc2, std::vector<int> idxs1, std::vector<int> idxs2, std::vector<cv::DMatch>& amatches) {

	std::vector<bool> vbTemp(kps2.size(), true);
	std::vector< std::vector<cv::DMatch> > tempmatches;
	matcher->knnMatch(desc1, desc2, tempmatches, 2);
	
	int n = 0;
	float thresh_diff = 30.0; //25.0

	for (unsigned long i = 0; i < tempmatches.size(); i++) {
		if (tempmatches[i][0].distance < nn_match_ratio * tempmatches[i][1].distance) {
			int idx1 = idxs1[tempmatches[i][0].queryIdx];
			int idx2 = idxs2[tempmatches[i][0].trainIdx];
			cv::Point2f pt1 = kps1[idx1].pt;
			cv::Point2f pt2 = kps2[idx2].pt;
			float diffX = abs(pt1.x - pt2.x);
			if (diffX > thresh_diff) { //25
				continue;
			}
			if (vbTemp[idx2]) {
				vbTemp[idx2] = false;
			}
			else {
				continue;
			}
			cv::DMatch matchInfo;
			matchInfo.queryIdx = idx1;
			matchInfo.trainIdx = idx2;
			amatches.push_back(matchInfo);

			//pCurr->mPlaneDescriptor.push_back(pCurr->matDescriptor.row(idx2));
			//pCurr->mPlaneIdxs.push_back(idx2);
			//pCurr->mLabelStatus.at<uchar>(idx2) = (int)ObjectType::OBJECT_FLOOR;
			n++;
		}
	}
	//std::cout << "matching test : " << n << ", " << desc1.rows << ", " << desc2.rows << std::endl;
	return n;
}

int UVR_SLAM::Matcher::MatchingWithLabeling(UVR_SLAM::Frame* pKF, UVR_SLAM::Frame* pCurr) {

	cv::Mat img1 = pKF->GetOriginalImage();
	cv::Mat img2 = pCurr->GetOriginalImage();
	cv::Point2f ptBottom = cv::Point2f(0, img1.rows);

	cv::Rect mergeRect1 = cv::Rect(0, 0, img1.cols, img1.rows);
	cv::Rect mergeRect2 = cv::Rect(0, img1.rows, img1.cols, img1.rows);
	cv::Mat debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	img1.copyTo(debugging(mergeRect1));
	img2.copyTo(debugging(mergeRect2));

	std::vector<bool> vbTemp(pCurr->mvKeyPoints.size(), true);
	std::vector< std::vector<cv::DMatch> > matches, matches2;

	std::vector<cv::DMatch> vMatchInfos;
	matcher->knnMatch(pKF->mPlaneDescriptor, pCurr->matDescriptor, matches, 2);
	matcher->knnMatch(pKF->mWallDescriptor,  pCurr->matDescriptor, matches2, 2);
	std::vector<cv::DMatch> vMatchInfosPlane, vMatchInfosWall;

	int n = 0;

	float thresh_diff = 30.0; //25.0

	for (unsigned long i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
			int idx = pKF->mPlaneIdxs[matches[i][0].queryIdx];
			int idx2 = matches[i][0].trainIdx;
			cv::Point2f pt1 = pKF->mvKeyPoints[idx].pt;
			cv::Point2f pt2 = pCurr->mvKeyPoints[idx2].pt;
			float diffX = abs(pt1.x - pt2.x);
			if (diffX > thresh_diff) { //25
				cv::line(debugging, pKF->mvKeyPoints[idx].pt, pCurr->mvKeyPoints[idx2].pt + ptBottom, cv::Scalar(0, 0, 255));
				continue;
			}
			if (vbTemp[idx2]) {
				vbTemp[idx2] = false;
			}
			else {
				continue;
			}
			cv::DMatch matchInfo;
			matchInfo.queryIdx = idx;
			matchInfo.trainIdx = idx2;
			vMatchInfosPlane.push_back(matchInfo);

			pCurr->mPlaneDescriptor.push_back(pCurr->matDescriptor.row(idx2));
			pCurr->mPlaneIdxs.push_back(idx2);
			pCurr->mLabelStatus.at<uchar>(idx2) = (int)ObjectType::OBJECT_FLOOR;
			n++;
		}
	}
	for (unsigned long i = 0; i < matches2.size(); i++) {
		if (matches2[i][0].distance < nn_match_ratio * matches2[i][1].distance) {
			int idx = pKF->mWallIdxs[matches2[i][0].queryIdx];
			int idx2 = matches2[i][0].trainIdx;
			cv::Point2f pt1 = pKF->mvKeyPoints[idx].pt;
			cv::Point2f pt2 = pCurr->mvKeyPoints[idx2].pt;
			float diffX = abs(pt1.x - pt2.x);
			if (diffX > thresh_diff){
				cv::line(debugging, pKF->mvKeyPoints[idx].pt, pCurr->mvKeyPoints[idx2].pt + ptBottom, cv::Scalar(0, 0, 255));
				continue;
			}
			if (vbTemp[idx2]) {
				vbTemp[idx2] = false;
			}
			else {
				continue;
			}
			cv::DMatch matchInfo;
			matchInfo.queryIdx = idx;
			matchInfo.trainIdx = idx2;
			vMatchInfosWall.push_back(matchInfo);

			pCurr->mWallDescriptor.push_back(pCurr->matDescriptor.row(idx2));
			pCurr->mWallIdxs.push_back(idx2);
			//n++;
		}
	}

	
	auto mvpMPs = pKF->GetMapPoints();
	for (int i = 0; i < vMatchInfosPlane.size(); i++) {
		int idx1 = vMatchInfosPlane[i].queryIdx;
		int idx2 = vMatchInfosPlane[i].trainIdx;

		UVR_SLAM::MapPoint* pMP = mvpMPs[idx1];
		bool bMatch = false;
		if (pMP) {
			if (!pMP->isDeleted())
				bMatch = true;
		}
		if(bMatch)
			cv::line(debugging, pKF->mvKeyPoints[idx1].pt, pCurr->mvKeyPoints[idx2].pt + ptBottom, cv::Scalar(255, 0, 0));
		else
			cv::line(debugging, pKF->mvKeyPoints[idx1].pt, pCurr->mvKeyPoints[idx2].pt + ptBottom, cv::Scalar(0, 255, 255));
	}
	/*for (int i = 0; i < vMatchInfosWall.size(); i++) {
		int idx1 = vMatchInfosWall[i].queryIdx;
		int idx2 = vMatchInfosWall[i].trainIdx;
		cv::line(debugging, pKF->mvKeyPoints[idx1].pt, pCurr->mvKeyPoints[idx2].pt + ptBottom, cv::Scalar(0, 255, 0));
	}*/
	cv::imshow("Output::Labeling", debugging);
	return n;
}

int UVR_SLAM::Matcher::MatchingWithPrevFrame(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr, std::vector<cv::DMatch>& mvMatches) {

	std::vector<bool> vbTemp(pCurr->mvKeyPoints.size(), true);
	std::vector< std::vector<cv::DMatch> > matches;
	
	std::vector<cv::DMatch> vMatchInfos;
	matcher->knnMatch(pPrev->mTrackedDescriptor, pCurr->matDescriptor, matches, 2);

	int Nf1 = 0;
	int Nf2 = 0;
	int count = 0;

	int nCurrFrameID = pCurr->GetFrameID();
	auto mvpPrevMPs = pPrev->GetMapPoints();
	cv::Mat R, t;
	pPrev->GetPose(R, t);

	for (unsigned long i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {

			//if (!pPrev->mvbMPInliers[matches[i][0].queryIdx]) {
			//	//vMatchInfos.push_back(matches[i][0]);
			//	continue;
			//}
			int idx = pPrev->mvTrackedIdxs[matches[i][0].queryIdx];
			UVR_SLAM::MapPoint* pMP = mvpPrevMPs[idx];
			if (!pMP)
				continue;
			if (pMP->isDeleted()) {
				//pPrev->mvbMPInliers[idx]= false;
				//pPrev->mvpMPs[idx] = nullptr;
				continue;
			}
			if (!pPrev->isInFrustum(pMP, 0.5))
				continue;
			int idx2 = matches[i][0].trainIdx;
			cv::Point2f pt1 = pPrev->mvKeyPoints[idx].pt;
			cv::Point2f pt2 = pCurr->mvKeyPoints[idx2].pt;
			float diffX = abs(pt1.x - pt2.x);
			if (diffX > 25.0)
				continue;
			if (vbTemp[idx2]) {
				vbTemp[idx2] = false;
			}
			else {
				Nf1++;
				continue;
			}
			vMatchInfos.push_back(matches[i][0]);
			//local map의 inlier를 확인할 때 이용함.
			pMP->SetRecentTrackingFrameID(nCurrFrameID);

			pCurr->mvpMPs[idx2] = pMP;
			pCurr->mvbMPInliers[idx2] = true;

			cv::DMatch matchInfo;
			matchInfo.queryIdx = idx;
			matchInfo.trainIdx = idx2;
			mvMatches.push_back(matchInfo);

			//매칭 성능 확인용
			count++;
		}
	}

	/*cv::Mat img1 = pPrev->GetOriginalImage();
	cv::Mat img2 = pCurr->GetOriginalImage();
	cv::Point2f ptBottom = cv::Point2f(0, img1.rows);

	cv::Rect mergeRect1 = cv::Rect(0, 0, img1.cols, img1.rows);
	cv::Rect mergeRect2 = cv::Rect(0, img1.rows, img1.cols, img1.rows);
	cv::Mat debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	img1.copyTo(debugging(mergeRect1));
	img2.copyTo(debugging(mergeRect2));

	

	for (int i = 0; i < pPrev->mvpMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = pPrev->mvpMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted()) {
			continue;
		}
		cv::Mat pCam;
		cv::Point2f p2D;
		pMP->Projection(p2D, pCam, pPrev->GetRotation(), pPrev->GetTranslation(), pCurr->mK, mWidth, mHeight);
		cv::circle(debugging, p2D, 3, cv::Scalar(0, 255, 0), -1);
	}
	for (int i = 0; i < vMatchInfos.size(); i++) {
		int idx = pPrev->mvTrackedIdxs[vMatchInfos[i].queryIdx];
		if (pCurr->mvbMPInliers[vMatchInfos[i].trainIdx]) {
			cv::line(debugging, pPrev->mvKeyPoints[idx].pt, pCurr->mvKeyPoints[vMatchInfos[i].trainIdx].pt + ptBottom, cv::Scalar(255, 0, 255));

			UVR_SLAM::MapPoint* pMP = pPrev->mvpMPs[idx];
			if (!pMP)
				continue;
			if (pMP->isDeleted()) {
				continue;
			}
			cv::Mat pCam;
			cv::Point2f p2D;
			pMP->Projection(p2D, pCam, pPrev->GetRotation(), pPrev->GetTranslation(), pCurr->mK, mWidth, mHeight);
			cv::line(debugging, pPrev->mvKeyPoints[idx].pt, p2D, cv::Scalar(0, 255, 255), 2);
			cv::circle(debugging, pPrev->mvKeyPoints[idx].pt, 2, cv::Scalar(255, 0, 255), -1);
		}
		else {
			cv::circle(debugging, pPrev->mvKeyPoints[idx].pt, 2, cv::Scalar(255, 0, 0), -1);
			cv::circle(debugging, pCurr->mvKeyPoints[vMatchInfos[i].trainIdx].pt + ptBottom, 3, cv::Scalar(255, 0, 0), -1);
			cv::line(debugging, pPrev->mvKeyPoints[idx].pt, pCurr->mvKeyPoints[vMatchInfos[i].trainIdx].pt + ptBottom, cv::Scalar(255, 255, 0));
		}
	}*/

	/*std::vector<cv::DMatch> vMatchInfosObj;
	if (pPrev->mPlaneDescriptor.rows > 0) {
		std::vector< std::vector<cv::DMatch> > matches;
		
		matcher->knnMatch(pPrev->mPlaneDescriptor, pCurr->matDescriptor, matches, 2);

		for (unsigned long i = 0; i < matches.size(); i++) {
			if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
				int idx = pPrev->mPlaneIdxs[matches[i][0].queryIdx];
				int idx2 = matches[i][0].trainIdx;
				cv::Point2f pt1 = pPrev->mvKeyPoints[idx].pt;
				cv::Point2f pt2 = pCurr->mvKeyPoints[idx2].pt;
				float diffX = abs(pt1.x - pt2.x);
				if (diffX > 25.0)
					continue;
				
				cv::DMatch matchInfo;
				matchInfo.queryIdx = idx;
				matchInfo.trainIdx = idx2;
				vMatchInfosObj.push_back(matchInfo);

				pCurr->mPlaneDescriptor.push_back(pCurr->matDescriptor.row(idx2));
				pCurr->mPlaneIdxs.push_back(idx2);
			}
		}

		for (int i = 0; i < vMatchInfosObj.size(); i++) {
			int idx1 = vMatchInfosObj[i].queryIdx;
			int idx2 = vMatchInfosObj[i].trainIdx;
			cv::line(debugging, pPrev->mvKeyPoints[idx1].pt, pCurr->mvKeyPoints[idx2].pt + ptBottom, cv::Scalar(0, 255, 255));
		}

	}
	cv::imshow("Output::Matching", debugging);
	*/

	return count;
}

int UVR_SLAM::Matcher::MatchingWithLocalMap(Frame* pF, std::vector<MapPoint*> mvpLocalMPs, cv::Mat mLocalMapDesc, float rr) {
	int nmatches = 0;
	int nf = 0;

	int nCurrID = pF->GetFrameID();

	cv::Mat R = pF->GetRotation();
	cv::Mat t = pF->GetTranslation();

	for (int i = 0; i < mvpLocalMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = mvpLocalMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		if (pMP->GetRecentTrackingFrameID() == nCurrID)
			continue;
		if (!pF->isInFrustum(pMP, 0.5))
			continue;

		cv::Mat pCam;
		cv::Point2f p2D;
		bool bProjection = pMP->Projection(p2D, pCam, R, t, pF->mK, mWidth, mHeight);
		if (!pF->isInImage(p2D.x, p2D.y))
			continue;

		std::vector<size_t> vIndices = pF->GetFeaturesInArea(p2D.x, p2D.y, rr);
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

			if (pF->mvbMPInliers[idx])
				continue;

			const cv::Mat &d = pF->matDescriptor.row(idx);

			const int dist = DescriptorDistance(MPdescriptor, d);

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

			pF->mvbMPInliers[bestIdx] = true;
			pF->mvpMPs[bestIdx] = pMP;
			pMP->SetRecentTrackingFrameID(nCurrID);

			//auto otype = pMP->GetObjectType();
			//pF->SetObjectType(otype, bestIdx);
			nmatches++;

		}
	}//pMP

	 //시각화
	//cv::Mat vis = pF->GetOriginalImage();
	//auto mvpMPs = pF->GetMapPoints();
	//cv::Mat R2 = pF->GetRotation();
	//cv::Mat t2 = pF->GetTranslation();
	//for (int i = 0; i < mvpMPs.size(); i++) {
	//	UVR_SLAM::MapPoint* pMP = mvpMPs[i];
	//	if (!pMP) {
	//		continue;
	//	}
	//	if (pMP->isDeleted()) {
	//		continue;
	//	}
	//	cv::Point2f p2D;
	//	cv::Mat pCam;
	//	pMP->Projection(p2D, pCam, R2, t2, pF->mK, 640, 360);

	//	if (!pF->mvbMPInliers[i]) {
	//		//if (pMP->GetPlaneID() > 0) {
	//		//	//circle(vis, p2D, 4, cv::Scalar(255, 0, 255), 2);
	//		//}
	//	}
	//	else {
	//		cv::circle(vis, pF->mvKeyPoints[i].pt, 2, cv::Scalar(255, 0, 255), -1);
	//		UVR_SLAM::ObjectType type = pMP->GetObjectType();
	//		cv::line(vis, p2D, pF->mvKeyPoints[i].pt, cv::Scalar(255, 255, 0), 2);
	//		if (type != OBJECT_NONE)
	//			circle(vis, p2D, 3, UVR_SLAM::ObjectColors::mvObjectLabelColors[type], -1);
	//	}
	//}
	//imshow("Matching::LocalMap", vis);

	return nmatches;
}

int UVR_SLAM::Matcher::SearchForInitialization(Frame* F1, Frame* F2, std::vector<cv::DMatch>& resMatches, int windowSize)
{
	int nmatches = 0;
	//vnMatches12 = vector<int>(F1.mvKeysUn.size(), -1);

	std::vector<int> rotHist[30];
	for (int i = 0; i < HISTO_LENGTH; i++)
		rotHist[i].reserve(500);
	const float factor = 1.0f / HISTO_LENGTH;

	std::vector<int> vMatchedDistance(F2->mvKeyPoints.size(), INT_MAX);
	std::vector<int> vnMatches21(F2->mvKeyPoints.size(), -1);
	std::vector<int> vnMatches12(F1->mvKeyPoints.size(), -1);

	for (size_t i1 = 0, iend1 = F1->mvKeyPoints.size(); i1 < iend1; i1++)
	{
		cv::KeyPoint kp1 = F1->mvKeyPoints[i1];
		int level1 = kp1.octave;
		if (level1 > 0)
			continue;

		std::vector<size_t> vIndices2 = F2->GetFeaturesInArea(kp1.pt.x, kp1.pt.y, windowSize, level1, level1);

		if (vIndices2.empty())
			continue;

		cv::Mat d1 = F1->matDescriptor.row(i1);

		int bestDist = INT_MAX;
		int bestDist2 = INT_MAX;
		int bestIdx2 = -1;

		for (std::vector<size_t>::iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
		{
			size_t i2 = *vit;

			cv::Mat d2 = F2->matDescriptor.row(i2);

			int dist = DescriptorDistance(d1, d2);

			if (vMatchedDistance[i2] <= dist)
				continue;

			if (dist < bestDist)
			{
				bestDist2 = bestDist;
				bestDist = dist;
				bestIdx2 = i2;
			}
			else if (dist < bestDist2)
			{
				bestDist2 = dist;
			}
		}

		if (bestDist <= TH_LOW)
		{
			if (bestDist < (float)bestDist2*mfNNratio)
			{
				if (vnMatches21[bestIdx2] >= 0)
				{
					vnMatches12[vnMatches21[bestIdx2]] = -1;
					nmatches--;
				}
				vnMatches12[i1] = bestIdx2;
				vnMatches21[bestIdx2] = i1;
				vMatchedDistance[bestIdx2] = bestDist;
				nmatches++;

				if (mbCheckOrientation)
				{
					float rot = F1->mvKeyPoints[i1].angle - F2->mvKeyPoints[bestIdx2].angle;
					if (rot < 0.0)
						rot += 360.0f;
					int bin = round(rot*factor);
					if (bin == HISTO_LENGTH)
						bin = 0;
					assert(bin >= 0 && bin < HISTO_LENGTH);
					rotHist[bin].push_back(i1);
				}
			}
		}

	}

	if (mbCheckOrientation)
	{
		int ind1 = -1;
		int ind2 = -1;
		int ind3 = -1;

		ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

		for (int i = 0; i < HISTO_LENGTH; i++)
		{
			if (i == ind1 || i == ind2 || i == ind3)
				continue;
			for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
			{
				int idx1 = rotHist[i][j];
				if (vnMatches12[idx1] >= 0)
				{
					vnMatches12[idx1] = -1;
					nmatches--;
				}
			}
		}

	}

	//Update prev matched
	for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
	{
		if (vnMatches12[i1] >= 0) {
			cv::DMatch tempMatch;
			tempMatch.queryIdx = i1;
			tempMatch.trainIdx = vnMatches12[i1];
			resMatches.push_back(tempMatch);
		}
	}
	//		vbPrevMatched[i1] = F2.mvKeysUn[vnMatches12[i1]].pt;

	return nmatches;
}

//Fuse 시 호출
//Fuse의 경우 동일한 실제 공간의 포인트에 대해 두 키프레임이 같은 맵포인트를 가지도록 하기 위함.
int UVR_SLAM::Matcher::MatchingForFuse(const std::vector<UVR_SLAM::MapPoint*> &vpMapPoints, UVR_SLAM::Frame *pKF, float th) {
	
	std::vector<bool> vbTemp(pKF->mvKeyPoints.size(), true);
	cv::Mat Rcw = pKF->GetRotation();
	cv::Mat tcw = pKF->GetTranslation();

	const float &fx = pKF->fx;
	const float &fy = pKF->fy;
	const float &cx = pKF->cx;
	const float &cy = pKF->cy;
	//const float &bf = pKF->mbf;

	cv::Mat Ow = pKF->GetCameraCenter();
	int nFused = 0;
	const int nMPs = vpMapPoints.size();
	for (int i = 0; i < nMPs; i++)
	{
		UVR_SLAM::MapPoint* pMP = vpMapPoints[i];

		if (!pMP)
			continue;
		if (pMP->isDeleted() || pMP->isInFrame(pKF))
			continue;

		cv::Mat p3Dw = pMP->GetWorldPos();
		cv::Mat p3Dc = Rcw*p3Dw + tcw;

		// Depth must be positive
		if (p3Dc.at<float>(2)<0.0f)
			continue;

		const float invz = 1 / p3Dc.at<float>(2);
		const float x = p3Dc.at<float>(0)*invz;
		const float y = p3Dc.at<float>(1)*invz;

		const float u = fx*x + cx;
		const float v = fy*y + cy;

		// Point must be inside the image
		if (!pKF->isInImage(u, v))
			continue;

		//const float ur = u - invz;

		//const float maxDistance = pMP->GetMaxDistance();
		//const float minDistance = pMP->GetMinDistance();
		//cv::Mat PO = p3Dw - Ow;
		//const float dist3D = cv::norm(PO);

		//// Depth must be inside the scale pyramid of the image
		//if (dist3D<minDistance || dist3D>maxDistance)
		//	continue;

		//// Viewing angle must be less than 60 deg
		//cv::Mat Pn = pMP->GetNormal();

		//if (PO.dot(Pn)<0.5*dist3D)
		//	continue;

		//int nPredictedLevel = pMP->PredictScale(dist3D, pKF);
		//const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

		const std::vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, th,0,pKF->mnScaleLevels);

		if (vIndices.empty())
			continue;

		// Match to the most similar keypoint in the radius

		const cv::Mat dMP = pMP->GetDescriptor();

		int bestDist = 256;
		int bestIdx = -1;
		for (std::vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
		{
			const size_t idx = *vit;

			const cv::KeyPoint &kp = pKF->mvKeyPoints[idx];

			const int &kpLevel = kp.octave;

			const float &kpx = kp.pt.x;
			const float &kpy = kp.pt.y;
			const float ex = u - kpx;
			const float ey = v - kpy;
			const float e2 = ex*ex + ey*ey;

			if (e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
				continue;

			const cv::Mat &dKF = pKF->matDescriptor.row(idx);

			const int dist = DescriptorDistance(dMP, dKF);

			if (dist<bestDist)
			{
				bestDist = dist;
				bestIdx = idx;
			}
		}//for matching

		 // If there is already a MapPoint replace otherwise add new measurement
		if (bestDist <= TH_LOW)
		{

			if (vbTemp[bestIdx])
				vbTemp[bestIdx] = false;
			else
				continue;
			
			MapPoint* pMPinKF = pKF->mvpMPs[bestIdx];
			if (pMPinKF)
			{
				if (!pMPinKF->isDeleted())
				{
					if (pMPinKF->GetConnedtedFrames()>pMP->GetConnedtedFrames())
						pMP->Fuse(pMPinKF);
					else
						pMPinKF->Fuse(pMP);
				}
			}
			else
			{
				pMP->AddFrame(pKF, bestIdx);
				pKF->AddMP(pMP, bestIdx);
			}
			nFused++;
		}
	}
	return nFused;
}


//pkf가 pneighkf이고
//targetkf가 vpmps
int UVR_SLAM::Matcher::MatchingForFuse(UVR_SLAM::Frame *pTargetKF, UVR_SLAM::Frame *pNeighKF, float th) {

	auto vpMapPoints = pTargetKF->GetMapPoints();
	std::vector<bool> vbTemp(pNeighKF->mvKeyPoints.size(), true);
	cv::Mat Rcw = pNeighKF->GetRotation();
	cv::Mat tcw = pNeighKF->GetTranslation();

	const float &fx = pNeighKF->fx;
	const float &fy = pNeighKF->fy;
	const float &cx = pNeighKF->cx;
	const float &cy = pNeighKF->cy;
	//const float &bf = pKF->mbf;

	cv::Mat Ow = pNeighKF->GetCameraCenter();
	int nFused = 0;
	const int nMPs = vpMapPoints.size();

	auto mvpTargetOPs = pTargetKF->GetObjectVector();
	auto mvpNeighOPs = pNeighKF->GetObjectVector();

	for (int i = 0; i < nMPs; i++)
	{
		UVR_SLAM::MapPoint* pMP = vpMapPoints[i];

		if (!pMP)
			continue;
		if (pMP->isDeleted() || pMP->isInFrame(pNeighKF))
			continue;

		cv::Mat p3Dw = pMP->GetWorldPos();
		cv::Mat p3Dc = Rcw*p3Dw + tcw;

		// Depth must be positive
		if (p3Dc.at<float>(2)<0.0f)
			continue;

		const float invz = 1 / p3Dc.at<float>(2);
		const float x = p3Dc.at<float>(0)*invz;
		const float y = p3Dc.at<float>(1)*invz;

		const float u = fx*x + cx;
		const float v = fy*y + cy;

		// Point must be inside the image
		if (!pNeighKF->isInImage(u, v))
			continue;

		//const float ur = u - invz;

		//const float maxDistance = pMP->GetMaxDistance();
		//const float minDistance = pMP->GetMinDistance();
		//cv::Mat PO = p3Dw - Ow;
		//const float dist3D = cv::norm(PO);

		//// Depth must be inside the scale pyramid of the image
		//if (dist3D<minDistance || dist3D>maxDistance)
		//	continue;

		//// Viewing angle must be less than 60 deg
		//cv::Mat Pn = pMP->GetNormal();

		//if (PO.dot(Pn)<0.5*dist3D)
		//	continue;

		//int nPredictedLevel = pMP->PredictScale(dist3D, pKF);
		//const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

		const std::vector<size_t> vIndices = pNeighKF->GetFeaturesInArea(u, v, th, 0, pNeighKF->mnScaleLevels);

		if (vIndices.empty())
			continue;

		// Match to the most similar keypoint in the radius

		const cv::Mat dMP = pMP->GetDescriptor();

		int bestDist = 256;
		int bestIdx = -1;
		for (std::vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
		{
			const size_t idx = *vit;

			const cv::KeyPoint &kp = pNeighKF->mvKeyPoints[idx];

			if (mvpTargetOPs[i] != mvpNeighOPs[idx])
				continue;

			const int &kpLevel = kp.octave;

			const float &kpx = kp.pt.x;
			const float &kpy = kp.pt.y;
			const float ex = u - kpx;
			const float ey = v - kpy;
			const float e2 = ex*ex + ey*ey;

			if (e2*pNeighKF->mvInvLevelSigma2[kpLevel]>5.99)
				continue;

			const cv::Mat &dKF = pNeighKF->matDescriptor.row(idx);

			const int dist = DescriptorDistance(dMP, dKF);

			if (dist<bestDist)
			{
				bestDist = dist;
				bestIdx = idx;
			}
		}//for matching

		 // If there is already a MapPoint replace otherwise add new measurement
		if (bestDist <= TH_LOW)
		{

			if (vbTemp[bestIdx])
				vbTemp[bestIdx] = false;
			else
				continue;

			MapPoint* pMPinKF = pNeighKF->mvpMPs[bestIdx];
			if (pMPinKF)
			{
				if (!pMPinKF->isDeleted())
				{
					if (pMPinKF->GetConnedtedFrames()>pMP->GetConnedtedFrames())
						pMP->Fuse(pMPinKF);
					else
						pMPinKF->Fuse(pMP);
				}
			}
			else
			{
				pMP->AddFrame(pNeighKF, bestIdx);
				pNeighKF->AddMP(pMP, bestIdx);
			}
			nFused++;
		}
	}
	return nFused;
}

int UVR_SLAM::Matcher::MatchingForFuse(const std::vector<MapPoint*> &vpMapPoints, Frame* pTargetKF, Frame *pNeighborKF, bool bOpt, float th){
	std::vector<bool> vbTemp(pNeighborKF->mvKeyPoints.size(), true);
	cv::Mat Rcw = pNeighborKF->GetRotation();
	cv::Mat tcw = pNeighborKF->GetTranslation();

	const float &fx = pNeighborKF->fx;
	const float &fy = pNeighborKF->fy;
	const float &cx = pNeighborKF->cx;
	const float &cy = pNeighborKF->cy;
	//const float &bf = pKF->mbf;

	cv::Mat Ow = pNeighborKF->GetCameraCenter();
	int nFused = 0;
	const int nMPs = vpMapPoints.size();

	auto mvpNeigOPs = pNeighborKF->GetObjectVector();
	auto mvpTargetOPs = pTargetKF->GetObjectVector();

	for (int i = 0; i < nMPs; i++)
	{
		UVR_SLAM::MapPoint* pMP = vpMapPoints[i];

		if (!pMP)
			continue;
		if (pMP->isDeleted() || pMP->isInFrame(pNeighborKF))
			continue;

		cv::Mat p3Dw = pMP->GetWorldPos();
		cv::Mat p3Dc = Rcw*p3Dw + tcw;

		// Depth must be positive
		if (p3Dc.at<float>(2)<0.0f)
			continue;

		const float invz = 1 / p3Dc.at<float>(2);
		const float x = p3Dc.at<float>(0)*invz;
		const float y = p3Dc.at<float>(1)*invz;

		const float u = fx*x + cx;
		const float v = fy*y + cy;

		// Point must be inside the image
		if (!pNeighborKF->isInImage(u, v))
			continue;

		//const float ur = u - invz;

		//const float maxDistance = pMP->GetMaxDistance();
		//const float minDistance = pMP->GetMinDistance();
		//cv::Mat PO = p3Dw - Ow;
		//const float dist3D = cv::norm(PO);

		//// Depth must be inside the scale pyramid of the image
		//if (dist3D<minDistance || dist3D>maxDistance)
		//	continue;

		//// Viewing angle must be less than 60 deg
		//cv::Mat Pn = pMP->GetNormal();

		//if (PO.dot(Pn)<0.5*dist3D)
		//	continue;

		//int nPredictedLevel = pMP->PredictScale(dist3D, pKF);
		//const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

		const std::vector<size_t> vIndices = pNeighborKF->GetFeaturesInArea(u, v, th, 0, pNeighborKF->mnScaleLevels);

		if (vIndices.empty())
			continue;

		// Match to the most similar keypoint in the radius

		const cv::Mat dMP = pMP->GetDescriptor();

		int bestDist = 256;
		int bestIdx = -1;
		for (std::vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
		{
			const size_t idx = *vit;

			const cv::KeyPoint &kp = pNeighborKF->mvKeyPoints[idx];

			const int &kpLevel = kp.octave;

			const float &kpx = kp.pt.x;
			const float &kpy = kp.pt.y;
			const float ex = u - kpx;
			const float ey = v - kpy;
			const float e2 = ex*ex + ey*ey;

			if (e2*pNeighborKF->mvInvLevelSigma2[kpLevel]>5.99)
				continue;

			const cv::Mat &dKF = pNeighborKF->matDescriptor.row(idx);

			const int dist = DescriptorDistance(dMP, dKF);

			if (dist<bestDist)
			{
				bestDist = dist;
				bestIdx = idx;
			}
		}//for matching

		 // If there is already a MapPoint replace otherwise add new measurement
		if (bestDist <= TH_LOW)
		{
			//merge object
			//타입이 다르면 그 타입을 추가, 같으면 값 더하기
			if (bOpt) {
				//target<-neighbor
				auto type = pNeighborKF->GetObjectType(bestIdx);
				auto iter = pTargetKF->mvMapObjects[i].find(type);
				if ( iter== pTargetKF->mvMapObjects[i].end()) {
					pTargetKF->mvMapObjects[i].insert(std::make_pair(type, 1));
				}
				else {
					iter->second++;
				}
			}
			else {
				//neighbor<-target
				auto type = pTargetKF->GetObjectType(i);
				auto iter = pNeighborKF->mvMapObjects[bestIdx].find(type);
				if (iter == pNeighborKF->mvMapObjects[bestIdx].end()) {
					pNeighborKF->mvMapObjects[bestIdx].insert(std::make_pair(type, 1));
				}
				else {
					iter->second++;
				}
			}

			if (vbTemp[bestIdx])
				vbTemp[bestIdx] = false;
			else
				continue;

			MapPoint* pMPinKF = pNeighborKF->mvpMPs[bestIdx];
			if (pMPinKF)
			{
				if (!pMPinKF->isDeleted())
				{
					if (pMPinKF->GetConnedtedFrames()>pMP->GetConnedtedFrames())
						pMP->Fuse(pMPinKF);
					else
						pMPinKF->Fuse(pMP);
				}
			}
			else
			{
				pMP->AddFrame(pNeighborKF, bestIdx);
				pNeighborKF->AddMP(pMP, bestIdx);
			}
			nFused++;
		}
	}
	return nFused;
}

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

int UVR_SLAM::Matcher::FeatureMatchingForPoseTrackingByProjection(UVR_SLAM::Frame* pF, std::vector<MapPoint*> mvpLocalMPs, cv::Mat mLocalMapDesc, std::vector<bool>& mvbLocalMapInliers, std::vector<cv::DMatch>& mvMatches, float rr) {
	int nmatches = 0;
	int nf = 0;

	int nCurrID = pF->GetFrameID();

	cv::Mat R = pF->GetRotation();
	cv::Mat t = pF->GetTranslation();

	for (int i = 0; i < mvpLocalMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = mvpLocalMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		if (mvbLocalMapInliers[i])
			continue;
		if (pMP->GetRecentTrackingFrameID() == nCurrID)
			continue;

		cv::Mat pCam;
		cv::Point2f p2D;
		bool bProjection = pMP->Projection(p2D, pCam, R, t, pF->mK, mWidth, mHeight);
		if (!pF->isInImage(p2D.x, p2D.y))
			continue;
		
		//중복 맵포인트 체크
		std::vector<size_t> vIndices = pF->GetFeaturesInArea(p2D.x, p2D.y, rr);
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
			
			if (pF->mvbMPInliers[idx])
				continue;

			const cv::Mat &d = pF->matDescriptor.row(idx);

			const int dist = DescriptorDistance(MPdescriptor, d);

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
			
			pF->mvbMPInliers[bestIdx] = true;
			pF->mvpMPs[bestIdx] = pMP;
			pMP->SetRecentTrackingFrameID(nCurrID);

			cv::DMatch tempMatch;
			tempMatch.queryIdx = i;
			tempMatch.trainIdx = bestIdx;
			mvMatches.push_back(tempMatch);
			mvbLocalMapInliers[i] = true;
			
			//auto otype = pMP->GetObjectType();
			//pF->SetObjectType(otype, bestIdx);
			nmatches++;
			
		}
	}//pMP
	
	//시각화
	cv::Mat vis = pF->GetOriginalImage();
	auto mvpMPs = pF->GetMapPoints();
	cv::Mat R2 = pF->GetRotation();
	cv::Mat t2 = pF->GetTranslation();
	for (int i = 0; i < mvpMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = mvpMPs[i];
		if (!pMP) {
			continue;
		}
		if (pMP->isDeleted()) {
			continue;
		}
		cv::Point2f p2D;
		cv::Mat pCam;
		pMP->Projection(p2D, pCam, R2, t2, pF->mK, 640, 360);

		if (!pF->mvbMPInliers[i]) {
			//if (pMP->GetPlaneID() > 0) {
			//	//circle(vis, p2D, 4, cv::Scalar(255, 0, 255), 2);
			//}
		}
		else {
			cv::circle(vis, pF->mvKeyPoints[i].pt, 2, cv::Scalar(255, 0, 255), -1);
			UVR_SLAM::ObjectType type = pMP->GetObjectType();
			cv::line(vis, p2D, pF->mvKeyPoints[i].pt, cv::Scalar(255, 255, 0), 2);
			if (type != OBJECT_NONE)
				circle(vis, p2D, 3, UVR_SLAM::ObjectColors::mvObjectLabelColors[type], -1);
		}
	}
	cv::imshow("abasdfasdf", vis);

	return nmatches;
}

//포즈  찾을 때 초기 매칭
int UVR_SLAM::Matcher::FeatureMatchingForInitialPoseTracking(UVR_SLAM::FrameWindow* pWindow, UVR_SLAM::Frame* pF, std::vector<MapPoint*> mvpLocalMPs, cv::Mat mLocalMapDesc, std::vector<bool>& mvbLocalMapInliers) {
	
	std::vector<bool> vbTemp(pF->mvKeyPoints.size(), true);
	std::vector< std::vector<cv::DMatch> > matches;
	matcher->knnMatch(mLocalMapDesc, pF->matDescriptor, matches, 2);

	int Nf1 = 0;
	int Nf2 = 0;
	int count = 0;
	//pWindow->mvMatchingInfo.clear();
	//pWindow->SetVectorInlier(pWindow->LocalMapSize, false);

	cv::Mat R = pWindow->GetRotation();
	cv::Mat t = pWindow->GetTranslation();
	
	for (unsigned long i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
			UVR_SLAM::MapPoint* pMP = mvpLocalMPs[matches[i][0].queryIdx];
			//if(pWindow->GetB)
			if (pF->mvpMPs[matches[i][0].trainIdx])
				continue;
			if (mvbLocalMapInliers[matches[i][0].queryIdx])
				continue;
			if (!pMP)
				continue;
			if (pMP->isDeleted())
				continue;
			if (vbTemp[matches[i][0].trainIdx]) {
				vbTemp[matches[i][0].trainIdx] = false;
			}
			else {
				Nf1++;
				continue;
			}
			
			/*cv::Point2f p2D;
			pF->Projection(pMP->GetWorldPos(), R, t, pF->mK);
			std::cout << p2D << std::endl;
			if (!pF->isInImage(p2D.x, p2D.y))
			{
				Nf2++;
				continue;
			}*/
			
			/*cv::Mat pCam;
			cv::Point2f p2D;
			if (!pMP->Projection(p2D, pCam, R, t, pF->mK, mWidth, mHeight)) {
				Nf2++;
				continue;
			}*/

			pF->mvbMPInliers[matches[i][0].trainIdx] = true;
			pF->mvpMPs[matches[i][0].trainIdx] = pMP;
			//std::cout << pF->mvpMPs[matches[i][0].trainIdx]->mnMapPointID <<", "<< pF->mvpMPs[matches[i][0].trainIdx] << std::endl;
			//pWindow->mvPairMatchingInfo.push_back(std::make_pair(matches[i][0], true));
			//pWindow->mvMatchInfos.push_back(matches[i][0]);
			mvbLocalMapInliers[matches[i][0].queryIdx] = true;
			count++;
		}
	}

	cv::Mat vis = pF->GetOriginalImage();
	auto mvpMPs = pF->GetMapPoints();
	/*cv::Mat R = pF->GetRotation();
	cv::Mat t = pF->GetTranslation();*/
	for (int i = 0; i < mvpMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = mvpMPs[i];
		if (!pMP) {
			continue;
		}
		if (pMP->isDeleted()) {
			continue;
		}
		cv::Point2f p2D;
		cv::Mat pCam;
		pMP->Projection(p2D, pCam, R, t, pF->mK, 640,360);

		if (!pF->mvbMPInliers[i]) {
			//if (pMP->GetPlaneID() > 0) {
			//	//circle(vis, p2D, 4, cv::Scalar(255, 0, 255), 2);
			//}
		}
		else {
			cv::circle(vis, pF->mvKeyPoints[i].pt, 2, cv::Scalar(255, 0, 255), -1);
			UVR_SLAM::ObjectType type = pMP->GetObjectType();
			cv::line(vis, p2D, pF->mvKeyPoints[i].pt, cv::Scalar(255, 255, 0), 2);
			if (type != OBJECT_NONE)
				circle(vis, p2D, 3, UVR_SLAM::ObjectColors::mvObjectLabelColors[type], -1);
		}
	}
	imshow("abasdfasdf", vis);

	//std::cout << "Matching::" << count << ", " << Nf1 << ", " << Nf2 << std::endl;
	return count;
}

int UVR_SLAM::Matcher::FeatureMatchingForInitialPoseTracking(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr, std::vector<MapPoint*> mvpLocalMPs, cv::Mat mLocalMapDesc, std::vector<bool>& mvbLocalMapInliers, std::vector<cv::DMatch>& mvMatches, int nLocalMapID) {

	std::vector<bool> vbTemp(pCurr->mvKeyPoints.size(), true);
	std::vector< std::vector<cv::DMatch> > matches;
	std::vector<cv::DMatch> vMatchInfos;
	matcher->knnMatch(pPrev->mTrackedDescriptor, pCurr->matDescriptor, matches, 2);
	
	int Nf1 = 0;
	int Nf2 = 0;
	int count = 0;
	
	int nCurrFrameID = pCurr->GetFrameID();

	for (unsigned long i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {

			//if (!pPrev->mvbMPInliers[matches[i][0].queryIdx]) {
			//	//vMatchInfos.push_back(matches[i][0]);
			//	continue;
			//}
			int idx = pPrev->mvTrackedIdxs[matches[i][0].queryIdx];
			UVR_SLAM::MapPoint* pMP = pPrev->mvpMPs[idx];
			if (!pMP)
				continue;
			if (pMP->isDeleted()){
				//pPrev->mvbMPInliers[idx]= false;
				//pPrev->mvpMPs[idx] = nullptr;
				continue;
			}
			if (pMP->GetRecentLocalMapID() != nLocalMapID)
				continue;

			if (vbTemp[matches[i][0].trainIdx]) {
				vbTemp[matches[i][0].trainIdx] = false;
			}
			else {
				Nf1++;
				continue;
			}
			vMatchInfos.push_back(matches[i][0]);
			pMP->SetRecentTrackingFrameID(nCurrFrameID);

			pCurr->mvpMPs[matches[i][0].trainIdx] = pMP;
			pCurr->mvbMPInliers[matches[i][0].trainIdx] = true;

			
			//labeling
			auto otype = pPrev->GetObjectType(idx);
			pCurr->SetObjectType(otype, matches[i][0].trainIdx);

			//매칭 성능 확인용
			count++;
		}
	}
	
	cv::Mat img1 = pPrev->GetOriginalImage();
	cv::Mat img2 = pCurr->GetOriginalImage();
	cv::Point2f ptBottom = cv::Point2f(0, img1.rows);

	cv::Rect mergeRect1 = cv::Rect(0, 0, img1.cols, img1.rows);
	cv::Rect mergeRect2 = cv::Rect(0, img1.rows, img1.cols, img1.rows);
	cv::Mat debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	img1.copyTo(debugging(mergeRect1));
	img2.copyTo(debugging(mergeRect2));

	for (int i = 0; i < pPrev->mvpMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = pPrev->mvpMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted()) {
			continue;
		}
		cv::Mat pCam;
		cv::Point2f p2D;
		pMP->Projection(p2D, pCam, pPrev->GetRotation(), pPrev->GetTranslation(), pCurr->mK, mWidth, mHeight);
		cv::circle(debugging, p2D, 3, cv::Scalar(0, 255, 0), -1);
	}
	for (int i = 0; i < vMatchInfos.size(); i++) {
		int idx = pPrev->mvTrackedIdxs[vMatchInfos[i].queryIdx];
		if (pCurr->mvbMPInliers[vMatchInfos[i].trainIdx]){
			cv::line(debugging, pPrev->mvKeyPoints[idx].pt, pCurr->mvKeyPoints[vMatchInfos[i].trainIdx].pt + ptBottom, cv::Scalar(255, 0, 255));
			
			UVR_SLAM::MapPoint* pMP = pPrev->mvpMPs[idx];
			if (!pMP)
				continue;
			if (pMP->isDeleted()) {
				continue;
			}
			cv::Mat pCam;
			cv::Point2f p2D;
			pMP->Projection(p2D, pCam, pPrev->GetRotation(), pPrev->GetTranslation(), pCurr->mK, mWidth, mHeight);
			cv::line(debugging, pPrev->mvKeyPoints[idx].pt, p2D, cv::Scalar(0, 255, 255), 2);
			cv::circle(debugging, pPrev->mvKeyPoints[idx].pt, 2, cv::Scalar(255, 0, 255), -1);
		}
		else{
			cv::circle(debugging, pPrev->mvKeyPoints[idx].pt, 2, cv::Scalar(255, 0, 0), -1);
			cv::circle(debugging, pCurr->mvKeyPoints[vMatchInfos[i].trainIdx].pt + ptBottom, 3, cv::Scalar(255, 0, 0), -1);
			cv::line(debugging, pPrev->mvKeyPoints[idx].pt, pCurr->mvKeyPoints[vMatchInfos[i].trainIdx].pt + ptBottom, cv::Scalar(255, 255, 0));
		}
	}
	cv::imshow("Test::Matching::Frame", debugging);
	//waitKey(0);

	return count;
}

//prev : target, curr : kf
//Fuse시 얘는 전체에 대해서 해야 함
//그래야 비어있는 애에 대해서도 알 수 있음.
int UVR_SLAM::Matcher::KeyFrameFuseFeatureMatching(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr) {

	std::vector<bool> vbTemp(pCurr->mvKeyPoints.size(), true);
	std::vector< std::vector<cv::DMatch> > matches;
	matcher->knnMatch(pPrev->mTrackedDescriptor, pCurr->matDescriptor, matches, 2);
	int Nf1 = 0;
	int Nf2 = 0;
	int count = 0;

	std::vector<cv::DMatch> vMatchInfos;

	for (unsigned long i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {

			vMatchInfos.push_back(matches[i][0]);

			int idx = pPrev->mvTrackedIdxs[matches[i][0].queryIdx];
			UVR_SLAM::MapPoint* pMP = pPrev->mvpMPs[idx];
			if (!pMP)
				continue;
			if (pMP->isDeleted()) {
				continue;
			}
			if (vbTemp[matches[i][0].trainIdx]) {
				vbTemp[matches[i][0].trainIdx] = false;
			}
			else {
				Nf1++;
				continue;
			}

			//pCurr->mvpMPs[matches[i][0].trainIdx] = pMP;
			//pCurr->mvbMPInliers[matches[i][0].trainIdx] = true;

			count++;
		}
	}

	/*cv::Mat img1 = pPrev->GetOriginalImage();
	cv::Mat img2 = pCurr->GetOriginalImage();
	cv::Point2f ptBottom = cv::Point2f(0, img1.rows);

	cv::Rect mergeRect1 = cv::Rect(0, 0, img1.cols, img1.rows);
	cv::Rect mergeRect2 = cv::Rect(0, img1.rows, img1.cols, img1.rows);
	cv::Mat debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	img1.copyTo(debugging(mergeRect1));
	img2.copyTo(debugging(mergeRect2));

	for (int i = 0; i < pPrev->mvpMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = pPrev->mvpMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted()) {
			continue;
		}
		cv::Mat pCam;
		cv::Point2f p2D;
		pMP->Projection(p2D, pCam, pPrev->GetRotation(), pPrev->GetTranslation(), pCurr->mK, mWidth, mHeight);
		cv::circle(debugging, p2D, 3, cv::Scalar(0, 255, 0), -1);
	}
	for (int i = 0; i < vMatchInfos.size(); i++) {
		int idx = pPrev->mvTrackedIdxs[vMatchInfos[i].queryIdx];
		if (pCurr->mvbMPInliers[vMatchInfos[i].trainIdx]) {
			cv::line(debugging, pPrev->mvKeyPoints[idx].pt, pCurr->mvKeyPoints[vMatchInfos[i].trainIdx].pt + ptBottom, cv::Scalar(255, 0, 255));

			UVR_SLAM::MapPoint* pMP = pPrev->mvpMPs[idx];
			if (!pMP)
				continue;
			if (pMP->isDeleted()) {
				continue;
			}
			cv::Mat pCam;
			cv::Point2f p2D;
			pMP->Projection(p2D, pCam, pPrev->GetRotation(), pPrev->GetTranslation(), pCurr->mK, mWidth, mHeight);
			cv::line(debugging, pPrev->mvKeyPoints[idx].pt, p2D, cv::Scalar(0, 255, 255), 2);
			cv::circle(debugging, pPrev->mvKeyPoints[idx].pt, 2, cv::Scalar(255, 0, 255), -1);
		}
		else {
			cv::circle(debugging, pPrev->mvKeyPoints[idx].pt, 1, cv::Scalar(255, 0, 0), -1);
			cv::line(debugging, pPrev->mvKeyPoints[idx].pt, pCurr->mvKeyPoints[vMatchInfos[i].trainIdx].pt + ptBottom, cv::Scalar(255, 255, 0));
		}
	}
	cv::imshow("Test::Matching::KeyFrame::Fuse", debugging);
	waitKey(100);*/

	return count;
}

int UVR_SLAM::Matcher::KeyFrameFuseFeatureMatching2(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr) {

	std::vector<bool> vbTemp(pCurr->mvKeyPoints.size(), true);
	std::vector< std::vector<cv::DMatch> > matches1, matches2;
	matcher->knnMatch(pPrev->mTrackedDescriptor, pCurr->mTrackedDescriptor, matches1, 2);
	matcher->knnMatch(pPrev->mTrackedDescriptor, pCurr->mNotTrackedDescriptor, matches2, 2);
	int Nf1 = 0;
	int Nf2 = 0;
	int count = 0;

	std::vector<cv::DMatch> vMatchInfos;
	
	pCurr->UpdateMapInfo();

	//두 맵포인트 결합
	for (unsigned long i = 0; i < matches1.size(); i++) {
		if (matches1[i][0].distance < nn_match_ratio * matches1[i][1].distance) {

			vMatchInfos.push_back(matches1[i][0]);

			int idx1 = pPrev->mvTrackedIdxs[matches1[i][0].queryIdx];
			int idx2 = pCurr->mvTrackedIdxs[matches1[i][0].trainIdx];
			UVR_SLAM::MapPoint* pMP = pPrev->mvpMPs[idx1];
			if (!pMP)
				continue;
			if (pMP->isDeleted()) {
				continue;
			}
			if (vbTemp[idx2]) {
				vbTemp[idx2] = false;
			}
			else {
				Nf1++;
				continue;
			}

			//pCurr->mvpMPs[matches[i][0].trainIdx] = pMP;
			//pCurr->mvbMPInliers[matches[i][0].trainIdx] = true;

			count++;
		}
	}

	//비어있는 맵포인트 확인
	for (unsigned long i = 0; i < matches2.size(); i++) {
		if (matches2[i][0].distance < nn_match_ratio * matches2[i][1].distance) {

			vMatchInfos.push_back(matches2[i][0]);

			int idx1 = pPrev->mvTrackedIdxs[matches2[i][0].queryIdx];
			int idx2 = pCurr->mvNotTrackedIdxs[matches2[i][0].trainIdx];
			UVR_SLAM::MapPoint* pMP = pPrev->mvpMPs[idx1];
			if (!pMP)
				continue;
			if (pMP->isDeleted()) {
				continue;
			}
			if (vbTemp[idx2]) {
				vbTemp[idx2] = false;
			}
			else {
				Nf1++;
				continue;
			}

			//pCurr->mvpMPs[matches[i][0].trainIdx] = pMP;
			//pCurr->mvbMPInliers[matches[i][0].trainIdx] = true;

			count++;
		}
	}

	return count;
}

//prev1, curr2
//1의 정보를 2에서 찾는 것임.
int UVR_SLAM::Matcher::KeyFrameFeatureMatching(UVR_SLAM::Frame* pKF1, UVR_SLAM::Frame* pKF2, cv::Mat desc1, cv::Mat desc2, std::vector<int> idxs1, std::vector<int> idxs2, std::vector<cv::DMatch>& vMatches) {

	std::vector<bool> vbTemp(pKF2->mvKeyPoints.size(), true);
	std::vector< std::vector<cv::DMatch> > matches;
	
	matcher->knnMatch(desc1, desc2, matches, 2);
	int count = 0;

	for (unsigned long i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {

			int idx1 = idxs1[matches[i][0].queryIdx];
			int idx2 = idxs2[matches[i][0].trainIdx]; //matches[i][0].trainIdx;
			
			if (vbTemp[idx2]) {
				vbTemp[idx2] = false;
			}
			else {
				continue;
			}
			cv::DMatch tempMatch;
			tempMatch.queryIdx = idx1;
			tempMatch.trainIdx = idx2;

			vMatches.push_back(tempMatch);

			count++;
		}
	}

	return count;
}


int UVR_SLAM::Matcher::KeyFrameFeatureMatching(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr, std::vector<cv::DMatch>& vMatches) {

	std::vector<bool> vbTemp(pCurr->mvKeyPoints.size(), true);
	std::vector< std::vector<cv::DMatch> > matches;
	matcher->knnMatch(pPrev->mNotTrackedDescriptor, pCurr->mNotTrackedDescriptor, matches, 2);
	
	int Nf1 = 0;
	int Nf2 = 0;
	int count = 0;

	std::vector<cv::DMatch> vMatchInfos;

	for (unsigned long i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {

			int idx1 = pPrev->mvNotTrackedIdxs[matches[i][0].queryIdx];
			int idx2 = pCurr->mvNotTrackedIdxs[matches[i][0].trainIdx];
			UVR_SLAM::MapPoint* pMP1 = pPrev->mvpMPs[idx1];
			UVR_SLAM::MapPoint* pMP2 = pCurr->mvpMPs[idx2];
			if (pMP1 || pMP2)
				continue;

			cv::DMatch tempMatch;
			tempMatch.queryIdx = idx1;
			tempMatch.trainIdx = idx2;

			vMatches.push_back(tempMatch);

			if (vbTemp[idx2]) {
				vbTemp[idx2] = false;
			}
			else {
				Nf1++;
				continue;
			}

			count++;
		}
	}

	/*cv::Mat img1 = pPrev->GetOriginalImage();
	cv::Mat img2 = pCurr->GetOriginalImage();
	cv::Point2f ptBottom = cv::Point2f(0, img1.rows);

	cv::Rect mergeRect1 = cv::Rect(0, 0, img1.cols, img1.rows);
	cv::Rect mergeRect2 = cv::Rect(0, img1.rows, img1.cols, img1.rows);
	cv::Mat debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	img1.copyTo(debugging(mergeRect1));
	img2.copyTo(debugging(mergeRect2));
	
	for (int i = 0; i < vMatchInfos.size(); i++) {
		int idx1 = pPrev->mvNotTrackedIdxs[vMatchInfos[i].queryIdx];
		int idx2 = pCurr->mvNotTrackedIdxs[vMatchInfos[i].trainIdx];
		cv::circle(debugging, pPrev->mvKeyPoints[idx1].pt, 1, cv::Scalar(255, 0, 0), -1);
		cv::line(debugging, pPrev->mvKeyPoints[idx1].pt, pCurr->mvKeyPoints[idx2].pt + ptBottom, cv::Scalar(255, 255, 0));
		
	}
	cv::imshow("Test::Matching::KeyFrame::CreateMPs", debugging);
	waitKey(100);*/

	return count;
}



//초기화에서 F를 계산하고 매칭
int UVR_SLAM::Matcher::MatchingProcessForInitialization(UVR_SLAM::Frame* init, UVR_SLAM::Frame* curr, cv::Mat& F, std::vector<cv::DMatch>& resMatches) {

	////Debuging
	cv::Point2f ptBottom = cv::Point2f(0, curr->GetOriginalImage().rows);
	cv::Rect mergeRect1 = cv::Rect(0, 0, curr->GetOriginalImage().cols, curr->GetOriginalImage().rows);
	cv::Rect mergeRect2 = cv::Rect(0, curr->GetOriginalImage().rows, curr->GetOriginalImage().cols, curr->GetOriginalImage().rows);
	cv::Mat featureImg = cv::Mat::zeros(curr->GetOriginalImage().rows *2, curr->GetOriginalImage().cols, curr->GetOriginalImage().type());

	curr->GetOriginalImage().copyTo(featureImg(mergeRect1));
	init->GetOriginalImage().copyTo(featureImg(mergeRect2));
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

	int res = 0;
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
			
			cv::Point2f pt1 = curr->mvKeyPoints[matches[i][0].trainIdx].pt;
			cv::Point2f pt2 = init->mvKeyPoints[matches[i][0].queryIdx].pt;
			float diffX = abs(pt1.x - pt2.x);
			bool bMatch = false;
			if(diffX < 15){
				bMatch = true;
				res++;
				cv::line(featureImg, pt1,  pt2 + ptBottom, cv::Scalar(255, 0, 255));
			}else if(diffX >= 15 && diffX < 90){
				res++;
				cv::line(featureImg, pt1, pt2 + ptBottom, cv::Scalar(0, 255, 255));
				bMatch = true;
			}
			else{
				
				cv::line(featureImg, pt1, pt2 + ptBottom, cv::Scalar(255, 255, 0));
			}

			if(bMatch)
				resMatches.push_back(matches[i][0]);
		}
	}
	
	cv::imshow("init", featureImg);
	
	return res; //190116 //inliers.size();
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

int UVR_SLAM::Matcher::GMSMatching(Frame* pFrame1, Frame* pFrame2, std::vector<cv::DMatch>& vMatchInfo) {

	std::chrono::high_resolution_clock::time_point fuse_start2 = std::chrono::high_resolution_clock::now();
	

	std::vector< std::vector<cv::DMatch> > matches;
	std::vector<cv::DMatch> vMatchInfos;

	vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	vector<DMatch> matches_all, matches_gms;

	Ptr<ORB> orb = ORB::create(10000);
	orb->setFastThreshold(0);
	orb->detectAndCompute(pFrame1->GetOriginalImage(), Mat(), kp1, d1);
	orb->detectAndCompute(pFrame2->GetOriginalImage(), Mat(), kp2, d2);

	//matcher->knnMatch(pFrame1->matDescriptor, pFrame2->matDescriptor, matches, 2);
	matcher->knnMatch(d1, d2, matches, 2);

	std::chrono::high_resolution_clock::time_point fuse_temp = std::chrono::high_resolution_clock::now();
	cv::Size imgSize = pFrame1->GetOriginalImage().size();
	//gms_matcher gms(pFrame1->mvKeyPoints, imgSize, pFrame2->mvKeyPoints, imgSize, matches[0]);
	gms_matcher gms(kp1, imgSize, kp2, imgSize, matches[0]);
	std::vector<bool> vbInliers;
	int num_inliers = gms.GetInlierMask(vbInliers, false, false);

	for (size_t i = 0; i < vbInliers.size(); i++) {
		if (vbInliers[i])
			vMatchInfo.push_back(matches[0][i]);
	}

	std::chrono::high_resolution_clock::time_point fuse_end2 = std::chrono::high_resolution_clock::now();
	auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(fuse_temp - fuse_start2).count();
	auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(fuse_end2 - fuse_start2).count();
	auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(fuse_end2 - fuse_temp).count();
	double tttt1 = duration1 / 1000.0;
	double tttt2 = duration2 / 1000.0;
	double tttt3 = duration3 / 1000.0;
	std::cout << "Matcher::GMS::" << tttt2 <<", "<<tttt1<<", "<<tttt3<<":: "<<vMatchInfo.size()<<", "<< num_inliers <<", "<< matches.size()<< std::endl;


	cv::Mat img1 = pFrame1->GetOriginalImage();
	cv::Mat img2 = pFrame2->GetOriginalImage();
	cv::Point2f ptBottom = cv::Point2f(0, img1.rows);

	cv::Rect mergeRect1 = cv::Rect(0, 0, img1.cols, img1.rows);
	cv::Rect mergeRect2 = cv::Rect(0, img1.rows, img1.cols, img1.rows);
	
	cv::Mat debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	img1.copyTo(debugging(mergeRect1));
	img2.copyTo(debugging(mergeRect2));

	for (size_t i = 0; i < vMatchInfo.size(); i++) {
		cv::line(debugging, pFrame1->mvKeyPoints[vMatchInfo[i].queryIdx].pt, ptBottom+pFrame2->mvKeyPoints[vMatchInfo[i].trainIdx].pt, cv::Scalar(255, 0, 255));
	}

	imshow("matcher::gms", debugging); cv::waitKey(1);
		
	return vMatchInfo.size();
}



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

bool CheckBoundary(float x, float y, int rows, int cols){
	if (x < 0 || y < 0 || y >= rows || x >= cols) {
		return false;
	}
	return true;
}

float CalcSSD(cv::Mat src1, cv::Mat src2) {
	cv::Mat diff = abs(src1 - src2);
	float sum = 0.0;
	int num = diff.cols*diff.rows;
	sum = sqrt(diff.dot(diff));

	/*int num = diff.cols*diff.rows*diff.channels();
	for (int x = 0; x < diff.cols; x++) {
		for (int y = 0; y < diff.rows; y++) {
			cv::Vec3b temp = diff.at<Vec3b>(y, x);
			sum += temp.val[0];
			sum += temp.val[1];
			sum += temp.val[2];
		}
	}*/
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
bool Projection(cv::Point2f& pt, float& depth, cv::Mat R, cv::Mat T, cv::Mat K,cv::Mat X3D) {
	cv::Mat prev = R*X3D + T;
	prev = K*prev;
	depth = prev.at<float>(2);
	pt = cv::Point2f(prev.at<float>(0) / prev.at<float>(2), prev.at<float>(1) / prev.at<float>(2));
	if (depth < 0.0) {
		return false;
	}
	return true;
}

////이것은 바닥만 매칭.
////벽의 경우 이것과 비슷한 걸로
//f1이 과거 f2가 현재 프레임
int UVR_SLAM::Matcher::DenseMatchingWithEpiPolarGeometry(Frame* f1, Frame* f2, int nPatchSize, int nHalfWindowSize, cv::Mat& debugging) {

	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	///////debug
	cv::Mat img1 = f2->GetOriginalImage();
	cv::Mat img2 = f1->GetOriginalImage();
	cv::Point2f ptBottom = cv::Point2f(0, img1.rows);

	cv::Rect mergeRect1 = cv::Rect(0, 0, img1.cols, img1.rows);
	cv::Rect mergeRect2 = cv::Rect(0, img1.rows, img1.cols, img1.rows);
	//cv::Mat debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	img1.copyTo(debugging(mergeRect1));
	img2.copyTo(debugging(mergeRect2));
	///////debug

	cv::Mat gray1, gray2;
	cv::cvtColor(img1, gray1, CV_BGR2GRAY);
	cv::cvtColor(img2, gray2, CV_BGR2GRAY);

	cv::Mat Rcurr, Tcurr, Rprev, Tprev;
	f1->GetPose(Rprev, Tprev);
	f2->GetPose(Rcurr, Tcurr);
	cv::Mat mK = f1->mK.clone();
	//cv::Mat F12 = CalcFundamentalMatrix(Rcurr, Tcurr, Rprev, Tprev, mK);
	cv::Mat F12 = CalcFundamentalMatrix(Rprev, Tprev, Rcurr, Tcurr, mK);

	////역변환을 위한 정뵤
	cv::Mat invP, invT, invK;
	f1->mpPlaneInformation->Calculate();
	f1->mpPlaneInformation->GetInformation(invP, invT, invK);

	int inc = nPatchSize / 2;
	int mX = f1->mnMaxX - inc;
	int mY = f1->mnMaxY - inc;
	
	int nFullWindow = nHalfWindowSize * 2 + 1;

	//f1비교해서 f2에 추가하기
	//여기에서 코드 재수정

	/////////////라인 관련
	std::vector<Point2f> vPts1, vPts2;
	std::vector<Point2f> vDepthPts; //vpts1의 2배
	/////////////라인 관련

	for (int i = 0; i < f1->mvX3Ds.size(); i++) {
		cv::Mat X3D = f1->mvX3Ds[i];

		cv::Point2f ptPrev;
		float depthPrev;
		if (!Projection(ptPrev, depthPrev, Rprev, Tprev, f1->mK, X3D))
			continue;
		//cv::Mat prev = Rprev*X3D + Tprev;
		//prev = f1->mK*prev;
		//float depth = prev.at<float>(2);
		//cv::Point2f ptPrev(prev.at<float>(0) / prev.at<float>(2), prev.at<float>(1) / prev.at<float>(2));
		////depth check
		//if (prev.at<float>(2) < 0.0) {
		//	//cv::circle(debugging, tpt, 3, cv::Scalar(0, 0, 0), 1);
		//	continue;
		//}
		//boundary
		if (!CheckBoundary(ptPrev.x, ptPrev.y, img1.rows, img1.cols)) {
			continue;
		}
		
		cv::Point2f ptCurr;
		float depthCurr;
		if (!Projection(ptCurr, depthCurr, Rcurr, Tcurr, f1->mK, X3D))
			continue;
		//cv::Mat curr = Rcurr*X3D + Tcurr;
		//curr = f1->mK*curr;
		//cv::Point2f ptCurr(curr.at<float>(0) / curr.at<float>(2), curr.at<float>(1) / curr.at<float>(2));
		////depth check
		//if (curr.at<float>(2) < 0.0) {
		//	//cv::circle(debugging, tpt, 3, cv::Scalar(0, 0, 0), 1);
		//	continue;
		//}
		//boundary
		if (!CheckBoundary(ptCurr.x, ptCurr.y, img1.rows, img1.cols)) {
			continue;
		}
		
		
		//////label check
		//int label1 = f1->matLabeled.at<uchar>(ptPrev.y / 2, ptPrev.x / 2);
		//int label2 = f2->matLabeled.at<uchar>(ptCurr.y / 2, ptCurr.x / 2);
		//if (label1 != label2)
		//	continue;
		//////label check
		
		bool b1 = CheckBoundary(ptPrev.x - nHalfWindowSize, ptPrev.y - nHalfWindowSize, img1.rows, img1.cols);
		bool b2 = CheckBoundary(ptPrev.x + nHalfWindowSize, ptPrev.y + nHalfWindowSize, img1.rows, img1.cols);
		bool b3 = CheckBoundary(ptCurr.x - nHalfWindowSize, ptCurr.y - nHalfWindowSize, img1.rows, img1.cols);
		bool b4 = CheckBoundary(ptCurr.x + nHalfWindowSize, ptCurr.y + nHalfWindowSize, img1.rows, img1.cols);
		cv::Rect rect1 = cv::Rect(ptPrev.x - nHalfWindowSize, ptPrev.y - nHalfWindowSize, nFullWindow, nFullWindow);
		cv::Rect rect2 = cv::Rect(ptCurr.x - nHalfWindowSize, ptCurr.y - nHalfWindowSize, nFullWindow, nFullWindow);
		float val = FLT_MAX;
		if (b1 && b2 && b3 && b4) {
			val = CalcSSD(f1->GetFrame()(rect1), f2->GetFrame()(rect2));
			if (val < 10.0) {
				cv::circle(debugging, ptCurr, 3, cv::Scalar(255, 0, 0), 1);
				cv::circle(debugging, ptPrev + ptBottom, 3, cv::Scalar(255, 0, 0), 1);
				//cv::line(debugging, ptPrev + ptBottom, ptCurr, cv::Scalar(255, 0, 255), 1);
				//f2->mvX3Ds.push_back(X3D);
			}
			else {
				cv::circle(debugging, ptCurr, 3, cv::Scalar(0, 0, 255), 1);
				cv::circle(debugging, ptPrev + ptBottom, 3, cv::Scalar(0, 0, 255), 1);
			}
			vPts1.push_back(ptPrev);
			vPts2.push_back(ptCurr);
			////depth 관련 복원
			//float depth1 = depthPrev - 5;
			//float depth2 = depthPrev + 5;
			//cv::Point2f pt1, pt2;
			//float td1, td2;
			//cv::Mat temp1 = CreateWorldPoint(ptPrev, invT, invK, depth1);
			//cv::Mat temp2 = CreateWorldPoint(ptPrev, invT, invK, depth2);
			//Projection(pt1, td1, Rcurr, Tcurr, f1->mK, temp1);
			//Projection(pt2, td2, Rcurr, Tcurr, f1->mK, temp2);
			//vDepthPts.push_back(pt1); 
			//vDepthPts.push_back(pt2);
		}
	}

	//////////////에피폴라 라인 출력
	//f1의 포인트를 f2에 라인으로 출력
	if (vPts1.size() < 10)
		return 0;
	vector<cv::Point3f> lines[2];
	cv::computeCorrespondEpilines(vPts1, 2, F12, lines[0]);
	for (int i = 0; i < vPts1.size(); i++) {
		float m = 9999.0;
		if (lines[0][i].x != 0)
			m = abs(lines[0][i].x / lines[0][i].y);
		bool opt = false;
		if (m > 1.0)
			opt = true;
		
		////////에피 라인 
		cv::Point2f spt, ept;
		if(opt){
			spt = CalcLinePoint(0.0, lines[0][i], opt);
			ept = CalcLinePoint(img1.rows, lines[0][i], opt);
		}
		else{
			spt = CalcLinePoint(0.0, lines[0][i], opt);
			ept = CalcLinePoint(img1.cols, lines[0][i], opt);
		}
		//cv::line(debugging, spt, ept, cv::Scalar(0, 255, 0), 1);
		////////에피 라인 

		float val;
		if (opt)
			val = vPts2[i].y;
		else
			val = vPts2[i].x;

		cv::Rect rect = cv::Rect(vPts1[i].x - nHalfWindowSize, vPts1[i].y - nHalfWindowSize, nFullWindow, nFullWindow);
		cv::Mat patch = f1->GetFrame()(rect);

		float minVal = 10.0;
		cv::Point2f minPt;
		bool bFind = false;
		for (float j = val - 5.0; j < val + 5.0; j += 1.0) {
			cv::Point2f tpt = CalcLinePoint(j, lines[0][i], opt);
			//cv::circle(debugging, tpt, 1, cv::Scalar(0, 0, 255), 1);

			//////ssd
			bool b1 = CheckBoundary(tpt.x - nHalfWindowSize, tpt.y - nHalfWindowSize, img1.rows, img1.cols);
			bool b2 = CheckBoundary(tpt.x + nHalfWindowSize, tpt.y + nHalfWindowSize, img1.rows, img1.cols);
			cv::Rect rect2 = cv::Rect(tpt.x - nHalfWindowSize, tpt.y - nHalfWindowSize, nFullWindow, nFullWindow);
			float val = FLT_MAX;
			if (b1 && b2) {
				val = CalcSSD(f2->GetFrame()(rect2), patch);
				if (val < minVal) {
					if (!bFind)
						bFind = true;
					minVal = val;
					minPt = tpt;
				}
			}
			//////ssd
		}
		if (bFind)
			cv::circle(debugging, minPt, 3, cv::Scalar(255, 0, 255), 1);
		//cv::line(debugging, vDepthPts[2*i], vDepthPts[2 * i + 1], cv::Scalar(0, 0, 255), 1);
	}
	//////////////에피폴라 라인 출력

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;

	//fuse time text 
	std::stringstream ss;
	ss << "Dense Matching Time = " << tttt;
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	//fuse time text

	cv::imshow("Output::Matching", debugging); waitKey(1);
	return 0;
}

////트래킹을 위한 것
int UVR_SLAM::Matcher::DenseMatchingWithEpiPolarGeometry(Frame* f1, Frame* f2, std::vector<UVR_SLAM::MapPoint*>& vPlanarMaps, std::vector<std::pair<int, cv::Point2f>>& mathes, int nPatchSize, int nHalfWindowSize, cv::Mat& debugging) {

	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	///////debug
	cv::Mat img1 = f2->GetOriginalImage();
	cv::Mat img2 = f1->GetOriginalImage();
	cv::Point2f ptBottom = cv::Point2f(0, img1.rows);

	cv::Rect mergeRect1 = cv::Rect(0, 0, img1.cols, img1.rows);
	cv::Rect mergeRect2 = cv::Rect(0, img1.rows, img1.cols, img1.rows);
	//cv::Mat debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	img1.copyTo(debugging(mergeRect1));
	img2.copyTo(debugging(mergeRect2));
	///////debug

	int nFullWindow = nHalfWindowSize * 2 + 1;
	auto mvpOPs1 = f1->GetObjectVector();
	auto mvpOPs2 = f2->GetObjectVector();

	//Fundamental matrix 및 keyframe pose 계산
	cv::Mat Rcurr, Tcurr, Rprev, Tprev;
	f1->GetPose(Rprev, Tprev);
	f2->GetPose(Rcurr, Tcurr);
	cv::Mat mK = f1->mK.clone();

	//에센셜 매트릭스 계산. curr(1) -> prev(2)
	cv::Mat F12 = CalcFundamentalMatrix(Rcurr, Tcurr, Rprev, Tprev, mK);

	std::vector<Point2f> vPrevPts, vCurrPts;
	std::vector<cv::Mat> vX3Ds;
	std::vector<int> vIdxs;

	//std::vector<bool> vbs;
	//std::vector<cv::DMatch> vMatches;
	int n1 = 0;
	int n2 = 0;

	for (int i = 0; i < vPlanarMaps.size(); i++) {
		
		UVR_SLAM::MapPoint* pMPi = vPlanarMaps[i];
		if (!pMPi)
			continue;
		if (pMPi->GetMapPointType() != UVR_SLAM::PLANE_DENSE_MP)
			continue;
		cv::Mat X3D = pMPi->GetWorldPos();

		////매칭 수행하기
		//이미지 프로젝션
		cv::Point2f ptPrev;
		float depthPrev;
		if (!Projection(ptPrev, depthPrev, Rprev, Tprev, f1->mK, X3D))
			continue;
		cv::Point2f ptCurr;
		float depthCurr;
		if (!Projection(ptCurr, depthCurr, Rcurr, Tcurr, f1->mK, X3D))
			continue;
		vPrevPts.push_back(ptPrev);
		vCurrPts.push_back(ptCurr);
		//vX3Ds.push_back(X3D);
		vIdxs.push_back(i);
		cv::circle(debugging, ptPrev + ptBottom, 1, cv::Scalar(255, 0, 0), 1);
		cv::circle(debugging, ptCurr, 1, cv::Scalar(255, 0, 0), 1);
		//이미지 프로젝션

	}

	//////////////에피폴라 라인 출력
	if (vPrevPts.size() < 10) {
		std::cout << "epi::error::" << vPrevPts.size() << std::endl;
		return 0;
	}
	auto currGray = f2->GetFrame();
	auto prevGray =  f1->GetFrame();
	vector<cv::Point3f> lines[2];
	cv::computeCorrespondEpilines(vCurrPts, 2, F12, lines[0]);

	float wsize = 10.0;
	for (int i = 0; i < vCurrPts.size(); i++) {
		//이미지 바운더리 안에 존재하는지 확인
		bool bc1 = CheckBoundary(vCurrPts[i].x - nHalfWindowSize, vCurrPts[i].y - nHalfWindowSize, img1.rows, img1.cols);
		bool bc2 = CheckBoundary(vCurrPts[i].x + nHalfWindowSize, vCurrPts[i].y + nHalfWindowSize, img1.rows, img1.cols);
		if (!bc1 || !bc2)
			continue;
		//현재 포인트 매칭을 위한 패치 획득
		cv::Rect rect = cv::Rect(vCurrPts[i].x - nHalfWindowSize, vCurrPts[i].y - nHalfWindowSize, nFullWindow, nFullWindow);
		cv::Mat patch = currGray(rect);
		//현재 포인트 매칭을 위한 패치 획득

		//기울기 확인 후 x축, y축 설정
		float m = 9999.0;
		if (lines[0][i].x != 0)
			m = abs(lines[0][i].x / lines[0][i].y);
		bool opt = false;
		if (m > 1.0)
			opt = true;

		float val;
		if (opt)
			val = vPrevPts[i].y;
		else
			val = vPrevPts[i].x;
		//기울기 확인 후 x축, y축 설정

		//에피 라인 따라서 매칭
		float minVal = 5.0;
		cv::Point2f minPt;
		bool bFind = false;
		//for (float j = val - 5.0; j < val + 5.0; j += 0.5) {
		for (float j = val - wsize; j < val + wsize; j += 0.5) {
			cv::Point2f tpt = CalcLinePoint(j, lines[0][i], opt);
			//////ssd
			bool b1 = CheckBoundary(tpt.x - nHalfWindowSize, tpt.y - nHalfWindowSize, img1.rows, img1.cols);
			bool b2 = CheckBoundary(tpt.x + nHalfWindowSize, tpt.y + nHalfWindowSize, img1.rows, img1.cols);
			cv::Rect rect2 = cv::Rect(tpt.x - nHalfWindowSize, tpt.y - nHalfWindowSize, nFullWindow, nFullWindow);
			float val = FLT_MAX;
			if (b1 && b2) {
				val = CalcSSD(prevGray(rect2), patch);
				if (val < minVal) {
					if (!bFind)
						bFind = true;
					minVal = val;
					minPt = tpt;
				}
			}
			//////ssd
		}
		if (bFind) {
			cv::circle(debugging, minPt + ptBottom, 1, cv::Scalar(255, 0, 255), 1);
			cv::line(debugging, minPt + ptBottom, vPrevPts[i] + ptBottom, cv::Scalar(255, 0, 255));

			vPlanarMaps[vIdxs[i]]->SetRecentTrackingFrameID(f1->GetFrameID());
			auto temp = std::make_pair(vIdxs[i], minPt);
			mathes.push_back(temp);

			//f1->AddDenseMP(vPlanarMaps[vIdxs[i]], minPt);
		}

	}
	//////////////에피폴라 라인 출력

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;

	//fuse time text 
	std::stringstream ss;
	ss << "Dense Matching Time = " << tttt<<" "<<vCurrPts.size();
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	//fuse time text

	imshow("Output::Matching::버리자", debugging);
	cv::waitKey(1);

	return 0;

}

////키프레임 매칭을 위한 것.
int UVR_SLAM::Matcher::DenseMatchingWithEpiPolarGeometry(Frame* f1, Frame* f2, std::vector<cv::Mat>& vPlanarMaps, std::vector<std::pair<int, cv::Point2f>>& mathes,int nPatchSize, int nHalfWindowSize, cv::Mat& debugging) {

	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	///////debug
	cv::Mat img1 = f2->GetOriginalImage();
	cv::Mat img2 = f1->GetOriginalImage();
	cv::Point2f ptBottom = cv::Point2f(0, img1.rows);

	cv::Rect mergeRect1 = cv::Rect(0, 0, img1.cols, img1.rows);
	cv::Rect mergeRect2 = cv::Rect(0, img1.rows, img1.cols, img1.rows);
	//cv::Mat debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	img1.copyTo(debugging(mergeRect1));
	img2.copyTo(debugging(mergeRect2));
	///////debug

	int nFullWindow = nHalfWindowSize * 2 + 1;
	auto mvpOPs1 = f1->GetObjectVector();
	auto mvpOPs2 = f2->GetObjectVector();

	//Fundamental matrix 및 keyframe pose 계산
	cv::Mat Rcurr, Tcurr, Rprev, Tprev;
	f1->GetPose(Rprev, Tprev);
	f2->GetPose(Rcurr, Tcurr);
	cv::Mat mK = f1->mK.clone();

	//에센셜 매트릭스 계산. curr(1) -> prev(2)
	cv::Mat F12 = CalcFundamentalMatrix(Rcurr, Tcurr, Rprev, Tprev, mK);

	cv::Mat invP1, invT1, invK, invP2, invT2;
	if (!f1->mpPlaneInformation || !f2->mpPlaneInformation) {
		//왜 에러인지 알아야 함.
		std::cout << "error case!!!!!!!!" << std::endl;
		return 0;
	}
	f1->mpPlaneInformation->Calculate();
	f2->mpPlaneInformation->Calculate();

	f1->mpPlaneInformation->GetInformation(invP1, invT1, invK);
	f2->mpPlaneInformation->GetInformation(invP2, invT2, invK);
	//벽의 경우
	//cv::Mat invPwawll = invT.t()*pWall->GetParam();

	std::vector<Point2f> vPrevPts, vCurrPts;
	std::vector<cv::Mat> vX3Ds;
	std::vector<int> vIdxs;

	//std::vector<bool> vbs;

	//std::vector<cv::DMatch> vMatches;
	int n1 = 0;
	int n2 = 0;
	//매칭 확인하기
	/*for (int i = 0; i < f1->mvKeyPoints.size(); i++) {
	if (mvpOPs1[i] != ObjectType::OBJECT_FLOOR && mvpOPs1[i] != ObjectType::OBJECT_WALL && mvpOPs1[i] != ObjectType::OBJECT_CEILING)
	continue;
	cv::circle(debugging, f1->mvKeyPoints[i].pt + ptBottom, 2, cv::Scalar(255, 0, 0), -1);
	}*/


	for (int i = 0; i < f2->mvKeyPoints.size(); i++) {
		if (mvpOPs2[i] != ObjectType::OBJECT_FLOOR && mvpOPs2[i] != ObjectType::OBJECT_WALL && mvpOPs2[i] != ObjectType::OBJECT_CEILING)
			continue;

		/*if (f2->mvpMPs[i])
		cv::circle(debugging, f2->mvKeyPoints[i].pt, 3, cv::Scalar(0, 255, 255), 1);
		else
		cv::circle(debugging, f2->mvKeyPoints[i].pt, 3, cv::Scalar(0, 255, 0), 1);*/

		if (vPlanarMaps[i].rows == 0)
			continue;

		cv::Mat X3D = vPlanarMaps[i];

		////매칭 수행하기
		//이미지 프로젝션
		cv::Point2f ptPrev;
		float depthPrev;
		if (!Projection(ptPrev, depthPrev, Rprev, Tprev, f1->mK, X3D))
			continue;
		cv::Point2f ptCurr;
		float depthCurr;
		if (!Projection(ptCurr, depthCurr, Rcurr, Tcurr, f1->mK, X3D))
			continue;

		////label check
		int label1 = f1->matLabeled.at<uchar>(ptPrev.y / 2, ptPrev.x / 2);
		int label2 = f2->matLabeled.at<uchar>(ptCurr.y / 2, ptCurr.x / 2);
		if (label1 != label2)
			continue;
		////label check

		////일단 매칭 실패한 애들만 저장하도록 변경
		vPrevPts.push_back(ptPrev);
		vCurrPts.push_back(ptCurr);
		//vX3Ds.push_back(X3D);
		vIdxs.push_back(i);
		cv::circle(debugging, ptPrev + ptBottom, 1, cv::Scalar(255, 0, 0), 1);
		cv::circle(debugging, ptCurr, 1, cv::Scalar(255, 0, 0), 1);
		//이미지 프로젝션

	}
	std::chrono::high_resolution_clock::time_point tracking_1 = std::chrono::high_resolution_clock::now();
	//////////////에피폴라 라인 출력
	if (vPrevPts.size() < 10) {
		std::cout << "epi::error::" << vPrevPts.size() << std::endl;
		return 0;
	}

	vector<cv::Point3f> lines[2];
	cv::computeCorrespondEpilines(vCurrPts, 2, F12, lines[0]);

	std::chrono::high_resolution_clock::time_point tracking_2 = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < vCurrPts.size(); i++) {

		//현재 포인트 매칭을 위한 패치 획득
		cv::Rect rect = cv::Rect(vCurrPts[i].x - nHalfWindowSize, vCurrPts[i].y - nHalfWindowSize, nFullWindow, nFullWindow);
		cv::Mat patch = f2->GetFrame()(rect);
		//현재 포인트 매칭을 위한 패치 획득

		//////기울기 확인 후 x축, y축 설정
		float m = 9999.0;
		if (lines[0][i].x != 0)
			m = abs(lines[0][i].x / lines[0][i].y);
		bool opt = false;
		if (m > 1.0)
			opt = true;

		float val;
		if (opt)
			val = vPrevPts[i].y;
		else
			val = vPrevPts[i].x;
		//////기울기 확인 후 x축, y축 설정
		/*bool opt = true;
		float val = vPrevPts[i].y;*/
		////////기울기 이용x

		//에피 라인 따라서 매칭
		float minVal = 10.0;
		cv::Point2f minPt;
		bool bFind = false;
		for (float j = val - 0.5; j < val + 0.5; j += 1.0) {
			cv::Point2f tpt = CalcLinePoint(j, lines[0][i], opt);
			//////ssd
			bool b1 = CheckBoundary(tpt.x - nHalfWindowSize, tpt.y - nHalfWindowSize, img1.rows, img1.cols);
			bool b2 = CheckBoundary(tpt.x + nHalfWindowSize, tpt.y + nHalfWindowSize, img1.rows, img1.cols);
			cv::Rect rect2 = cv::Rect(tpt.x - nHalfWindowSize, tpt.y - nHalfWindowSize, nFullWindow, nFullWindow);
			float val = FLT_MAX;
			/*if (b1 && b2) {
				
			}*/
			val = CalcSSD(f1->GetFrame()(rect2), patch);
			if (val < minVal) {
				if (!bFind)
					bFind = true;
				minVal = val;
				minPt = tpt;
			}
			//////ssd
		}
		if (bFind) {
			cv::circle(debugging, minPt + ptBottom, 1, cv::Scalar(255, 0, 255), 1);
			cv::line(debugging, minPt + ptBottom, vPrevPts[i] + ptBottom, cv::Scalar(255, 0, 255));

			auto temp = std::make_pair(vIdxs[i], minPt);
			mathes.push_back(temp);
		}
		//cv::line(debugging, vDepthPts[2*i], vDepthPts[2 * i + 1], cv::Scalar(0, 0, 255), 1);
		//에피 라인 따라서 매칭

		//////////에피 라인 
		//cv::Point2f spt, ept;
		//if (opt) {
		//	spt = CalcLinePoint(0.0, lines[0][i], opt);
		//	ept = CalcLinePoint(img1.rows, lines[0][i], opt);
		//}
		//else {
		//	spt = CalcLinePoint(0.0, lines[0][i], opt);
		//	ept = CalcLinePoint(img1.cols, lines[0][i], opt);
		//}
		////cv::line(debugging, spt, ept, cv::Scalar(0, 255, 0), 1);
		//////////에피 라인 

	}
	//////////////에피폴라 라인 출력

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;

	auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_1 - tracking_start).count();
	double tttt1 = duration1 / 1000.0;

	auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_2 - tracking_1).count();
	double tttt2 = duration2 / 1000.0;

	auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_2).count();
	double tttt3 = duration3 / 1000.0;

	//fuse time text 
	std::stringstream ss;
	ss << "Matching = " << tttt<<", "<<tttt2<<", "<<tttt3;
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	//fuse time text

	imshow("matching test : ", debugging);
	cv::waitKey(1);

	return 0;

}

///Optical flow와 epipolar geometry의 혼합
//f1은 prev
//f2는 curr
int UVR_SLAM::Matcher::MatchingWithOptiNEpi(Frame* prev, Frame* curr, std::vector<cv::Mat>& vPlanarMaps, std::vector<bool>& vbInliers, std::vector<cv::DMatch>& vMatches, std::vector<std::pair<int, cv::Point2f>>& mathes, int nPatchSize, int nHalfWindowSize, cv::Mat& debugging) {
	//////////////////////////
	////Optical flow
	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	std::vector<cv::Mat> currPyr, prevPyr;
	std::vector<uchar> status;
	std::vector<float> err;
	std::vector<cv::Point2f> prevPts, currPts;
	cv::Mat prevImg = prev->GetOriginalImage();
	cv::Mat currImg = curr->GetOriginalImage();

	cv::Mat prevGray, currGray;
	cvtColor(prevImg, prevGray, CV_BGR2GRAY);
	cvtColor(currImg, currGray, CV_BGR2GRAY);
	///////debug
	cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
	cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
	debugging = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
	currImg.copyTo(debugging(mergeRect1));
	prevImg.copyTo(debugging(mergeRect2));
	///////debug

	//////타겟프레임에 키포인트 값 설정
	//키포인트의 인덱스값 저장 +1로
	cv::Mat matPrevKeypoints = cv::Mat::zeros(currImg.size(), CV_16UC1);
	for (int i = 0; i < prev->mvKeyPoints.size(); i++) {
		matPrevKeypoints.at<ushort>(prev->mvKeyPoints[i].pt) = i+1;
	}
	//////타겟프레임에 키포인트 값 설정
	////이전 프레임의 키포인트를 벡터화
	for (int i = 0; i < curr->mvKeyPoints.size(); i++) {
		currPts.push_back(curr->mvKeyPoints[i].pt);
	}
	////이전 프레임의 키포인트를 벡터화

	//Fundamental matrix 및 keyframe pose 계산
	cv::Mat Rcurr, Tcurr, Rprev, Tprev;
	prev->GetPose(Rprev, Tprev);
	curr->GetPose(Rcurr, Tcurr);
	cv::Mat mK = curr->mK.clone();

	//에센셜 매트릭스 계산. curr(1) -> prev(2)
	cv::Mat F12 = CalcFundamentalMatrix(Rcurr, Tcurr, Rprev, Tprev, mK);
	//에피폴라 라인 계산
	vector<cv::Point3f> lines[2];
	cv::computeCorrespondEpilines(currPts, 2, F12, lines[0]);
	
	int maxLvl = 3;
	int searchSize = 21;
	maxLvl = cv::buildOpticalFlowPyramid(currGray, currPyr, cv::Size(searchSize, searchSize), maxLvl);
	cv::buildOpticalFlowPyramid(prevGray, prevPyr, cv::Size(searchSize, searchSize), maxLvl);
	cv::calcOpticalFlowPyrLK(currPyr, prevPyr, currPts, prevPts, status, err, cv::Size(searchSize, searchSize), maxLvl);
	//바운더리 에러도 고려해야 함.
	int nTotal = 0;
	int nKeypoint = 0;
	int nBad = 0;
	int nEpi = 0;
	int n3D = 0;
	for (int i = 0; i < currPts.size(); i++) {
		if (status[i] == 0) {
			nBad++;
			continue;
		}

		//추가적인 에러처리
		//레이블드에서 255 150 100 벽 바닥 천장
		int currLabel = curr->matLabeled.at<uchar>(currPts[i].y/2, currPts[i].x/2);
		if (currLabel != 255 && currLabel != 150 && currLabel != 100){
			nBad++;
			continue;
		}
		int prevLabel = prev->matLabeled.at<uchar>(prevPts[i].y / 2, prevPts[i].x / 2);
		if (prevLabel != currLabel) {
			nBad++;
			continue;
		}

		

		////curr pts를 prev image에 projection
		////3d pts가 있는 경우만
		if (vPlanarMaps[i].rows >0){
			cv::Mat X3D = vPlanarMaps[i];
			////매칭 수행하기
			//이미지 프로젝션
			cv::Point2f ptPrev;
			float depthPrev;
			if (Projection(ptPrev, depthPrev, Rprev, Tprev, mK, X3D)) {
				cv::Point2f diffPt = ptPrev - prevPts[i];
				float dist = diffPt.dot(diffPt);
				if (dist > 25.0) {
					nBad++;
					continue;
				}
				n3D++;
				cv::circle(debugging, ptPrev + ptBottom, 1, cv::Scalar(0, 255, 0), -1);
				cv::line(debugging, prevPts[i] + ptBottom, ptPrev+ptBottom, cv::Scalar(255, 255, 0), 1);
			}
			/*cv::Point2f ptCurr;
			float depthCurr;
			if (!Projection(ptCurr, depthCurr, Rcurr, Tcurr, mK, X3D))
				continue;*/
			//continue;
		}
		else {
			//에피폴라 거리 체크 & 타입체크
			cv::Mat tempPrev = (cv::Mat_<float>(3, 1) << prevPts[i].x, prevPts[i].y, 1);
			cv::Mat tempCurr = (cv::Mat_<float>(3, 1) << currPts[i].x, currPts[i].y, 1);
			cv::Mat temp = tempPrev.t()*F12*tempCurr;
			float epiDist = temp.at<float>(0);
			float epiDist2 = lines[0][i].x*prevPts[i].x + lines[0][i].y*prevPts[i].y + lines[0][i].z;
			if (epiDist2 > 1.0) {
				nEpi++;
				//std::cout << "epi error : " <<epiDist2<<", "<< epiDist << std::endl;
				continue;
			}
		}
		


		//cv::line(debugging, prevPts[i]+ ptBottom, currPts[i], cv::Scalar(255, 255, 0), 1);
		if (matPrevKeypoints.at<ushort>(prevPts[i]) > 0) {
			//indirect
			cv::circle(debugging, prevPts[i] + ptBottom, 1, cv::Scalar(255, 0, 255), -1);
			cv::circle(debugging, currPts[i], 1, cv::Scalar(255, 0, 255), -1);
			nKeypoint++;
			nTotal++;
		}
		else {
			//direct
			nTotal++;
			cv::circle(debugging, currPts[i], 1, cv::Scalar(0, 255, 255), -1);
			cv::circle(debugging, prevPts[i] + ptBottom, 1, cv::Scalar(0, 255, 255), -1);
		}
	}

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;

	//fuse time text 
	std::stringstream ss;
	ss << "Optical flow = " << tttt << "::" << nKeypoint << ", " << nTotal << "||" << nBad<<"||"<< n3D;
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	imshow("optical test : ", debugging);
	/////////////////////////
}

//f2가 타겟, curr
//f1이 인접한 포인트, prev
//img1 = target, 위
//img2 = prev, 아래
//하늘색 선 : indiret
//분홍색 선 : direct
int UVR_SLAM::Matcher::MatchingWithEpiPolarGeometry(Frame* f1, Frame* f2, std::vector<cv::Mat>& vPlanarMaps, std::vector<bool>& vbInliers, std::vector<cv::DMatch>& vMatches, std::vector<std::pair<int, cv::Point2f>>& vPairs, int nPatchSize, int nHalfWindowSize, cv::Mat& debugging){

	vbInliers = std::vector<bool>(f2->mvKeyPoints.size(), false);
	vMatches = std::vector<cv::DMatch>(vbInliers.size());
	
	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	///////debug
	cv::Mat img1 = f2->GetOriginalImage();
	cv::Mat img2 = f1->GetOriginalImage();
	cv::Point2f ptBottom = cv::Point2f(0, img1.rows);

	cv::Rect mergeRect1 = cv::Rect(0, 0, img1.cols, img1.rows);
	cv::Rect mergeRect2 = cv::Rect(0, img1.rows, img1.cols, img1.rows);
	//cv::Mat debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	img1.copyTo(debugging(mergeRect1));
	img2.copyTo(debugging(mergeRect2));
	///////debug

	int nFullWindow = nHalfWindowSize * 2 + 1;
	auto mvpOPs1 = f1->GetObjectVector();
	auto mvpOPs2 = f2->GetObjectVector();
	
	//Fundamental matrix 및 keyframe pose 계산
	cv::Mat Rcurr, Tcurr, Rprev, Tprev;
	f1->GetPose(Rprev, Tprev);
	f2->GetPose(Rcurr, Tcurr);
	cv::Mat mK = f1->mK.clone();

	//에센셜 매트릭스 계산. curr(1) -> prev(2)
	cv::Mat F12 = CalcFundamentalMatrix(Rcurr, Tcurr, Rprev, Tprev, mK);
	

	std::vector<Point2f> vPrevPts, vCurrPts;
	std::vector<cv::Mat> vX3Ds;
	std::vector<int> vIdxs;
	//std::vector<bool> vbs;
	
	auto currGray = f2->GetFrame();
	auto prevGray = f1->GetFrame();

	//std::vector<cv::DMatch> vMatches;
	int n1 = 0;
	int n2 = 0;
	
	for (int i = 0; i < f2->mvKeyPoints.size(); i++) {
		if (mvpOPs2[i] != ObjectType::OBJECT_FLOOR && mvpOPs2[i] != ObjectType::OBJECT_WALL && mvpOPs2[i] != ObjectType::OBJECT_CEILING)
			continue;
		
		/*if (f2->mvpMPs[i])
			cv::circle(debugging, f2->mvKeyPoints[i].pt, 3, cv::Scalar(0, 255, 255), 1);
		else
			cv::circle(debugging, f2->mvKeyPoints[i].pt, 3, cv::Scalar(0, 255, 0), 1);*/
		
		if (vPlanarMaps[i].rows == 0)
			continue;
		
		cv::Mat X3D = vPlanarMaps[i];
				
		////매칭 수행하기
		//이미지 프로젝션
		cv::Point2f ptPrev;
		float depthPrev;
		if (!Projection(ptPrev, depthPrev, Rprev, Tprev, f1->mK, X3D))
			continue;
		cv::Point2f ptCurr;
		float depthCurr;
		if (!Projection(ptCurr, depthCurr, Rcurr, Tcurr, f1->mK, X3D))
			continue;

		////일단 매칭 실패한 애들만 저장하도록 변경
		/*vPrevPts.push_back(ptPrev);
		vCurrPts.push_back(ptCurr);
		*/
		cv::circle(debugging, ptPrev+ptBottom, 1, cv::Scalar(255, 0, 0), 1);
		cv::circle(debugging, ptCurr, 1, cv::Scalar(255, 0, 0), 1);
		//이미지 프로젝션

		////calc ssd
		//여기 바운더리 에러 수정하기
		/*bool b1 = CheckBoundary(f2->mvKeyPoints[i].pt.x - 1, f2->mvKeyPoints[i].pt.y - 1, img1.rows, img1.cols);
		bool b2 = CheckBoundary(f2->mvKeyPoints[i].pt.x + 1, f2->mvKeyPoints[i].pt.y + 1, img1.rows, img1.cols);
		bool b3 = CheckBoundary(tpt.x - 1, tpt.y - 1, img1.rows, img1.cols);
		bool b4 = CheckBoundary(tpt.x + 1, tpt.y + 1, img1.rows, img1.cols);
		cv::Rect rect1 = cv::Rect(f2->mvKeyPoints[i].pt.x-1, f2->mvKeyPoints[i].pt.y-1, 3, 3);
		cv::Rect rect2 = cv::Rect(tpt.x-1, tpt.y - 1, 3, 3);
		float val = FLT_MAX;
		if (b1 && b2 && b3 && b4) {
			val = CalcSSD(img1(rect1), img2(rect2));
			if (val < 30.0) {
				cv::circle(debugging, tpt + ptBottom, 3, cv::Scalar(255, 0, 0), 1);
			}
			else {
				cv::circle(debugging, tpt + ptBottom, 3, cv::Scalar(0, 255, 0), 1);
			}
		}
		else {
			cv::circle(debugging, tpt + ptBottom, 3, cv::Scalar(0, 255, 0), 1);
		}
		std::cout << "ssd::" << val << std::endl;*/
		////calc ssd

		////인접한 특징 찾기
		std::vector<size_t> vIndices = f1->GetFeaturesInArea(ptPrev.x, ptPrev.y, 3.0);
		if (vIndices.empty()){
			vPrevPts.push_back(ptPrev);
			vCurrPts.push_back(ptCurr);
			vIdxs.push_back(i);
			continue;
		}
		cv::Mat desc1 = f2->matDescriptor.row(i);
		//매칭 수행
		//디스크립터 또는 에피폴라 라인으로 매칭
		
		float nMinDist = FLT_MAX;
		int bestIdx = -1;
		n1++;
		for (std::vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
		{
			const size_t idx = *vit;
			if (mvpOPs1[idx] != ObjectType::OBJECT_FLOOR && mvpOPs1[idx] != ObjectType::OBJECT_WALL && mvpOPs1[idx] != ObjectType::OBJECT_CEILING)
				continue;
			//const cv::Mat &d = pF->matDescriptor.row(idx);
			cv::KeyPoint kp = f1->mvKeyPoints[idx];
			float sigma = f1->mvLevelSigma2[kp.octave];
			float epiDist;
			if (!CheckEpiConstraints(F12, ptPrev, kp.pt, sigma, epiDist) || (mvpOPs1[idx] != mvpOPs2[i]))
				continue;
			
			cv::Mat desc2 = f1->matDescriptor.row(idx);
			int descDist = DescriptorDistance(desc1, desc2);
			if (nMinDist > descDist) {
				nMinDist = descDist;
				bestIdx = idx;
			}
			/*if (epiDist < nMinDist) {
				nMinDist = epiDist;
				bestIdx = idx;
			}*/
		}
		if(bestIdx >=0){
		//if (nMinDist < TH_HIGH) {
			/*if (f1->mvpMPs[bestIdx])
				continue;*/
			cv::DMatch tempMatch;
			tempMatch.queryIdx = i;
			tempMatch.trainIdx = bestIdx;

			vbInliers[i] = true;
			vMatches[i] = tempMatch;
			//cv::line(debugging, tpt + ptBottom, f1->mvKeyPoints[bestIdx].pt + ptBottom, cv::Scalar(0, 255, 0));
			cv::circle(debugging, f1->mvKeyPoints[bestIdx].pt + ptBottom, 1, cv::Scalar(0, 255, 0), 1);
			cv::line(debugging, f1->mvKeyPoints[bestIdx].pt + ptBottom, ptPrev + ptBottom, cv::Scalar(255, 255, 0));
		}
		else {
			vPrevPts.push_back(ptPrev);
			vCurrPts.push_back(ptCurr);
			vIdxs.push_back(i);
		}
		
	}
	n2 = vMatches.size();

	//////////////에피폴라 라인 출력
	if (vPrevPts.size() < 10) {
		std::cout << "epi::error::" << vPrevPts.size() << std::endl;
		return 0;
	}

	vector<cv::Point3f> lines[2];
	cv::computeCorrespondEpilines(vCurrPts, 2, F12, lines[0]);

	for (int i = 0; i < vCurrPts.size(); i++) {
		
		//현재 포인트 매칭을 위한 패치 획득
		bool bc1 = CheckBoundary(vCurrPts[i].x - nHalfWindowSize, vCurrPts[i].y - nHalfWindowSize, img1.rows, img1.cols);
		bool bc2 = CheckBoundary(vCurrPts[i].x + nHalfWindowSize, vCurrPts[i].y + nHalfWindowSize, img1.rows, img1.cols);
		if (!bc1 || !bc2)
			continue;
		cv::Rect rect = cv::Rect(vCurrPts[i].x - nHalfWindowSize, vCurrPts[i].y - nHalfWindowSize, nFullWindow, nFullWindow);
		cv::Mat patch = currGray(rect);
		//현재 포인트 매칭을 위한 패치 획득

		//기울기 확인 후 x축, y축 설정
		float m = 9999.0;
		if (lines[0][i].x != 0)
			m = abs(lines[0][i].x / lines[0][i].y);
		bool opt = false;
		if (m > 1.0)
			opt = true;

		float val;
		if (opt)
			val = vPrevPts[i].y;
		else
			val = vPrevPts[i].x;
		//기울기 확인 후 x축, y축 설정
		
		//에피 라인 따라서 매칭
		float minVal = 10.0;
		cv::Point2f minPt;
		bool bFind = false;
		//for (float j = val - 5.0; j < val + 5.0; j += 0.5) {
		for (float j = val - 5.0; j < val + 5.0; j += 0.5) {
			cv::Point2f tpt = CalcLinePoint(j, lines[0][i], opt);
			//////ssd
			bool b1 = CheckBoundary(tpt.x - nHalfWindowSize, tpt.y - nHalfWindowSize, img1.rows, img1.cols);
			bool b2 = CheckBoundary(tpt.x + nHalfWindowSize, tpt.y + nHalfWindowSize, img1.rows, img1.cols);
			cv::Rect rect2 = cv::Rect(tpt.x - nHalfWindowSize, tpt.y - nHalfWindowSize, nFullWindow, nFullWindow);
			float val = FLT_MAX;
			if (b1 && b2) {
				val = CalcSSD(prevGray(rect2), patch);
				if (val < minVal) {
					if (!bFind)
						bFind = true;
					minVal = val;
					minPt = tpt;
				}
			}
			//////ssd
		}
		if (bFind){
			cv::circle(debugging, minPt + ptBottom, 1, cv::Scalar(255, 0, 255), 1);
			cv::line(debugging, minPt + ptBottom, vPrevPts[i] + ptBottom, cv::Scalar(255, 0, 255));
			auto temp = std::make_pair(vIdxs[i], minPt);
			vPairs.push_back(temp);
		}
		//cv::line(debugging, vDepthPts[2*i], vDepthPts[2 * i + 1], cv::Scalar(0, 0, 255), 1);
		//에피 라인 따라서 매칭

		//////////에피 라인 
		//cv::Point2f spt, ept;
		//if (opt) {
		//	spt = CalcLinePoint(0.0, lines[0][i], opt);
		//	ept = CalcLinePoint(img1.rows, lines[0][i], opt);
		//}
		//else {
		//	spt = CalcLinePoint(0.0, lines[0][i], opt);
		//	ept = CalcLinePoint(img1.cols, lines[0][i], opt);
		//}
		////cv::line(debugging, spt, ept, cv::Scalar(0, 255, 0), 1);
		//////////에피 라인 

	}
	//////////////에피폴라 라인 출력

	

	
	//std::cout << "matching test : " << n2 << " " << n1 << std::endl;
	
	///////debug
	/*for (int i = 0; i < vMatches.size(); i++) {
		int idx1 = vMatches[i].queryIdx;
		int idx2 = vMatches[i].trainIdx;
		if(f2->mvpMPs[idx1])
			cv::line(debugging, f2->mvKeyPoints[idx1].pt, f1->mvKeyPoints[idx2].pt + ptBottom, cv::Scalar(0, 255, 255));
		else
			cv::line(debugging, f2->mvKeyPoints[idx1].pt, f1->mvKeyPoints[idx2].pt + ptBottom, cv::Scalar(0, 255, 0));
	}*/
	
	//////debug

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;
	
	//fuse time text 
	std::stringstream ss;
	ss << "Dense Matching Time = " << tttt;
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	//fuse time text

	imshow("matching test : ", debugging);
	cv::waitKey(1);

	return 0;

}

int UVR_SLAM::Matcher::MatchingWithEpiPolarGeometry(Frame* pKF, Frame* pF, std::vector<cv::DMatch>& vMatches) {

	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();

	/////debug
	cv::Mat img1 = pKF->GetOriginalImage();
	cv::Mat img2 = pF->GetOriginalImage();
	cv::Point2f ptBottom = cv::Point2f(0, img1.rows);

	cv::Rect mergeRect1 = cv::Rect(0, 0, img1.cols, img1.rows);
	cv::Rect mergeRect2 = cv::Rect(0, img1.rows, img1.cols, img1.rows);
	cv::Mat debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	img1.copyTo(debugging(mergeRect1));
	img2.copyTo(debugging(mergeRect2));
	/////debug

	int nCurrID = pF->GetFrameID();

	auto mvpOPs = pKF->GetObjectVector();
	//auto mvpOPs2 = f2->GetObjectVector();

	//Fundamental matrix 및 keyframe pose 계산
	cv::Mat Rcurr, Tcurr, Rprev, Tprev;
	pKF->GetPose(Rprev, Tprev);
	pF->GetPose(Rcurr, Tcurr);
	cv::Mat mK = pKF->mK.clone();
	cv::Mat F12 = CalcFundamentalMatrix(Rprev, Tprev, Rcurr, Tcurr, mK);

	std::vector<Point2f> vPts1, vPts2;
	std::vector<cv::Mat> vX3Ds;
	//std::vector<bool> vbs;

	std::cout << "matching::start" << std::endl;
	//std::vector<cv::DMatch> vMatches;
	int n1 = 0;
	int n2 = 0;
	//매칭 확인하기
	for (int i = 0; i < pKF->mvKeyPoints.size(); i++) {
		UVR_SLAM::MapPoint* pMP = pKF->mvpMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		if (pMP->GetRecentTrackingFrameID() == nCurrID)
			continue;
		if (mvpOPs[i] != ObjectType::OBJECT_FLOOR && mvpOPs[i] != ObjectType::OBJECT_WALL)
			continue;
		
		cv::Mat X3D = pMP->GetWorldPos();
		
		////매칭 수행하기
		//이미지 프로젝션
		cv::Mat temp = Rcurr*X3D + Tcurr;
		temp = mK*temp;
		cv::Point2f tpt(temp.at<float>(0) / temp.at<float>(2), temp.at<float>(1) / temp.at<float>(2));
		//인접한 특징 찾기
		std::vector<size_t> vIndices = pF->GetFeaturesInArea(tpt.x, tpt.y, 3.0);

		float nMinDist = FLT_MAX;
		int bestIdx = -1;
		int count = 0;
		for (std::vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
		{
			const size_t idx = *vit;

			//const cv::Mat &d = pF->matDescriptor.row(idx);
			cv::KeyPoint kp = pF->mvKeyPoints[idx];
			float sigma = pF->mvLevelSigma2[kp.octave];
			float epiDist;
			if (!CheckEpiConstraints(F12, tpt, kp.pt, sigma, epiDist))
				continue;
			count++;
			
			if (epiDist < nMinDist) {
				nMinDist = epiDist;
				bestIdx = idx;
			}
		}
		if (count > 0)
			n1++;

		if (bestIdx >= 0) {
			if (pF->mvpMPs[bestIdx])
				continue;
			cv::DMatch tempMatch;
			tempMatch.queryIdx = i;
			tempMatch.trainIdx = bestIdx;
			vMatches.push_back(tempMatch);
			pF->mvbMPInliers[bestIdx] = true;
			pF->mvpMPs[bestIdx] = pMP;
			pMP->SetRecentTrackingFrameID(nCurrID);
		}
		cv::circle(debugging, pKF->mvKeyPoints[i].pt, 3, cv::Scalar(255, 0, 255), -1);
	}
	n2 = vMatches.size();

	if (vPts1.size() < 10) {
		std::cout << "epi::error::" << vPts1.size() << std::endl;
		return 0;
	}

	std::cout << "matching test : " << n2 << " " << n1 << std::endl;

	/////debug
	for (int i = 0; i < vMatches.size(); i++) {
		int idx1 = vMatches[i].queryIdx;
		int idx2 = vMatches[i].trainIdx;

		cv::line(debugging, pKF->mvKeyPoints[idx1].pt, pF->mvKeyPoints[idx2].pt + ptBottom, cv::Scalar(0, 0, 255));
	}
	imshow("matching test : ", debugging);
	cv::waitKey(1);
	////debug

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;
	std::cout << "epitime::" << tttt << std::endl;
	return 0;
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
	imshow("Init::OpticalFlow ", debugging);
	/////////////////////////

	return res;
}

int UVR_SLAM::Matcher::OpticalMatchingForInitialization(Frame* init, Frame* curr, std::vector<cv::Point2f>& vpPts2, std::vector<bool>& vbInliers, std::vector<int>& vnIDXs, cv::Mat& debugging) {

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
	/*
	cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
	cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
	debugging = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
	prevImg.copyTo(debugging(mergeRect1));
	currImg.copyTo(debugging(mergeRect2));*/
	///////debug

	///////////
	//matKPs, mvKPs
	//init -> curr로 매칭
	////////
	std::vector<cv::Point2f> prevPts, currPts;
	prevPts = init->mvMatchingPts;
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
		if (!curr->isInImage(currPts[i].x, currPts[i].y)){
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
			//cv::line(debugging, prevPts[i], currPts[i] + ptBottom, cv::Scalar(255, 0, 255));
		}
		else if (diffX >= 15 && diffX < 90) {
			//cv::line(debugging, prevPts[i], currPts[i] + ptBottom, cv::Scalar(0, 255, 255));
			bMatch = true;
		}
		else {
			//cv::line(debugging, prevPts[i], currPts[i] + ptBottom, cv::Scalar(255, 255, 0));
		}
		//cv::circle(debugging, prevPts[i], 1, cv::Scalar(255), -1);
		if (bMatch) {
			//vpPts1.push_back(prevPts[i]);
			vpPts2.push_back(currPts[i]);
			vbInliers.push_back(true);
			vnIDXs.push_back(init->mvMatchingIdxs[i]);
		}
		//매칭 결과
		////

	}
	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;

	////fuse time text 
	//std::stringstream ss;
	//ss << "Optical flow init= " << res << ", " << tttt;
	//cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	//cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	//imshow("Init::OpticalFlow ", debugging);
	///////////////////////////

	return vpPts2.size();
}

int UVR_SLAM::Matcher::OpticalMatchingForTracking(Frame* prev, Frame* curr, std::vector<UVR_SLAM::MapPoint*>& vpMPs, std::vector<cv::Point2f>& vpPts, std::vector<bool>& vbInliers, cv::Mat& overlap) {
	
	//////////////////////////
	////Optical flow
	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	std::vector<cv::Mat> currPyr, prevPyr;
	std::vector<uchar> status;
	std::vector<float> err;
	cv::Mat prevImg = prev->GetOriginalImage();
	cv::Mat currImg = curr->GetOriginalImage();
	/*cv::Mat prevGray, currGray;
	cvtColor(prevImg, prevGray, CV_BGR2GRAY);
	cvtColor(currImg, currGray, CV_BGR2GRAY);*/
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
	prevPts = prev->mvMatchingPts;
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
		std::cout << "1" << std::endl;
		UVR_SLAM::MapPoint* pMPi = prev->mvpMatchingMPs[i];
		std::cout << "2" << std::endl;
		if (!pMPi || pMPi->isDeleted())
		{
			continue;
		}
		/*if (!curr->isInFrustum(pMPi, 0.5)) {
			continue;
		}*/

		if (!curr->isInImage(currPts[i].x, currPts[i].y)) {
			continue;
		}
		if (!CheckOpticalPointOverlap(overlap, 2, currPts[i])) {
			nBad++;
			continue;
		}

		if (pMPi->GetRecentTrackingFrameID() == nCurrFrameID)
		{
			//nBad++;
			continue;
		}

		//매칭 결과
		float diffX = abs(prevPts[i].x - currPts[i].x);
		if (diffX > 25) {
			continue;
		}
		
		pMPi->SetRecentTrackingFrameID(nCurrFrameID);
		vpMPs.push_back(pMPi);
		vpPts.push_back(currPts[i]);
		vbInliers.push_back(true);
		
		cv::line(debugging, prevPts[i], currPts[i] + ptBottom, cv::Scalar(255, 255, 0));
		cv::circle(debugging, prevPts[i], 1, cv::Scalar(255, 0, 255),-1);
		cv::circle(debugging, currPts[i] + ptBottom, 1, cv::Scalar(255, 0, 255), -1);
		res++;
	}
	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;

	//fuse time text 
	std::stringstream ss;
	ss << "Optical flow tracking= " << res <<", "<<nBad<< "::" << tttt;
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	imshow("Output::Matching", debugging);
	/////////////////////////

	return res;
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

	cv::Mat prevGray, currGray;
	cvtColor(prevImg, prevGray, CV_BGR2GRAY);
	cvtColor(currImg, currGray, CV_BGR2GRAY);
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
	prevPts = prev->mvPts;
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
		if (diffX < 15) {
			bMatch = true;
			res++;
			//cv::line(debugging, prevPts[i], currPts[i] + ptBottom, cv::Scalar(255, 0, 255));
		}
		else if (diffX >= 15 && diffX < 90) {
			res++;
			//cv::line(debugging, prevPts[i], currPts[i] + ptBottom, cv::Scalar(0, 255, 255));
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
	ss << "Optical flow mapping= " << res << ", " << tttt;
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(debugging.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	imshow("Mapping::OpticalFlow ", debugging);
	/////////////////////////
	return res;
}
////200410 Optical flow
///////////////////////////////////////////////////////////////////////////////////////////////////////