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
			//local map�� inlier�� Ȯ���� �� �̿���.
			pMP->SetRecentTrackingFrameID(nCurrFrameID);

			pCurr->mvpMPs[idx2] = pMP;
			pCurr->mvbMPInliers[idx2] = true;

			cv::DMatch matchInfo;
			matchInfo.queryIdx = idx;
			matchInfo.trainIdx = idx2;
			mvMatches.push_back(matchInfo);

			//��Ī ���� Ȯ�ο�
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

	 //�ð�ȭ
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

//Fuse �� ȣ��
//Fuse�� ��� ������ ���� ������ ����Ʈ�� ���� �� Ű�������� ���� ������Ʈ�� �������� �ϱ� ����.
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


//pkf�� pneighkf�̰�
//targetkf�� vpmps
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
			//Ÿ���� �ٸ��� �� Ÿ���� �߰�, ������ �� ���ϱ�
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
		
		//�ߺ� ������Ʈ üũ
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
	
	//�ð�ȭ
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
	imshow("abasdfasdf", vis);

	return nmatches;
}

//����  ã�� �� �ʱ� ��Ī
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

			//��Ī ���� Ȯ�ο�
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
//Fuse�� ��� ��ü�� ���ؼ� �ؾ� ��
//�׷��� ����ִ� �ֿ� ���ؼ��� �� �� ����.
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

	//�� ������Ʈ ����
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

	//����ִ� ������Ʈ Ȯ��
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
//1�� ������ 2���� ã�� ����.
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


//�ʱ�ȭ���� F�� ����ϰ� ��Ī
int UVR_SLAM::Matcher::MatchingProcessForInitialization(UVR_SLAM::Frame* init, UVR_SLAM::Frame* curr, cv::Mat& F, std::vector<cv::DMatch>& resMatches) {

	////Debuging
	cv::Point2f ptBottom = cv::Point2f(0, curr->GetOriginalImage().rows);
	cv::Rect mergeRect1 = cv::Rect(0, 0, curr->GetOriginalImage().cols, curr->GetOriginalImage().rows);
	cv::Rect mergeRect2 = cv::Rect(0, curr->GetOriginalImage().rows, curr->GetOriginalImage().cols, curr->GetOriginalImage().rows);
	cv::Mat featureImg = cv::Mat::zeros(curr->GetOriginalImage().rows *2, curr->GetOriginalImage().cols, curr->GetOriginalImage().type());

	//std::stringstream sfile;
	//sfile << "/keyframe_" << 0;

	curr->GetOriginalImage().copyTo(featureImg(mergeRect1));
	init->GetOriginalImage().copyTo(featureImg(mergeRect2));
	//cvtColor(featureImg, featureImg, CV_RGBA2BGR);
	//featureImg.convertTo(featureImg, CV_8UC3);
	//////Debuging

	//�ߺ� ���ſ�
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
	
	imshow("init", featureImg);
	
	return res; //190116 //inliers.size();
}


////////Fundamental Matrix�� ���� �̿�
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



cv::Point2f CalcLinePoint(float y, Point3f mLine) {
	float x = 0.0;
	if (mLine.x != 0)
		x = (-mLine.z - mLine.y*y) / mLine.x;
	return cv::Point2f(x, y);
}

float CalcSSD(cv::Mat src1, cv::Mat src2) {
	cv::Mat diff = abs(src1 - src2);
	float sum = 0.0;
	int num = diff.cols*diff.rows*diff.channels();
	for (int x = 0; x < diff.cols; x++) {
		for (int y = 0; y < diff.rows; y++) {
			cv::Vec3b temp = diff.at<Vec3b>(y, x);
			sum += temp.val[0];
			sum += temp.val[1];
			sum += temp.val[2];
		}
	}
	return sum / num;
}

////�̰��� �ٴڸ� ��Ī.
////���� ��� �̰Ͱ� ����� �ɷ�
int UVR_SLAM::Matcher::MatchingWithEpiPolarGeometry(Frame* f1, Frame* f2, PlaneInformation* pFloor, std::vector<cv::Mat>& vPlanarMaps, std::vector<bool>& vbInliers, std::vector<cv::DMatch>& vMatches, cv::Mat& debugging){

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
		
	auto mvpOPs1 = f1->GetObjectVector();
	auto mvpOPs2 = f2->GetObjectVector();
	
	//Fundamental matrix �� keyframe pose ���
	cv::Mat Rcurr, Tcurr, Rprev, Tprev;
	f1->GetPose(Rprev, Tprev);
	f2->GetPose(Rcurr, Tcurr);
	cv::Mat mK = f1->mK.clone();
	cv::Mat F12 = CalcFundamentalMatrix(Rcurr, Tcurr, Rprev, Tprev, mK);
	
	cv::Mat invP1, invT1, invK, invP2, invT2;
	if (!f1->mpPlaneInformation || !f2->mpPlaneInformation) {
		//�� �������� �˾ƾ� ��.
		std::cout << "error case!!!!!!!!" << std::endl;
		return 0;
	}
	f1->mpPlaneInformation->Calculate();
	f2->mpPlaneInformation->Calculate();
	
	f1->mpPlaneInformation->GetInformation(invP1, invT1, invK);
	f2->mpPlaneInformation->GetInformation(invP2, invT2, invK);
	//���� ���
	//cv::Mat invPwawll = invT.t()*pWall->GetParam();

	std::vector<Point2f> vPts1, vPts2;
	std::vector<cv::Mat> vX3Ds;
	//std::vector<bool> vbs;
	
	//std::vector<cv::DMatch> vMatches;
	int n1 = 0;
	int n2 = 0;
	//��Ī Ȯ���ϱ�
	/*for (int i = 0; i < f1->mvKeyPoints.size(); i++) {
		if (mvpOPs1[i] != ObjectType::OBJECT_FLOOR && mvpOPs1[i] != ObjectType::OBJECT_WALL && mvpOPs1[i] != ObjectType::OBJECT_CEILING)
			continue;
		cv::circle(debugging, f1->mvKeyPoints[i].pt + ptBottom, 2, cv::Scalar(255, 0, 0), -1);
	}*/
	for (int i = 0; i < f2->mvKeyPoints.size(); i++) {
		if (mvpOPs2[i] != ObjectType::OBJECT_FLOOR && mvpOPs2[i] != ObjectType::OBJECT_WALL && mvpOPs2[i] != ObjectType::OBJECT_CEILING)
			continue;
		
		if (f2->mvpMPs[i])
			cv::circle(debugging, f2->mvKeyPoints[i].pt, 3, cv::Scalar(0, 255, 255), 1);
		else
			cv::circle(debugging, f2->mvKeyPoints[i].pt, 3, cv::Scalar(0, 255, 0), 1);
		
		if (vPlanarMaps[i].rows == 0)
			continue;
		
		cv::Mat X3D = vPlanarMaps[i];
				
		////��Ī �����ϱ�
		//�̹��� ��������
		cv::Mat temp = Rprev*X3D + Tprev;
		temp = f1->mK*temp;
		cv::Point2f tpt(temp.at<float>(0) / temp.at<float>(2), temp.at<float>(1) / temp.at<float>(2));
		cv::circle(debugging, tpt+ ptBottom, 3, cv::Scalar(0, 0, 255), 1);

		cv::Mat temp2 = Rcurr*X3D + Tcurr;
		temp2 = f1->mK*temp2;
		cv::Point2f tpt2(temp2.at<float>(0) / temp2.at<float>(2), temp2.at<float>(1) / temp2.at<float>(2));
		cv::circle(debugging, tpt2, 3, cv::Scalar(0, 0, 255), 1);

		////calc ssd
		//���� �ٿ���� ���� �����ϱ�
		cv::Rect rect1 = cv::Rect(f2->mvKeyPoints[i].pt.x-1, f2->mvKeyPoints[i].pt.y-1, 3, 3);
		cv::Rect rect2 = cv::Rect(tpt.x-1, tpt.y - 1, 3, 3);
		float val = CalcSSD(img1(rect1), img2(rect2));
		if (val < 30.0) {
			cv::circle(debugging, tpt + ptBottom, 3, cv::Scalar(255, 0, 0), 1);
		}
		else {
			cv::circle(debugging, tpt + ptBottom, 3, cv::Scalar(0, 255, 0), 1);
		}

		
		std::cout << "ssd::" << val << std::endl;
		////calc ssd

		//������ Ư¡ ã��
		std::vector<size_t> vIndices = f1->GetFeaturesInArea(tpt.x, tpt.y, 5.0);
		if (vIndices.empty())
			continue;
		cv::Mat desc1 = f2->matDescriptor.row(i);
		//��Ī ����
		//��ũ���� �Ǵ� �������� �������� ��Ī
		
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
			if (!CheckEpiConstraints(F12, tpt, kp.pt, sigma, epiDist) || (mvpOPs1[idx] != mvpOPs2[i]))
				continue;
			
			/*cv::Mat desc2 = f1->matDescriptor.row(idx);
			int descDist = DescriptorDistance(desc1, desc2);
			if (nMinDist > descDist) {
				nMinDist = descDist;
				bestIdx = idx;
			}*/
			if (epiDist < nMinDist) {
				nMinDist = epiDist;
				bestIdx = idx;
			}
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
		}
		
	}
	n2 = vMatches.size();

	/*if (vPts1.size() < 10){
		std::cout << "epi::error::" << vPts1.size() << std::endl;
		return 0;
	}*/
	
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
	imshow("matching test : ", debugging);
	cv::waitKey(1);
	//////debug

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	double tttt = duration / 1000.0;
	std::cout << "epitime::" << tttt << std::endl;
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

	//Fundamental matrix �� keyframe pose ���
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
	//��Ī Ȯ���ϱ�
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
		
		////��Ī �����ϱ�
		//�̹��� ��������
		cv::Mat temp = Rcurr*X3D + Tcurr;
		temp = mK*temp;
		cv::Point2f tpt(temp.at<float>(0) / temp.at<float>(2), temp.at<float>(1) / temp.at<float>(2));
		//������ Ư¡ ã��
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