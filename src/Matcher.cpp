#include <Matcher.h>
#include <omp.h>
#include <random>
#include <Frame.h>
#include <MapPoint.h>
#include <FrameWindow.h>
#include <MatrixOperator.h>

UVR_SLAM::Matcher::Matcher(){}
UVR_SLAM::Matcher::Matcher(cv::Ptr < cv::DescriptorMatcher> _matcher, int w, int h)
	:mWidth(w), mHeight(h), TH_HIGH(100), TH_LOW(50), HISTO_LENGTH(30), mfNNratio(0.8), mbCheckOrientation(true), matcher(_matcher)
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

int UVR_SLAM::Matcher::FeatureMatchingForPoseTrackingByProjection(UVR_SLAM::FrameWindow* pWindow, UVR_SLAM::Frame* pF, std::vector<MapPoint*> mvpLocalMPs, cv::Mat mLocalMapDesc, std::vector<bool>& mvbLocalMapInliers,float rr) {
	int nmatches = 0;
	int nf = 0;

	cv::Mat R = pWindow->GetRotation();
	cv::Mat t = pWindow->GetTranslation();

	for (int i = 0; i < mvpLocalMPs.size(); i++) {
		UVR_SLAM::MapPoint* pMP = mvpLocalMPs[i];
		if (!pMP)
			continue;
		if (pMP->isDeleted())
			continue;
		if (mvbLocalMapInliers[i])
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
			
			cv::DMatch tempMatch;
			tempMatch.queryIdx = i;
			tempMatch.trainIdx = bestIdx;
			pWindow->mvMatchInfos[i] = tempMatch;
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
	imshow("abasdfasdf", vis);

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
			pWindow->mvMatchInfos[matches[i][0].queryIdx] = matches[i][0];
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

int UVR_SLAM::Matcher::FeatureMatchingForInitialPoseTracking(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr, UVR_SLAM::FrameWindow* pWindow, std::vector<MapPoint*> mvpLocalMPs, cv::Mat mLocalMapDesc, std::vector<bool>& mvbLocalMapInliers, std::vector<cv::DMatch>& vMatchInfos) {

	std::vector<bool> vbTemp(pCurr->mvKeyPoints.size(), true);
	std::vector< std::vector<cv::DMatch> > matches;
	matcher->knnMatch(pPrev->mTrackedDescriptor, pCurr->matDescriptor, matches, 2);
	
	int Nf1 = 0;
	int Nf2 = 0;
	int count = 0;
	
	for (unsigned long i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {

			vMatchInfos.push_back(matches[i][0]);

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
			if (vbTemp[matches[i][0].trainIdx]) {
				vbTemp[matches[i][0].trainIdx] = false;
			}
			else {
				Nf1++;
				continue;
			}
			
			/*cv::Mat pCam;
			cv::Point2f p2D;
			if (!pMP->Projection(p2D, pCam, pWindow->GetRotation(), pWindow->GetTranslation(), pCurr->mK, mWidth, mHeight)) {
				Nf2++;
				continue;
			}*/

			pCurr->mvpMPs[matches[i][0].trainIdx] = pMP;
			pCurr->mvbMPInliers[matches[i][0].trainIdx] = true;

			cv::DMatch tempMatch;
			tempMatch.queryIdx = pMP->GetFrameWindowIndex();
			tempMatch.trainIdx = matches[i][0].trainIdx;

			//pWindow->mvMatchInfos.push_back(tempMatch);
			mvbLocalMapInliers[tempMatch.queryIdx] = true;
			pWindow->mvMatchInfos[tempMatch.queryIdx] = tempMatch;
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
			cv::circle(debugging, pPrev->mvKeyPoints[idx].pt, 1, cv::Scalar(255, 0, 0), -1);
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
			resMatches.push_back(matches[i][0]);
		}
	}
	
	//std::vector<bool> mvInliers;
	//float score;

	//if ((int)vMatches.size() >= 8) {
	//	FindFundamental(init, curr, vMatches, mvInliers, score, F);
	//	F.convertTo(F, CV_32FC1);
	//}
	//if ((int)vMatches.size() < 8 || F.empty()) {
	//	F.release();
	//	F = cv::Mat::zeros(0, 0, CV_32FC1);
	//	return 0;
	//}

	//int count = 0;
	//

	//for (unsigned long i = 0; i < vMatches.size(); i++) {
	//	//if(inlier_mask.at<uchar>((int)i)) {
	//	if (mvInliers[i]) {

	//		cv::Point2f pt1 = init->mvKeyPoints[vMatches[i].queryIdx].pt;
	//		cv::Point2f pt2 = curr->mvKeyPoints[vMatches[i].trainIdx].pt;
	//		//init->mvnCPMatchingIdx.push_back(vMatches[i].queryIdx);
	//		//curr->mvnCPMatchingIdx.push_back(vMatches[i].trainIdx);
	//		resMatches.push_back(vMatches[i]);
	//		count++;
	//	}
	//}
	return resMatches.size(); //190116 //inliers.size();
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