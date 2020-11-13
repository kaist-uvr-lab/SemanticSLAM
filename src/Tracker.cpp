#include <Tracker.h>
#include <System.h>
#include <Map.h>
#include <Plane.h>
#include <Frame.h>
#include <Matcher.h>
#include <Initializer.h>
#include <LocalMapper.h>
#include <SemanticSegmentator.h>
#include <SegmentationData.h>
#include <PlaneEstimator.h>
#include <FrameVisualizer.h>
#include <Visualizer.h>
#include <CandidatePoint.h>
#include <MapPoint.h>
#include <FrameGrid.h>
#include <DepthFilter.h>
#include <ZMSSD.h>

cv::Mat ComputeLineEquation(cv::Point2f pt1, cv::Point2f pt2) {
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

//std::vector<cv::Vec3b> UVR_SLAM::ObjectColors::mvObjectLabelColors;
cv::Point2f CalcLinePoint(float val, cv::Mat mLine, bool opt) {
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
cv::Mat ComputeWarpAffineMatrix(cv::Mat ray, cv::Point2f pt, float depth, cv::Mat Rrel, cv::Mat Trel, cv::Mat K, cv::Mat invK) {
	int halfpatch_size = 5;
	cv::Mat Xcam = ray*depth;
	cv::Point2f pt_du_ref = pt + cv::Point2f(halfpatch_size, 0);
	cv::Point2f pt_dv_ref = pt + cv::Point2f(0, halfpatch_size);
	cv::Mat Xcam_du_ref = invK*(cv::Mat_<float>(3, 1) << pt_du_ref.x, pt_du_ref.y, 1.0);
	cv::Mat Xcam_dv_ref = invK*(cv::Mat_<float>(3, 1) << pt_dv_ref.x, pt_dv_ref.y, 1.0);
	Xcam_du_ref *= (Xcam.at<float>(2) / Xcam_du_ref.at<float>(2));
	Xcam_dv_ref *= (Xcam.at<float>(2) / Xcam_dv_ref.at<float>(2));
	
	cv::Mat temp1 = K*(Rrel*Xcam + Trel);		 cv::Point2f px(temp1.at<float>(0) / temp1.at<float>(2), temp1.at<float>(1) / temp1.at<float>(2));
	cv::Mat temp2 = K*(Rrel*Xcam_du_ref + Trel); cv::Point2f px_du(temp2.at<float>(0) / temp2.at<float>(2), temp2.at<float>(1) / temp2.at<float>(2));
	cv::Mat temp3 = K*(Rrel*Xcam_dv_ref + Trel); cv::Point2f px_dv(temp3.at<float>(0) / temp3.at<float>(2), temp3.at<float>(1) / temp3.at<float>(2));

	px_du -= px; px_du /=halfpatch_size;
	px_dv -= px; px_dv /= halfpatch_size;

	/*cv::Mat px    = temp1 / temp1.at<float>(2);
	cv::Mat px_du = temp1 / temp2.at<float>(2);
	cv::Mat px_dv = temp1 / temp3.at<float>(2);
	
	px_du = (px_du - px) / halfpatch_size;
	px_dv = (px_dv - px) / halfpatch_size;*/

	cv::Mat res = cv::Mat::zeros(2, 2, CV_32FC1);
	//px_du.copyTo(res.col(0));
	res.at<float>(0, 0) = px_du.x;
	res.at<float>(1, 0) = px_du.y;
	res.at<float>(0, 1) = px_dv.x;
	res.at<float>(1, 1) = px_dv.y;
	return res;
}


UVR_SLAM::Tracker::Tracker() {}
UVR_SLAM::Tracker::Tracker(int w, int h, cv::Mat K):mnWidth(w), mnHeight(h), mK(K), mbInitializing(false), mbFirstFrameAfterInit(false), mbInitilized(false){}
UVR_SLAM::Tracker::Tracker(System* pSys, std::string strPath) : mpSystem(pSys), mbInitializing(false), mbFirstFrameAfterInit(false), mbInitilized(false) {
	FileStorage fs(strPath, FileStorage::READ);

	float fx = fs["Camera.fx"];
	float fy = fs["Camera.fy"];
	float cx = fs["Camera.cx"];
	float cy = fs["Camera.cy"];

	mK = cv::Mat::eye(3, 3, CV_32F);
	mK.at<float>(0, 0) = fx;
	mK.at<float>(1, 1) = fy;
	mK.at<float>(0, 2) = cx;
	mK.at<float>(1, 2) = cy;

	cv::Mat DistCoef(4, 1, CV_32F);
	DistCoef.at<float>(0) = fs["Camera.k1"];
	DistCoef.at<float>(1) = fs["Camera.k2"];
	DistCoef.at<float>(2) = fs["Camera.p1"];
	DistCoef.at<float>(3) = fs["Camera.p2"];
	const float k3 = fs["Camera.k3"];
	if (k3 != 0)
	{
		DistCoef.resize(5);
		DistCoef.at<float>(4) = k3;
	}
	DistCoef.copyTo(mD);

	float fps = fs["Camera.fps"];
	mnMaxFrames = 5;// 10;//fps;
	mnMinFrames = 3; //fps / 3;//3

	mnThreshMinCPs	 = fs["Tracker.MinCP"];
	mnThreshMinMPs	 = fs["Tracker.MinMP"];
	mnThreshDiff	 = fs["Tracker.MinDiff"];
	mnThreshDiffPose = fs["Tracker.MinPoseHandle"];

	mnWidth = fs["Image.width"];
	mnHeight = fs["Image.height"];
	mK2 = (cv::Mat_<float>(3, 3) << fx, 0, 0, 0, fy, 0, -fy*cx, -fx*cy, fx*fy); //line projection
	fs.release();
}
UVR_SLAM::Tracker::~Tracker() {}

bool UVR_SLAM::Tracker::isInitialized() {
	return mbInitilized;
}
void UVR_SLAM::Tracker::Init() {
	mpMap = mpSystem->mpMap;
	mpVisualizer = mpSystem->mpVisualizer;
	mpFrameVisualizer = mpSystem->mpFrameVisualizer;
	mpMatcher = mpSystem->mpMatcher;
	mpInitializer = mpSystem->mpInitializer;
	mpSegmentator = mpSystem->mpSegmentator;
	mpLocalMapper = mpSystem->mpLocalMapper;
	mpPlaneEstimator = mpSystem->mpPlaneEstimator;
}
bool UVR_SLAM::Tracker::CheckNeedKeyFrame(Frame* pCurr, bool &bNeedCP, bool &bNeedMP, bool &bNeedPoseHandle, bool &bNeedNewKF) {
	///////////////
	//keyframe process

	int nHalf = mpSystem->mnRadius;
	int nSize = nHalf * 2;
	int a = mnWidth / nSize;
	int b = mnHeight / nSize;
	int nTotal = a*b;
	float fRatioCP = ((float)mnPointMatching) / nTotal;
	float fRatioMP = ((float)mnMapPointMatching) / nTotal;

	//int nDiffCP = abs(mnPointMatching - mnPrevPointMatching);
	//int nDiffMP = abs(mnMapPointMatching - mnPrevMapPointMatching);
	//int nPoseFail = abs(mnPointMatching - mnMapPointMatching);
	//bool bDiffCP = nDiffCP > mnThreshDiff;
	//bool bDiffMP = nDiffMP > mnThreshDiff;
	//bool bPoseFail = mnMapPointMatching < 80;//nPoseFail > mnThreshDiffPose;
	//bool bMatchMP = mnMapPointMatching < mnThreshMinMPs;
	//bool bMatchCP = mnPointMatching < mnThreshMinCPs;
	//bool bDoingMapping = !mpLocalMapper->isDoingProcess();
	//bool bMaxFrames = pCurr->mnFrameID >= mpRefKF->mnFrameID + mnMaxFrames;//mnMinFrames;
	//bool bMinFrames = pCurr->mnFrameID >= mpRefKF->mnFrameID + mnMinFrames;
	//bNeedCP = bDiffCP || bMatchCP;
	//bNeedMP = (bDiffMP || bMatchMP) && bMinFrames;
	//bNeedPoseHandle = bPoseFail;
	//bNeedNewKF = bMinFrames;
	
	bool bDoingMapping = !mpLocalMapper->isDoingProcess();
	bool bMaxFrames = pCurr->mnFrameID >= mpRefKF->mnFrameID + mnMaxFrames;//mnMinFrames;
	bool bMinFrames = pCurr->mnFrameID >= mpRefKF->mnFrameID + mnMinFrames;
	bNeedCP = fRatioCP < 0.2f;
	bNeedMP = fRatioMP < 0.1f && bMinFrames;
	bNeedPoseHandle = fRatioMP < 0.1f;
	bNeedNewKF = bMinFrames;
	return bDoingMapping && (bNeedCP || bNeedMP || bNeedPoseHandle || bNeedNewKF);
}
UVR_SLAM::Frame* UVR_SLAM::Tracker::CheckNeedKeyFrame(Frame* pCurr, Frame* pPrev) {

	///////////////
	//keyframe process
	int nDiffCP = abs(mnPointMatching - mnPrevPointMatching);
	int nDiffMP = abs(mnMapPointMatching - mnPrevMapPointMatching);
	int nPoseFail = abs(mnPointMatching - mnMapPointMatching);

	//bool bDiff = nDiff > 50;
	bool bDiffCP = nDiffCP > 50;
	bool bDiffMp = nDiffMP > 50;
	bool bPoseFail = nPoseFail > 100;

	//1 : rotation angle
	bool bDoingMapping = !mpLocalMapper->isDoingProcess();
	bool bRotation = pCurr->CalcDiffAngleAxis(mpRefKF) > 10.0;
	bool bMaxFrames = pCurr->mnFrameID >= mpRefKF->mnFrameID + mnMaxFrames;//mnMinFrames;
	bool bMinFrames = pCurr->mnFrameID < mpRefKF->mnFrameID + mnMinFrames;

	bool bMatchMapPoint = mnMapPointMatching < 200;
	bool bMatchPoint = mnPointMatching < 350;
	

	//if ((bRotation || bMatchMapPoint || bMatchPoint || bMaxFrames) && bDoingSegment)
	/*if ((bRotation || bMatchMapPoint || bKF || bAVG || bMaxFrames) && !bMinFrames && bDoingMapping)
	{
	if(pCurr->CheckBaseLine(mpRefKF))
	return true;
	return false;
	}
	else
	return false;*/
	UVR_SLAM::Frame* pRes = nullptr;
	if (!bMinFrames && bDoingMapping) {
		//if (bRotation || bMaxFrames) {
		//	pRes = pCurr;
		//}
		//else if (bMatchMapPoint || bDiff) {//bKF
		//	pRes = pPrev;
		//}
		//else
		//	pRes = pCurr;
		pRes = pCurr;
		return pRes;
		//baseline이 중요하지 않은듯 여기서는
		/*if (pRes->CheckBaseLine(mpRefKF))
			return pRes;
		else
			return nullptr;*/
	}
	else
		return nullptr;
}

//bool bRefKF = false;
void UVR_SLAM::Tracker::Tracking(Frame* pPrev, Frame* pCurr) {
	if(!mbInitializing){
		bool bReset = false;
		mbInitializing = mpInitializer->Initialize(pCurr, bReset, mnWidth, mnHeight);
		
		if (bReset){
			mpSystem->Reset();
		}
		//mbInit = bInit;
		mbFirstFrameAfterInit = false;
		
		if (mbInitializing){
			mpRefKF = pCurr;
			mbInitilized = true;
			mpSystem->SetBoolInit(true);
			mnPrevMapPointMatching = pCurr->mpMatchInfo->GetNumMPs();
			mnPrevPointMatching = mnPrevMapPointMatching;
			std::cout << "INIT::" << mnPrevMapPointMatching << ", " << mnPrevPointMatching << std::endl;
		}
	}
	else {
		std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////Optical Flow Matching
		////MatchInfo 설정
		mpRefKF->SetRecentTrackedFrameID(pCurr->mnFrameID);
		pCurr->mpMatchInfo = new UVR_SLAM::MatchInfo(mpSystem, pCurr, mpRefKF, mnWidth, mnHeight);
		
		{
			std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateCP);
			mpSystem->cvUseCreateCP.wait(lock, [&] {return mpSystem->mbCreateCP;});
		}
		////MatchInfo 설정
		//초기 매칭 테스트
		std::vector<UVR_SLAM::MapPoint*> vpTempMPs;
		std::vector<UVR_SLAM::CandidatePoint*> vpTempCPs;
		std::vector<cv::Point2f> vTempCurrPts, vTempPrevPts;
		std::vector<cv::Point3f> vpTempPts2;
		std::vector<uchar> vcInliers;
		std::vector<bool> vbTempInliers;// = std::vector<bool>(pPrev->mvpMatchingMPs.size(), false);
		std::vector<int> vnIDXs, vnMPIDXs;
		cv::Mat debugImg;
		cv::Mat overlap = cv::Mat::zeros(pCurr->mnHeight, pCurr->mnWidth, CV_8UC1);
		mnPointMatching = mpMatcher->OpticalMatchingForTracking(pPrev, pCurr, vpTempCPs, vTempPrevPts, vTempCurrPts, vbTempInliers);
		//mpMatcher->OpticalGridsMatching(pPrev, pCurr, vpTempPts1);
		//pCurr->mpMatchInfo->InitMapPointInlierVector(mnPointMatching); //삭제 예정
		std::chrono::high_resolution_clock::time_point tracking_a = std::chrono::high_resolution_clock::now();
		{
			std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateMP);
			mpSystem->cvUseCreateMP.wait(lock, [&] {return mpSystem->mbCreateMP; });
		}
		cv::Mat prevR, prevT;
		pPrev->GetPose(prevR, prevT);
		pCurr->SetPose(prevR, prevT);
		mnMapPointMatching = Optimization::PoseOptimization(mpMap, pCurr, vpTempCPs, vTempCurrPts, vbTempInliers);

		//////임시
		cv::Mat prevImg = pPrev->GetOriginalImage().clone();
		cv::Mat currImg = pCurr->GetOriginalImage().clone();
		cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
		cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);

		//에피폴라 관련 파라메터
		cv::Mat invK = mK.inv();
		cv::Mat F12 = pCurr->ComputeFundamentalMatrix(pPrev);

		/*cv::Mat F12cv = cv::findFundamentalMat(vTempPrevPts, vTempCurrPts);
		std::vector<cv::Point3f> vLines;
		cv::computeCorrespondEpilines(vTempCurrPts, 2, F12, vLines);*/

		//cv::Mat F12 = pPrev->ComputeFundamentalMatrix(pCurr);
		cv::Mat Rrel, Trel;
		pCurr->GetRelativePoseFromTargetFrame(pPrev, Rrel, Trel);
		float fx = mK.at<float>(0, 0);
		float fy = mK.at<float>(1, 1);
		float cx = mK.at<float>(0, 2);
		float cy = mK.at<float>(1, 2);
		int patch_size = UVR_SLAM::ZMSSD::patch_size_;
		int patch_half_size = patch_size / 2;
		int patch_area = patch_size * patch_size;
		int mzssd_thresh = 2000 * patch_area;
		cv::Point2f patch_pt(patch_half_size, patch_half_size);
		//에피폴라 관련 파라메터

		////그리드 파라메터
		int nGridSize = mpSystem->mnRadius*2;
		float fGridDistThresh = nGridSize*nGridSize*4;
		////그리드 파라메터
		for (size_t i = 0, iend = vpTempCPs.size(); i < iend; i++) {
			auto pCP = vpTempCPs[i];
			auto currPt = vTempCurrPts[i];
			auto prevPt = vTempPrevPts[i];
			auto pMP = pCP->GetMP();
			bool bMP = pMP && !pMP->isDeleted();
			
			if (!vbTempInliers[i])
				continue;

			auto gridPt = pPrev->GetGridBasePt(currPt, nGridSize);
			if (pCurr->mmbFrameGrids[gridPt]) {
				vbTempInliers[i] = false;
				continue;
			}

			auto prevGridPt = pPrev->GetGridBasePt(prevPt, nGridSize);
			auto diffx = abs(prevGridPt.x - gridPt.x);
			auto diffy = abs(prevGridPt.y - gridPt.y);

			if (diffx > 2 * nGridSize || diffy > 2 * nGridSize) {
				vbTempInliers[i] = false;
				continue;
			}

			float epi_res;
			bool bEpiLine;
			cv::Mat epiLine;
			bool bTestLine = false;
			bool bEpiConstraints = mpMatcher->CheckEpiConstraints(F12, currPt, prevPt, 1.0, epiLine, epi_res, bEpiLine);
			
			if (!bEpiConstraints)
			{
				cv::circle(currImg, currPt, 4, cv::Scalar(0, 0, 0), -1);
				cv::circle(prevImg, prevPt, 4, cv::Scalar(0, 0, 0), -1);
				/*vbTempInliers[i] = false;
				continue;*/
			}

			auto rect = cv::Rect(gridPt, std::move(cv::Point2f(gridPt.x + nGridSize, gridPt.y + nGridSize)));
			//////grid matching
			//auto prevGrid = pPrev->mmpFrameGrids[prevGridPt];
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
			pCurr->mmbFrameGrids[gridPt] = true;
			auto currGrid = new FrameGrid(gridPt, rect);
			//currGrid->vecPTs = vGridPTs;
			pCurr->mmpFrameGrids[gridPt] = currGrid;
			pCurr->mmpFrameGrids[gridPt]->mpCP = pCP;
			pCurr->mmpFrameGrids[gridPt]->pt = currPt;
			//////grid 추가
			
			//////epipolar
			////ray test
			cv::Mat ray = mpSystem->mInvK*(cv::Mat_<float>(3, 1) << prevPt.x, prevPt.y, 1.0);
			float z_min, z_max;
			/*auto pSeeda = pCP->mpSeed;
			if (pCP->mpSeed) {
				float z_inv_min = pSeeda->mu + sqrt(pSeeda->sigma2);
				float z_inv_max = max(pSeeda->mu - sqrt(pSeeda->sigma2), 0.00000001f);
				z_min = 1. / z_inv_min;
				z_max = 1. / z_inv_max;
			}
			else
			{
				z_min = 0.01f;
				z_max = 1.0f;
			}*/
			z_min = 0.01f;
			z_max = 1.0f;

			cv::Mat Xcmin = Rrel*ray*z_min + Trel;
			cv::Mat Xcmax = Rrel*ray*z_max + Trel;
			cv::Point2f XprojMin(Xcmin.at<float>(0) / Xcmin.at<float>(2), Xcmin.at<float>(1) / Xcmin.at<float>(2));
			cv::Point2f XprojMax(Xcmax.at<float>(0) / Xcmax.at<float>(2), Xcmax.at<float>(1) / Xcmax.at<float>(2));
			cv::Point2f epi_dir = XprojMin - XprojMax;
			cv::Point2f XimgMin(XprojMin.x*fx + cx, XprojMin.y*fy + cy);
			cv::Point2f XimgMax(XprojMax.x*fx + cx, XprojMax.y*fy + cy);
			cv::Mat lineEqu = ComputeLineEquation(XimgMin, XimgMax);
			float epi_length = cv::norm(XimgMin - XimgMax) / 2.0;
			size_t n_steps = epi_length / 0.7; // one step per pixel
			cv::Point2f step(epi_dir.x / n_steps, epi_dir.y / n_steps);
			cv::Point2f uv = XprojMax - step;
			cv::Point2f uv_best;
			float zmssd_best = 0.0;//mzssd_thresh;
			auto refLeftPt = prevPt - patch_pt;
			auto refRightPt = prevPt + patch_pt;
			bool bRefLeft = pCurr->isInImage(refLeftPt.x, refLeftPt.y, 10);
			bool bRefRight = pCurr->isInImage(refRightPt.x, refRightPt.y, 10);
			//if (bRefLeft && bRefRight) {
			//	cv::Rect refRect(prevPt - patch_pt, prevPt + patch_pt);

			//	cv::Mat ray = invK*(cv::Mat_<float>(3, 1) << prevPt.x, prevPt.y, 1.0);
			//	cv::Mat A = ComputeWarpAffineMatrix(ray, prevPt, 1.0, Rrel, Trel, mK, invK);
			//	cv::Mat ref_warp;
			//	//cv::warpAffine(pPrev->GetOriginalImage()(refRect), ref_warp, A, cv::Size(patch_size, patch_size));

			//	auto refZMSSD = new NCC(pPrev->GetOriginalImage()(refRect).clone());
			//	//imshow("ref::", pPrev->GetOriginalImage()(refRect));
			//	//imshow("ref::warp", ref_warp);
			//	for (size_t i = 0; i < n_steps; ++i, uv += step)
			//	{
			//		cv::Point2f pt(uv.x*fx + cx, uv.y*fy + cy);

			//		auto patchGridPt = pPrev->GetGridBasePt(pt, nGridSize);
			//		auto diffPt = prevPt - pt;
			//		float dist = diffPt.dot(diffPt);

			//		if (dist > fGridDistThresh) {
			//			continue;
			//		}
			//		/*if (pCurr->mmbFrameGrids[patchGridPt]) {wwwwwwwwww
			//			continue;
			//		}*/
			//		
			//		/*if(bTestLine)
			//			cv::circle(currImg, pt, 1, cv::Scalar(0 , 0, 255), -1);
			//		else*/
			//			cv::circle(currImg, pt, 1, cv::Scalar(255, 255, 255), -1);

			//		auto leftPt = pt - patch_pt;
			//		auto rightPt = pt + patch_pt;
			//		bool bPatchLeft = pCurr->isInImage(leftPt.x, leftPt.y, 10);
			//		bool bPatchRight = pCurr->isInImage(rightPt.x, rightPt.y, 10);
			//		if (bPatchLeft && bPatchRight) {
			//			cv::Rect patchRect(leftPt, rightPt);
			//			float val = refZMSSD->computeScore(pCurr->GetOriginalImage()(patchRect).clone());
			//			if (val > zmssd_best) {
			//				uv_best = pt;
			//				zmssd_best = val;
			//			}
			//			//imshow("cur::", pCurr->GetOriginalImage()(patchRect));
			//		}//if patch
			//	}//for
			//	if (zmssd_best > 0.1) {//mzssd_thresh
			//		//cv::line(currImg, prevPt, uv_best, cv::Scalar(0, 255, 255), 1);
			//		cv::circle(currImg, uv_best, 2, cv::Scalar(0, 255, 0), -1);
			//	}
			//}//if ref

			if (i % 50 == 0 && bEpiLine) {
				/*bTestLine = true;
				auto sPt = CalcLinePoint(0, epiLine, true);
				auto ePt = CalcLinePoint(mnHeight, epiLine, true);
				cv::line(currImg, sPt, ePt, cv::Scalar(0, 255, 0), 1);
				cv::Mat epiLines2(vLines[i]);
				auto sPt2 = CalcLinePoint(0, epiLines2, true);
				auto ePt2 = CalcLinePoint(mnHeight, epiLines2, true);
				cv::line(currImg, sPt2, ePt2, cv::Scalar(255, 0, 0), 1);*/

				//cv::Mat epiLines3(Xcmin- Xcmin);
				auto sPt3 = CalcLinePoint(0, lineEqu, true);
				auto ePt3 = CalcLinePoint(mnHeight, lineEqu, true);
				cv::line(currImg, sPt3, ePt3, cv::Scalar(0, 0, 255), 1);
				cv::line(currImg, XimgMin, XimgMax, cv::Scalar(0, 255, 255), 1);
			}

			

			//auto pSeed = pCP->mpSeed;
			//if (pSeed) {

			//	/*float z_inv_min = pSeed->mu + sqrt(pSeed->sigma2);
			//	float z_inv_max = max(pSeed->mu - sqrt(pSeed->sigma2), 0.00000001f);
			//	float z_min = 1. / z_inv_min;
			//	float z_max = 1. / z_inv_max;
			//	cv::Mat Xcam4 = pSeed->ray*z_min;
			//	cv::Mat Xcam5 = pSeed->ray*z_max;
			//	cv::Mat Rrel2, Trel2;
			//	pCurr->GetRelativePoseFromTargetFrame(pCP->mpRefKF, Rrel2, Trel2);
			//	{
			//		cv::Mat projFromRef  = mK*(Rrel2*Xcam4 + Trel2);
			//		cv::Mat projFromRef2 = mK*(Rrel2*Xcam5 + Trel2);
			//		cv::Point2f ptFromRef(projFromRef.at<float>(0) / projFromRef.at<float>(2), projFromRef.at<float>(1) / projFromRef.at<float>(2));
			//		cv::Point2f ptFromRef2(projFromRef2.at<float>(0) / projFromRef2.at<float>(2), projFromRef2.at<float>(1) / projFromRef2.at<float>(2));
			//		cv::line(currImg, ptFromRef, ptFromRef2, cv::Scalar(0, 255, 0), 1);
			//	}*/
			//}

			if (bMP)
			{
				cv::line(currImg, currPt, prevPt, cv::Scalar(255, 0, 255), 1);
				cv::circle(prevImg, prevPt, 2,    cv::Scalar(0, 0, 255), -1);
				cv::circle(currImg, currPt, 2,    cv::Scalar(0, 0, 255), -1);
			}
			else {
				cv::line(currImg, currPt, prevPt, cv::Scalar(255, 255, 0), 1);
				cv::circle(prevImg, prevPt, 2, cv::Scalar(255, 0, 0), -1);
				cv::circle(currImg, currPt, 2, cv::Scalar(255, 0, 0), -1);
			}
			/*if (!bEpiConstraints) {
				cv::circle(currImg, currPt, 4, cv::Scalar(0, 0, 0), 1);
				cv::circle(prevImg, prevPt, 4, cv::Scalar(0, 0, 0), 1);
			}*/
			
		}
		cv::Mat debugMatch = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
		prevImg.copyTo(debugMatch(mergeRect1));
		currImg.copyTo(debugMatch(mergeRect2));
		cv::moveWindow("Output::MatchTest2", mpSystem->mnDisplayX+prevImg.cols*2, mpSystem->mnDisplayY);
		cv::imshow("Output::MatchTest2", debugMatch); //cv::waitKey();
		int nMP = UpdateMatchingInfo(pCurr, vpTempCPs, vTempCurrPts, vbTempInliers);
		////여기서 시각화


		///////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////키프레임 체크
		bool bNeedCP, bNeedMP, bNeedPoseHandle, bNeedNewKF;
		auto bNewKF = CheckNeedKeyFrame(pCurr, bNeedCP, bNeedMP, bNeedPoseHandle, bNeedNewKF);
		if (bNewKF) {
			if (bNeedCP) {
				std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateCP);
				mpSystem->mbCreateCP = false;
			}
			/*if (bNeedMP) {
				std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateMP);
				mpSystem->mbCreateMP = false;
			}*/
			if(bNeedNewKF || bNeedMP)
				mpRefKF = pCurr;
			mpLocalMapper->InsertKeyFrame(pCurr, bNeedCP, bNeedMP, bNeedPoseHandle, bNeedNewKF);
		}
		mnPrevPointMatching = mnPointMatching;
		mnPrevMapPointMatching = mnMapPointMatching;
		////////Visualization & 시간 계산
		std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
		auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_a - tracking_start).count();
		double t1 = duration1 / 1000.0;
		auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
		double t2 = duration2 / 1000.0;

		///////시각화
		if (!mpFrameVisualizer->isVisualize()) {
			mpFrameVisualizer->SetFrameMatchingInformation(mpRefKF, pCurr, t2);//vpTempMPs, vpTempPts, vbTempInliers,
		}

		/////////트래킹 결과 이미지 저장
		//visualizer thread
		if (!mpVisualizer->isDoingProcess()) {
			mpVisualizer->SetMatchInfo(pCurr->mpMatchInfo);
			mpVisualizer->SetBoolDoingProcess(true);
		}
		//visualizer thread
	}
}

//void UVR_SLAM::Tracker::Run() {
//	while (1) {
//		////Frame 정보와 Queue 정보를 받아야 함.
//		if (mbInit) {
//
//		}		else {
//
//		}
//	}
//}

////MP와 PT가 대응함.
////pPrev가 mpRefKF가 됨
//매칭정보에 비율을 아예 추가하기
int UVR_SLAM::Tracker::UpdateMatchingInfo(UVR_SLAM::Frame* pCurr, std::vector<UVR_SLAM::CandidatePoint*> vpCPs, std::vector<cv::Point2f> vpPts, std::vector<bool> vbInliers) {
	
	auto pMatchInfo = pCurr->mpMatchInfo;
	
	for (size_t i = 0, iend = vpCPs.size(); i < iend; i++) {
		auto pCP = vpCPs[i];
		auto pt = vpPts[i];
		auto pMP = pCP->GetMP();
		bool bMP = pMP && !pMP->isDeleted();
		if (bMP) {
			pMP->IncreaseVisible();
		}
		if (vbInliers[i]) {
			if(bMP)
				pMP->IncreaseFound();
			if (pMatchInfo->CheckOpticalPointOverlap(pt, mpSystem->mnRadius) < 0) {
				int idx = pMatchInfo->AddCP(pCP, pt);
				//pCP->ConnectFrame(pMatchInfo, idx);
			}
		}
	}
	
	return 0;
}

int UVR_SLAM::Tracker::UpdateMatchingInfo(UVR_SLAM::Frame* pPrev, UVR_SLAM::Frame* pCurr, std::vector<UVR_SLAM::CandidatePoint*> vpCPs, std::vector<UVR_SLAM::MapPoint*> vpMPs, std::vector<cv::Point2f> vpPts, std::vector<bool> vbInliers, std::vector<int> vnIDXs, std::vector<int> vnMPIDXs) {
	auto pMatchInfo = pCurr->mpMatchInfo;
	auto pPrevMatchInfo = pPrev->mpMatchInfo;
	int nCurrID = pCurr->mnFrameID;
	int nres = 0;
	int nFail = 0;

	for (int i = 0; i < vpPts.size(); i++) {
		auto pCP = vpCPs[i];
		auto pMP = vpMPs[i];
		if (!vbInliers[i]){
			//pMP->AddFail();
			nFail++;
			continue;
		}
		int prevIdx = vnIDXs[i];
		auto pt = vpPts[i];
		if (pMatchInfo->CheckOpticalPointOverlap(pt, mpSystem->mnRadius) < 0) {
			//pMP->AddSuccess();
			//pMP->SetLastSuccessFrame(pCurr->GetFrameID());
			pMatchInfo->AddCP(pCP, pt);
			nres++;
		}
	}

	pMatchInfo->mfLowQualityRatio = ((float)nFail)/ vpPts.size();
	//std::cout << "Tracking::ID=" << pPrev->GetKeyFrameID() <<", "<< nCurrID << " matching = " << nres <<", Quality = "<< pMatchInfo->mfLowQualityRatio << std::endl;
	return nres;
}