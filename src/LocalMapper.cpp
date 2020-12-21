#include <LocalMapper.h>
#include <CandidatePoint.h>
#include <Frame.h>
#include <FrameGrid.h>
#include <System.h>
#include <Map.h>

#include <MapPoint.h>
#include <Matcher.h>
#include <LoopCloser.h>
#include <Optimization.h>
#include <PlaneEstimator.h>
#include <Plane.h>
#include <Visualizer.h>
#include <MapOptimizer.h>
#include <SemanticSegmentator.h>
#include <DepthFilter.h>

#include <opencv2/core/mat.hpp>
#include <ctime>
#include <direct.h>

UVR_SLAM::LocalMapper::LocalMapper(){}
UVR_SLAM::LocalMapper::LocalMapper(System* pSystem, std::string strPath, int w, int h):mnWidth(w), mnHeight(h), mbStopBA(false), mbDoingProcess(false), mbStopLocalMapping(false), mpTempFrame(nullptr),mpTargetFrame(nullptr), mpPrevKeyFrame(nullptr), mpPPrevKeyFrame(nullptr){
	mpSystem = pSystem;

	cv::FileStorage fs(strPath, cv::FileStorage::READ);

	float fx = fs["Camera.fx"];
	float fy = fs["Camera.fy"];
	float cx = fs["Camera.cx"];
	float cy = fs["Camera.cy"];

	fs.release();

	mK = cv::Mat::eye(3, 3, CV_32F);
	mK.at<float>(0, 0) = fx;
	mK.at<float>(1, 1) = fy;
	mK.at<float>(0, 2) = cx;
	mK.at<float>(1, 2) = cy;

	mInvK = mK.inv();
	mnThreshMinKF = mpSystem->mnThreshMinKF;
}
UVR_SLAM::LocalMapper::~LocalMapper() {}

void UVR_SLAM::LocalMapper::Init() {
	mpMap = mpSystem->mpMap;
	mpMatcher = mpSystem->mpMatcher;
	mpMapOptimizer = mpSystem->mpMapOptimizer;

	mpVisualizer = mpSystem->mpVisualizer;
	mpSegmentator = mpSystem->mpSegmentator;
	mpPlaneEstimator = mpSystem->mpPlaneEstimator;
	mpLoopCloser = mpSystem->mpLoopCloser;
	mpDepthFilter = mpSystem->mpDepthFilter;
}

void UVR_SLAM::LocalMapper::SetInitialKeyFrame(UVR_SLAM::Frame* pKF1, UVR_SLAM::Frame* pKF2) {
	mpPrevKeyFrame = pKF1;
	mpTargetFrame = pKF2;
}
void UVR_SLAM::LocalMapper::InsertKeyFrame(UVR_SLAM::Frame *pKF, bool bNeedCP, bool bNeedMP, bool bNeedPoseHandle, bool bNeedNewKF)
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	mKFQueue.push(pKF);
	//std::cout << "insertkeyframe::queue size = " << mKFQueue.size() << std::endl;
	mbStopBA = true;
	mbNeedCP = bNeedCP;
	mbNeedMP = bNeedMP;
	mbNeedPoseHandle = bNeedPoseHandle;
	mbNeedNewKF = bNeedNewKF;
	/*if (mbNeedPoseHandle){
		std::cout << "Need Pose Handler!!!::"<<pKF->mnFrameID << std::endl;
	}*/
}

bool UVR_SLAM::LocalMapper::CheckNewKeyFrames()
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	return(!mKFQueue.empty());
}

void UVR_SLAM::LocalMapper::AcquireFrame() {
	{
		std::unique_lock<std::mutex> lock(mMutexNewKFs);
		mpTempFrame = mKFQueue.front();
		mKFQueue.pop();
	}
	mpTempFrame->Init(mpSystem->mpORBExtractor, mpSystem->mK, mpSystem->mD);
}

void UVR_SLAM::LocalMapper::ProcessNewKeyFrame()
{
	{
		std::unique_lock<std::mutex> lock(mMutexNewKFs);
		mbStopBA = false;
	}
	mpPPrevKeyFrame = mpPrevKeyFrame;
	mpPrevKeyFrame = mpTargetFrame;
	mpTargetFrame = mpTempFrame;
	mpTargetFrame->mnKeyFrameID = UVR_SLAM::System::nKeyFrameID++;
}

bool  UVR_SLAM::LocalMapper::isStopLocalMapping(){
	std::unique_lock<std::mutex> lock(mMutexStopLocalMapping);
	return mbStopLocalMapping;
}
void  UVR_SLAM::LocalMapper::StopLocalMapping(bool flag){
	std::unique_lock<std::mutex> lock(mMutexStopLocalMapping);
	mbStopLocalMapping = flag;
}

bool UVR_SLAM::LocalMapper::isDoingProcess(){
	std::unique_lock<std::mutex> lock(mMutexDoingProcess);
	return mbDoingProcess;
}
void UVR_SLAM::LocalMapper::SetDoingProcess(bool flag){
	std::unique_lock<std::mutex> lock(mMutexDoingProcess);
	mbDoingProcess = flag;
}

//void UVR_SLAM::LocalMapper::InterruptLocalMapping() {
//	std::unique_lock<std::mutex> lock(mMutexNewKFs);
//	mbStopBA = true;
//}
void UVR_SLAM::LocalMapper::Reset() {
	mpTargetFrame = mpPrevKeyFrame;
	mpPrevKeyFrame = nullptr;
	mpPPrevKeyFrame = nullptr;
}
void UVR_SLAM::LocalMapper::Run() {
	
	int nMinMapPoints = 1000;

	int numActive = 0;
	float totalActive = 0.f;

	int numLM = 0;
	float totalLM = 0.f;

	while (1) {

		if (CheckNewKeyFrames()) {
			SetDoingProcess(true);
			std::chrono::high_resolution_clock::time_point lm_start = std::chrono::high_resolution_clock::now();
			
			double time1 = 0.0;
			double time2 = 0.0;

			AcquireFrame();
			bool bNeedCP, bNeedMP, bNeedNewKF, bPoseHandle;
			
			{
				std::unique_lock<std::mutex> lock(mMutexNewKFs);
				bNeedMP = mbNeedMP;
				bNeedNewKF = mbNeedNewKF;
				bNeedCP = mbNeedCP;
				bPoseHandle = mbNeedPoseHandle;
			}

			auto mpTargetMatchInfo = mpTempFrame->mpMatchInfo;
			int nTargetID = mpTempFrame->mnFrameID;
			
			////키프레임의 mean & min depth 계산
			mpTempFrame->ComputeSceneDepth();
			//mpDepthFilter->Update(mpTempFrame, mpPrevKeyFrame);

			int mnLabel_floor = 4;
			int mnLabel_ceil = 6;
			int mnLabel_wall = 1;

			if (bNeedCP) {
				/*mpTargetFrame->DetectFeature();
				mpTargetFrame->DetectEdge();
				mpTargetFrame->SetBowVec(mpSystem->fvoc);*/
				{
					std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateCP);
					std::chrono::high_resolution_clock::time_point lm_start = std::chrono::high_resolution_clock::now();
					mpTempFrame->SetGrids();
					//mpTargetFrame->mpMatchInfo->SetMatchingPoints();
					mpSystem->mbCreateCP = true;
					//std::cout << "LM::CP::" << mpTargetFrame->mnFrameID << "::" << mpTargetFrame->mpMatchInfo->mvpMatchingCPs.size() << std::endl;

					std::chrono::high_resolution_clock::time_point lm_end = std::chrono::high_resolution_clock::now();
					auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(lm_end - lm_start).count();
					float t_test1 = du_test1 / 1000.0;
					numActive++;
					totalActive += t_test1;
					lock.unlock();
					mpSystem->cvUseCreateCP.notify_all();
				}
				mpSegmentator->InsertKeyFrame(mpTempFrame);
			}
			else {
				std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateCP);
				mpSystem->mbCreateCP = true;
				////tempolar point reprojection error check
				////해당 그리드 정보도 필요함.
				{

					auto pFloorParam = mpPlaneEstimator->GetPlaneParam();
					if (pFloorParam->mbInit) {
						int nSize = mpSystem->mnRadius * 2;
						int nGridSize = mpSystem->mnRadius * 2;
						cv::Mat testImg = mpTargetFrame->GetOriginalImage().clone();
						std::vector<cv::Mat> vpTempPlaneVisPTs;
						std::vector<FrameGrid*> vpTempGrids;
						mpPlaneEstimator->GetTempPTs(vpTempGrids, vpTempPlaneVisPTs);
						cv::Mat R, t;
						mpTargetFrame->GetPose(R, t);
						cv::Scalar tempPlaneColor(255, 0, 255);
						for (size_t i = 0, iend = vpTempPlaneVisPTs.size(); i < iend; i++) {
							cv::Mat x3D = vpTempPlaneVisPTs[i];
							cv::Mat temp = mK*(R*x3D + t);
							float depth = temp.at<float>(2);
							cv::Point2f pt(temp.at<float>(0) / depth, temp.at<float>(1) / depth);
							if (!mpTargetFrame->isInImage(pt.x, pt.y, 5.0))
								continue;
							cv::Point2f ptRight(pt.x + nSize, pt.y + nSize);
							if (ptRight.x > mnWidth || ptRight.y > mnHeight) {
								continue;
							}
							auto gridPt = mpTargetFrame->GetGridBasePt(pt, nGridSize);
							auto pGrid = mpTargetFrame->mmpFrameGrids[gridPt];
							if (!pGrid) {
								pGrid = new UVR_SLAM::FrameGrid(gridPt, nGridSize);
								mpTargetFrame->mmpFrameGrids[gridPt] = pGrid;
								mpTargetFrame->mmbFrameGrids[gridPt] = false;
							}
							auto pPrevGrid = vpTempGrids[i];
							pPrevGrid->mObjCount.at<int>(mnLabel_floor)++;
							pGrid->mObjCount = pPrevGrid->mObjCount.clone();
							pGrid->mObjArea = pPrevGrid->mObjArea.clone();
							circle(testImg, pt, 3, tempPlaneColor);
						}
						cv::imshow("plane test projection", testImg); cv::waitKey(1);
					}
				}
				////tempolar point reprojection error check

				lock.unlock();
				mpSystem->cvUseCreateCP.notify_all();
			}
			float timeAP = 0.0f;
			if (numActive > 0)
				timeAP = (totalActive / numActive);
			
			if (bNeedMP || bNeedNewKF || bPoseHandle) {

				ProcessNewKeyFrame();
				mpTargetFrame->mpMatchInfo->UpdateKeyFrame();
				//NewMapPointMarginalization();
				
				////Planar Refinement
				{
					////이왕이면 현재 포인트가 속하는 프레임 정보를 미리 알고, 현재 프레임셋에서 3개 이하이면 없애기.
					////각 프레임별로 인버스 매트릭스 미리 계산해놓기

					std::vector<UVR_SLAM::MapPoint*> vpTempFloorMPs, vpTempObjectMPs;
					std::vector<MapPoint*> vpMPs;
					std::map<MapPoint*, cv::Mat> mpMapPointLabels;
					std::map<MapPoint*,int> mpMapPointCounts;
					std::map<MapPoint*, std::set<Frame*>> mpMapPointKFs;
					std::map<Frame*, std::tuple<cv::Mat, cv::Mat, cv::Mat>> mInverseInformation;
					auto pFloorParam = mpPlaneEstimator->GetPlaneParam();
					if (pFloorParam->mbInit) {
						auto spGraphKFs = mpMap->GetWindowFramesSet();
						auto vpGraphKFs = mpMap->GetWindowFramesVector();
						cv::Mat invK = mpSystem->mInvK;
						////미리 인버스 플레인값 획득하기
						
						for (auto iter = spGraphKFs.begin(), iend = spGraphKFs.end(); iter != iend; iter++) {
							auto pKFi = *iter;

							auto vpGrids = pKFi->mmpFrameGrids;
							auto vbGrids = pKFi->mmbFrameGrids;

							////평면 정보를 미리 계산해놓기
							cv::Mat Rinv, Tinv, PlaneInv;
							cv::Mat tmep;
							pKFi->GetInversePose(Rinv, Tinv);
							auto pPlaneInformation = new UVR_SLAM::PlaneProcessInformation(pKFi, pFloorParam);
							pPlaneInformation->Calculate(pFloorParam);
							pPlaneInformation->GetInformation(PlaneInv, tmep, invK);
							mInverseInformation[pKFi] = std::make_tuple(PlaneInv, Rinv, Tinv);

							for (auto iter = vpGrids.begin(), iend = vpGrids.end(); iter != iend; iter++) {
								auto pGrid = iter->second;
								auto pt = iter->first;
								if (!pGrid)
									continue;

								auto pCPi = pGrid->mpCP;
								if (!pCPi)
									continue;
								auto pMPi = pCPi->GetMP();
								bool bMP = !pMPi || pMPi->isDeleted() || !pMPi->GetQuality();
								if (bMP)
									continue;

								if (!mpMapPointCounts.count(pMPi)) {
									cv::Mat temp = cv::Mat::zeros(1, ObjectColors::mvObjectLabelColors.size(), CV_32SC1);
									mpMapPointLabels.insert(std::make_pair(pMPi, temp));
								}
								mpMapPointCounts[pMPi]++;
								mpMapPointKFs[pMPi].insert(pKFi);
								mpMapPointLabels[pMPi] += pGrid->mObjCount;

								//std::cout << mpMapPointCounts[pMPi] << " " << mpMapPointKFs[pMPi].size() << std::endl;
								//int nCountFloor = pGrid->mObjCount.at<int>(mnLabel_floor);//pGrid->mmObjCounts.count(mnLabel_floor);
								//float fWallArea = pGrid->mObjArea.at<float>(mnLabel_wall);
								//float fFloorArea = pGrid->mObjArea.at<float>(mnLabel_floor);
								//bool bFloor = nCountFloor > 0 && fFloorArea > fWallArea*5.0;
								//if (bFloor) {
								//	vpTempFloorMPs.push_back(pMPi);
								//}
								//else {
								//	vpTempObjectMPs.push_back(pMPi);
								//}
								////vpMPs.push_back(pMPi);
								//spMPs.insert(pMPi);
								////int nCountFloor = pGrid->mmObjCounts[mnLabel_floor];
							}//for grid
						}//for frame

						////평면 포인트 생성.
						int nFail = 0;
						for (auto iter = mpMapPointCounts.begin(), iend = mpMapPointCounts.end(); iter != iend; iter++) {
							auto pMPi = iter->first;
							int count = iter->second;
							/*if (count < 3){
								nFail++;
								continue;
							}*/
							cv::Mat label = mpMapPointLabels[pMPi];
							////wall, ceil, obj 분류 추가 예정
							int nCountFloor = label.at<int>(mnLabel_floor);

							cv::Point maxIdx;
							double maxVal;
							cv::minMaxLoc(label, NULL, &maxVal, NULL, &maxIdx);
							//std::cout << maxIdx <<"::"<<maxVal<<"::"<< nCountFloor << std::endl;
							if (maxIdx.x != mnLabel_floor){
								vpTempObjectMPs.push_back(pMPi);
								continue;
							}
							
							auto pKF = *(mpMapPointKFs[pMPi].begin());
							auto data = mInverseInformation[pKF];
							cv::Mat Pinv = std::get<0>(data);
							cv::Mat Rinv = std::get<1>(data);
							cv::Mat Tinv = std::get<2>(data);
							float fMaxDepth = pKF->mfMedianDepth + pKF->mfRange;

							int idx = pMPi->GetPointIndexInFrame(pKF->mpMatchInfo);
							if (idx >= 0 && count > 2) {
								cv::Mat Xw;
								auto pt = pKF->mpMatchInfo->mvMatchingPts[idx];
								bool b = PlaneInformation::CreatePlanarMapPoint(Xw, pt, Pinv, invK, Rinv, Tinv, fMaxDepth);
								if (b) {
									pMPi->SetWorldPos(Xw);
									vpTempFloorMPs.push_back(pMPi);
									//std::cout << "???????????????????22222222222222222222222" << std::endl;
									//cv::circle(testImg, pt, 3, cv::Scalar(255));
								}
							}else{
								nFail++;
							}
							/*else {
								std::cout << "???????????????????????????" << std::endl;
							}*/
						}//for mp

						////평면 포인트 보정
						//다만, 노말로 값을 줄이는게 맞는가??
						////이걸 줄이는 것부터 제대로 작성해야 함.
						cv::Mat pNormal;
						float pDist;
						pFloorParam->GetParam(pNormal, pDist);
						
						//cv::Mat invP, invT, invK2;
						//cv::Mat Rinv, Tinv;
						//mpPrevKeyFrame->GetInversePose(Rinv, Tinv);
						//float mfMaxDepth = mpPrevKeyFrame->mfMedianDepth + mpPrevKeyFrame->mfRange;

						//auto pPlaneInformation = new UVR_SLAM::PlaneProcessInformation(mpPrevKeyFrame, pFloorParam);
						//pPlaneInformation->Calculate(pFloorParam);
						//pPlaneInformation->GetInformation(invP, invT, invK2);

						//int N = 0;
						//for (size_t i = 0, iend = vpTempFloorMPs.size(); i < iend; i++) {
						//	cv::Mat s;
						//	auto pMPi = vpTempFloorMPs[i];
						//	int idx = pMPi->GetPointIndexInFrame(mpPrevKeyFrame->mpMatchInfo);
						//	if (idx >= 0) {
						//		auto pt = mpPrevKeyFrame->mpMatchInfo->mvMatchingPts[idx];
						//		bool b = PlaneInformation::CreatePlanarMapPoint(s, pt, invP, invK, Rinv, Tinv, mfMaxDepth);
						//		if (b) {
						//			N++;
						//			pMPi->SetWorldPos(s);
						//			//std::cout << "???????????????????22222222222222222222222" << std::endl;
						//			//cv::circle(testImg, pt, 3, cv::Scalar(255));
						//		}
						//	}
						//}

						std::cout << "Refinement::" <<  vpTempFloorMPs.size() <<"::"<<vpTempObjectMPs.size()<<", "<<nFail<<"::"<<mpMapPointCounts.size()<< std::endl;
						////포즈 보정
						//pNormal, pDist 삭제
						//Optimization::PlanarPoseRefinement(mpMap, vpTempFloorMPs, vpGraphKFs);
						//Optimization::ObjectPointRefinement(mpMap, vpTempObjectMPs, vpGraphKFs);
						////오브젝트 포인트 보정(포즈 고정)

					}//if plane
				}
				////Planar Refinement

				/////디버깅 이미지
				int nCreated = 0;
				cv::Mat debugMatch;
				cv::Mat prevImg = mpPrevKeyFrame->GetOriginalImage();
				cv::Mat currImg = mpTargetFrame->GetOriginalImage().clone();
				cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
				cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
				cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
				debugMatch = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
				prevImg.copyTo(debugMatch(mergeRect1));
				currImg.copyTo(debugMatch(mergeRect2));
				/////디버깅 이미지

				if (bNeedMP) {

					/*std::cout << "Optimization::start" << std::endl;
					Optimization::LocalOptimization(mpSystem, mpMap, mpTargetFrame);
					std::cout << "Optimization::end" << std::endl;*/

					////////RefKeyFrame 관련 작업
					auto mpRefKeyFrame = mpPPrevKeyFrame;
					//auto pCurrMatch = mpTargetFrame->mpMatchInfo;
					//auto pRefMatch = mpRefKeyFrame->mpMatchInfo;
					//auto vpTempCPs = pCurrMatch->mvpMatchingCPs;
					//auto vpRefCPs = pRefMatch->mvpMatchingCPs;
					//auto vpTempPTsCurr = pCurrMatch->mvMatchingPts;
					//auto vpTempPTsRef  = pRefMatch->mvMatchingPts;

					//cv::Mat Rref, Tref, Rcurr, Tcurr;
					//mpTargetFrame->GetPose(Rcurr, Tcurr);
					//mpRefKeyFrame->GetPose(Rref, Tref);

					//cv::Mat Pcurr, Pref;
					//cv::hconcat(Rcurr, Tcurr, Pcurr);
					//cv::hconcat(Rref, Tref, Pref);
					////1) 스케일 계산
					//cv::Mat Rcw = Rref.row(2);
					//Rcw = Rcw.t();
					//float zcw = Tref.at<float>(2);
					//std::vector<float> vfRefDepths;
					//float fMedianDepth, fMeanDepth, fStdDev;
					//for (size_t i = 0, iend = vpRefCPs.size(); i < iend; i++) {
					//	auto pCPi = vpRefCPs[i];
					//	auto pMPi = pCPi->GetMP();
					//	if (!pMPi || pMPi->isDeleted()) {
					//		continue;
					//	}
					//	cv::Mat x3Dw = pMPi->GetWorldPos();
					//	float z = (float)Rcw.dot(x3Dw) + zcw;
					//	vfRefDepths.push_back(z);
					//}
					////median
					//std::nth_element(vfRefDepths.begin(), vfRefDepths.begin() + vfRefDepths.size() / 2, vfRefDepths.end());
					//fMedianDepth = vfRefDepths[vfRefDepths.size() / 2];
					////mean & stddev
					//cv::Mat mMean, mDev;
					//meanStdDev(vfRefDepths, mMean, mDev);
					//fMeanDepth = (float)mMean.at<double>(0);
					//fStdDev = (float)mDev.at<double>(0);
					//float fRange = 1.15;
					////float fMaxTh = fMean + 1.15*fStdDev; //1.654		//float thresh = 1.15;//1.284;// *dStdDev + dMean; //1.654
					////float fMinTh = fMean - 1.15*fStdDev;

					////2) 연결된 포인트 찾기
					//std::vector<CandidatePoint*> vMatchCPs;
					//std::vector<cv::Point2f> vMatchRefPTs, vMatchCurrPTs;
					//for (size_t i = 0, iend = vpTempPTsCurr.size(); i < iend; i++) {
					//	auto pCPi = vpTempCPs[i];
					//	int prevIDX = pCPi->GetPointIndexInFrame(pRefMatch);
					//	if (prevIDX == -1)
					//		continue;
					//	auto currPt = vpTempPTsCurr[i];
					//	auto refPt  = vpTempPTsRef[prevIDX];
					//	vMatchRefPTs.push_back(refPt);
					//	vMatchCurrPTs.push_back(currPt);
					//	vMatchCPs.push_back(pCPi);
					//}
					//////pt 확보 과정

					std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateMP);
					//nCreated = MappingProcess(mpMap, mpTargetFrame, mpPPrevKeyFrame, time2, currImg);
					nCreated = MappingProcess(mpMap, mpTargetFrame, mpPPrevKeyFrame, mpRefKeyFrame->mfMedianDepth, mpRefKeyFrame->mfMeanDepth, mpRefKeyFrame->mfStdDev, time2, currImg);
					mpSystem->mbCreateMP = true;
					lock.unlock();
					mpSystem->cvUseCreateMP.notify_all();
					////std::cout << "LM::MP::" <<mpTargetFrame->mnFrameID<<"::"<< nCreated << std::endl;

					/*cv::Mat resized;
					cv::resize(currImg, resized, cv::Size(currImg.cols / 2, currImg.rows / 2));
					mpVisualizer->SetOutputImage(resized, 2);*/
				}
				//else if (bPoseHandle) {
				//	std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateMP);
				//	
				//	auto pCurrMatch = mpTargetFrame->mpMatchInfo;
				//	auto pPrevMatch = mpPPrevKeyFrame->mpMatchInfo;
				//	auto vpTempCPs = pCurrMatch->mvpMatchingCPs;
				//	auto vpTempPTsCurr = pCurrMatch->mvMatchingPts;
				//	auto vpTempPTsPrev = pPrevMatch->mvMatchingPts;

				//	std::vector<CandidatePoint*> vMatchCPs;
				//	std::vector<cv::Point2f> vMatchPrevPTs, vMatchCurrPTs;

				//	////pt 확보 과정
				//	for (size_t i = 0, iend = vpTempPTsCurr.size(); i < iend; i++) {
				//		auto pCPi = vpTempCPs[i];
				//		int prevIDX = pCPi->GetPointIndexInFrame(pPrevMatch);
				//		if (prevIDX == -1)
				//			continue;
				//		auto currPt = vpTempPTsCurr[i];
				//		auto prevPt = vpTempPTsPrev[prevIDX];
				//		vMatchPrevPTs.push_back(prevPt);
				//		vMatchCurrPTs.push_back(currPt);
				//		vMatchCPs.push_back(pCPi);
				//	}
				//	cv::Mat R, t;
				//	double time = 0.0;
				//	cv::Mat img1 = mpTargetFrame->GetOriginalImage().clone();
				//	cv::Mat img2 = mpPPrevKeyFrame->GetOriginalImage().clone();
				//	int a = RecoverPose(mpTargetFrame, mpPPrevKeyFrame, vMatchPrevPTs, vMatchCurrPTs, pPrevMatch->mvpMatchingCPs, R, t, time, img2, img1);
				//	std::cout << "Recover pose : " << a << std::endl;

				//	mpSystem->mbCreateMP = true;
				//	lock.unlock();
				//}

				//////평면 관련 프로세스
				////평면 레이블링
				{
					int mnLabel_floor = 4;
					int mnLabel_ceil = 6;
					int mnLabel_wall = 1;

					auto pFloorParam = mpPlaneEstimator->GetPlaneParam();
					//if (pFloorParam->mbInit) {
					//	cv::Mat testImg = mpTargetFrame->GetOriginalImage().clone();

					//	cv::Mat normal;
					//	float dist;
					//	pFloorParam->GetParam(normal, dist);
					//	//std::cout << "LM::" << normal <<", "<<dist<< std::endl;
					//	auto vpGrids = mpTargetFrame->mmpFrameGrids;
					//	auto vbGrids = mpTargetFrame->mmbFrameGrids;

					//	for (auto iter = vpGrids.begin(), iend = vpGrids.end(); iter != iend; iter++) {
					//		auto pGrid = iter->second;
					//		auto pt = iter->first;
					//		if (!pGrid)
					//			continue;
					//		
					//		auto pCPi = pGrid->mpCP;
					//		if (!pCPi)
					//			continue;
					//		auto pMPi = pCPi->GetMP();
					//		bool bMP = !pMPi || pMPi->isDeleted() || !pMPi->GetQuality();
					//		if (bMP)
					//			continue;
					//		
					//		cv::Mat X3D = pMPi->GetWorldPos();
					//		float val = abs(normal.dot(X3D) + dist);

					//		float fWallArea = pGrid->mObjArea.at<float>(mnLabel_wall);
					//		float fFloorArea = pGrid->mObjArea.at<float>(mnLabel_floor);

					//		int count = pGrid->mObjCount.at<int>(mnLabel_floor);
					//		if (count > 0 && fFloorArea > fWallArea) {
					//			cv::circle(testImg, pGrid->pt, 4, cv::Scalar(0, 255, 255), -1);
					//		}
					//		if (val < 0.02) {
					//			pGrid->mObjCount.at<int>(mnLabel_floor)++;
					//			cv::circle(testImg, pGrid->pt, 2, cv::Scalar(255, 0, 255), -1);
					//			count++;
					//		}
					//		else {
					//			cv::circle(testImg, pGrid->pt, 2, cv::Scalar(255, 255, 0), -1);
					//		}
					//		
					//	}//for iter grid
					//	cv::imshow("plane test", testImg); cv::waitKey(1);
					//}//if

					
				}
				//////평면 관련 프로세스

				auto pTarget = mpMap->AddWindowFrame(mpTargetFrame);
				//if (pTarget) {
				//	mpSegmentator->InsertKeyFrame(pTarget);

				//	//mpPlaneEstimator->InsertKeyFrame(pTarget);
				//	//mpLoopCloser->InsertKeyFrame(pTarget);
				//}

				if (mpMapOptimizer->isDoingProcess()) {
					//std::cout << "lm::ba::busy" << std::endl;
					mpMapOptimizer->StopBA(true);
				}
				else {
					mpMapOptimizer->InsertKeyFrame(mpTargetFrame);
				}
			}
			
			/////프레임 퀄리티 계산
			//bool bLowQualityFrame = mpTargetFrame->mpMatchInfo->UpdateFrameQuality();
			//
			///////프레임 퀄리티 계산
			///////////KF-KF 매칭
			//std::vector<cv::Point2f> vOpticalMatchPPrevPts, vOpticalMatchPrevPts, vOpticalMatchCurrPts;
			//std::vector<CandidatePoint*> vOpticalMatchCPs;
			////int nMatch = mpMatcher->OpticalMatchingForMapping(mpMap, mpTargetFrame, mpPrevKeyFrame, vOpticalMatchPrevPts, vOpticalMatchCurrPts, vOpticalMatchCPs, mK, mInvK, time1, debugMatch);
			//int nMatch = mpMatcher->OpticalMatchingForMapping(mpMap, mpTargetFrame, mpPrevKeyFrame, mpPPrevKeyFrame, vOpticalMatchPPrevPts, vOpticalMatchPrevPts, vOpticalMatchCurrPts, vOpticalMatchCPs, mK, mInvK, time1, debugMatch);
			//mpTargetFrame->mpMatchInfo->ConnectAll();
			////NewMapPointMarginalization();

			//std::vector<cv::Point2f> vMappingPPrevPts, vMappingPrevPts, vMappingCurrPts;
			//std::vector<CandidatePoint*> vMappingCPs;
			//int nMapping = MappingProcess(mpMap, mpTargetFrame, mpPrevKeyFrame, vMappingPrevPts, vMappingCurrPts, vMappingCPs, vOpticalMatchPrevPts, vOpticalMatchCurrPts, vOpticalMatchCPs, time2, debugMatch);
			//			
			////std::cout << "LM::MappingProcess::" << nMapping <<", new = "<< nMapping << " , optical = " << nMatch << std::endl;
			//cv::Mat resized;
			//cv::resize(debugMatch, resized, cv::Size(debugMatch.cols / 2, debugMatch.rows / 2));
			//mpVisualizer->SetOutputImage(resized, 3);

			/////Create Map Points
			/////////KF-KF 매칭
			
			std::chrono::high_resolution_clock::time_point lm_end = std::chrono::high_resolution_clock::now();
			auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(lm_end - lm_start).count();
			float t_test1 = du_test1 / 1000.0;
			numLM++;
			totalLM += t_test1;
			
			std::stringstream ssa;
			ssa << "LocalMapping : " << mpTargetFrame->mnKeyFrameID << "::" << (totalLM / numLM) <<"::"<< timeAP << "::" << "::" << 0 <<", "<< time1 << ", " << time2 << std::endl;;// << ", " << nMinKF << ", " << nMaxKF;
			mpSystem->SetLocalMapperString(ssa.str());
			
			SetDoingProcess(false);
			continue;
			//////200412
		}
	}//while
}

//맵포인트가 삭제 되면 현재 프레임에서도 해당 맵포인트를 삭제 해야 하며, 
//이게 수행되기 전에는 트래킹이 동작하지 않도록 막아야 함.
//
void UVR_SLAM::LocalMapper::NewMapPointMarginalization() {
	//std::cout << "Maginalization::Start" << std::endl;
	//mvpDeletedMPs.clear();
	int nMarginalized = 0;
	int nNumRequireKF = mnThreshMinKF;
	float mfRatio = 0.5f;

	std::list<UVR_SLAM::MapPoint*>::iterator lit = mpSystem->mlpNewMPs.begin();
	while (lit != mpSystem->mlpNewMPs.end()) {
		UVR_SLAM::MapPoint* pMP = *lit;
		int nDiffKF = mpTargetFrame->mnKeyFrameID - pMP->mnFirstKeyFrameID;
		bool bBad = false;
		if (pMP->isDeleted()) {
			//already deleted
			lit = mpSystem->mlpNewMPs.erase(lit);
		}
		else if (pMP->GetFVRatio() < mfRatio) {
			bBad = true;
			lit = mpSystem->mlpNewMPs.erase(lit);
		}
		else if (nDiffKF < nNumRequireKF && pMP->GetNumConnectedFrames()-mnThreshMinKF+1 < nDiffKF) {
			bBad = true;
			lit = mpSystem->mlpNewMPs.erase(lit);
		}
		else if (nDiffKF >= nNumRequireKF) {
			lit = mpSystem->mlpNewMPs.erase(lit);
			pMP->SetNewMP(false);
		}
		else
			lit++;
		if (bBad) {
			pMP->Delete();
		}
	}
	return;
}
int UVR_SLAM::LocalMapper::RecoverPose(Frame* pCurrKF, Frame* pPrevKF, std::vector<cv::Point2f> vMatchPrevPts, std::vector<cv::Point2f> vMatchCurrPts, std::vector<CandidatePoint*> vPrevCPs, cv::Mat& R, cv::Mat& T, double& ftime, cv::Mat& prevImg, cv::Mat& currImg) {
	
	//Find fundamental matrix & matching
	std::vector<uchar> vFInliers;
	std::vector<cv::Point2f> vTempFundPrevPts, vTempFundCurrPts;
	std::vector<int> vTempMatchIDXs;
	cv::Mat E12 = cv::findEssentialMat(vMatchPrevPts, vMatchCurrPts, mK, cv::FM_RANSAC, 0.999, 1.0, vFInliers);
	for (unsigned long i = 0; i < vFInliers.size(); i++) {
		if (vFInliers[i]) {
			vTempFundPrevPts.push_back(vMatchPrevPts[i]);
			vTempFundCurrPts.push_back(vMatchCurrPts[i]);
			vTempMatchIDXs.push_back(i);//vTempIndexs[i]
		}
	}
	
	////////F, E를 통한 매칭 결과 반영
	/////////삼각화 : OpenCV
	cv::Mat matTriangulateInliers;
	cv::Mat Map3D;
	cv::Mat K;
	mK.convertTo(K, CV_64FC1);
	int res2 = cv::recoverPose(E12, vTempFundPrevPts, vTempFundCurrPts, mK, R, T, 50.0, matTriangulateInliers, Map3D);
	R.convertTo(R, CV_32FC1);
	T.convertTo(T, CV_32FC1);
	Map3D.convertTo(Map3D, CV_32FC1);

	cv::Mat Rcurr, Tcurr;
	pCurrKF->GetPose(Rcurr, Tcurr);
	cv::Mat Rprev, Tprev;
	pPrevKF->GetPose(Rprev, Tprev);
	std::vector<float> vScales;
	float sumScale = 0.0;
	std::vector<float> vPrevScales;
	float meanPrevScale = 0.0;

	std::vector<CandidatePoint*> vpTempCPs;
	std::vector<cv::Mat> vX3Ds;

	cv::Mat Rinv = Rprev.t();
	cv::Mat Tinv = -Rinv*Tprev;

	for (int i = 0; i < matTriangulateInliers.rows; i++) {
		int val = matTriangulateInliers.at<uchar>(i);
		int idx = vTempMatchIDXs[i]; //cp idx
		if (val == 0)
			continue;

		cv::Mat X3D = Map3D.col(i).clone();
		//if (abs(X3D.at<float>(3)) < 0.0001) {
		//	/*std::cout << "test::" << X3D.at<float>(3) << std::endl;
		//	cv::circle(debugging, currPt + ptBottom, 2, cv::Scalar(0, 0, 0), -1);
		//	cv::circle(debugging, prevPt, 2, cv::Scalar(0, 0, 0), -1);*/
		//	continue;
		//}
		X3D /= X3D.at<float>(3);
		X3D = X3D.rowRange(0, 3);

		auto currPt = vTempFundCurrPts[i];
		auto prevPt = vTempFundPrevPts[i];

		////reprojection error
		cv::Mat proj1 = X3D.clone();
		cv::Mat proj2 = R*X3D + T;
		proj1 = mK*proj1;
		proj2 = mK*proj2;
		float depth1 = proj1.at<float>(2);
		float depth2 = proj2.at<float>(2);
		cv::Point2f projected1(proj1.at<float>(0) / depth1, proj1.at<float>(1) / depth1);
		cv::Point2f projected2(proj2.at<float>(0) / depth2, proj2.at<float>(1) / depth2);

		auto diffPt1 = projected1 - prevPt;
		auto diffPt2 = projected2 - currPt;
		float err1 = (diffPt1.dot(diffPt1));
		float err2 = (diffPt2.dot(diffPt2));
		
		cv::circle(currImg, currPt, 3, cv::Scalar(0, 255, 0),-1);
		cv::circle(prevImg, prevPt, 3, cv::Scalar(0, 255, 0), -1);
		cv::line(prevImg, prevPt, projected1, cv::Scalar(255, 0, 0));
		cv::line(currImg, currPt, projected2, cv::Scalar(255, 0, 0));
		
		if (err1 > 9.0 || err2 > 9.0) {
			continue;
		}
		////reprojection error

		//scale 계산
		
		auto pCPi = vPrevCPs[idx];
		auto pMPi = pCPi->GetMP();

		vpTempCPs.push_back(pCPi);
		vX3Ds.push_back(proj1);

		if (pMPi) {
			cv::Mat Xw = pMPi->GetWorldPos();
			if (pPrevKF && pMPi->isInFrame(pPrevKF->mpMatchInfo))
			{
				cv::Mat proj3 = Rprev*Xw +Tprev;
				//proj3 = mK*proj3;
				float depth3 = proj3.at<float>(2);
				float scale = depth3 / depth1;
				vPrevScales.push_back(scale);
				meanPrevScale += scale;

				/*cv::Mat Temp = mInvK*proj1*scale;
				Temp = Rinv*(Temp)+Tinv;
				std::cout << Temp.t() << ", " << Xw.t() << std::endl;*/
			}
			//if (!pMPi->isInFrame(pCurrKF->mpMatchInfo))
			//	continue;
			//cv::Mat proj3 = Rcurr*Xw + Tcurr;
			//proj3 = mK*proj3;
			//float depth3 = proj3.at<float>(2);
			//float scale = depth3 / depth2;
			//vScales.push_back(scale);
			//sumScale += scale;

			//cv::Mat Temp = mInvK*proj2*scale;//X3D*scale;
			//cv::Mat Temp2 = mInvK*proj3;
			//cv::Mat T2 = T*scale;
			//std::cout << scale<<"::"<<Temp.t()<<", "<< Temp2.t()<< std::endl;
			////std::cout << scale<<"::"<<T2.t() << ", " << Tcurr.t() << std::endl;
		}
	}
	////scale

	if (vPrevScales.size() < 10)
		return -1;

	/*float meanScalae = sumScale / vScales.size();
	int nidx = vScales.size() / 2;
	std::nth_element(vScales.begin(), vScales.begin() + nidx, vScales.end());
	float medianScale = vScales[(nidx)];*/

	std::nth_element(vPrevScales.begin(), vPrevScales.begin()+ vPrevScales.size()/2, vPrevScales.end());
	float medianPrevScale = vPrevScales[vPrevScales.size() / 2];
	cv::Mat scaled = R*Tprev+T*medianPrevScale;
	//Map3D *= medianPrevScale;
	std::cout << "RecoverPose = "<< vpTempCPs .size()<<":: scale : "  <<"||"<<medianPrevScale<< "::" << scaled.t() << ", " << Tcurr.t() << std::endl;

	//포즈 변경
	R = R*Rprev;
	T = scaled;
	mpTargetFrame->GetPose(R, T);
	mpMap->ClearReinit();
	std::vector<bool> vbInliers(vX3Ds.size(), true);
	std::vector<bool> vbInliers2(vX3Ds.size(), true);
	//Optimization::LocalOptimization(mpSystem, mpMap, pCurrKF, vX3Ds, vpTempCPs, vbInliers, vbInliers2, 1.0);
	//MP 생성
	
	//for (int i = 0; i < vpTempCPs.size(); i++) {
	//	auto pCPi = vpTempCPs[i];
	//	cv::Mat X3D = mInvK*vX3Ds[i]* medianPrevScale;
	//	X3D = Rinv*(X3D) + Tinv;
	//	
	//	//MP fuse나 replace 함수가 필요해짐. 아니면, world pos만 변경하던가
	//	//빈곳만 채우던가
	//	auto pMPi = pCPi->GetMP();
	//	if (pMPi) {
	//		pMPi->SetWorldPos(X3D);
	//	}
	//	else {
	//		int label = pCPi->GetLabel();
	//		auto pMP = new UVR_SLAM::MapPoint(mpMap, mpTargetFrame, pCPi, X3D, cv::Mat(), label, pCPi->octave);
	//		//여기서 모든 CP 다 연결하기?
	//		auto mmpFrames = pCPi->GetFrames();
	//		for (auto iter = mmpFrames.begin(); iter != mmpFrames.end(); iter++) {
	//			auto pMatch = iter->first;
	//			if (pMatch->mpRefFrame->mnKeyFrameID % 3 != 0)
	//				continue;
	//			int idx = iter->second;
	//			pMP->ConnectFrame(pMatch, idx);
	//		}
	//		/*pMP->AddFrame(pCurrKF->mpMatchInfo, pt1);
	//		pMP->AddFrame(pPrevKF->mpMatchInfo, pt2);*/
	//	}
	//}

	return res2;
}

int UVR_SLAM::LocalMapper::RecoverPose(Frame* pCurrKF, Frame* pPrevKF, Frame* pPPrevKF, std::vector<cv::Point2f> vCurrPts, std::vector<cv::Point2f> vPrevPts, std::vector<cv::Point2f> vPPrevPts, std::vector<CandidatePoint*> vpCPs, std::vector<bool>& vbInliers, cv::Mat& R, cv::Mat& T, double& ftime,
	cv::Mat& currImg, cv::Mat& prevImg, cv::Mat& pprevImg) {

	//Find fundamental matrix & matching
	std::vector<uchar> vFInliers;
	std::vector<cv::Point2f> vTempFundPPrevPts, vTempFundPrevPts, vTempFundCurrPts;
	std::vector<int> vTempMatchIDXs;
	cv::Mat E12 = cv::findEssentialMat(vPrevPts, vCurrPts, mK, cv::FM_RANSAC, 0.999, 1.0, vFInliers);
	for (unsigned long i = 0; i < vFInliers.size(); i++) {
		if (vFInliers[i]) {
			vTempFundCurrPts.push_back(std::move(vCurrPts[i]));
			vTempFundPrevPts.push_back(std::move(vPrevPts[i]));
			vTempFundPPrevPts.push_back(std::move(vPPrevPts[i]));
			vTempMatchIDXs.push_back(std::move(i));//vTempIndexs[i]
		}
	}
	
	if (vTempMatchIDXs.size() < 200)
		return -1;
	////////F, E를 통한 매칭 결과 반영
	/////////삼각화 : OpenCV
	cv::Mat matTriangulateInliers;
	cv::Mat Map3D;
	cv::Mat K;
	mK.convertTo(K, CV_64FC1);

	int res2 = cv::recoverPose(E12, vTempFundPrevPts, vTempFundCurrPts, mK, R, T, 50.0, matTriangulateInliers, Map3D);
	if (countNonZero(matTriangulateInliers) < 100)
		return -1;
	R.convertTo(R, CV_32FC1);
	T.convertTo(T, CV_32FC1);
	Map3D.convertTo(Map3D, CV_32FC1);

	cv::Mat Rcurr, Tcurr;
	pCurrKF->GetPose(Rcurr, Tcurr);
	cv::Mat Rprev, Tprev;
	pPrevKF->GetPose(Rprev, Tprev);
	cv::Mat Rpprev, Tpprev;
	pPPrevKF->GetPose(Rpprev, Tpprev);

	cv::Mat Rpinv = Rprev.t();
	cv::Mat Tpinv = -Rpinv*Tprev;

	//Tprev ->Tcurr로 가는 변환 매트릭스, T를 이용하여 스케일을 전환
	cv::Mat Rdiff = Rcurr*Rpinv;
	cv::Mat Tdiff = Rcurr*Tpinv + Tcurr;
	float scale = sqrt(Tdiff.dot(Tdiff));
	
	/////Optimize용
	std::vector<cv::Point2f> vMapCurrPTs, vMapPrevPTs, vMapPPrevPTs;
	std::vector<CandidatePoint*> vMapCPs;
	std::vector<cv::Mat> vX3Ds;

	/////TEST CODE
	std::vector<float> vPrevScales;
	mpMap->ClearReinit();
	int nTest = 0;

	for (int i = 0; i < matTriangulateInliers.rows; i++) {
		int val = matTriangulateInliers.at<uchar>(i);
		if (val == 0)
			continue;
		int idx = vTempMatchIDXs[i]; //cp idx
		cv::Mat X3D = Map3D.col(i).clone();
		X3D /= X3D.at<float>(3);
		X3D = X3D.rowRange(0, 3);

		auto currPt = std::move(vTempFundCurrPts[i]);
		auto prevPt = std::move(vTempFundPrevPts[i]);
		auto pprevPt = std::move(vTempFundPPrevPts[i]);

		////reprojection error
		cv::Mat proj1 = X3D.clone();
		cv::Mat proj2 = R*X3D + T;
		proj1 = mK*proj1;
		proj2 = mK*proj2;
		float depth1 = proj1.at<float>(2);
		float depth2 = proj2.at<float>(2);
		cv::Point2f projected1(proj1.at<float>(0) / depth1, proj1.at<float>(1) / depth1);
		cv::Point2f projected2(proj2.at<float>(0) / depth2, proj2.at<float>(1) / depth2);

		auto diffPt1 = projected1 - prevPt;
		auto diffPt2 = projected2 - currPt;
		float err1 = (diffPt1.dot(diffPt1));
		float err2 = (diffPt2.dot(diffPt2));

		if (err1 > 9.0 || err2 > 9.0) {
			continue;
		}
		
		////Xscaled 에 대해서 reprojection test
		////처리는 카메라 좌표계가지 변환 후 다시 해야 함.
		cv::Mat Xscaled = Rpinv*(X3D*scale) + Tpinv;//proj1*scale;
		//mpMap->AddReinit(Xscaled);

		////////최적화를 위한 추가
		vMapCPs.push_back(vpCPs[idx]);
		vMapCurrPTs.push_back(std::move(currPt));
		vMapPrevPTs.push_back(std::move(prevPt));
		vMapPPrevPTs.push_back(std::move(pprevPt));
		vX3Ds.push_back(std::move(Xscaled));
		////////시각화
		//cv::Mat newProj1 = Rprev*Xscaled + Tprev;
		//newProj1 = mK*newProj1;
		//float newDepth1 = newProj1.at<float>(2);
		//cv::Point2f newProjected1(newProj1.at<float>(0) / newDepth1, newProj1.at<float>(1) / newDepth1);
		//cv::circle(prevImg, newProjected1, 3, cv::Scalar(255, 0, 0), -1);

		//cv::Mat newProj2 = Rcurr*Xscaled + Tcurr;
		//newProj2 = mK*newProj2;
		//float newDepth2 = newProj2.at<float>(2);
		//cv::Point2f newProjected2(newProj2.at<float>(0) / newDepth2, newProj2.at<float>(1) / newDepth2);
		//cv::circle(currImg, newProjected2, 3, cv::Scalar(255, 0, 0), -1);
		//
		//cv::Mat newProj3 = Rpprev*Xscaled + Tpprev;
		//newProj3 = mK*newProj3;
		//float newDepth3 = newProj3.at<float>(2);
		//cv::Point2f newProjected3(newProj3.at<float>(0) / newDepth3, newProj3.at<float>(1) / newDepth3);
		//cv::circle(pprevImg, newProjected3, 3, cv::Scalar(255, 0, 0), -1);
		////////시각화
		nTest++;
	}

	UVR_SLAM::Optimization::PoseRecoveryOptimization(pCurrKF, pPrevKF, pPPrevKF, vMapCurrPTs, vMapPrevPTs, vMapPPrevPTs, vX3Ds);
	pCurrKF->GetPose(Rcurr, Tcurr);

	int nMP = 0;
	float nSuccess = 0;

	/////시각화 확인
	for (int i = 0; i < vX3Ds.size(); i++) {
		cv::Mat X3D = vX3Ds[i];
		mpMap->AddReinit(X3D);

		////////시각화
		cv::Mat newProj1 = Rcurr*X3D + Tcurr;
		newProj1 = mK*newProj1;
		float newDepth1 = newProj1.at<float>(2);
		cv::Point2f newProjected1(newProj1.at<float>(0) / newDepth1, newProj1.at<float>(1) / newDepth1);
		cv::circle(currImg, newProjected1, 3, cv::Scalar(255, 0, 0), -1);
		
		cv::Mat newProj2 = Rprev*X3D + Tprev;
		newProj2 = mK*newProj2;
		float newDepth2 = newProj2.at<float>(2);
		cv::Point2f newProjected2(newProj2.at<float>(0) / newDepth2, newProj2.at<float>(1) / newDepth2);
		cv::circle(prevImg, newProjected2, 3, cv::Scalar(255, 0, 0), -1);
		
		cv::Mat newProj3 = Rpprev*X3D + Tpprev;
		newProj3 = mK*newProj3;
		float newDepth3 = newProj3.at<float>(2);
		cv::Point2f newProjected3(newProj3.at<float>(0) / newDepth3, newProj3.at<float>(1) / newDepth3);
		cv::circle(pprevImg, newProjected3, 3, cv::Scalar(255, 0, 0), -1);

		auto pCPi = vMapCPs[i];
		auto pMPi = pCPi->GetMP();
		if (pMPi && !pMPi->isDeleted()) {
			nMP++;
			cv::Mat Xw = pMPi->GetWorldPos();
			pMPi->SetWorldPos(X3D);
			{
				cv::Mat proj = Rcurr*Xw + Tcurr;
				proj = mK*proj;
				float depth = proj.at<float>(2);
				//std::cout << "diff::" << newDepth2 - depth <<"::"<<X3D.t()<<Xw.t()<< std::endl;
				cv::Point2f projPt(proj.at<float>(0) / depth, proj.at<float>(1) / depth);
				cv::line(currImg, projPt, newProjected1, cv::Scalar(0, 0, 255), 2);
			}
			
			{
				cv::Mat proj = Rprev*Xw + Tprev;
				proj = mK*proj;
				float depth = proj.at<float>(2);
				//std::cout << "diff::" << newDepth2 - depth <<"::"<<X3D.t()<<Xw.t()<< std::endl;
				cv::Point2f projPt(proj.at<float>(0) / depth, proj.at<float>(1) / depth);
				if (pMPi->isInFrame(pPrevKF->mpMatchInfo)){
					cv::line(prevImg, projPt, newProjected2, cv::Scalar(0, 0, 255), 2);
				}
				else
					cv::circle(prevImg, projPt, 2, cv::Scalar(0, 255, 0), -1);
			}
			
			{
				cv::Mat proj = Rpprev*Xw + Tpprev;
				proj = mK*proj;
				float depth = proj.at<float>(2);
				//std::cout << "diff::" << newDepth2 - depth <<"::"<<X3D.t()<<Xw.t()<< std::endl;
				cv::Point2f projPt(proj.at<float>(0) / depth, proj.at<float>(1) / depth);
				if (pMPi->isInFrame(pPPrevKF->mpMatchInfo)){
					cv::line(pprevImg, projPt, newProjected3, cv::Scalar(0, 0, 255), 2);
				}
				else
					cv::circle(pprevImg, projPt, 2, cv::Scalar(0, 255, 0), -1);
			}
		}
		else {
			////new mp test
			int label = pCPi->GetLabel();
			auto pMP = new UVR_SLAM::MapPoint(mpMap, mpTargetFrame, pCPi, X3D, cv::Mat(), label, pCPi->octave);
			auto mmpFrames = pCPi->GetFrames();
			for (auto iter = mmpFrames.begin(); iter != mmpFrames.end(); iter++) {
				auto pMatch = iter->first;
				if (pMatch->mpRefFrame->mnKeyFrameID % 3 != 0) {
					continue;
				}
				int idx = iter->second;
				//pMatch->AddMP(pMP, idx);
				pMP->ConnectFrame(pMatch, idx);
			}
		}
		////////시각화
	}

	/*if (vPrevScales.size() < 10)
		return -1;*/
	//////메디안 스케일 계산
	//std::nth_element(vPrevScales.begin(), vPrevScales.begin() + vPrevScales.size() / 2, vPrevScales.end());
	//float medianPrevScale = vPrevScales[vPrevScales.size() / 2];

	//////스케일 보정
	//for (int i = 0; i < vX3Ds.size(); i++) {
	//	////처리는 카메라 좌표계가지 변환 후 다시 해야 함.
	//	cv::Mat Xscaled = Rinv*(vX3Ds[i]* medianPrevScale) + Tinv;//proj1*scale;
	//	mpMap->AddReinit(Xscaled);

	//	//Xscaled 에 대해서 reprojection test
	//	cv::Mat newProj1 = Rprev*Xscaled + Tprev;
	//	newProj1 = mK*newProj1;
	//	float newDepth1 = newProj1.at<float>(2);
	//	cv::Point2f newProjected1(newProj1.at<float>(0) / newDepth1, newProj1.at<float>(1) / newDepth1);
	//	cv::circle(prevImg, newProjected1, 2, cv::Scalar(255, 0, 0), -1);

	//	cv::Mat newProj2 = Rcurr*Xscaled + Tcurr;
	//	newProj2 = mK*newProj2;
	//	float newDepth2 = newProj2.at<float>(2);
	//	cv::Point2f newProjected2(newProj2.at<float>(0) / newDepth2, newProj2.at<float>(1) / newDepth2);
	//	cv::circle(currImg, newProjected2, 2, cv::Scalar(255, 0, 0), -1);
	//	//Xscaled 에 대해서 reprojection test
	//	//시각화
	//	
	//}
	//std::cout << "recover pose::candidate points::" << nTest << std::endl;
	imshow("recover::1", currImg);
	imshow("recover::2", prevImg);
	imshow("recover::3", pprevImg);
	cv::waitKey(1);

}
////////////200722 수정 필요
int UVR_SLAM::LocalMapper::MappingProcess(Map* pMap, Frame* pCurrKF, Frame* pPrevKF, double& dtime, cv::Mat& debugging) {
	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	int N;
	auto pCurrMatch = pCurrKF->mpMatchInfo;
	auto pPrevMatch = pPrevKF->mpMatchInfo;
	auto vpTempCPs = pCurrMatch->mvpMatchingCPs;
	auto vpTempPTsCurr = pCurrMatch->mvMatchingPts;
	auto vpTempPTsPrev = pPrevKF->mpMatchInfo->mvMatchingPts;

	cv::Mat Rprev, Tprev, Rcurr, Tcurr;
	pCurrKF->GetPose(Rcurr, Tcurr);
	pPrevKF->GetPose(Rprev, Tprev);

	cv::Mat Pcurr, Pprev;
	cv::hconcat(Rcurr, Tcurr, Pcurr);
	cv::hconcat(Rprev, Tprev, Pprev);

	std::vector<CandidatePoint*> vMatchCPs;
	std::vector<cv::Point2f> vMatchPrevPTs, vMatchCurrPTs;

	////뎁스 필터 1픽셀 에러 관련
	float fx = mpSystem->mK.at<float>(0, 0);
	float noise = 1.0;
	float px_err_angle = atan(noise / (2.0*fx))*2.0;
	////뎁스 필터 1픽셀 에러 관련

	//////////테스트용도
	cv::Mat debugMatch;
	cv::Mat prevImg = pPrevKF->GetOriginalImage().clone();
	cv::Mat currImg = pCurrKF->GetOriginalImage().clone();
	cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
	cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
	debugMatch = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());

	//////////테스트용도

	std::cout << "MappingProcess::" << pCurrKF->mpMatchInfo->mvpMatchingCPs.size() << ", " << vpTempCPs.size() << ", " << vpTempPTsCurr.size() << std::endl;

	////pt 확보 과정
	for (size_t i = 0, iend = vpTempPTsCurr.size(); i < iend; i++) {
		auto pCPi = vpTempCPs[i];
		int prevIDX = pCPi->GetPointIndexInFrame(pPrevMatch);
		if (prevIDX == -1)
			continue;
		if (pCPi->GetNumSize() < mnThreshMinKF)
			continue;
		auto pMPi = pCPi->GetMP();
		bool bOldMP = pMPi && pMPi->GetQuality() && !pMPi->isDeleted();
		if (bOldMP)
			continue;
		auto currPt = vpTempPTsCurr[i];
		auto prevPt = vpTempPTsPrev[prevIDX];
		vMatchPrevPTs.push_back(prevPt);
		vMatchCurrPTs.push_back(currPt);
		vMatchCPs.push_back(pCPi);

		//cv::line(debugMatch, prevPt, currPt+ptBottom, cv::Scalar(255, 0, 255), 2);
		cv::circle(prevImg, prevPt, 4, cv::Scalar(255, 0, 0), 1);
		cv::circle(currImg, currPt, 4, cv::Scalar(255, 0, 0), 1);
	}

	if (vMatchCPs.size() < 10) {
		std::cout << "포인트 부족 000" << std::endl;
		//return -1;
	}

	cv::Mat TempMap;
	cv::triangulatePoints(mK*Pprev, mK*Pcurr, vMatchPrevPTs, vMatchCurrPTs, TempMap);

	cv::Scalar color1(255, 0, 0);
	cv::Scalar color2(255, 255, 0);
	cv::Scalar color3(0, 0, 255);
	cv::Scalar color4(0, 255, 255);
	cv::Scalar color5(0, 255, 0);

	float fMaxTH1 = mpTargetFrame->mfMedianDepth + mpTargetFrame->mfRange;// fMeanDepth + fRange * fStdDev;
	float fMaxTH2 = pPrevKF->mfMedianDepth + pPrevKF->mfRange;// fMeanDepth + fRange * fStdDev;

	std::vector<CandidatePoint*> vMappingCPs;
	std::vector<cv::Point2f> vPTs;
	std::vector<cv::Mat> vX3Ds;

	std::vector<float> vfScales;
	float thresh = 9.0;
	for (size_t i = 0, iend = TempMap.cols; i < iend; i++) {

		////Seed 확인이 필요. 초기화 & 업데이트를 여기서 수행하고 확인도 여기서 하기.

		auto currPt = std::move(vMatchCurrPTs[i]);
		auto prevPt = std::move(vMatchPrevPTs[i]);
		auto pCPi = std::move(vMatchCPs[i]);
		

		cv::Mat X3D;
		float depth;
		bool bNewMP = true;
		float depth1, depth2;
		{
			X3D = std::move(TempMap.col(i));
			if (abs(X3D.at<float>(3)) < 0.0001) {
				bNewMP = false;
				continue;
			}
			X3D /= X3D.at<float>(3);
			X3D = X3D.rowRange(0, 3);
			//New MP 조정

			cv::Mat proj1 = Rcurr*X3D + Tcurr;
			cv::Mat proj2 = Rprev*X3D + Tprev;

			depth1 = proj1.at<float>(2);
			depth2 = proj2.at<float>(2);

			if (depth1  < 0.0 || depth2 < 0.0) {
				bNewMP = false;
				continue;
			}
			if (depth1 > fMaxTH1 || depth2 > fMaxTH2) {
				bNewMP = false;
				continue;
			}
			////depth test


			////reprojection error
			proj1 = mK*proj1;
			proj2 = mK*proj2;
			cv::Point2f projected1(proj1.at<float>(0) / proj1.at<float>(2), proj1.at<float>(1) / proj1.at<float>(2));
			cv::Point2f projected2(proj2.at<float>(0) / proj2.at<float>(2), proj2.at<float>(1) / proj2.at<float>(2));

			auto diffPt1 = projected1 - currPt;
			auto diffPt2 = projected2 - prevPt;
			float err1 = (diffPt1.dot(diffPt1));
			float err2 = (diffPt2.dot(diffPt2));
			if (err1 > thresh || err2 > thresh) {
				bNewMP = false;
				continue;
			}
		}
		vMappingCPs.push_back(pCPi);
		vX3Ds.push_back(X3D);
		vPTs.push_back(currPt);
	}

	////////////////////////////////////////최적화 진행
	if (vX3Ds.size() < 30) {
		std::cout << "포인트 부족11" << std::endl;
		//return -1;
	}

	mpMap->ClearReinit();
	std::vector<bool> vbInliers(vX3Ds.size(), true);
	Optimization::LocalOptimization(mpSystem, mpMap, pCurrKF, vX3Ds, vMappingCPs, vbInliers);

	int nFail = vX3Ds.size();
	for (size_t i = 0, iend = vX3Ds.size(); i < iend; i++) {
		if (!vbInliers[i]) {
			nFail--;
			cv::circle(debugging, vPTs[i], 3, color3, -1);
		}
	}
	/*if (nFail < 50) {
		std::cout << "Map Creation Fail case::" << pCurrKF->mnFrameID << std::endl;
	}*/

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	dtime = duration / 1000.0;
	return vX3Ds.size() - nFail;

}
int UVR_SLAM::LocalMapper::MappingProcess(Map* pMap, Frame* pCurrKF, Frame* pPrevKF, float fMedianDepth, float fMeanDepth, float fStdDev, double& dtime, cv::Mat& debugging) {
	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();
	int N;
	auto pCurrMatch = pCurrKF->mpMatchInfo;
	auto pPrevMatch = pPrevKF->mpMatchInfo;
	auto vpTempCPs = pCurrMatch->mvpMatchingCPs;
	auto vpTempPTsCurr = pCurrMatch->mvMatchingPts;
	auto vpTempPTsPrev = pPrevKF->mpMatchInfo->mvMatchingPts;

	cv::Mat Rprev, Tprev, Rcurr, Tcurr;
	pCurrKF->GetPose(Rcurr, Tcurr);
	pPrevKF->GetPose(Rprev, Tprev);

	cv::Mat Pcurr, Pprev;
	cv::hconcat(Rcurr, Tcurr, Pcurr);
	cv::hconcat(Rprev, Tprev, Pprev);

	std::vector<CandidatePoint*> vMatchCPs;
	std::vector<cv::Point2f> vMatchPrevPTs, vMatchCurrPTs;

	////뎁스 필터 1픽셀 에러 관련
	float fx = mpSystem->mK.at<float>(0, 0);
	float noise = 1.0;
	float px_err_angle = atan(noise / (2.0*fx))*2.0;
	////뎁스 필터 1픽셀 에러 관련

	//////////테스트용도
	cv::Mat debugMatch;
	cv::Mat prevImg = pPrevKF->GetOriginalImage().clone();
	cv::Mat currImg = pCurrKF->GetOriginalImage().clone();
	cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
	cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
	cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
	debugMatch = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());

	//////////테스트용도

	////pt 확보 과정
	for (size_t i = 0, iend = vpTempPTsCurr.size(); i < iend; i++) {
		auto pCPi = vpTempCPs[i];
		int prevIDX = pCPi->GetPointIndexInFrame(pPrevMatch);
		if (prevIDX == -1)
			continue;
		auto currPt = vpTempPTsCurr[i];
		auto prevPt = vpTempPTsPrev[prevIDX];
		vMatchPrevPTs.push_back(prevPt);
		vMatchCurrPTs.push_back(currPt);
		vMatchCPs.push_back(pCPi);

		//cv::line(debugMatch, prevPt, currPt+ptBottom, cv::Scalar(255, 0, 255), 2);
		cv::circle(prevImg, prevPt, 4, cv::Scalar(255, 0, 0), 1);
		cv::circle(currImg, currPt, 4, cv::Scalar(255, 0, 0), 1);
	}

	if (vMatchCPs.size() < 10) {
		std::cout << "포인트 부족 000" << std::endl;
		return -1;
	}

	cv::Mat TempMap;
	cv::triangulatePoints(mK*Pprev, mK*Pcurr, vMatchPrevPTs, vMatchCurrPTs, TempMap);

	cv::Scalar color1(255, 0, 0);
	cv::Scalar color2(255, 255, 0);
	cv::Scalar color3(0, 0, 255);
	cv::Scalar color4(0, 255, 255);
	cv::Scalar color5(0, 255, 0);

	float fMinTH = mpTargetFrame->mfMedianDepth - mpTargetFrame->mfRange;// fMeanDepth - fRange * fStdDev;
	float fMaxTH = mpTargetFrame->mfMedianDepth + mpTargetFrame->mfRange;// fMeanDepth + fRange * fStdDev;
	float fMaxTH2 = pPrevKF->mfMedianDepth + pPrevKF->mfRange;// fMeanDepth + fRange * fStdDev;

	std::vector<CandidatePoint*> vMappingCPs;
	std::vector<cv::Point2f> vPTs;
	std::vector<cv::Mat> vX3Ds;

	std::vector<float> vfScales;
	float thresh = 25.0;
	for (size_t i = 0, iend = TempMap.cols; i < iend; i++) {

		////Seed 확인이 필요. 초기화 & 업데이트를 여기서 수행하고 확인도 여기서 하기.

		auto currPt = std::move(vMatchCurrPTs[i]);
		auto prevPt = std::move(vMatchPrevPTs[i]);
		auto pCPi = std::move(vMatchCPs[i]);
		auto pMPi = pCPi->GetMP();

		//////시드 관련
		//{
		//	if (pCPi->mpSeed) {

		//	}
		//	else {
		//		int idx = pCPi->GetPointIndexInFrame(pCPi->mpRefKF->mpMatchInfo);
		//		auto refPt = pCPi->mpRefKF->mpMatchInfo->mvMatchingPts[idx];
		//		cv::Mat a = (cv::Mat_<float>(3, 1) << refPt.x, refPt.y, 1);
		//		pCPi->mpSeed = new Seed(std::move(mpSystem->mInvK*a), px_err_angle, pCPi->mpRefKF->mfMedianDepth, pCPi->mpRefKF->mfMinDepth);
		//	}
		//}
		//////시드 관련

		cv::Mat X3D;
		float depth;
		if (pCPi->GetNumSize() < mnThreshMinKF)
			continue;
		bool bNewMP = true;
		float depth1, depth2;
		{
			X3D = std::move(TempMap.col(i));
			if (abs(X3D.at<float>(3)) < 0.0001) {
				bNewMP = false;
			}
			X3D /= X3D.at<float>(3);
			X3D = X3D.rowRange(0, 3);
			//New MP 조정

			cv::Mat proj1 = Rcurr*X3D + Tcurr;
			cv::Mat proj2 = Rprev*X3D + Tprev;

			depth1 = proj1.at<float>(2);
			depth2 = proj2.at<float>(2);

			if (depth1  < 0.0 || depth2 < 0.0) {
				bNewMP = false;
			}
			if (depth1 > fMaxTH || depth2 > fMaxTH2) {
				bNewMP = false;
			}
			////depth test


			////reprojection error
			if (bNewMP) {
				proj1 = mK*proj1;
				proj2 = mK*proj2;
				cv::Point2f projected1(proj1.at<float>(0) / proj1.at<float>(2), proj1.at<float>(1) / proj1.at<float>(2));
				cv::Point2f projected2(proj2.at<float>(0) / proj2.at<float>(2), proj2.at<float>(1) / proj2.at<float>(2));

				auto diffPt1 = projected1 - currPt;
				auto diffPt2 = projected2 - prevPt;
				float err1 = (diffPt1.dot(diffPt1));
				float err2 = (diffPt2.dot(diffPt2));
				if (err1 > thresh || err2 > thresh) {
					bNewMP = false;
				}

				////CReateMP test
				cv::Mat X3D2;
				float depthRef;
				auto bNewMP2 = pCPi->CreateMapPoint(X3D2, depthRef, mK, mInvK, Pcurr, Rcurr, Tcurr, currPt);
				//if(bNewMP2){
				//	cv::Point2f ptFromRefB = pCurrKF->Projection(X3D2);
				//	cv::Mat Rrel, Trel;
				//	pCurrKF->GetRelativePoseFromTargetFrame(pCPi->mpRefKF, Rrel, Trel);
				//	cv::Mat Xcam2 = pCPi->mpSeed->ray*depthRef;
				//	cv::Mat Xcam3 = pCPi->mpSeed->ray*(depthRef - 1.0f);
				//	cv::Mat Xcam4 = pCPi->mpSeed->ray*(depthRef + 1.0f);
				//	cv::Mat projFromRef = mK*(Rrel*Xcam2 + Trel);
				//	cv::Mat projFromRef2 = mK*(Rrel*Xcam3 + Trel);
				//	cv::Mat projFromRef3 = mK*(Rrel*Xcam4 + Trel);
				//	cv::Point2f ptFromRef(projFromRef.at<float>(0) / projFromRef.at<float>(2), projFromRef.at<float>(1) / projFromRef.at<float>(2));
				//	cv::Point2f ptFromRef2(projFromRef2.at<float>(0) / projFromRef2.at<float>(2), projFromRef2.at<float>(1) / projFromRef2.at<float>(2));
				//	cv::Point2f ptFromRef3(projFromRef3.at<float>(0) / projFromRef3.at<float>(2), projFromRef3.at<float>(1) / projFromRef3.at<float>(2));
				//
				//	//cv::circle(prevImg, prevPt, 3, color2, -1);
				//	/*cv::line(currImg, currPt, ptFromRefB, color5, 2);*/
				//	cv::line(currImg, currPt, ptFromRef, color3, 2);
				//	cv::line(currImg, ptFromRef2, ptFromRef3, color5, 1);
				//	//cv::circle(currImg, currPt, 3, color2, -1);
				//	//cv::circle(currImg, ptFromRef, 2, color4, -1);
				//
				//	//cv::Mat ray = mInvK*(cv::Mat_<float>(3, 1) << prevPt.x, prevPt.y, 1);
				//	//cv::Mat Xcam2 = ray*depth2;
				//	//////Seed Init & Update
				//	//cv::Mat Rrel, Trel;
				//	//pCurrKF->GetRelativePoseFromTargetFrame(pPrevKF, Rrel, Trel);
				//	//cv::Mat projFromRef = mK*(Rrel*Xcam2 + Trel);
				//	//cv::Point2f ptFromRef(projFromRef.at<float>(0) / projFromRef.at<float>(2), projFromRef.at<float>(1) / projFromRef.at<float>(2));
				//	//cv::circle(currImg, projected1, 3, color2, -1);
				//	//cv::line(currImg, projected1, ptFromRef, color4, 2);

				//	////Seed Init & Update
				//	/*cv::Mat Rrel, Trel;
				//	pCPi->mpRefKF->GetRelativePoseFromTargetFrame(pCurrKF, Rrel, Trel);*/
				//	//float depthRef = pCPi->mpRefKF->GetDepth(X3D);
				//	//cv::Point2f refPt1 = pCPi->mpRefKF->Projection(X3D);
				//	//int idx = pCPi->GetPointIndexInFrame(pCPi->mpRefKF->mpMatchInfo);
				//	//auto refPt2 = pCPi->mpRefKF->mpMatchInfo->mvMatchingPts[idx];
				//	//auto diffPt3 = refPt1 - refPt2;
				//	//float errref = (diffPt3.dot(diffPt3));
				//	//std::cout << "ref::err::" << errref << std::endl;
				//	//if (pCPi->mpSeed) {
				//	//	cv::Mat Xcam2 = pCPi->mpSeed->ray*depthRef;
				//	//	cv::Mat projFromRef = mK*(Rrel*Xcam2 + Trel);
				//	//	cv::Point2f ptFromRef(projFromRef.at<float>(0) / projFromRef.at<float>(2), projFromRef.at<float>(1) / projFromRef.at<float>(2));
				//	//	cv::circle(currImg, projected1, 3, color2, -1);
				//	//	cv::line(currImg, projected1, ptFromRef, color4, 2);
				//	//}
				//	////Seed Init & Update
				//}
				//else {
				//	cv::circle(prevImg, prevPt, 3, cv::Scalar(0,0,0), -1);
				//	cv::circle(currImg, currPt, 3, cv::Scalar(0, 0, 0), -1);
				//}

			}//reproj
			if (bNewMP) {
				cv::circle(debugging, currPt, 3, color2, -1);
			}
		}

		bool bOldMP = pMPi && pMPi->GetQuality() && !pMPi->isDeleted();
		if (bOldMP) {
			//MP가 존재하면 얘를 이용함.
			//여기서 스케일을 다시 계산하자.
			X3D = std::move(pMPi->GetWorldPos());
			//pMPi->SetLastVisibleFrame(nCurrKeyFrameID);
			cv::circle(debugging, currPt, 3, color4, -1);
			if (bNewMP) {
				////scale
				cv::Mat proj3 = Rprev*X3D + Tprev;
				float depth3 = proj3.at<float>(2);
				float scale = depth3 / depth2;
				vfScales.push_back(scale);
			}
		}
		if (!bNewMP && !bOldMP) {
			cv::circle(debugging, currPt, 3, color1, -1);
			continue;
		}
		vMappingCPs.push_back(pCPi);
		vX3Ds.push_back(X3D);
		vPTs.push_back(currPt);
	}

	////////////테스트용도
	//cv::Mat resized, r1, r2;
	///*cv::resize(debugMatch, resized, cv::Size(debugMatch.cols / 2, debugMatch.rows / 2));
	//cv::Rect mergeRect3 = cv::Rect(0, 0, prevImg.cols/2, prevImg.rows / 2);
	//cv::Rect mergeRect4 = cv::Rect(0, prevImg.rows / 2, prevImg.cols / 2, prevImg.rows / 2);*/
	//cv::resize(prevImg, r1, cv::Size(prevImg.cols / 2, prevImg.rows / 2));
	//cv::resize(currImg, r2, cv::Size(currImg.cols / 2, currImg.rows / 2));
	//mpVisualizer->SetOutputImage(r1, 2);
	//mpVisualizer->SetOutputImage(r2, 3);

	//prevImg.copyTo(debugMatch(mergeRect1));
	//currImg.copyTo(debugMatch(mergeRect2));
	//cv::moveWindow("Output::DepthFilter", mpSystem->mnDisplayX, mpSystem->mnDisplayY);
	//cv::imshow("Output::DepthFilter", debugMatch); 
	//cv::waitKey(1);
	////////////테스트용도

	////////////////////////////////////////최적화 진행
	if (vX3Ds.size() < 30) {
		std::cout << "포인트 부족11" << std::endl;
		return -1;
	}

	/////////Scale 계산
	if (vfScales.size() < 10) {
		//std::cout << "포인트 부족22" << std::endl;
		return -1;
	}
	std::nth_element(vfScales.begin(), vfScales.begin() + vfScales.size() / 2, vfScales.end());
	float medianPrevScale = vfScales[vfScales.size() / 2];
	/////////Scale 계산

	mpMap->ClearReinit();
	std::vector<bool> vbInliers(vX3Ds.size(), true);
	std::vector<bool> vbInliers2(vX3Ds.size(), true);
	Optimization::LocalOptimization(mpSystem, mpMap, pCurrKF, vX3Ds, vMappingCPs, vbInliers, vbInliers2, 1.0, fMedianDepth, fMeanDepth, fStdDev);

	int nFail = vX3Ds.size();
	for (size_t i = 0, iend = vX3Ds.size(); i < iend; i++) {
		if (!vbInliers[i]) {
			nFail--;
			cv::circle(debugging, vPTs[i], 3, color3, -1);
		}
		if (!vbInliers2[i])
			cv::circle(debugging, vPTs[i], 3, color5, -1);
	}
	if (nFail < 50) {
		std::cout << "Map Creation Fail case::" << pCurrKF->mnFrameID << std::endl;
	}

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	dtime = duration / 1000.0;
	return vX3Ds.size() - nFail;

}

int UVR_SLAM::LocalMapper::MappingProcess(Map* pMap, Frame* pCurrKF, double& dtime, cv::Mat& debugging) {
	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();

	int N;
	auto pMatch = pCurrKF->mpMatchInfo;
	auto vpTempCPs = pMatch->mvpMatchingCPs;
	auto vpTempPTs = pMatch->mvMatchingPts;
	cv::Mat Rcurr, Tcurr, Pcurr;
	mpTargetFrame->GetPose(Rcurr, Tcurr);
	cv::hconcat(Rcurr, Tcurr, Pcurr);
	auto spWindowKFs = mpMap->GetWindowFramesSet(2);
	cv::Point2f ptBottom = cv::Point2f(0, debugging.rows/2);

	std::vector<CandidatePoint*> vMappingCPs;
	std::vector<cv::Point2f> vPTs;
	std::vector<cv::Mat> vX3Ds;
	std::vector<float> vfScales;

	cv::Scalar color1(255, 0,   0);
	cv::Scalar color2(255, 255, 0);
	cv::Scalar color3(0,  0, 255);
	cv::Scalar color4(0,255, 255);
	cv::Scalar color5(0, 255, 0);
		
	//mpMap->ClearReinit();
	for (size_t i = 0, iend = vpTempPTs.size(); i < iend; i++) {
		auto pCPi = vpTempCPs[i];
		auto pMPi = pCPi->GetMP();
		auto currPt = vpTempPTs[i];

		cv::Mat X3D;
		float depth;
		if (pCPi->GetNumSize() < mnThreshMinKF)
			continue;
		bool bNewMP = pCPi->CreateMapPoint(X3D, depth, mK, mInvK, Pcurr, Rcurr, Tcurr, currPt);
		bool bOldMP = pMPi && pMPi->GetQuality() && !pMPi->isDeleted();
		if (bNewMP){
			cv::circle(debugging, currPt, 3, color2, -1);
		}
		//
		if (bOldMP) {
			//MP가 존재하면 얘를 이용함.
			//여기서 스케일을 다시 계산하자.
			X3D = std::move(pMPi->GetWorldPos());
			//pMPi->SetLastVisibleFrame(nCurrKeyFrameID);
			cv::circle(debugging, currPt, 3, color4, -1);
			if(bNewMP){
				////scale
				cv::Mat proj3 = Rcurr*X3D + Tcurr;
				float depth2 = proj3.at<float>(2);
				float scale = depth2 / depth;
				vfScales.push_back(scale);
			}
		}
		if (!bNewMP && !bOldMP){
			cv::circle(debugging, currPt, 3, color1, -1);
			continue;
		}
		vMappingCPs.push_back(pCPi);
		vX3Ds.push_back(X3D);
		vPTs.push_back(currPt);
		
	}

	////////////////////////////////////////최적화 진행
	if (vX3Ds.size() < 10) {
		//std::cout << "포인트 부족11" << std::endl;
		return -1;
	}

	/////////Scale 계산
	if (vfScales.size() < 10){
		//std::cout << "포인트 부족22" << std::endl;
		return -1;
	}
	std::nth_element(vfScales.begin(), vfScales.begin() + vfScales.size() / 2, vfScales.end());
	float medianPrevScale = vfScales[vfScales.size() / 2];
	/////////Scale 계산

	mpMap->ClearReinit();
	std::vector<bool> vbInliers(vX3Ds.size(), true);
	std::vector<bool> vbInliers2(vX3Ds.size(), true);
	//Optimization::LocalOptimization(mpSystem, mpMap, pCurrKF, vX3Ds, vMappingCPs, vbInliers, vbInliers2, medianPrevScale);

	for (size_t i = 0, iend = vX3Ds.size(); i < iend; i++) {
		if (!vbInliers[i])
			cv::circle(debugging, vPTs[i], 3, color3, -1);
		if (!vbInliers2[i])
			cv::circle(debugging, vPTs[i], 3, color5, -1);
	}

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	dtime = duration / 1000.0;
	return vX3Ds.size();
}
int UVR_SLAM::LocalMapper::MappingProcess(Map* pMap, Frame* pCurrKF, Frame* pPrevKF,
	std::vector<cv::Point2f>& vMappingPrevPts, std::vector<cv::Point2f>& vMappingCurrPts, std::vector<CandidatePoint*>& vMappingCPs,
	std::vector<cv::Point2f>  vMatchedPrevPts, std::vector<cv::Point2f>  vMatchedCurrPts, std::vector<CandidatePoint*>  vMatchedCPs,
	double& dtime, cv::Mat& debugging) {

	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();

	if (vMatchedPrevPts.size() < 10) {
		std::cout << "LM::Matching::error" << std::endl;
		return -1;
	}
	int nCurrKeyFrameID = pCurrKF->mnKeyFrameID;
	///////////////////projection based matching
	cv::Mat Rprev, Tprev, Rcurr, Tcurr;
	pCurrKF->GetPose(Rcurr, Tcurr);
	pPrevKF->GetPose(Rprev, Tprev);

	cv::Mat Pcurr, Pprev;
	cv::hconcat(Rcurr, Tcurr, Pcurr);
	cv::hconcat(Rprev, Tprev, Pprev);
	cv::Mat TempMap;
	cv::triangulatePoints(mK*Pprev, mK*Pcurr, vMatchedPrevPts, vMatchedCurrPts, TempMap);

	///////데이터 전처리
	std::vector<cv::Mat> vX3Ds;
	cv::Mat Rcfromc = Rcurr.t();
	cv::Mat Rpfromc = Rprev.t();

	cv::Mat prevImg = pPrevKF->GetOriginalImage();
	cv::Mat currImg = pCurrKF->GetOriginalImage();
	cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);

	float thresh = 5.0*5.0;

	int nRes = 0;
	int nTargetID = pPrevKF->mnKeyFrameID;
	std::vector<float> vfScales;
	for (int i = 0; i < TempMap.cols; i++) {

		auto currPt = std::move(vMatchedCurrPts[i]);
		auto prevPt = std::move(vMatchedPrevPts[i]);
		auto pCPi = std::move(vMatchedCPs[i]);
		auto pMPi = pCPi->GetMP();
		cv::Mat X3D;
		bool bMP = false;

		//New MP 조정
		X3D = std::move(TempMap.col(i));
		if (abs(X3D.at<float>(3)) < 0.0001) {
			continue;
		}
		X3D /= X3D.at<float>(3);
		X3D = X3D.rowRange(0, 3);
		//New MP 조정

		cv::Mat proj1 = Rcurr*X3D + Tcurr;
		cv::Mat proj2 = Rprev*X3D + Tprev;

		float depth1 = proj1.at<float>(2);
		float depth2 = proj2.at<float>(2);

		if (pMPi && pMPi->GetQuality() && !pMPi->isDeleted()) {
			//MP가 존재하면 얘를 이용함.
			//여기서 스케일을 다시 계산하자.
			X3D = std::move(pMPi->GetWorldPos());
			pMPi->SetLastVisibleFrame(nCurrKeyFrameID);

			////scale
			cv::Mat proj3 = Rprev*X3D + Tprev;
			float depth3  = proj3.at<float>(2);
			float scale = depth3 / depth2;
			vfScales.push_back(scale);
			////scale
			bMP = true;

		}
		
		////depth test
		if ( depth1  < 0.0 || depth2 < 0.0) {
			//cv::circle(debugging, currPt + ptBottom, 2, cv::Scalar(0, 255, 0), -1);
			//cv::circle(debugging, prevPt, 2, cv::Scalar(0, 255, 0), -1);
			/*if (proj1.at<float>(0) < 0 && proj1.at<float>(1) < 0 && proj1.at<float>(2) < 0) {
			cv::circle(debugMatch, pt1 + ptBottom, 2, cv::Scalar(255, 0, 0), -1);
			cv::circle(debugMatch, pt2, 2, cv::Scalar(255, 0, 0), -1);
			}*/
			continue;
		}
		////depth test

		////reprojection error
		proj1 = mK*proj1;
		proj2 = mK*proj2;
		cv::Point2f projected1(proj1.at<float>(0) / proj1.at<float>(2), proj1.at<float>(1) / proj1.at<float>(2));
		cv::Point2f projected2(proj2.at<float>(0) / proj2.at<float>(2), proj2.at<float>(1) / proj2.at<float>(2));

		auto diffPt1 = projected1 - currPt;
		auto diffPt2 = projected2 - prevPt;
		float err1 = (diffPt1.dot(diffPt1));
		float err2 = (diffPt2.dot(diffPt2));
		if (err1 > thresh || err2 > thresh) {
			continue;
		}
		////reprojection error

		//////////////////////////////////
		////이미 CP에 연결된 것들은 연결하지 않음.
		////현재 프레임에 CP 연결하기
		int idxa = pCPi->GetPointIndexInFrame(pCurrKF->mpMatchInfo);
		if(idxa == -1){
			int idxi = pCurrKF->mpMatchInfo->AddCP(pCPi, currPt);
			pCPi->ConnectFrame(pCurrKF->mpMatchInfo, idxi);
			////현재 프레임에 MP 연결하기
			if (bMP) {
				pMPi->SetLastSuccessFrame(nCurrKeyFrameID);
				pMPi->ConnectFrame(pCurrKF->mpMatchInfo, idxi);
			}
		}
		//////////////////////////////////

		////커넥트가 최소 3개인 CP들은 전부 참여
		if(pCPi->GetNumSize() >= 3){
			//nRes++;
			vMappingCurrPts.push_back(std::move(currPt));
			vMappingPrevPts.push_back(std::move(prevPt));
			vMappingCPs.push_back(std::move(pCPi));
			vX3Ds.push_back(std::move(X3D));
		}
		////커넥트가 최소 3개인 CP들은 전부 참여

		//////시각화
		if (pCPi->mnFirstID == nTargetID) {
			cv::circle(debugging, prevPt, 2, cv::Scalar(0, 0, 255), -1);
			cv::circle(debugging, currPt + ptBottom, 2, cv::Scalar(0, 0, 255), -1);
		}
		else {
			cv::circle(debugging, prevPt, 2, cv::Scalar(255, 0, 0), -1);
			cv::circle(debugging, currPt + ptBottom, 2, cv::Scalar(255, 0, 0), -1);
		}
		if (pCPi->GetMP()) {
						
			cv::line(debugging, currPt + ptBottom, projected1 + ptBottom, cv::Scalar(0, 255, 0), 1);
			cv::line(debugging, prevPt, projected2, cv::Scalar(0, 255, 0), 1);
			/*cv::circle(debugging, projected1 + ptBottom, 2, cv::Scalar(0, 255, 0), -1);
			cv::circle(debugging, projected2, 2, cv::Scalar(0, 255, 0), -1);*/
			cv::circle(debugging, prevPt, 3, cv::Scalar(0, 255, 0));
			cv::circle(debugging, currPt + ptBottom, 3, cv::Scalar(0, 255, 0));
		}
		//////시각화
		//nRes++;
	}

	////////////////////////////////////////최적화 진행
	if (vX3Ds.size() < 20){
		std::cout << "포인트 부족" << std::endl;
		return -1;
	}

	/////////Scale 계산
	std::nth_element(vfScales.begin(), vfScales.begin() + vfScales.size() / 2, vfScales.end());
	float medianPrevScale = vfScales[vfScales.size() / 2];
	/////////Scale 계산

	mpMap->ClearReinit();
	std::vector<bool> vbInliers(vX3Ds.size(), true);
	std::vector<bool> vbInliers2(vX3Ds.size(), true);
	//Optimization::LocalOptimization(mpSystem, mpMap, pCurrKF, vX3Ds, vMappingCPs, vbInliers, vbInliers2, medianPrevScale);

	///////////////////New Mp Creation
	////기존 MP도 여기 결과에 따라서 커넥션이 가능해야 할 듯함.
	
	auto spWindowKFs = mpMap->GetWindowFramesSet(3);
	/////시각화 확인
	for (int i = 0; i < vMappingCPs.size(); i++) {
		if (!vbInliers[i]){
			continue;
		}
		nRes++;
		/*cv::Mat X3D = std::move(vX3Ds[i]);
		mpMap->AddReinit(X3D);
		auto pCPi = std::move(vMappingCPs[i]);
		auto pMPi = pCPi->GetMP();
		if (pMPi && pMPi->GetQuality() && !pMPi->isDeleted()){
			pMPi->SetWorldPos(std::move(X3D));
			continue;
		}
		
		int label = pCPi->GetLabel();
		auto pMP = new UVR_SLAM::MapPoint(mpMap, mpTargetFrame, pCPi, X3D, cv::Mat(), label, pCPi->octave);
		auto mmpFrames = pCPi->GetFrames();
		for (auto iter = mmpFrames.begin(); iter != mmpFrames.end(); iter++) {
			auto pMatch = iter->first;
			int idx = iter->second;
			auto pKF = pMatch->mpRefFrame;
			if (spWindowKFs.find(pKF) != spWindowKFs.end()) {
				pMatch->AddMP();
				pMP->ConnectFrame(pMatch, idx);
				pMP->IncreaseVisible();
				pMP->IncreaseFound();
			}
		}
		pMP->SetOptimization(true);
		mpSystem->mlpNewMPs.push_back(pMP);
		*/
	}

	std::chrono::high_resolution_clock::time_point tracking_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_end - tracking_start).count();
	dtime = duration / 1000.0;
	return nRes;
	
}

int UVR_SLAM::LocalMapper::CreateMapPoints(std::vector<cv::Point2f> vMatchCurrPts, std::vector<CandidatePoint*> vMatchCPs, double& ftime, cv::Mat& debugMatch) {
	
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	
	auto spWindowKFs = mpMap->GetWindowFramesSet(2);
	auto pTargetKF = mpMap->GetReverseWindowFrame(1);
	auto pTargetMatch = pTargetKF->mpMatchInfo;
	auto pCurrKF = mpTargetFrame;

	cv::Mat Rcurr, Tcurr;
	pCurrKF->GetPose(Rcurr, Tcurr);
	cv::Mat Rprev, Tprev;
	pTargetKF->GetPose(Rprev, Tprev);
	cv::Mat Pcurr, Pprev;
	cv::hconcat(Rcurr, Tcurr, Pcurr);
	cv::hconcat(Rprev, Tprev, Pprev);

	std::vector<cv::Point2f> vNewPrevPTs, vNewCurrPTs;
	std::vector<CandidatePoint*> vNewCPs;

	for (int i = 0; i < vMatchCPs.size(); i++) {
		auto pCPi = vMatchCPs[i];
		int idx = pCPi->GetPointIndexInFrame(pTargetMatch);
		if (idx < 0)
			continue;
		auto pt = pTargetMatch->mvMatchingPts[idx];
		vNewPrevPTs.push_back(pt);
		vNewCurrPTs.push_back(vMatchCurrPts[i]);
		vNewCPs.push_back(pCPi);
	}



}

int UVR_SLAM::LocalMapper::CreateMapPoints(Frame* pCurrKF, std::vector<cv::Point2f> vMatchCurrPts, std::vector<CandidatePoint*> vMatchPrevCPs, double& ftime, cv::Mat& debugMatch){

	cv::Point2f ptBottom = cv::Point2f(0, mnHeight);
	cv::Mat Rcurr, Tcurr;
	pCurrKF->GetPose(Rcurr, Tcurr);

	cv::Mat Pcurr, Pprev;
	cv::hconcat(Rcurr, Tcurr, Pcurr);

	///////데이터 전처리
	cv::Mat Rcfromc = Rcurr.t();

	//set kf 제거 필요함

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	auto spWindowKFs = mpMap->GetWindowFramesSet(2);
	int N = vMatchCurrPts.size();
	int nRes = 0;
	for (int i = 0; i < N; i++) {
		auto pCPi = vMatchPrevCPs[i];
		int nConnCP = pCPi->GetNumSize();
		auto pMPinCP = pCPi->GetMP();
		auto currPt = vMatchCurrPts[i];
	
		if (nConnCP > 2 && !pMPinCP) {
			cv::Mat Xw;
			bool b1, b2;
			b1 = false;
			pCPi->CreateMapPoint(Xw, mK, mInvK, Pcurr, Rcurr, Tcurr, currPt, b1, b2, debugMatch);
			if (b1) {
				int label = pCPi->GetLabel();
				auto pMP = new UVR_SLAM::MapPoint(mpMap, mpTargetFrame, pCPi, Xw, cv::Mat(), label, pCPi->octave);
				//여기서 모든 CP 다 연결하기?
				auto mmpFrames = pCPi->GetFrames();
				for (auto iter = mmpFrames.begin(); iter != mmpFrames.end(); iter++) {
					auto pMatch = iter->first;
					auto pKF = pMatch->mpRefFrame;
					if (spWindowKFs.find(pKF) != spWindowKFs.end()) {
						int idx = iter->second;
						//pMatch->AddMP(pMP, idx);
						pMP->ConnectFrame(pMatch, idx);
					}
				}
				nRes++;
				cv::circle(debugMatch, currPt+ptBottom, 3, cv::Scalar(0, 255, 255));
				//cv::circle(debugMatch, prevPt, 3, cv::Scalar(0, 255, 255));
			}
		}
		
	}
	//std::cout << "mapping::kf::" << spMatches.size() << std::endl;
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	ftime = duration1 / 1000.0;
	return nRes;
}
