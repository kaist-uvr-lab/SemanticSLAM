#include <LocalMapper.h>
#include <CandidatePoint.h>
#include <Frame.h>
#include <System.h>
#include <Map.h>
#include <MapGrid.h>
#include <MapPoint.h>
#include <Matcher.h>
#include <LoopCloser.h>
#include <Optimization.h>
#include <PlaneEstimator.h>
#include <Plane.h>
#include <Visualizer.h>
#include <MapOptimizer.h>
#include <SemanticSegmentator.h>
#include <opencv2/core/mat.hpp>

#include <ctime>
#include <direct.h>

UVR_SLAM::LocalMapper::LocalMapper(){}
UVR_SLAM::LocalMapper::LocalMapper(Map* pMap, std::string strPath, int w, int h):mnWidth(w), mnHeight(h), mbStopBA(false), mbDoingProcess(false), mbStopLocalMapping(false), mpTargetFrame(nullptr), mpPrevKeyFrame(nullptr), mpPPrevKeyFrame(nullptr){
	mpMap = pMap;

	FileStorage fs(strPath, FileStorage::READ);

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

}
UVR_SLAM::LocalMapper::~LocalMapper() {}

void UVR_SLAM::LocalMapper::SetSystem(System* pSystem) {
	mpSystem = pSystem;
}
void UVR_SLAM::LocalMapper::SetLoopCloser(LoopCloser* pVis)
{
	mpLoopCloser = pVis;
}
void UVR_SLAM::LocalMapper::SetMapOptimizer(MapOptimizer* pMapOptimizer) {
	mpMapOptimizer = pMapOptimizer;
}
void UVR_SLAM::LocalMapper::SetVisualizer(Visualizer* pVis)
{
	mpVisualizer = pVis;
}
void UVR_SLAM::LocalMapper::SetInitialKeyFrame(UVR_SLAM::Frame* pKF1, UVR_SLAM::Frame* pKF2) {
	mpPrevKeyFrame = pKF1;
	mpTargetFrame = pKF2;
}
void UVR_SLAM::LocalMapper::InsertKeyFrame(UVR_SLAM::Frame *pKF)
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	mKFQueue.push(pKF);
	//std::cout << "insertkeyframe::queue size = " << mKFQueue.size() << std::endl;
	mbStopBA = true;
}

bool UVR_SLAM::LocalMapper::CheckNewKeyFrames()
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	return(!mKFQueue.empty());
}

void UVR_SLAM::LocalMapper::ProcessNewKeyFrame()
{
	//if (mpPrevKeyFrame)
	mpPPrevKeyFrame = mpPrevKeyFrame;
	//if (mpTargetFrame)
	mpPrevKeyFrame = mpTargetFrame;

	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	mpTargetFrame = mKFQueue.front();
	/*mpTargetFrame->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
	mpSystem->SetDirPath(mpTargetFrame->GetKeyFrameID());
	mpTargetFrame->SetBowVec(mpSystem->fvoc);*/
	mKFQueue.pop();
	mbStopBA = false;

	mpTargetFrame->Init(mpSystem->mpORBExtractor, mpSystem->mK, mpSystem->mD);
	//mpTargetFrame->mpMatchInfo->SetKeyFrame();
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

	while (1) {

		if (CheckNewKeyFrames()) {
			SetDoingProcess(true);
			std::chrono::high_resolution_clock::time_point lm_start = std::chrono::high_resolution_clock::now();
			////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////200412
			
			ProcessNewKeyFrame();

			int nTargetID = mpTargetFrame->GetFrameID();
			//std::cout << "lm::start::" << mpTargetFrame->GetFrameID() << std::endl;

			////Median Depth 게산
			mpTargetFrame->ComputeSceneMedianDepth();
			//////////////업데이트 맵포인트

			///////////////////////////////////////////////////////
			/////Delayed Triangulation
			cv::Mat ddddbug;
			mpPrevKeyFrame->mpMatchInfo->SetMatchingPoints();
			//이전 MP, CP, 새로운 CP가 같이 포함되어야 함.

			int nCreated = 0;
			////////New Matching & Create & Delayed CP test
			cv::Mat debugMatch;
			std::vector<cv::Point2f> vMatchPPrevPts, vMatchPrevPts, vMatchCurrPts;
			std::vector<CandidatePoint*> vMatchPrevCPs;
			/*std::vector<bool> vbInliers;
			std::vector<int> vnIDXs;
			std::vector<int> vnPrevOctaves;*/
			double time1 = 0.0;
			double time2 = 0.0;
			/////////KF-KF 매칭
			mpMatcher->OpticalMatchingForMapping(mpMap, mpTargetFrame, mpPrevKeyFrame, vMatchPrevPts, vMatchCurrPts, vMatchPrevCPs, mK, mInvK, time1, debugMatch);
			//if (mpTargetFrame->mpMatchInfo->GetNumMapPoints() < nMinMapPoints){
			//nCreated = CreateMapPoints(mpPrevKeyFrame, vMatchPrevPts, vMatchPrevCPs, time2, debugMatch); //제대로 동작안함.
			nCreated = CreateMapPoints(mpTargetFrame, vMatchCurrPts, vMatchPrevCPs, time2, debugMatch); //왜인지는 모르겟으나 잘 동작함
			//}
			mpPrevKeyFrame->mpMatchInfo->mMatchedImage = debugMatch.clone();

			///////매칭 정보 저장
			std::stringstream sstdir;
			sstdir << mpSystem->GetDirPath(0) << "/kfmatching/kfmatching_" << mpTargetFrame->GetKeyFrameID() << "_" << mpPrevKeyFrame->GetKeyFrameID() << ".jpg";
			cv::imwrite(sstdir.str(), debugMatch);
			///////매칭 정보 저장
			/////////KF-KF 매칭

			////KF 조절
			if (mpTargetFrame->GetKeyFrameID() % 3 != 0)
			{
				SetDoingProcess(false);
				continue;
			}
			////Segmentation 수행 및 Bow 설정
			mpMap->AddFrame(mpTargetFrame);
			mpTargetFrame->SetBowVec(mpSystem->fvoc); //키프레임 파트로 옮기기
			mpSegmentator->InsertKeyFrame(mpTargetFrame);
			mpPlaneEstimator->InsertKeyFrame(mpTargetFrame);
			//mpLoopCloser->InsertKeyFrame(mpTargetFrame);

			//////////////////업데이트 맵포인트
			float fratio = 0.0f;
			auto matchInfo = mpTargetFrame->mpMatchInfo;
			matchInfo->UpdateFrame();
			//mpPrevKeyFrame->mpMatchInfo->UpdateFrame();
			//////////////////업데이트 맵포인트

			/////////////////////Window Test
			//////1
			//std::vector<CandidatePoint*> vpCurrCPs;
			//auto vCurrPTs = mpTargetFrame->mpMatchInfo->GetMatchingPts(vpCurrCPs);
			//auto vpFrameWindows = mpMap->GetWindowFrames();
			///*cv::Mat currImg = mpTargetFrame->GetOriginalImage();
			//cv::Rect currRect(0, 0, mnWidth, mnHeight);*/
			//for (auto iter = vpFrameWindows.begin(); iter != vpFrameWindows.end(); iter++) {
			//	auto pKF = *iter;
			//	///////Matching Test
			//	//if (pKF->GetKeyFrameID() + 3 == mpTargetFrame->GetKeyFrameID()) {
			//	//	cv::Mat tempImg;
			//	//	std::vector<cv::Point2f> vTempMatchPrevPts, vTempMatchCurrPts;
			//	//	std::vector<CandidatePoint*> vTempMatchPrevCPs;
			//	//	mpMatcher->OpticalMatchingForMapping(mpMap, mpPrevKeyFrame, pKF, vTempMatchPrevPts, vTempMatchCurrPts, vTempMatchPrevCPs, mK, mInvK, time1, tempImg);
			//	//	/*if (mpTargetFrame->mpMatchInfo->GetNumMapPoints() < nMinMapPoints)
			//	//	nCreated = CreateMapPoints(mpTargetFrame, mpPrevKeyFrame, vMatchPrevPts, vMatchCurrPts, vMatchPrevCPs, time2, debugMatch);*/
			//	//	std::stringstream asstdir;
			//	//	asstdir << mpSystem->GetDirPath(0) << "/fuse/kfmatching_" << mpPrevKeyFrame->GetKeyFrameID() << "_" << pKF->GetKeyFrameID() << ".jpg";
			//	//	cv::imwrite(asstdir.str(), tempImg);
			//	//}
			//	///////Matching Test
			//	

			//	/*cv::Mat windowImg = cv::Mat::zeros(mnHeight * 2, mnWidth, CV_8UC3);
			//	cv::Mat img = pKF->GetOriginalImage();
			//	cv::Rect tmpRect(0, mnHeight, mnWidth, mnHeight);
			//	img.copyTo(windowImg(tmpRect));
			//	currImg.clone().copyTo(windowImg(currRect));
			//	cv::Mat img1 = windowImg(tmpRect);
			//	cv::Mat img2 = windowImg(currRect);*/

			//	auto pTargetMatch = pKF->mpMatchInfo;
			//	for (int i = 0; i < vCurrPTs.size(); i++) {
			//		auto pCPi = vpCurrCPs[i];
			//		auto pMPi = pCPi->mpMapPoint;
			//		int idx = pCPi->GetPointIndexInFrame(pTargetMatch);
			//		int idx2 = pCPi->GetPointIndexInFrame(mpTargetFrame->mpMatchInfo);
			//		if (idx >= 0) {
			//			auto pt = pTargetMatch->GetPt(idx);
			//			/*cv::circle(img1, pt, 2, cv::Scalar(255, 0, 255), -1);
			//			cv::circle(img2, vCurrPTs[i], 2, cv::Scalar(255, 0, 255), -1);*/
			//			if (pCPi->bCreated && pMPi) {
			//				bool bCurr = pMPi->isInFrame(mpTargetFrame->mpMatchInfo);
			//				bool bPrev = pMPi->isInFrame(pTargetMatch);
			//				if (bPrev && !bCurr) {
			//					pMPi->AddFrame(mpTargetFrame->mpMatchInfo, i);
			//				}
			//			}
			//		}
			//	}
			//	/*std::stringstream sstdir;
			//	sstdir << mpSystem->GetDirPath(0) << "/fuse/fuse_" << mpTargetFrame->GetKeyFrameID() << "_" << pKF->GetKeyFrameID() << ".jpg";
			//	cv::imwrite(sstdir.str(), windowImg);*/
			//}
			//////1
			////std::cout << "LM::Fusing::Start" << std::endl;
			//cv::Mat windowImg = cv::Mat::zeros(mnHeight * 2, mnWidth * 4, CV_8UC3);
			//std::chrono::high_resolution_clock::time_point fuse_start = std::chrono::high_resolution_clock::now();
			//int Nimg = 0;
			//cv::Rect currRect(mnWidth * 3, mnHeight, mnWidth, mnHeight);
			//mpTargetFrame->GetOriginalImage().copyTo(windowImg(currRect));
			//cv::Point2f ptBase(mnWidth * 3, mnHeight);
			//bool bRecover = false;
			//for (int i = 0; i < vMatchCurrPts.size(); i++) {
			//	cv::circle(windowImg, vMatchCurrPts[i] + ptBase, 1, cv::Scalar(255, 0, 0), -1);
			//}
			//std::vector<int> vnMatches;
			//for (auto iter = mpMap->mQueueFrameWindows.begin(); iter != mpMap->mQueueFrameWindows.end(); iter++) {
			//	auto pKF = *iter;
			//		
			//	int h = Nimg / 4;
			//	int w = Nimg % 4;
			//	cv::Mat img = pKF->GetOriginalImage();
			//	cv::Rect tmpRect(mnWidth*w, mnHeight*h, mnWidth, mnHeight);
			//	img.copyTo(windowImg(tmpRect));
			//	Nimg++;

			//	int nTemp = 0;
			//	auto pTargetMatch = pKF->mpMatchInfo;
			//	cv::Point2f ptTarget(mnWidth*w, mnHeight*h);

			//	auto ttttPts = pTargetMatch->GetMatchingPts();
			//	for (int i = 0; i < ttttPts.size(); i++) {
			//		cv::circle(windowImg, ttttPts[i] + ptTarget, 1, cv::Scalar(255, 0, 0), -1);
			//	}

			//	std::vector<cv::Point2f> vTempCurrPTs, vTempPrevPTs;
			//	std::vector<CandidatePoint*> vTempCPs;
			//	cv::Mat R, T;
			//	//bool bTest = pKF->GetKeyFrameID() + 3 == mpTargetFrame->GetKeyFrameID() || pKF->GetKeyFrameID() + 6 == mpTargetFrame->GetKeyFrameID() || pKF->GetKeyFrameID() + 9 == mpTargetFrame->GetKeyFrameID();
			//	//bool bTest = false;
			//	if(pKF->GetKeyFrameID() +3 == mpTargetFrame->GetKeyFrameID()){
			//		//pKF->GetKeyFrameID() + 3 == mpTargetFrame->GetKeyFrameID()
			//		fratio = ((float)mpTargetFrame->mpMatchInfo->GetNumMapPoints()) / pKF->mpMatchInfo->GetNumMapPoints();
			//		if (fratio < 0.8)
			//			bRecover = true;
			//	}

			//	for (int i = 0; i < vMatchCurrPts.size(); i++) {
			//		auto pCPi = vMatchPrevCPs[i];
			//		int idx = pCPi->GetPointIndexInFrame(pTargetMatch);
			//		if (idx >= 0) {
			//			auto pt = pTargetMatch->GetPt(idx);
			//			nTemp++;
			//			//bool bUsed1 = pTargetMatch->CheckOpticalPointOverlap(used, Frame::mnRadius, 10, pt); //used //얘는 왜 used 따로 만듬???
			//			//bool bUsed2 = pTargetMatch->CheckOpticalPointOverlap(usedCurr, Frame::mnRadius, 10, vMatchCurrPts[i]); //used //얘는 왜 used 따로 만듬???
			//			//if (!bUsed1 || !bUsed2) {
			//			//	continue;
			//			//}
			//			//cv::circle(used, pt, 2, cv::Scalar(255, 255, 255), -1);
			//			//cv::circle(usedCurr, vMatchCurrPts[i], 2, cv::Scalar(255, 255, 255), -1);

			//			cv::circle(windowImg, vMatchCurrPts[i] + ptBase, 2, cv::Scalar(255, 0, 255), -1);
			//			cv::circle(windowImg, pt+ ptTarget, 2, cv::Scalar(255, 0, 255), -1);

			//			if (bRecover) {
			//				vTempCurrPTs.push_back(vMatchCurrPts[i]);
			//				vTempPrevPTs.push_back(pt);
			//				vTempCPs.push_back(pCPi);
			//			}

			//			bool b = pCPi->bCreated;
			//			auto pMPi = pCPi->mpMapPoint;
			//			if (b && pMPi) {
			//				//둘다 존재하는지 체크하자
			//				bool bCurr = pMPi->isInFrame(mpTargetFrame->mpMatchInfo);
			//				bool bPrev = pMPi->isInFrame(pTargetMatch);

			//				//cv::Point2f projPt;
			//				//bool bProj = pMPi->Projection(projPt, pKF, mnWidth, mnHeight);
			//				//if (bProj) {
			//				//	//cv::circle(windowImg, projPt + ptTarget, 2, cv::Scalar(255, 0, 0), -1);
			//				//	cv::line(windowImg, projPt + ptTarget, pt + ptTarget, cv::Scalar(0, 0, 255));
			//				//}
			//				//else {
			//				//	cv::line(windowImg, projPt + ptTarget, pt + ptTarget, cv::Scalar(0, 0, 255));
			//				//}

			//				//cv::Point2f projPt2;
			//				//bool bProj2 = pMPi->Projection(projPt2, mpTargetFrame, mnWidth, mnHeight);
			//				//if (bProj2) {
			//				//	//cv::circle(windowImg, projPt + ptTarget, 2, cv::Scalar(255, 0, 0), -1);
			//				//	cv::line(windowImg, projPt2 + ptBase, vMatchCurrPts[i] + ptBase, cv::Scalar(0, 0, 255));
			//				//}
			//				
			//				if (bCurr) {
			//					if (pMPi->mnFirstKeyFrameID == mpTargetFrame->GetKeyFrameID()-3)
			//						cv::circle(windowImg, vMatchCurrPts[i] + ptBase, 3, cv::Scalar(0,0,255));
			//					else
			//						cv::circle(windowImg, vMatchCurrPts[i] + ptBase, 2, cv::Scalar(255, 255, 0), -1);
			//				}
			//				if (bPrev) {
			//					if (pMPi->mnFirstKeyFrameID == mpTargetFrame->GetKeyFrameID()-3)
			//						cv::circle(windowImg, pt + ptTarget, 3, cv::Scalar(0, 0, 255), -1);
			//					else
			//						cv::circle(windowImg, pt+ ptTarget, 2, cv::Scalar(255, 255, 0), -1);
			//				}
			//				if (!bCurr && !bPrev) {
			//					cv::circle(windowImg, vMatchCurrPts[i] + ptBase, 2, cv::Scalar(0, 255, 255), -1);
			//					cv::circle(windowImg, pt+ ptTarget, 2, cv::Scalar(0, 255, 255), -1);
			//				}

			//				if (bCurr && !bPrev) {
			//					//pMPi->AddFrame(pTargetMatch, pt);
			//					//check projection pt
			//					cv::circle(windowImg, vMatchCurrPts[i] + ptBase, 3, cv::Scalar(255, 0, 0));
			//					cv::circle(windowImg, pt+ ptTarget, 3, cv::Scalar(255, 0, 0));
			//				}
			//				if (bPrev && !bCurr) {
			//					//pMPi->AddFrame(mpTargetFrame->mpMatchInfo, vMatchCurrPts[i]);
			//					//check projection pt
			//					cv::circle(windowImg, vMatchCurrPts[i] + ptBase, 3, cv::Scalar(0, 255, 255));
			//					cv::circle(windowImg, pt + ptTarget, 3, cv::Scalar(0, 255, 255));
			//				}
			//			}//맵포인트가 존재하는 경우
			//		}
			//	}//for curr pts
			//	mpTargetFrame->AddKF(pKF, nTemp);
			//	vnMatches.push_back(nTemp);

			//	if (bRecover) {
			//		double d3;
			//		RecoverPose(mpTargetFrame, pKF, vTempPrevPTs, vTempCurrPTs, vTempCPs, R, T, d3, windowImg(tmpRect), windowImg(currRect));
			//		
			//		/*cv::Mat Rcurr, Tcurr, Rprev, Tprev;
			//		pKF->GetPose(Rprev, Tprev);
			//		mpTargetFrame->GetPose(Rcurr, Tcurr);
			//		cv::Mat Rinv = Rprev.t();
			//		cv::Mat Tinv = -Rinv*Tprev;
			//		cv::Mat Rdiff = Rcurr*Rinv;
			//		cv::Mat Tdiff = Rcurr*Tinv + Tcurr;
			//		std::cout << Rdiff << R << std::endl;
			//		std::cout << Tdiff.t() << T.t() << std::endl;*/
			//	}

			//}//for iter
			////std::cout << "LM::Fusing::End" << std::endl;
			//std::chrono::high_resolution_clock::time_point fuse_end = std::chrono::high_resolution_clock::now();
			//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(fuse_end - fuse_start).count();
			//double fuse_time = duration / 1000.0;
			//std::stringstream ss;
			//ss << "WindowFusing= " << fratio << "::" << fuse_time;
			//for (int i = 0; i < vnMatches.size(); i++)
			//	ss << "::" << vnMatches[i];
			//cv::rectangle(windowImg, cv::Point2f(0, 0), cv::Point2f(windowImg.cols, 30), cv::Scalar::all(0), -1);
			//cv::putText(windowImg, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));

			//ss.str("");
			//ss << "Optical flow Mapping2= " << mpTargetFrame->GetFrameID() << ", " << mpPrevKeyFrame->GetFrameID() << ", " << "::" << time1 << ", " << time2;
			//cv::rectangle(debugMatch, cv::Point2f(0, 0), cv::Point2f(debugMatch.cols, 30), cv::Scalar::all(0), -1);
			//cv::putText(debugMatch, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
			/////////////////////Window Test

			//////맵포인트 생성
			/*if (matchInfo->GetNumMapPoints() < nMinMapPoints)
				nCreated = CreateMapPoints(mpTargetFrame, mpPrevKeyFrame, vMatchPrevPts, vMatchCurrPts, vMatchPrevCPs, time2, debugMatch);*/
			mpPrevKeyFrame->mpMatchInfo->mMatchedImage = debugMatch.clone();
			//////맵포인트 생성

			///////매칭 정보 저장
			//cv::Mat resImg = cv::Mat::zeros(mnHeight * 2, mnWidth * 5, CV_8UC3);
			//windowImg.copyTo(resImg(cv::Rect(0, 0, windowImg.cols, windowImg.rows)));
			//debugMatch.copyTo(resImg(cv::Rect(mnWidth*4, 0, debugMatch.cols, debugMatch.rows)));
			//std::stringstream sstdir;
			//sstdir << mpSystem->GetDirPath(0) << "/testmatching/fuse_" << mpTargetFrame->GetKeyFrameID() << "_" << mpPrevKeyFrame->GetKeyFrameID() << ".jpg";
			//cv::imwrite(sstdir.str(), resImg);
			///////매칭 정보 저장

			////프레임 윈도우에 추가
			mpMap->AddWindowFrame(mpTargetFrame);

			//////////////업데이트 키프레임
			////이건 단순히 해당 키프레임에서 추적되는 맵포인트가 연결되어 있는 프레임들의 리스트에 불과함.
			//std::map<UVR_SLAM::Frame*, int> mmpCandidateKFs;
			////int nTargetID = mpTargetFrame->GetFrameID();
			//int nMaxKF = 0;
			//int nMinKF = INT_MAX;
			//for (int i = 0; i <  targetMatchingMPs.size(); i++) {
			//	UVR_SLAM::MapPoint* pMP = targetMatchingMPs[i];
			//	if (!pMP || pMP->isDeleted())
			//		continue;
			//	auto mmpMP = pMP->GetConnedtedFrames();
			//	if (mmpMP.size() > nMaxKF)
			//		nMaxKF = mmpMP.size();
			//	for (auto biter = mmpMP.begin(), eiter = mmpMP.end(); biter != eiter; biter++) {
			//		auto pMatchInfo = biter->first;
			//		auto pkF = pMatchInfo->mpRefFrame;
			//		//UVR_SLAM::Frame* pCandidateKF = biter->first;
			//		if (nTargetID == pkF->GetFrameID())
			//			continue;
			//		mmpCandidateKFs[pkF]++;
			//	}
			//}

			//nMaxKF = 0;
			//for (auto biter = mmpCandidateKFs.begin(), eiter = mmpCandidateKFs.end(); biter != eiter; biter++) {
			//	UVR_SLAM::Frame* pKF = biter->first;
			//	int nCount = biter->second;
			//	if (nMinKF > nCount)
			//		nMinKF = nCount;
			//	if (nMaxKF < nCount)
			//		nMaxKF = nCount;
			//	if (nCount > 10) {
			//		//mpTargetFrame->AddKF(pKF);
			//		//vPairs.push_back(std::make_pair(nCount, pKF));
			//		mpTargetFrame->AddKF(pKF, nCount);
			//		pKF->AddKF(mpTargetFrame, nCount);
			//	}
			//}
			//////////////업데이트 키프레임

			////////////////그리드 테스트
			//for (int i = 0; i < targetMatchingMPs.size(); i++) {
			//	auto pMPi = targetMatchingMPs[i];
			//	if (!pMPi || pMPi->isDeleted())
			//		continue;
			//	auto pt1 = mpMap->ProjectMapPoint(pMPi, mpMap->mfMapGridSize);
			//	auto pMG1 = mpMap->GetGrid(pMPi); //원래 포함된 곳
			//	auto pMG2 = mpMap->GetGrid(pt1);  //바뀐 곳
			//	if (!pMG1) {
			//		std::cout << "LocalMap::MapGrid::Check::error case111" << std::endl << std::endl << std::endl << std::endl;
			//	}
			//	if (pMG2) {
			//		if (pMG1 != pMG2) {
			//			mpMap->UpdateMapPoint(pMPi, pMG2);
			//		}
			//	}
			//	else {
			//		//pMG2가 존재하지 않는 경우.
			//		mpMap->DeleteMapPoint(pMPi);
			//		pMG2 = mpMap->InsertGrid(pt1);
			//		/*if (mpMap->CheckGrid(pt1, pMG1->mInitPt))
			//		{
			//			std::cout << "LocalMap::MapGrid::Check::error case" << std::endl << std::endl << std::endl << std::endl;
			//		}*/
			//		mpMap->InsertMapPoint(pMPi, pMG2);
			//	}
			//	
			//	//if (mpMap->CheckGrid(pt1)) {
			//	//	mpMap->InsertGrid(pt1);
			//	//	//std::cout << "map grid : " << pt << std::endl;
			//	//}
			//	//else {

			//	//}
			//}

			////그리드 테스트
			////auto mvGrids = mpMap->GetMapGrids();
			////cv::Mat gridImg = mpTargetFrame->GetOriginalImage();
			////std::set<UVR_SLAM::MapPoint*> mspMPs, mspTargetMPs;

			////auto mvpConnectedKFs = mpTargetFrame->GetConnectedKFs(7);
			////mvpConnectedKFs.push_back(mpTargetFrame);
			////for (int i = 0; i < mvpConnectedKFs.size(); i++) {
			////	auto pKF = mvpConnectedKFs[i];
			////	for (int j = 0; j < pKF->mpMatchInfo->mvpMatchingMPs.size(); j++) {
			////		auto pMPj = pKF->mpMatchInfo->mvpMatchingMPs[j];
			////		if (!pMPj || pMPj->isDeleted())
			////			continue;
			////		auto findres = mspTargetMPs.find(pMPj);
			////		if (findres == mspTargetMPs.end())
			////			mspTargetMPs.insert(pMPj);
			////	}
			////}

			////for (int i = 0; i < mvGrids.size(); i++) {
			////	auto pMGi = mvGrids[i];
			////	int nCount = pMGi->Count();
			////	if (nCount < 10)
			////		continue;
			////	cv::Mat x3D = pMGi->Xw.clone();
			////		
			////	auto pt = mpTargetFrame->Projection(x3D);
			////	if (!mpTargetFrame->isInImage(pt.x, pt.y, 10.0f))
			////		continue;
			////	auto mvpMPs = pMGi->GetMPs();
			////	for (int j = 0; j < mvpMPs.size(); j++) {
			////		auto pMPj = mvpMPs[j];
			////		if (!pMPj || pMPj->isDeleted())
			////			continue;
			////		auto findres2 = mspTargetMPs.find(pMPj);
			////		if (findres2 != mspTargetMPs.end())
			////			continue;
			////		auto findres = mspMPs.find(mvpMPs[j]);
			////		if (findres == mspMPs.end())
			////			mspMPs.insert(mvpMPs[j]);
			////	}
			////}
			////for (auto iter = mspMPs.begin(); iter != mspMPs.end(); iter++) {
			////	auto pMP = *iter;
			////	cv::Mat x3D = pMP->GetWorldPos();
			////	auto pt = mpTargetFrame->Projection(x3D);
			////	cv::circle(gridImg, pt, 3, cv::Scalar(255, 0, 255), -1);
			////}
			////for (auto iter = mspTargetMPs.begin(); iter != mspTargetMPs.end(); iter++) {
			////	auto pMP = *iter;
			////	cv::Mat x3D = pMP->GetWorldPos();
			////	auto pt = mpTargetFrame->Projection(x3D);
			////	cv::circle(gridImg, pt, 2, cv::Scalar(255, 0, 0), -1);
			////}
			////
			////////////////////
			/////////KF 누적////
			////std::set<UVR_SLAM::Frame*> mspKeyFrameWindows(mvpConnectedKFs.begin(), mvpConnectedKFs.end());
			////std::map<UVR_SLAM::Frame*, int> mmpGridKFs;
			////std::multimap<int, UVR_SLAM::Frame*, std::greater<int>> mmpCountGridKFs;
			////for (int i = 0; i < mvGrids.size(); i++) {
			////	auto pMGi = mvGrids[i];
			////	int nCount = pMGi->Count();
			////	if (nCount < 10)
			////		continue;

			////	auto tmmpGridKFs = pMGi->GetKFs();
			////	for (auto iter = tmmpGridKFs.begin(); iter != tmmpGridKFs.end(); iter++) {
			////		auto pKF = iter->first;
			////		auto count = iter->second;
			////		auto findres1 = mmpGridKFs.find(pKF);
			////		if (findres1 != mmpGridKFs.end() && count > 0) {
			////			mmpGridKFs[pKF] += count;
			////		}
			////		else {
			////			auto findres = mspKeyFrameWindows.find(pKF);
			////			if (findres == mspKeyFrameWindows.end() && count > 0) {
			////				mmpGridKFs.insert(std::make_pair(pKF, count));
			////			}
			////		}
			////	}
			////}
			////std::cout << "grid kf ::" << mmpGridKFs.size() <<", "<<mpTargetFrame->GetKeyFrameID()<< std::endl;

			////for (auto iter = mmpGridKFs.begin(); iter != mmpGridKFs.end(); iter++) {
			////	auto pKF = iter->first;
			////	auto count = iter->second;
			////	mmpCountGridKFs.insert(std::make_pair(count, pKF));
			////	//std::cout << "grid kf ::" << pKF->GetKeyFrameID() << "::" << count << std::endl;
			////}
			////int nKFSearch = 3;
			////int nKFSearchIDX = 0;
			////UVR_SLAM::Frame* mMaxGridKF = nullptr;
			////for (auto iter = mmpCountGridKFs.begin(); iter != mmpCountGridKFs.end() && nKFSearchIDX < nKFSearch; iter++, nKFSearchIDX++) {
			////	auto pKF = iter->second;
			////	auto count = iter->first;
			////	if (nKFSearchIDX == 0)
			////		mMaxGridKF = pKF;
			////}
			////if (mMaxGridKF)
			////{
			////	std::stringstream ss;
			////	cv::Mat gImg = mMaxGridKF->GetOriginalImage();
			////	ss << "Grid = " << mpTargetFrame->GetKeyFrameID() << ", " << mMaxGridKF->GetKeyFrameID() << std::endl;
			////	cv::rectangle(gImg, cv::Point2f(0, 0), cv::Point2f(gImg.cols, 30), cv::Scalar::all(0), -1);
			////	cv::putText(gImg, ss.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
			////	cv::imshow("Grid Image", gImg); waitKey(1);
			////}
			///////KF 누적////
			//////////////////

			////for (int i = 0; i < mvGrids.size(); i++) {
			////	auto pMGi = mvGrids[i];
			////	int nCount = pMGi->Count();
			////	if (nCount < 10)
			////		continue;
			////	cv::Mat x3D = pMGi->Xw.clone();
			////	
			////	auto pt = mpTargetFrame->Projection(x3D);
			////	if (!mpTargetFrame->isInImage(pt.x, pt.y, 10.0f))
			////		continue;

			////	cv::Mat a = cv::Mat::zeros(3, 1, CV_32FC1); 
			////	a.at<float>(1) = mpMap->mfMapGridSize;
			////	a.at<float>(2) = mpMap->mfMapGridSize;
			////	auto pt1 = mpTargetFrame->Projection(x3D + a);

			////	cv::Mat b = cv::Mat::zeros(3, 1, CV_32FC1);
			////	b.at<float>(0) = mpMap->mfMapGridSize;
			////	b.at<float>(1) = mpMap->mfMapGridSize;
			////	b.at<float>(2) = mpMap->mfMapGridSize;
			////	auto pt2 = mpTargetFrame->Projection(x3D + b);

			////	cv::Mat c = cv::Mat::zeros(3, 1, CV_32FC1);
			////	c.at<float>(0) = mpMap->mfMapGridSize;
			////	c.at<float>(1) = mpMap->mfMapGridSize;
			////	auto pt3 = mpTargetFrame->Projection(x3D + c);

			////	cv::Mat d = cv::Mat::zeros(3, 1, CV_32FC1);
			////	d.at<float>(1) = mpMap->mfMapGridSize;
			////	auto pt4 = mpTargetFrame->Projection(x3D + d);

			////	cv::Mat e = cv::Mat::zeros(3, 1, CV_32FC1);
			////	e.at<float>(2) = mpMap->mfMapGridSize;
			////	auto pt5 = mpTargetFrame->Projection(x3D + e);

			////	cv::Mat f = cv::Mat::zeros(3, 1, CV_32FC1);
			////	f.at<float>(0) = mpMap->mfMapGridSize;
			////	f.at<float>(2) = mpMap->mfMapGridSize;
			////	auto pt6= mpTargetFrame->Projection(x3D + f);

			////	cv::Mat g = cv::Mat::zeros(3, 1, CV_32FC1);
			////	g.at<float>(0) = mpMap->mfMapGridSize;
			////	auto pt7 = mpTargetFrame->Projection(x3D + g);

			////	auto pt8 = mpTargetFrame->Projection(x3D);

			////	line(gridImg, pt1, pt2, cv::Scalar(255, 0, 0));
			////	line(gridImg, pt2, pt3, cv::Scalar(255, 0, 0));
			////	line(gridImg, pt3, pt4, cv::Scalar(255, 0, 0));
			////	line(gridImg, pt4, pt1, cv::Scalar(255, 0, 0));

			////	line(gridImg, pt5, pt6, cv::Scalar(255, 0, 0));
			////	line(gridImg, pt6, pt7, cv::Scalar(255, 0, 0));
			////	line(gridImg, pt7, pt8, cv::Scalar(255, 0, 0));
			////	line(gridImg, pt8, pt5, cv::Scalar(255, 0, 0));

			////	line(gridImg, pt1, pt5, cv::Scalar(255, 0, 0));
			////	line(gridImg, pt2, pt6, cv::Scalar(255, 0, 0));
			////	line(gridImg, pt3, pt7, cv::Scalar(255, 0, 0));
			////	line(gridImg, pt4, pt8, cv::Scalar(255, 0, 0));
			////	
			////	//circle(gridImg, pt1, 2, cv::Scalar(255, 0, 0), -1);
			////}
			////imshow("grid grid : ", gridImg); waitKey(1);
			////////////////그리드 테스트

			

			

			/////////////////
			////맵포인트 생성
			cv::Mat ddebug;
			//CreateMapPoints(mpTargetFrame->mpMatchInfo, ddebug);
			//std::stringstream ssdir;
			//ssdir << mpSystem->GetDirPath(0) << "/kfmatching";// << mpTargetFrame->GetKeyFrameID() << "_" << mpPrevFrame->GetKeyFrameID() << ".jpg";
			/*ssdir << mpSystem->GetDirPath(0) << "/kfmatching/" << mpTargetFrame->GetKeyFrameID() << ".jpg";
			imwrite(ssdir.str(), ddebug);*/
			////맵포인트 생성
			/////////////////
			
			//////////////////////////KF-KF Matching
			/*{
				auto vpKFs = mpTargetFrame->GetConnectedKFs(7);
				if (vpKFs.size() > 3) {
					for (int i = 3; i < vpKFs.size(); i++) {
						cv::Mat dddddbug;
						auto pKFi = vpKFs[i];
						mpMatcher->OpticalMatchingForTracking2(mpTargetFrame, pKFi, dddddbug);
						std::stringstream ssdira;
						ssdira << mpSystem->GetDirPath(0) << "/kfmatching/" << mpTargetFrame->GetFrameID() << "_" << pKFi->GetFrameID() << "_localmapping.jpg";
						imwrite(ssdira.str(), dddddbug);
					}
				}
			}*/
			//////////////////////////KF-KF Matching

			////////////////
			////Fuse :: Three view :: base version
			//쓰리뷰는 당연히 두 이미지의 동일한 곳에서 포인트가 있어야 의미가 있는 것이라 버린다...
			//{
			//	auto vpKFs = mpTargetFrame->GetConnectedKFs(7);
			//	if (vpKFs.size() > 4) {
			//		cv::Mat debug;
			//		auto pFuseKF1 = vpKFs[1];//vpKFs[3];
			//		auto pFuseKF2 = vpKFs[3];//vpKFs.size() - 1
			//		mpMatcher->Fuse(mpTargetFrame, pFuseKF1, pFuseKF2, debug);

			//		//cv::Mat Rtarget, Ttarget, Rfuse1, Tfuse1, Rfuse2, Tfuse2;
			//		//mpTargetFrame->GetPose(Rtarget, Ttarget);
			//		//pFuseKF1->GetPose(Rfuse1, Tfuse1);
			//		//pFuseKF2->GetPose(Rfuse2, Tfuse2);
			//		//cv::Mat mK = mpTargetFrame->mK.clone();
			//		//cv::Mat FfromFuse1toTarget = mpMatcher->CalcFundamentalMatrix(Rfuse1, Tfuse1, Rtarget, Ttarget, mK);
			//		//cv::Mat FfromFuse2toTarget = mpMatcher->CalcFundamentalMatrix(Rfuse2, Tfuse2, Rtarget, Ttarget, mK);

			//		//cv::Mat imgFuse1 = pFuseKF1->GetOriginalImage();
			//		//cv::Mat imgFuse2 = pFuseKF2->GetOriginalImage();
			//		//cv::Mat imgTarget = mpTargetFrame->GetOriginalImage();

			//		//std::vector<cv::Point3f> lines[2];
			//		//cv::Mat matLines;
			//		//std::vector<cv::Point2f> pts1, pts2, ptsInFuse1;
			//		//std::vector<UVR_SLAM::MapPoint*> vpMPs;
			//		//for (int i = 0; i < pFuseKF1->mpMatchInfo->mnTargetMatch; i++) {
			//		//	auto pMPi = pFuseKF1->mpMatchInfo->mvpMatchingMPs[i];
			//		//	if (!pMPi || pMPi->isDeleted())
			//		//		continue;
			//		//	auto X3D = pMPi->GetWorldPos();
			//		//	auto currPt = mpTargetFrame->Projection(X3D);
			//		//	if (!mpTargetFrame->isInImage(currPt.x, currPt.y, 10.0)) {
			//		//		continue;
			//		//	}
			//		//	vpMPs.push_back(pMPi);
			//		//	pts1.push_back(pFuseKF1->mpMatchInfo->mvMatchingPts[i]);
			//		//}
			//		//cv::computeCorrespondEpilines(pts1, 2, FfromFuse1toTarget, lines[0]);
			//		//
			//		//for (int i = 0; i < lines[0].size(); i++) {
			//		//	float a1 = lines[0][i].x;
			//		//	float b1 = lines[0][i].y;
			//		//	float c1 = lines[0][i].z;

			//		//	auto pMPi = vpMPs[i];
			//		//	auto X3D = pMPi->GetWorldPos();
			//		//	auto currPt = mpTargetFrame->Projection(X3D);
			//		//	float dist = currPt.x*a1 + currPt.y*b1 + c1;
			//		//	if (pMPi->isInFrame(mpTargetFrame->mpMatchInfo))
			//		//		continue;
			//		//	if (dist < 1.0) {
			//		//		cv::circle(imgTarget, currPt, 3, cv::Scalar(255, 0, 255), -1);
			//		//		cv::circle(imgFuse2, pts1[i], 3, cv::Scalar(255, 0, 255), -1);
			//		//	}
			//		//	else {
			//		//		cv::circle(imgTarget, currPt, 3, cv::Scalar(0, 0, 255), -1);
			//		//		cv::circle(imgFuse2, pts1[i], 3, cv::Scalar(0, 0, 255), -1);
			//		//	}
			//		//}
			//		///*for (int i = 0; i < pFuseKF2->mpMatchInfo->mnTargetMatch; i++) {
			//		//	auto pMPi = pFuseKF2->mpMatchInfo->mvpMatchingMPs[i];
			//		//	if (!pMPi || pMPi->isDeleted())
			//		//		continue;
			//		//	auto X3D = pMPi->GetWorldPos();
			//		//	auto currPt = mpTargetFrame->Projection(X3D);
			//		//	if (!mpTargetFrame->isInImage(currPt.x, currPt.y, 10.0)) {
			//		//		continue;
			//		//	}
			//		//	pts2.push_back(pFuseKF2->mpMatchInfo->mvMatchingPts[i]);
			//		//}
			//		//cv::computeCorrespondEpilines(pts2, 2, FfromFuse2toTarget, lines[1]);*/
			//		//
			//		///*float a1 = lines[0][0].x;
			//		//float b1 = lines[0][0].y;
			//		//float c1 = lines[0][0].z;
			//		//if(abs(b1) > 0.00001){

			//		//	cv::circle(imgFuse1, pts1[0], 3, cv::Scalar(255, 0, 255), -1);

			//		//	for (int j = 0; j < lines[1].size(); j++) {
			//		//		float a2 = lines[1][j].x;
			//		//		float b2 = lines[1][j].y;
			//		//		float c2 = lines[1][j].z;
			//		//		float a = a1*b2 - a2*b1;
			//		//		if (abs(a) < 0.00001)
			//		//			continue;
			//		//		float b = b1*c2 - b2*c1;
			//		//		float x = b / a;
			//		//		float y = -a1 / b1*x - c1 / b1;
			//		//	
			//		//		cv::Point2f pt(x, y);
			//		//		if (!mpTargetFrame->isInImage(x, y, 10.0)) {
			//		//			continue;
			//		//		}
			//		//		cv::circle(imgTarget, pt, 3, cv::Scalar(255, 0, 255), -1);
			//		//	}
			//		//}*/

			//		//cv::imshow("fuse::1", imgFuse1);
			//		////cv::imshow("fuse::2", imgFuse2);
			//		//cv::imshow("fuase::curr", imgTarget);
			//		//cv::waitKey(1);
			//	}
			//}
			////Fuse :: Three view :: base version
			////////////////
			
			////////////////
			////KF-KF rectification
			/*{
				auto vpKFs = mpTargetFrame->GetConnectedKFs(7);
				if (vpKFs.size() > 4) {
					auto pPrevKF = mpTargetFrame->mpMatchInfo->mpTargetFrame;
					auto pPrevPrevKF = pPrevKF->mpMatchInfo->mpTargetFrame;
					auto pLastKF = vpKFs[4];
					cv::Mat imgKFNF;
					mpMatcher->OpticalMatchingForFuseWithEpipolarGeometry(pLastKF, mpTargetFrame, imgKFNF);
					std::stringstream ssdira;
					ssdira << mpSystem->GetDirPath(0) << "/kfmatching/rectification_" << mpTargetFrame->GetFrameID() << "_" << pLastKF->GetFrameID() << ".jpg";
					imwrite(ssdira.str(), imgKFNF);
				}
			}*/
			////KF-KF rectification
			///////////////

			////////////////////////////////////////////////////////////////////////////////////
			/////KF-kF 매칭 성능 확인
			//auto mvpConnectedKFs = mpTargetFrame->GetConnectedKFs(10);
			//auto pTargetMatch = mpTargetFrame->mpMatchInfo;
			//auto mK = mpTargetFrame->mK.clone();
			//auto mvpMPs = std::vector<UVR_SLAM::MapPoint*>(pTargetMatch->mvpMatchingMPs.begin(), pTargetMatch->mvpMatchingMPs.end());
			//auto mvKPs = std::vector<cv::Point2f>(pTargetMatch->mvMatchingPts.begin(), pTargetMatch->mvMatchingPts.end());
			//cv::Mat prevImg = mpTargetFrame->GetOriginalImage();
			//cv::Mat tdebugging = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
			//cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
			//cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
			//cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
			//std::stringstream ssdir;
			//for (int i = 0; i < mvpConnectedKFs.size(); i++) {

			//	auto pKFi = mvpConnectedKFs[i];
			//	auto pMatch = pKFi->mpMatchInfo;
			//	cv::Mat R, t;
			//	pKFi->GetPose(R, t);

			//	/////debug image
			//	cv::Mat currImg = pKFi->GetOriginalImage();
			//	prevImg.copyTo(tdebugging(mergeRect1));
			//	currImg.copyTo(tdebugging(mergeRect2));
			//	/////debug image
			//	for (int j = 0; j <mvpMPs.size(); j++) {
			//		auto pMPj = mvpMPs[j];
			//		if (!pMPj || pMPj->isDeleted() || !pMPj->isInFrame(pMatch))
			//			continue;
			//		auto idx = pMPj->GetPointIndexInFrame(pMatch);
			//		if (idx == -1)
			//			std::cout << "lm::kf::error::1!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
			//		if(idx >= pMatch->mvMatchingPts.size())
			//			std::cout << "lm::kf::error::2!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
			//		auto pt1 = mvKPs[j];
			//		auto pt2 = pMatch->mvMatchingPts[idx]+ptBottom;
			//		auto X3D = pMPj->GetWorldPos();

			//		cv::Mat proj = mK*(R*X3D + t);
			//		auto p2D = cv::Point2f(proj.at<float>(0) / proj.at<float>(2), proj.at<float>(1) / proj.at<float>(2));
			//		if(!pKFi->isInImage(p2D.x, p2D.y))
			//			std::cout << "lm::kf::error::3!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
			//		p2D += ptBottom;
			//		
			//		cv::circle(tdebugging, pt1, 2, cv::Scalar(255, 255, 0), -1);
			//		cv::line(tdebugging, pt2, p2D, cv::Scalar(255, 255, 0), 2);
			//	}
			//	////파일 저장
			//	/*ssdir.str("");
			//	ssdir << mpSystem->GetDirPath(0) << "/kfmatching/" << mpTargetFrame->GetKeyFrameID() << "_" << pKFi->GetKeyFrameID() << ".jpg";
			//	if(i%5==0)
			//		imwrite(ssdir.str(), tdebugging);*/
			//	////파일 저장
			//}
			/////KF-kF 매칭 성능 확인
			////////////////////////////////////////////////////////////////////////////////////

			/////////////////Save Keyframe optical flow matching results
			/*auto prevFrame = mpTargetFrame->mpMatchInfo->mpTargetFrame;
			std::vector<std::pair<cv::Point2f, cv::Point2f>> vPairs1, vPairs2;
			cv::Mat kfdebug1, kfdebug2;
			mpMatcher->OpticalMatchingForMapping(mpTargetFrame, prevFrame, vPairs1, kfdebug1);
			std::stringstream ssdir;
			ssdir << mpSystem->GetDirPath(0) << "/kfmatching/" << mpTargetFrame->GetFrameID() <<"_"<<prevFrame->GetFrameID()<< ".jpg";
			imwrite(ssdir.str(), kfdebug1);*/
			/*
			auto prevPrevFrame = mpTargetFrame->mpMatchInfo->mpTargetFrame->mpMatchInfo->mpTargetFrame;
			mpMatcher->OpticalMatchingForMapping(mpTargetFrame, prevPrevFrame, vPairs2, kfdebug2);
			ssdir.str("");
			ssdir << mpSystem->GetDirPath(0) << "/kfmatching/" << mpTargetFrame->GetFrameID() << "_" << prevPrevFrame->GetFrameID() << ".jpg";
			imwrite(ssdir.str(), kfdebug2);
			*/
			/////////////////Save Keyframe optical flow matching results

			///////////////////////Fuse Map Points
			//////빨간 : 이전 프레임에는 존재하나, 현재 프레임에서는 포인트도 존재하지 않음.
			//////녹색 : 포인트는 존재하나, MP가 없음.
			//////분홍 : 맵포인트도 있음. 둘이 일치.
			//////노랑 : 맵포인트도 있음. 둘이 다름.
			//auto mvpKFs = mpTargetFrame->GetConnectedKFs();
			//cv::Mat fuseMap = mpTargetFrame->GetOriginalImage();
			//cv::Mat fuseIdx = cv::Mat::ones(fuseMap.size(), CV_16SC1)*-1;
			//for (int i = 0; i < mpTargetFrame->mpMatchInfo->mvMatchingPts.size(); i++) {
			//	auto pt = mpTargetFrame->mpMatchInfo->mvMatchingPts[i];
			//	cv::circle(fuseMap, pt, 1, cv::Scalar(255, 255, 0), -1);
			//	fuseIdx.at<short>(pt) = i;
			//}
			//cv::Mat R, t;
			//mpTargetFrame->GetPose(R, t);
			//for (int i = 0; i < mvpKFs.size(); i++) {
			//	auto pMatchInfo = mvpKFs[i]->mpMatchInfo;
			//	for (int j = 0; j < pMatchInfo->mvpMatchingMPs.size(); j++) {
			//		auto pMP = pMatchInfo->mvpMatchingMPs[j];
			//		if (!pMP || pMP->isDeleted())
			//			continue;
			//		if (pMP->isInFrame(mpTargetFrame->mpMatchInfo))
			//			continue;
			//		auto X3D = pMP->GetWorldPos();
			//		cv::Mat proj = mpTargetFrame->mK*(R*X3D + t);
			//		//proj = *proj;
			//		cv::Point2f pt(proj.at<float>(0) / proj.at<float>(2), proj.at<float>(1) / proj.at<float>(2));
			//		if (!mpTargetFrame->isInImage(pt.x, pt.y, 10.0))
			//			continue;
			//		int idx = fuseIdx.at<short>(pt);
			//		if (idx < 0) {
			//			cv::circle(fuseMap, pt, 2, cv::Scalar(0, 0, 255), -1);
			//			continue;
			//		}
			//		if (idx >= mpTargetFrame->mpMatchInfo->mvpMatchingMPs.size())
			//			std::cout << "???????????????????????????????????????????????????????????????????????????????????????????????????????????????" << std::endl;
			//		auto pMPi = mpTargetFrame->mpMatchInfo->mvpMatchingMPs[idx];
			//		if (!pMPi || pMPi->isDeleted()) {
			//			cv::circle(fuseMap, pt, 2, cv::Scalar(255, 0, 0), -1);
			//			pMP->AddFrame(mpTargetFrame->mpMatchInfo, idx);
			//			//fuseIdx.at<short>(pt) = idx;
			//		}
			//		else if (pMPi && !pMPi->isDeleted()) {
			//			//if (pMP->mnMapPointID == pMPi->mnMapPointID) {
			//			//	cv::circle(fuseMap, pt, 2, cv::Scalar(255, 0, 255), -1);
			//			//}
			//			//else
			//			//{
			//			//	cv::circle(fuseMap, pt, 2, cv::Scalar(0, 255, 255), -1);
			//			//	//fuse
			//			//}
			//		}
			//	}
			//}
			//std::cout << "lm::fuse::end" << std::endl;
			//imshow("fuse:", fuseMap); waitKey(1);
			/////////////////////Fuse Map Points

			//mpPlaneEstimator->InsertKeyFrame(mpTargetFrame);
			if (mpMapOptimizer->isDoingProcess()) {
				//std::cout << "lm::ba::busy" << std::endl;
				mpMapOptimizer->StopBA(true);
			}
			else {
				//std::cout << "lm::ba::idle" << std::endl;
				mpMapOptimizer->InsertKeyFrame(mpTargetFrame);
			}
			std::chrono::high_resolution_clock::time_point lm_end = std::chrono::high_resolution_clock::now();
			auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(lm_end - lm_start).count();
			float t_test1 = du_test1 / 1000.0;

			std::stringstream ssa;
			ssa << "LocalMapping : " << mpTargetFrame->GetKeyFrameID() << "::" << t_test1 << "::" << ", " << nCreated << "::" << fratio;// << ", " << nMinKF << ", " << nMaxKF;
			mpSystem->SetLocalMapperString(ssa.str());

			//std::cout << "lm::end::" <<mpTargetFrame->GetFrameID()<<"::"<<nCreated<< std::endl;
			SetDoingProcess(false);
			continue;
			//////200412
		}
	}//while
}

void UVR_SLAM::LocalMapper::SetMatcher(UVR_SLAM::Matcher* pMatcher) {
	mpMatcher = pMatcher;
}

void UVR_SLAM::LocalMapper::SetLayoutEstimator(SemanticSegmentator* pEstimator) {
	mpSegmentator = pEstimator;
}

void UVR_SLAM::LocalMapper::SetPlaneEstimator(PlaneEstimator* pPlaneEstimator) {
	mpPlaneEstimator = pPlaneEstimator;
}

//맵포인트가 삭제 되면 현재 프레임에서도 해당 맵포인트를 삭제 해야 하며, 
//이게 수행되기 전에는 트래킹이 동작하지 않도록 막아야 함.
//
void UVR_SLAM::LocalMapper::NewMapPointMarginalization() {
	//std::cout << "Maginalization::Start" << std::endl;
	//mvpDeletedMPs.clear();
	int nMarginalized = 0;
	int mnMPThresh = 2;
	float mfRatio = 0.25f;

	std::list<UVR_SLAM::MapPoint*>::iterator lit = mpSystem->mlpNewMPs.begin();
	while (lit != mpSystem->mlpNewMPs.end()) {
		UVR_SLAM::MapPoint* pMP = *lit;

		int nMPThresh = mnMPThresh;
		float fRatio = mfRatio;
		//if (pMP->GetMapPointType() == UVR_SLAM::PLANE_MP) {
		//	//nMPThresh = 0;
		//	fRatio = 0.01;
		//}
		bool bBad = false;
		if (pMP->isDeleted()) {
			//already deleted
			lit = mpSystem->mlpNewMPs.erase(lit);
		}
		else if (pMP->GetFVRatio() < fRatio) {
			bBad = true;
			lit = mpSystem->mlpNewMPs.erase(lit);
		}
		//else if (pMP->mnFir//KeyFrameID + 2 <= mpTargetFrame->GetKeyFrameID() && pMP->GetNumConnectedFrames() <= nMPThresh && pMP->GetMapPointType() != UVR_SLAM::PLANE_MP) {
		/*else if (pMP->GetMapPointType() == UVR_SLAM::PLANE_MP && pMP->mnFirstKeyFrameID + 1 > mpTargetFrame->GetKeyFrameID() && pMP->mnFirstKeyFrameID + 2 <= mpTargetFrame->GetKeyFrameID() && pMP->GetNumConnectedFrames() <= 1) {
			bBad = true;
			lit = mpSystem->mlpNewMPs.erase(lit);
		}*/
		//else if (pMP->mnFirstKeyFrameID + 2 <= mpTargetFrame->GetKeyFrameID() && pMP->GetNumConnectedFrames() <= nMPThresh != UVR_SLAM::PLANE_MP)
		else if (pMP->mnFirstKeyFrameID + 2 <= mpTargetFrame->GetKeyFrameID() && pMP->GetNumConnectedFrames() <= nMPThresh)
		{
			lit = mpSystem->mlpNewMPs.erase(lit);
		}
		else if (pMP->mnFirstKeyFrameID + 3 <= mpTargetFrame->GetKeyFrameID()){
			lit = mpSystem->mlpNewMPs.erase(lit);
			pMP->SetNewMP(false);
		}else
			lit++;
		if (bBad) {
			//mpFrameWindow->SetMapPoint(nullptr, i);
			//mpFrameWindow->SetBoolInlier(false, i);
			//frame window와 현재 키프레임에 대해서 삭제될 포인트 처리가 필요할 수 있음.
			//pMP->SetDelete(true);
			pMP->Delete();
		}
	}

	return;
}
int UVR_SLAM::LocalMapper::RecoverPose(Frame* pCurrKF, Frame* pPrevKF, std::vector<cv::Point2f> vMatchPrevPts, std::vector<cv::Point2f> vMatchCurrPts, std::vector<CandidatePoint*> vPrevCPs, cv::Mat& R, cv::Mat& T, double& ftime, cv::Mat& prevImg, cv::Mat& currImg) {
	
	//스케일과 맵 그리고 포즈를 복원

	cv::Point2f ptBottom(0, mnHeight);

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
		int idx = vTempMatchIDXs[i]; //cp idx
		auto pCPi = vPrevCPs[idx];
		auto pMPi = pCPi->mpMapPoint;

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
	/*float meanScalae = sumScale / vScales.size();
	int nidx = vScales.size() / 2;
	std::nth_element(vScales.begin(), vScales.begin() + nidx, vScales.end());
	float medianScale = vScales[(nidx)];*/

	std::nth_element(vPrevScales.begin(), vPrevScales.begin()+ vPrevScales.size()/2, vPrevScales.end());
	float medianPrevScale = vPrevScales[vPrevScales.size() / 2];
	cv::Mat scaled = R*Tprev+T*medianPrevScale;
	//Map3D *= medianPrevScale;
	std::cout << "scale : "  <<"||"<<medianPrevScale<< "::" << scaled.t() << ", " << Tcurr.t() << std::endl;

	//포즈 변경
	R = R*Rprev;
	T = scaled;
	mpTargetFrame->GetPose(R, T);
	//MP 생성
	
	for (int i = 0; i < vpTempCPs.size(); i++) {
		auto pCPi = vpTempCPs[i];
		cv::Mat X3D = mInvK*vX3Ds[i]* medianPrevScale;
		X3D = Rinv*(X3D) + Tinv;
		
		//MP fuse나 replace 함수가 필요해짐. 아니면, world pos만 변경하던가
		//빈곳만 채우던가
		bool bMP = pCPi->bCreated;
		if (bMP) {
			pCPi->mpMapPoint->SetWorldPos(X3D);
		}
		else {
			int label = pCPi->GetLabel();
			auto pMP = new UVR_SLAM::MapPoint(mpMap, mpTargetFrame, pCPi, X3D, cv::Mat(), label, pCPi->octave);
			//여기서 모든 CP 다 연결하기?
			auto mmpFrames = pCPi->GetFrames();
			for (auto iter = mmpFrames.begin(); iter != mmpFrames.end(); iter++) {
				auto pMatch = iter->first;
				if (pMatch->mpRefFrame->GetKeyFrameID() % 3 != 0)
					continue;
				int idx = iter->second;
				pMP->AddFrame(pMatch, idx);
			}
			/*pMP->AddFrame(pCurrKF->mpMatchInfo, pt1);
			pMP->AddFrame(pPrevKF->mpMatchInfo, pt2);*/
		}
	}

	return res2;
}

////////////200722 수정 필요
int UVR_SLAM::LocalMapper::CreateMapPoints(Frame* pCurrKF, std::vector<cv::Point2f> vMatchCurrPts, std::vector<CandidatePoint*> vMatchPrevCPs, double& ftime, cv::Mat& debugMatch){
	
	std::set<UVR_SLAM::MatchInfo*> spMatches;

	cv::Point2f ptBottom = cv::Point2f(0, mnHeight);
	cv::Mat Rcurr, Tcurr;
	pCurrKF->GetPose(Rcurr, Tcurr);

	cv::Mat Pcurr, Pprev;
	cv::hconcat(Rcurr, Tcurr, Pcurr);

	///////데이터 전처리
	cv::Mat Rcfromc = Rcurr.t();

	//set kf 제거 필요함

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	int N = vMatchCurrPts.size();
	int nRes = 0;
	for (int i = 0; i < N; i++) {
		auto pCPi = vMatchPrevCPs[i];
		int nConnCP = pCPi->GetNumSize();
		bool bMP = pCPi->bCreated; 
		MapPoint* pMPinCP = pCPi->mpMapPoint;
		auto currPt = vMatchCurrPts[i];
	
		if (nConnCP > 2 && !bMP) {
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
					if (pMatch->mpRefFrame->GetKeyFrameID() % 3 != 0){
						////KF id를 조정하는 것이 필요함
						//std::cout << "LocalMapper::asdj;asjd;lkasdj;flkasjdlkasdf"<<std::endl;
						continue;
					}
					int idx = iter->second;
					pMP->AddFrame(pMatch, idx);
					//auto pt = pMatch->GetPt();
					//pMP->AddFrame(pMatch, pt);
					spMatches.insert(pMatch);
				}
				
				cv::circle(debugMatch, currPt+ptBottom, 3, cv::Scalar(0, 255, 255));
				//cv::circle(debugMatch, prevPt, 3, cv::Scalar(0, 255, 255));
			}
		}
		nRes++;
	}
	//std::cout << "mapping::kf::" << spMatches.size() << std::endl;
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	ftime = duration1 / 1000.0;
	return spMatches.size();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////코드 백업
///////////200512
////////////////////////
///////매칭 결과 저장
//auto currFrame = mpTargetFrame;
//auto targetFrame = mpTargetFrame->mpMatchInfo->mpTargetFrame;;
//auto targettargetFrame = targetFrame->mpMatchInfo->mpTargetFrame;
//cv::Mat targettargetImg = targettargetFrame->GetOriginalImage();
//cv::Mat currImg = currFrame->GetOriginalImage();
//cv::Point2f ptBottom = cv::Point2f(0, currImg.rows);
//cv::Rect mergeRect1 = cv::Rect(0, 0, currImg.cols, currImg.rows);
//cv::Rect mergeRect2 = cv::Rect(0, currImg.rows, currImg.cols, currImg.rows);
//cv::Mat debug = cv::Mat::zeros(currImg.rows * 2, currImg.cols, currImg.type());
//targettargetImg.copyTo(debug(mergeRect1));
//currImg.copyTo(debug(mergeRect2));

//std::vector<cv::Point2f> vPts1, vPts2, vPts3;
//std::vector<int> vIDXs1, vIDXs2, vIDXs3;
//auto currInfo = mpTargetFrame->mpMatchInfo;
//auto targetInfo = targetFrame->mpMatchInfo;
//auto targetTargetInfo = targettargetFrame->mpMatchInfo;
//int n = targetInfo->nMatch;
//for (int i = 0; i < currInfo->mvnTargetMatchingPtIDXs.size(); i++) {
//	/*if (matchInfo->mvpMatchingMPs[i])
//		continue;*/
//	//이전 프레임 매칭 중 특징점으로 새로 추가된 포인트는 전전프레임에 존재하지 않기 때문
//	int tidx1 = currInfo->mvnTargetMatchingPtIDXs[i];
//	if (tidx1 >= n) {
//		continue;
//	}
//	//전프레임 매칭 결과와 전전프레임 매칭 결과를 연결
//	int tidx2 = targetInfo->mvnTargetMatchingPtIDXs[tidx1];

//	//전전프레임의 매칭값 중 
//	//전프레임의 매칭에서 포인트 위치
//	int idx2 = targetInfo->mvnMatchingPtIDXs[tidx1];
//	//전전프레임에서 매칭 위치
//	int idx3 = targetTargetInfo->mvnMatchingPtIDXs[tidx2];

//	/*if (targetInfo->mvpMatchingMPs[idx2]) {
//		continue;
//	}*/
//	//맵포인트가 존재하는 곳은 하늘색, 존재하지 않으면 분홍색
//	cv::Scalar color(255, 0, 255);
//	if (targetTargetInfo->mvpMatchingMPs[idx3]) {
//		color = cv::Scalar(255, 255, 0);
//	}
//	////visualize
//	cv::circle(debug, targetTargetInfo->mvMatchingPts[idx3], 2, color, -1);
//	cv::circle(debug, currInfo->mvMatchingPts[i] + ptBottom, 2, color, -1);
//}
//std::stringstream ssdir;
////ssdir << mpSystem->GetDirPath(0) << "/kfmatching";// << mpTargetFrame->GetKeyFrameID() << "_" << mpPrevFrame->GetKeyFrameID() << ".jpg";
//ssdir << mpSystem->GetDirPath(0) << "/kfmatching/" << mpTargetFrame->GetKeyFrameID() << ".jpg";
//imwrite(ssdir.str(), debug);
///////매칭 결과 저장
////////////////////////
///////////////////Fuse Map Points
//auto refMatch = matchInfo;
//auto target = matchInfo->mpTargetFrame;
//auto ref = mpTargetFrame;
//auto base = mpTargetFrame;
//std::vector<int> vRefIDXs;// = ref->mpMatchInfo->mvnTargetMatchingPtIDXs;
//std::vector<cv::Point2f> vRefPts;// = ref->mpMatchInfo->mvMatchingPts;
//std::vector<UVR_SLAM::MapPoint*> vRefMPs;

////매칭이 존재하는 포인트를 저장
//for (int i = 0; i < ref->mpMatchInfo->mvnTargetMatchingPtIDXs.size(); i++) {
//	if (!ref->mpMatchInfo->mvpMatchingMPs[i])
//		continue;
//	vRefIDXs.push_back(ref->mpMatchInfo->mvnTargetMatchingPtIDXs[i]);
//	vRefPts.push_back(ref->mpMatchInfo->mvMatchingPts[i]);
//	vRefMPs.push_back(ref->mpMatchInfo->mvpMatchingMPs[i]);
//}
//int nMatch = vRefPts.size();
//int nTotalTest = 0;
//while (target && nMatch > 50)
//	//while (target && nRes > 100)
//{
//	auto refMatchInfo = ref->mpMatchInfo;
//	auto targetMatchInfo = target->mpMatchInfo;
//	std::vector<cv::Point2f> tempPts;
//	std::vector<int> tempIDXs;
//	std::vector<UVR_SLAM::MapPoint*> tempMPs;

//	int n = targetMatchInfo->nMatch;
//	//////////포인트 매칭
//	//새로 생기기 전이고 mp가 없으면 연결
//	for (int i = 0; i < vRefIDXs.size(); i++) {
//		int baseIDX = vRefIDXs[i]; //base의 pt
//		int targetIDX = targetMatchInfo->mvnMatchingPtIDXs[baseIDX];

//		auto pt1 = vRefPts[i];
//		auto pMP1 = vRefMPs[i];
//		if (targetIDX < n)
//			continue;
//		auto pMP2 = targetMatchInfo->mvpMatchingMPs[targetIDX];

//		//auto pt2 = targetMatchInfo->mvMatchingPts[targetIDX];
//		//cv::line(debugging, pt1, pt2, cv::Scalar(255, 255, 0), 1);

//		////여기서 MP 확인 후 ID가 다르면 교체.
//		////N이 작아야 이전 프레임으로 이동 가능함.
//		if (pMP2 && !pMP2->isDeleted()) {
//			int id1 = pMP1->mnMapPointID;
//			int id2 = pMP2->mnMapPointID;
//			if(id1 != id2){
//				std::cout <<"fUSE ::"<< pMP1->mnMapPointID << ", " << pMP2->mnMapPointID << std::endl;
//				nTotalTest++;
//			}
//			int idx = targetMatchInfo->mvnTargetMatchingPtIDXs[targetIDX];
//			tempIDXs.push_back(idx);
//			tempPts.push_back(pt1);
//			tempMPs.push_back(pMP1);
//		}
//		/*if (!targetMatchInfo->mvpMatchingMPs[targetIDX] && targetIDX < n) {
//		int idx = targetMatchInfo->mvnTargetMatchingPtIDXs[targetIDX];
//		tempIDXs.push_back(idx);
//		tempPts.push_back(pt1);
//		}
//		else if (targetIDX >= n) {
//		}*/
//	}
//	//update
//	vRefPts = tempPts;
//	vRefIDXs = tempIDXs;
//	vRefMPs = tempMPs;
//	ref = target;
//	target = target->mpMatchInfo->mpTargetFrame;
//	nMatch = vRefIDXs.size();
//}
///////////////////Fuse Map Points
////200512




//window에 포함되는 KF를 설정하기.
//너무 많은 KF가 포함안되었으면 하고, 
//MP들이 잘 분배되었으면 함.
//lastframeid의 역할은?
//void UVR_SLAM::LocalMapper::UpdateKFs() {
//	mpFrameWindow->ClearLocalMapFrames();
//	auto mvpConnectedKFs = mpTargetFrame->GetConnectedKFs(15);
//	mpFrameWindow->AddFrame(mpTargetFrame);
//	int n = mpTargetFrame->GetFrameID();
//	for (auto iter = mvpConnectedKFs.begin(); iter != mvpConnectedKFs.end(); iter++) {
//		auto pKFi = *iter;
//		mpFrameWindow->AddFrame(pKFi);
//		if (pKFi->GetFrameID() == mpTargetFrame->GetFrameID())
//			std::cout << "??????????????" << std::endl;
//		(*iter)->mnLocalMapFrameID = n;
//	}
//}
//
//void UVR_SLAM::LocalMapper::UpdateMPs() {
//	int nUpdated = 0;
//	std::vector<cv::Point2f> vMatchingPts = std::vector<cv::Point2f>(mpTargetFrame->mvMatchingPts.begin(), mpTargetFrame->mvMatchingPts.end());
//	
//	for (int i = 0; i < vMatchingPts.size(); i++) {
//		UVR_SLAM::MapPoint* pMP = mpTargetFrame->mvpMatchingMPs[i];
//		if (!pMP || pMP->isDeleted()) {
//			continue;
//		}
//		
//		auto pt = vMatchingPts[i];
//		if (!mpTargetFrame->isInImage(pt.x, pt.y)) {
//			std::cout << "lm::updatemp::이미지밖::" << pt << std::endl;
//			continue;
//		}
//		pMP->AddDenseFrame(mpTargetFrame, pt);
//	}
//	/*for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {
//		UVR_SLAM::MapPoint* pMP = mpTargetFrame->mvpMPs[i];
//		if (pMP) {
//			if (pMP->isDeleted()) {
//				mpTargetFrame->RemoveMP(i);
//			}
//			else {
//				nUpdated++;
//				pMP->AddFrame(mpTargetFrame, i);
//				pMP->UpdateNormalAndDepth();
//			}
//		}
//	}*/
//	//std::cout << "Update MPs::" << nUpdated << std::endl;
//}
//
//void UVR_SLAM::LocalMapper::DeleteMPs() {
//	for (int i = 0; i < mvpDeletedMPs.size(); i++) {
//		delete mvpDeletedMPs[i];
//	}
//}
//
//void UVR_SLAM::LocalMapper::FuseMapPoints(int nn) {
//	int nTargetID = mpTargetFrame->GetFrameID();
//	const auto vpNeighKFs = mpTargetFrame->GetConnectedKFs(nn);
//
//	int n1 = 0;
//	std::chrono::high_resolution_clock::time_point fuse_start1 = std::chrono::high_resolution_clock::now();
//	for (int i = 0; i < vpNeighKFs.size(); i++) {
//		UVR_SLAM::Frame* pKFi = vpNeighKFs[i];
//		n1+=mpMatcher->KeyFrameFuseFeatureMatching2(mpTargetFrame, pKFi);
//	}
//	std::chrono::high_resolution_clock::time_point fuse_end1 = std::chrono::high_resolution_clock::now();
//	auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(fuse_end1 - fuse_start1).count();
//	double tttt1 = duration1 / 1000.0;
//
//	int n2 = 0;
//	std::chrono::high_resolution_clock::time_point fuse_start2 = std::chrono::high_resolution_clock::now();
//	for (int i = 0; i < vpNeighKFs.size(); i++) {
//		UVR_SLAM::Frame* pKFi = vpNeighKFs[i];
//		n2 += mpMatcher->KeyFrameFuseFeatureMatching(mpTargetFrame, pKFi);
//	}
//	std::chrono::high_resolution_clock::time_point fuse_end2 = std::chrono::high_resolution_clock::now();
//	auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(fuse_end2 - fuse_start2).count();
//	double tttt2 = duration2 / 1000.0;
//
//	for (int i = 0; i < vpNeighKFs.size(); i++) {
//		UVR_SLAM::Frame* pKFi = vpNeighKFs[i];
//		std::vector<cv::DMatch> vMatchInfoi;
//		mpMatcher->GMSMatching(mpTargetFrame, pKFi, vMatchInfoi);
//	}
//
//	std::cout << "New Fuse Test : " << tttt1 << ", " << tttt2<<", "<<"::"<<n1<<", "<<n2<< std::endl;
//}
//
//void UVR_SLAM::LocalMapper::FuseMapPoints()
//{
//
//	std::string mStrPath = mpSystem->GetDirPath(mpTargetFrame->GetKeyFrameID());;
//
//	int nn = 15;
//	int nTargetID = mpTargetFrame->GetFrameID();
//	const auto vpNeighKFs = mpTargetFrame->GetConnectedKFs(nn);
//
//	std::vector<UVR_SLAM::Frame*> vpTargetKFs;
//	for (std::vector<UVR_SLAM::Frame*>::const_iterator vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++)
//	{
//		UVR_SLAM::Frame* pKF = *vit;
//		if (pKF->mnFuseFrameID == nTargetID)
//			continue;
//		pKF->mnFuseFrameID = nTargetID;
//		vpTargetKFs.push_back(pKF);
//
//		const auto vpTempNeighKFs = pKF->GetConnectedKFs(10);
//		for (std::vector<UVR_SLAM::Frame*>::const_iterator vit2 = vpTempNeighKFs.begin(), vend2 = vpTempNeighKFs.end(); vit2 != vend2; vit2++)
//		{
//			UVR_SLAM::Frame* pKF2 = *vit2;
//			if (pKF2->mnFuseFrameID == nTargetID || pKF2->GetFrameID() == nTargetID)
//				continue;
//			pKF2->mnFuseFrameID = nTargetID;
//			vpTargetKFs.push_back(pKF2);
//		}
//	}
//	int nTotal = 0;
//	//std::cout << "LocalMapper::Fuse::" << vpTargetKFs.size() << std::endl;
//	std::vector<MapPoint*> vpMapPointMatches = mpTargetFrame->GetMapPoints();
//	for (int i = 0; i < vpTargetKFs.size(); i++) {
//		if (isStopLocalMapping())
//			break;
//		////int n1 = mpMatcher->MatchingForFuse(vpMapPointMatches, mpTargetFrame, vpTargetKFs[i], true);
//		//int n1 = mpMatcher->MatchingForFuse(vpMapPointMatches, vpTargetKFs[i]);
//		//std::vector<MapPoint*> vpMapPointMatches2 = vpTargetKFs[i]->GetMapPoints();
//		////int n2 = mpMatcher->MatchingForFuse(vpMapPointMatches2, vpTargetKFs[i], mpTargetFrame, false);
//		//int n2 = mpMatcher->MatchingForFuse(vpMapPointMatches2,mpTargetFrame);
//		////std::cout << "LocalMapper::MatchingFuse::" <<vpTargetKFs[i]->GetFrameID()<<"::"<< n1<<", "<<n2 << std::endl;
//		int n1 = mpMatcher->MatchingForFuse(mpTargetFrame, vpTargetKFs[i]);
//		int n2 = mpMatcher->MatchingForFuse(vpTargetKFs[i], mpTargetFrame);
//
//		////plane matching test
//		//std::vector<cv::DMatch> atempMatches;
//		//int n = mpMatcher->MatchingWithLabeling(mpTargetFrame->mvKeyPoints, vpTargetKFs[i]->mvKeyPoints, mpTargetFrame->mPlaneDescriptor, vpTargetKFs[i]->mPlaneDescriptor, mpTargetFrame->mPlaneIdxs, vpTargetKFs[i]->mPlaneIdxs, atempMatches);
//		//{
//		//	cv::Mat img1 = mpTargetFrame->GetOriginalImage();
//		//	cv::Mat img2 = vpTargetKFs[i]->GetOriginalImage();
//
//		//	cv::Point2f ptBottom = cv::Point2f(0, img1.rows);
//
//		//	cv::Rect mergeRect1 = cv::Rect(0, 0, img1.cols, img1.rows);
//		//	cv::Rect mergeRect2 = cv::Rect(0, img1.rows, img1.cols, img1.rows);
//
//		//	cv::Mat debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
//		//	img1.copyTo(debugging(mergeRect1));
//		//	img2.copyTo(debugging(mergeRect2));
//
//		//	for (int j = 0; j < atempMatches.size(); j++) {
//		//		cv::line(debugging, mpTargetFrame->mvKeyPoints[atempMatches[j].queryIdx].pt, vpTargetKFs[i]->mvKeyPoints[atempMatches[j].trainIdx].pt + ptBottom, cv::Scalar(255), 1);
//		//	}
//		//	std::stringstream ss;
//		//	ss << mStrPath.c_str() << "/floor_" << mpTargetFrame->GetFrameID() << "_" << vpTargetKFs[i]->GetFrameID() << ".jpg";
//		//	imwrite(ss.str(), debugging);
//		//}
//		nTotal += (n1 + n2);
//	}
//	//std::cout << "Original Fuse : " << nTotal << std::endl;
//}
//int UVR_SLAM::LocalMapper::CreateMapPoints() {
//	int nRes = 0;
//	auto mvpConnectedKFs = mpTargetFrame->GetConnectedKFs();
//	for (int i = 0; i < mvpConnectedKFs.size(); i++) {
//		if (CheckNewKeyFrames()){
//			std::cout << "LocalMapping::CreateMPs::Break::" << std::endl;
//			break;
//		}
//		nRes += CreateMapPoints(mpTargetFrame, mvpConnectedKFs[i]);
//	}
//	std::cout << "LocalMapping::CreateMPs::End::" << nRes << std::endl;
//	return nRes;
//}
//
//int UVR_SLAM::LocalMapper::CreateMapPoints(UVR_SLAM::Frame* pCurrKF, UVR_SLAM::Frame* pLastKF) {
//	//debugging image
//	cv::Mat lastImg = pLastKF->GetOriginalImage();
//	cv::Mat currImg = pCurrKF->GetOriginalImage();
//	//cv::Rect mergeRect1 = cv::Rect(0, 0, lastImg.cols, lastImg.rows);
//	//cv::Rect mergeRect2 = cv::Rect(lastImg.cols, 0, lastImg.cols, lastImg.rows);
//	//cv::Mat debugging = cv::Mat::zeros(lastImg.rows, lastImg.cols * 2, lastImg.type());
//	cv::Rect mergeRect1 = cv::Rect(0, 0,			lastImg.cols, lastImg.rows);
//	cv::Rect mergeRect2 = cv::Rect(0, lastImg.rows, lastImg.cols, lastImg.rows);
//	cv::Mat debugging = cv::Mat::zeros(lastImg.rows * 2, lastImg.cols, lastImg.type());
//	lastImg.copyTo(debugging(mergeRect1));
//	currImg.copyTo(debugging(mergeRect2));
//	//cv::cvtColor(debugging, debugging, CV_RGBA2BGR);
//	//debugging.convertTo(debugging, CV_8UC3);
//
//	//preprocessing
//	bool bNearBaseLine = false;
//	if (!pLastKF->CheckBaseLine(pCurrKF)) {
//		std::cout << "CreateMapPoints::Baseline error" << std::endl;
//		bNearBaseLine = true;
//		return 0;
//	}
//
//	cv::Mat Rprev, Tprev, Rcurr, Tcurr;
//	pLastKF->GetPose(Rprev, Tprev);
//	pCurrKF->GetPose(Rcurr, Tcurr);
//
//	cv::Mat mK = pCurrKF->mK.clone();
//
//	cv::Mat RprevInv = Rprev.t();
//	cv::Mat RcurrInv = Rcurr.t();
//	float invfx = 1.0 / mK.at<float>(0, 0);
//	float invfy = 1.0 / mK.at<float>(1, 1);
//	float cx = mK.at<float>(0, 2);
//	float cy = mK.at<float>(1, 2);
//	float ratioFactor = 1.5f*pCurrKF->mfScaleFactor;
//
//	cv::Mat P0 = cv::Mat::zeros(3, 4, CV_32FC1);
//	Rprev.copyTo(P0.colRange(0, 3));
//	Tprev.copyTo(P0.col(3));
//	cv::Mat P1 = cv::Mat::zeros(3, 4, CV_32FC1);
//	Rcurr.copyTo(P1.colRange(0, 3));
//	Tcurr.copyTo(P1.col(3));
//	
//	cv::Mat O1 = pLastKF->GetCameraCenter();
//	cv::Mat O2 = pCurrKF->GetCameraCenter();
//
//	cv::Mat F12 = mpMatcher->CalcFundamentalMatrix(Rcurr, Tcurr, Rprev, Tprev, mK);
//	int thresh_epi_dist = 50;
//	float thresh_reprojection = 16.0;
//	int count = 0;
//
//	cv::RNG rng(12345);
//
//	//두 키프레임 사이의 매칭 정보 초기화
//	//mpFrameWindow->mvMatchInfos.clear();
//
//	
//	for (int i = 0; i < pCurrKF->mvKeyPoints.size(); i++) {
//		if (pCurrKF->mvbMPInliers[i])
//			continue;
//		int matchIDX, kidx;
//		cv::KeyPoint kp = pCurrKF->mvKeyPoints[i];
//		cv::Point2f pt = pCurrKF->mvKeyPoints[i].pt;
//		cv::Mat desc = pCurrKF->matDescriptor.row(i);
//
//		float sigma = pCurrKF->mvLevelSigma2[kp.octave];
//		bool bMatch = mpMatcher->FeatureMatchingWithEpipolarConstraints(matchIDX, pLastKF, F12, kp, desc, sigma, thresh_epi_dist);
//		if (bMatch) {
//			if (!pLastKF->mvbMPInliers[matchIDX]) {
//				
//				cv::KeyPoint kp2 = pLastKF->mvKeyPoints[matchIDX];
//				cv::Mat X3D;
//				if(!Triangulate(kp2.pt, kp.pt, mK*P0, mK*P1, X3D))
//					continue;
//				cv::Mat Xcam1 = Rprev*X3D + Tprev;
//				cv::Mat Xcam2 = Rcurr*X3D + Tcurr;
//				//SetLogMessage("Triangulation\n");
//				if (!CheckDepth(Xcam1.at<float>(2)) || !CheckDepth(Xcam2.at<float>(2))) {
//					continue;
//				}
//
//				if (!CheckReprojectionError(Xcam1, mK, kp2.pt, 5.991*mpTargetFrame->mvLevelSigma2[kp.octave]) || !CheckReprojectionError(Xcam2, mK, kp.pt, thresh_reprojection))
//				{
//					continue;
//				}
//
//				if (!CheckScaleConsistency(X3D, O2, O1, ratioFactor, pCurrKF->mvScaleFactors[kp.octave], pLastKF->mvScaleFactors[kp2.octave]))
//				{
//					continue;
//				}
//
//				UVR_SLAM::MapPoint* pMP2 = new UVR_SLAM::MapPoint(mpTargetFrame,X3D, desc);
//				pMP2->AddFrame(pLastKF, matchIDX);
//				pMP2->AddFrame(pCurrKF, i);
//				pMP2->mnFirstKeyFrameID = pCurrKF->GetKeyFrameID();
//				mpSystem->mlpNewMPs.push_back(pMP2);
//				//mvpNewMPs.push_back(pMP2);
//				//mDescNewMPs.push_back(pMP2->GetDescriptor());
//
//				//매칭 정보 추가
//				cv::DMatch tempMatch;
//				tempMatch.queryIdx = i;
//				tempMatch.trainIdx = matchIDX;
//				//mpFrameWindow->mvMatchInfos.push_back(tempMatch);
//
//				cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//				//cv::line(debugging, pLastKF->mvKeyPoints[matchIDX].pt, pCurrKF->mvKeyPoints[i].pt + cv::Point2f(lastImg.cols, 0), cv::Scalar(255, 0, 255), 1);
//				cv::line(debugging, pLastKF->mvKeyPoints[matchIDX].pt, pCurrKF->mvKeyPoints[i].pt + cv::Point2f(0, lastImg.rows), color, 1);
//				count++;
//			}
//		}
//	}
//	cv::imshow("LocalMapping::CreateMPs", debugging); cv::waitKey(10);
//	//std::cout << "CreateMapPoints=" << count << std::endl;
//	return count;
//}
//
//
//bool UVR_SLAM::LocalMapper::Triangulate(cv::Point2f pt1, cv::Point2f pt2, cv::Mat P1, cv::Mat P2, cv::Mat& X3D) {
//
//	cv::Mat A(4, 4, CV_32F);
//	A.row(0) = pt1.x*P1.row(2) - P1.row(0);
//	A.row(1) = pt1.y*P1.row(2) - P1.row(1);
//	A.row(2) = pt2.x*P2.row(2) - P2.row(0);
//	A.row(3) = pt2.y*P2.row(2) - P2.row(1);
//
//	cv::Mat u, w, vt;
//	cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
//	
//	float a = w.at<float>(3);
//	if (a <= 0.0001) {
//		//std::cout << a << ":" << X3D << std::endl;
//		return false;
//	}
//	/*cv::Mat X2 = vt.row(3).t();
//	X2 = X2.rowRange(0, 3) / X2.at<float>(3);
//
//	cv::Mat B;
//	cv::reduce(abs(A), B, 0, CV_REDUCE_MAX);
//	for (int i = 0; i < 4; i++)
//		if (B.at<float>(i) < 1.0)
//			B.at<float>(i) = 1.0;
//	B = 1.0 / B;
//	B = cv::Mat::diag(B);
//	
//	cv::SVD::compute(A*B, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
//	vt = B*vt;*/
//
//	cv::Mat x3D = vt.row(3).t();
//	
//	/*
//	if (abs(x3D.at<float>(3)) < 0.01){
//		std::cout << "abc:" << x3D.at<float>(3) <<"::"<< x3D.rowRange(0, 3) / x3D.at<float>(3)<< std::endl;
//		return false;
//	}
//	else if (abs(x3D.at<float>(3)) == 0.0)
//		std::cout << "cccc:" << x3D.at<float>(3) << std::endl;
//	*/
//
//	if (x3D.at<float>(3) == 0.0){
//		return false;
//	};
//	X3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
//	//std::cout <<"X3D = "<<X2.t()<< X3D.t() << std::endl;
//	return true;
//}
//
//bool UVR_SLAM::LocalMapper::CheckDepth(float depth) {
//	if (depth < 0)
//		return false;
//	return true;
//}
//
//bool UVR_SLAM::LocalMapper::CheckReprojectionError(cv::Mat x3D, cv::Mat K, cv::Point2f pt, float thresh) {
//	cv::Mat reproj1 = K*x3D;
//	reproj1 /= x3D.at<float>(2);
//	float squareError1 = (reproj1.at<float>(0) - pt.x)*(reproj1.at<float>(0) - pt.x) + (reproj1.at<float>(1) - pt.y)*(reproj1.at<float>(1) - pt.y);
//	if (squareError1>thresh)
//		return false;
//	return true;
//}
//
//bool UVR_SLAM::LocalMapper::CheckScaleConsistency(cv::Mat x3D, cv::Mat Ow1, cv::Mat Ow2, float fRatioFactor, float fScaleFactor1, float fScaleFactor2) {
//	cv::Mat normal1 = x3D - Ow1;
//	float dist1 = cv::norm(normal1);
//
//	cv::Mat normal2 = x3D - Ow2;
//	float dist2 = cv::norm(normal2);
//
//	if (dist1 == 0 || dist2 == 0)
//		return false;
//
//	const float ratioDist = dist2 / dist1;
//	const float ratioOctave = fScaleFactor1 / fScaleFactor2;
//
//	if (ratioDist*fRatioFactor<ratioOctave || ratioDist>ratioOctave*fRatioFactor)
//		return false;
//	return true;
//}
//
//void UVR_SLAM::LocalMapper::CalculateKFConnections() {
//	std::map<UVR_SLAM::Frame*, int> mmpCandidateKFs;
//	int nTargetID = mpTargetFrame->GetFrameID();
//	
//	auto mvpDenseMPs = mpTargetFrame->GetDenseVectors();
//	for (int i = 0; i < mvpDenseMPs.size(); i++) {
//		UVR_SLAM::MapPoint* pMP = mvpDenseMPs[i];
//		if (!pMP || pMP->isDeleted())
//			continue;
//		auto mmpMP = pMP->GetConnedtedDenseFrames();
//		for (auto biter = mmpMP.begin(), eiter = mmpMP.end(); biter != eiter; biter++) {
//			UVR_SLAM::Frame* pCandidateKF = biter->first;
//			if (nTargetID == pCandidateKF->GetFrameID())
//				continue;
//			/*if (mmpCandidateKFs.find(pCandidateKF) == mmpCandidateKFs.end()) {
//			std::cout << "LocalMapping::Not connected kf" << std::endl;
//			}*/
//			mmpCandidateKFs[pCandidateKF]++;
//		}
//	}
//
//	/*auto mvpTemporalCandidateKFs = mpFrameWindow->GetLocalMapFrames();
//	
//	for (int i = 0; i < mvpTemporalCandidateKFs.size(); i++) {
//		if (nTargetID == mvpTemporalCandidateKFs[i]->GetFrameID())
//			continue;
//		mmpCandidateKFs[mvpTemporalCandidateKFs[i]] = 0;
//		auto mvpTemp2 = mvpTemporalCandidateKFs[i]->GetConnectedKFs();
//		for (int j = 0; j < mvpTemp2.size(); j++) {
//			if (nTargetID == mvpTemp2[j]->GetFrameID())
//				continue;
//			mmpCandidateKFs[mvpTemp2[j]] = 0;
//		}
//	}*/
//	
//	int Nkf = mmpCandidateKFs.size();
//	//auto mvpLocalMPs = mpTargetFrame->GetMapPoints();
//	//for (int i = 0; i < mvpLocalMPs.size(); i++) {
//	//	
//	//	UVR_SLAM::MapPoint* pMP = mvpLocalMPs[i];
//	//	if (!pMP)
//	//		continue;
//	//	if (pMP->isDeleted())
//	//		continue;
//	//	auto mmpMP = pMP->GetConnedtedFrames();
//	//	for (auto biter = mmpMP.begin(), eiter = mmpMP.end(); biter != eiter; biter++) {
//	//		UVR_SLAM::Frame* pCandidateKF = biter->first;
//	//		if (nTargetID == pCandidateKF->GetFrameID())
//	//			continue;
//	//		/*if (mmpCandidateKFs.find(pCandidateKF) == mmpCandidateKFs.end()) {
//	//			std::cout << "LocalMapping::Not connected kf" << std::endl;
//	//		}*/
//	//		mmpCandidateKFs[pCandidateKF]++;
//	//	}
//	//}
//	//sort mmp
//	std::vector<std::pair<int,UVR_SLAM::Frame*>> vPairs;
//
//	/*if (mpPrevKeyFrame) {
//		mpTargetFrame->AddKF(mpPrevKeyFrame, 0);
//		mpPrevKeyFrame->AddKF(mpTargetFrame, 0);
//	}
//	if (mpPPrevKeyFrame) {
//		mpTargetFrame->AddKF(mpPPrevKeyFrame, 0);
//		mpPPrevKeyFrame->AddKF(mpTargetFrame, 0);
//	}*/
//
//	for (auto biter = mmpCandidateKFs.begin(), eiter = mmpCandidateKFs.end(); biter != eiter; biter++) {
//		UVR_SLAM::Frame* pKF = biter->first;
//		int nCount = biter->second;
//		if (nCount > 10) {
//			//mpTargetFrame->AddKF(pKF);
//			//vPairs.push_back(std::make_pair(nCount, pKF));
//			mpTargetFrame->AddKF(pKF, nCount);
//			pKF->AddKF(mpTargetFrame, nCount);
//		}
//	}
//	//test
//	/*std::cout << "tttttttt" << std::endl;
//	auto temp1 = mpTargetFrame->GetConnectedKFs();
//	auto temp2 = mpTargetFrame->GetConnectedKFsWithWeight();
//	for (auto iter = temp2.begin(); iter != temp2.end(); iter++) {
//		std::cout << iter->second->GetFrameID() << "::" << iter->first << std::endl;
//	}
//	for (auto iter = temp1.begin(); iter != temp1.end(); iter++) {
//		std::cout << (*iter)->GetFrameID() << std::endl;
//	}
//	std::cout << "???????" << std::endl;*/
//}
//
//int UVR_SLAM::LocalMapper::Test() {
//	auto mvpLocalFrames = mpTargetFrame->GetConnectedKFs();
//
//	cv::Mat mK = mpTargetFrame->mK.clone();
//	cv::Mat Rcurr, Tcurr;
//	mpTargetFrame->GetPose(Rcurr, Tcurr);
//	cv::Mat O2 = mpTargetFrame->GetCameraCenter();
//	cv::Mat P1 = cv::Mat::zeros(3, 4, CV_32FC1);
//	Rcurr.copyTo(P1.colRange(0, 3));
//	Tcurr.copyTo(P1.col(3));
//
//	int nTotal = 0;
//
//	//target keyframe information
//	auto mvpMPs = mpTargetFrame->GetMapPoints();
//	auto mvpCurrOPs = mpTargetFrame->GetObjectVector();
//	float fMinCurr, fMaxCurr;
//	mpTargetFrame->GetDepthRange(fMinCurr, fMaxCurr);
//	//std::cout << "Depth::Curr::" <<mpTargetFrame->GetKeyFrameID()<<"::"<< fMaxCurr << std::endl << std::endl << std::endl;
//	
//	for (int i = 0; i < mvpLocalFrames.size(); i++) {
//		if (isStopLocalMapping())
//			break;
//		UVR_SLAM::Frame* pKF = mvpLocalFrames[i];
//
//		//neighbor keyframe information
//		auto mvpMPs2 = pKF->GetMapPoints();
//		auto mvpPrevOPs = pKF->GetObjectVector();
//		float fMinNeighbor, fMaxNeighbor;
//		pKF->GetDepthRange(fMinNeighbor, fMaxNeighbor);
//		//std::cout << "Depth::Neighbor::" << fMaxNeighbor << std::endl << std::endl << std::endl;
//		//preprocessing
//		/*bool bNearBaseLine = false;
//		if (!pKF->CheckBaseLine(mpTargetFrame)) {
//			std::cout << "CreateMapPoints::Baseline error" << std::endl;
//			bNearBaseLine = true;
//			continue;
//		}*/
//
//		cv::Mat Rprev, Tprev;
//		pKF->GetPose(Rprev, Tprev);
//
//		cv::Mat RprevInv = Rprev.t();
//		cv::Mat RcurrInv = Rcurr.t();
//		float invfx = 1.0 / mK.at<float>(0, 0);
//		float invfy = 1.0 / mK.at<float>(1, 1);
//		float cx = mK.at<float>(0, 2);
//		float cy = mK.at<float>(1, 2);
//		float ratioFactor = 1.5f*mpTargetFrame->mfScaleFactor;
//
//		cv::Mat P0 = cv::Mat::zeros(3, 4, CV_32FC1);
//		Rprev.copyTo(P0.colRange(0, 3));
//		Tprev.copyTo(P0.col(3));
//		
//		cv::Mat O1 = pKF->GetCameraCenter();
//
//		//cv::Mat F12 = mpMatcher->CalcFundamentalMatrix(Rcurr, Tcurr, Rprev, Tprev, mK);
//		int thresh_epi_dist = 50;
//		float thresh_reprojection = 9.0;
//		int count = 0;
//
//		//tracked와 non-tracked를 구분하는 것
//		pKF->UpdateMapInfo(true);
//		
//		std::vector<cv::DMatch> vMatches;
//		mpMatcher->KeyFrameFeatureMatching(mpTargetFrame, pKF, vMatches);
//		int nTemp = 0;
//		for (int j = 0; j < vMatches.size(); j++) {
//			int idx1 = vMatches[j].queryIdx;
//			int idx2 = vMatches[j].trainIdx;
//			if (mpTargetFrame->mvpMPs[idx1])
//				continue;
//			//if (mvpCurrOPs[idx1] != mvpPrevOPs[idx2] || mvpCurrOPs[idx1] == ObjectType::OBJECT_FLOOR || mvpPrevOPs[idx2] == ObjectType::OBJECT_FLOOR)
//			if ((mvpCurrOPs[idx1] != mvpPrevOPs[idx2]) || mvpCurrOPs[idx1] == ObjectType::OBJECT_PERSON || mvpPrevOPs[idx2] == ObjectType::OBJECT_PERSON)
//				continue;
//			cv::KeyPoint kp1 = mpTargetFrame->mvKeyPoints[idx1];
//			cv::KeyPoint kp2 = pKF->mvKeyPoints[idx2];
//
//			//epi constraints
//			/*float fepi;			
//			mpMatcher->CheckEpiConstraints(F12, kp1.pt, kp2.pt, pKF->mvLevelSigma2[kp2.octave], fepi);
//			if (fepi > 2.0)
//				continue;*/
//
//			cv::Mat X3D;
//			if (!Triangulate(kp2.pt, kp1.pt, mK*P0, mK*P1, X3D)){
//				continue;
//			}
//			cv::Mat Xcam1 = Rcurr*X3D + Tcurr; 
//			cv::Mat Xcam2 = Rprev*X3D + Tprev;
//			float depth1 = Xcam1.at<float>(2);
//			float depth2 = Xcam2.at<float>(2);
//			//SetLogMessage("Triangulation\n");
//			if (!CheckDepth(depth1) || !CheckDepth(depth2)) {
//				continue;
//			}
//			//if (depth1 > fMaxCurr || depth2 > fMaxNeighbor){
//			//	//std::cout << depth1 <<", "<< fMaxCurr <<", "<< depth2 << ", " << fMaxNeighbor << std::endl;
//			//	continue;
//			//}
//			if (!CheckReprojectionError(Xcam2, mK, kp2.pt, 5.991*pKF->mvLevelSigma2[kp2.octave]) || !CheckReprojectionError(Xcam1, mK, kp1.pt, 5.991*mpTargetFrame->mvLevelSigma2[kp1.octave]))
//			{
//				//std::cout << "LocalMapping::CreateMP::Reprojection" << std::endl;
//				continue;
//			}
//
//			//if (!CheckScaleConsistency(X3D, O2, O1, ratioFactor, mpTargetFrame->mvScaleFactors[kp1.octave], pKF->mvScaleFactors[kp2.octave]))
//			//{
//			//	//std::cout << "LocalMapping::CreateMP::Scale" << std::endl;
//			//	//continue;
//			//}
//
//			cv::Mat desc = mpTargetFrame->matDescriptor.row(idx1);
//			UVR_SLAM::MapPoint* pMP = new UVR_SLAM::MapPoint(mpTargetFrame, X3D, desc);
//			pMP->AddFrame(pKF, idx2);
//			pMP->AddFrame(mpTargetFrame, idx1);
//			pMP->UpdateNormalAndDepth();
//			pMP->mnFirstKeyFrameID = mpTargetFrame->GetKeyFrameID();
//			mpSystem->mlpNewMPs.push_back(pMP);
//
//			//mpTargetFrame->mTrackedDescriptor.push_back(mpTargetFrame->matDescriptor.row(idx1));
//			//mpTargetFrame->mvTrackedIdxs.push_back(idx1);
//			pKF->mTrackedDescriptor.push_back(pKF->matDescriptor.row(idx2));
//			pKF->mvTrackedIdxs.push_back(idx2);
//
//			nTemp++;
//
//			////save image
//			//line(debugging, mpTargetFrame->mvKeyPoints[idx1].pt, pKF->mvKeyPoints[idx2].pt + ptBottom, cv::Scalar(255, 0, 255), 1);
//			
//		}
//		nTotal += nTemp;
//
//		////save image
//		//std::stringstream ss;
//		//ss << "../../bin/SLAM/kf_matching/" << mpTargetFrame->GetFrameID() << "_" << pKF->GetFrameID() << ".jpg";
//		//imwrite(ss.str(), debugging);
//		////save image
//	}
//	return nTotal;
//}
//
//void UVR_SLAM::LocalMapper::KeyframeMarginalization() {
//
//	int nThreshKF = 5;
//
//	//auto mvpLocalMPs = mpTargetFrame->GetMapPoints();
//	auto mvpLocalFrames = mpTargetFrame->GetConnectedKFs();
//	int nKFs = mvpLocalFrames.size();
//	int nMPs = 0;
//	if (nKFs < nThreshKF)
//		return;
//	//여기에 true는 계속 나오는 MP이고 false는 별로 나오지 않는 MP이다.
//	//없애는게 나을지도 모르는 것들
//	/*std::vector<bool> mvbMPs(mvpLocalMPs.size(), false);
//	for (int i = 0; i < mvpLocalMPs.size(); i++) {
//		UVR_SLAM::MapPoint* pMP = mvpLocalMPs[i];
//		if (!pMP)
//			continue;
//		if (pMP->isDeleted())
//			continue;
//		int nObs = pMP->GetNumConnectedFrames();
//		double ratio = ((double)nObs) / nKFs;
//		if (nObs > 2) {
//			mvbMPs[i] = true;
//			
//		}
//		else {
//			nMPs++;
//		}
//	}*/
//
//	auto mvpConnectedKFs = mpTargetFrame->GetConnectedKFs();
//	for (int i = 0; i < mvpConnectedKFs.size(); i++) {
//		mpMatcher->KeyFrameFuseFeatureMatching(mpTargetFrame, mvpConnectedKFs[i]);
//	}
//
//	std::cout << "TESt:::" << nMPs <<", "<< mvpConnectedKFs.size()<< std::endl;
//}
//////////////////코드 백업
///////////////////////////////////////////////////////////////////////////////////////////////////////
