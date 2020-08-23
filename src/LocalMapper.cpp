#include <LocalMapper.h>
#include <CandidatePoint.h>
#include <Frame.h>
#include <System.h>
#include <Map.h>
#include <MapGrid.h>
#include <MapPoint.h>
#include <Matcher.h>
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
	mpMap->AddFrame(mpTargetFrame);
	mpTargetFrame->SetBowVec(mpSystem->fvoc); //키프레임 파트로 옮기기
	
	////이게 필요한지?
	//이전 키프레임 정보 획득 후 현재 프레임을 윈도우에 추가
	//mpPrevKeyFrame = mpFrameWindow->back();
	//mpFrameWindow->push_back(mpTargetFrame);
	
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

	while (1) {

		if (CheckNewKeyFrames()) {
			SetDoingProcess(true);
			std::chrono::high_resolution_clock::time_point lm_start = std::chrono::high_resolution_clock::now();
			////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////200412
			
			ProcessNewKeyFrame();
			int nTargetID = mpTargetFrame->GetFrameID();
			std::cout << "lm::start::" << mpTargetFrame->GetFrameID() << std::endl;

			///////debug
			/*cv::Mat prevImg = mpPrevKeyFrame->GetOriginalImage();
			cv::Mat currImg = mpTargetFrame->GetOriginalImage();
			cv::Point2f ptBottom = cv::Point2f(prevImg.cols,0);
			cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
			cv::Rect mergeRect2 = cv::Rect(prevImg.cols, 0, prevImg.cols, prevImg.rows);
			cv::Mat debugImg = cv::Mat::zeros(prevImg.rows, prevImg.cols*2, prevImg.type());
			prevImg.copyTo(debugImg(mergeRect1));
			currImg.copyTo(debugImg(mergeRect2));
			std::vector<cv::Point2f> vTempPts1, vTempPts2, vTempPts3, vTempPts4;
			std::vector<uchar> vbTempInliers;*/
			///////debug

			//////////////업데이트 맵포인트
			auto matchInfo = mpTargetFrame->mpMatchInfo;
			auto targetMatchingPTs = matchInfo->GetMatchingPts();
			auto targetMatchingMPs = matchInfo->GetMatchingMPs();
			for (int i = 0; i < targetMatchingPTs.size(); i++) {
				auto pMPi = targetMatchingMPs[i];
				auto pt = targetMatchingPTs[i];
				if (pMPi && !pMPi->isDeleted()) {
					pMPi->AddFrame(matchInfo, i);
					
					//int idx = matchInfo->mvnMatchingIDXs[i];
					//auto prevPt = mpPrevKeyFrame->mpMatchInfo->GetMatchingPt(idx);
					//circle(debugImg, prevPt, 2, cv::Scalar(255, 255, 0), -1);
					//circle(debugImg, pt+ ptBottom, 2, cv::Scalar(255, 255, 0), -1);
					//vTempPts1.push_back(prevPt);
					//vTempPts2.push_back(pt);
				}
			}

			//cv::Mat E12 = cv::findEssentialMat(vTempPts1, vTempPts2, mK, cv::FM_RANSAC, 0.999, 1.0, vbTempInliers);
			//for (unsigned long i = 0; i < vbTempInliers.size(); i++) {
			//	if (vbTempInliers[i]) {
			//		vTempPts3.push_back(vTempPts1[i]);
			//		vTempPts4.push_back(vTempPts2[i]);
			//		//cv::circle(debugImg, vTempPts1[i], 2, cv::Scalar(255, 0, 255), -1);
			//		//cv::circle(debugImg, vTempPts2[i] + ptBottom, 2, cv::Scalar(255, 0, 255), -1);
			//	}
			//}
			//std::stringstream ssatdir;
			//ssatdir << mpSystem->GetDirPath(0) << "/testmatching/mapping_test2_" << mpTargetFrame->GetKeyFrameID() << "_" << mpPrevKeyFrame->GetKeyFrameID() << ".jpg";
			////imwrite(ssatdir.str(), debugImg);

			//mpMap->AddFrame(mpTargetFrame);
			mpTargetFrame->ComputeSceneMedianDepth();
			//////////////업데이트 맵포인트

			//매핑 테스트
			/*if(mpPPrevKeyFrame && mpPrevKeyFrame){
				
			}*/
			///////////////////////////////////////////////////////
			/////Delayed Triangulation
			cv::Mat ddddbug;
			mpPrevKeyFrame->mpMatchInfo->SetMatchingPoints();
			//이전 MP, CP, 새로운 CP가 같이 포함되어야 함.

			int nCreated = 0;
			{
				////////New Matching & Create & Delayed CP test
				cv::Mat debugMatch;
				std::vector<cv::Point2f> vMatchPPrevPts, vMatchPrevPts, vMatchCurrPts;
				std::vector<cv::Point2f> vMappingPPrevPts, vMappingPrevPts, vMappingCurrPts;
				std::vector<bool> vbInliers;
				std::vector<int> vnIDXs;
				std::vector<int> vnPrevOctaves;
				auto vPrevCPs = mpPrevKeyFrame->mpMatchInfo->GetMatchingCPs();
				auto vPrevCandidatePoints = mpPrevKeyFrame->mpMatchInfo->GetMatchingPts(vnPrevOctaves);
				mpMatcher->OpticalMatchingForMapping(mpTargetFrame, mpPrevKeyFrame, vPrevCandidatePoints, vMatchPrevPts, vMatchCurrPts, vnIDXs, vbInliers, debugMatch);
				std::cout << "LM::" << vMatchPrevPts.size() << std::endl;
				nCreated = CreateMapPoints(mpTargetFrame, mpPrevKeyFrame, vMatchPrevPts, vMatchCurrPts, vPrevCPs, vnIDXs, debugMatch);
				mpPrevKeyFrame->mpMatchInfo->mMatchedImage = debugMatch.clone();

				cv::Mat ttttddddebug = cv::Mat::zeros(debugMatch.rows, debugMatch.cols * 2, debugMatch.type());
				mpPPrevKeyFrame->mpMatchInfo->mMatchedImage.copyTo(ttttddddebug(cv::Rect(0, 0, debugMatch.cols, debugMatch.rows)));
				mpPrevKeyFrame->mpMatchInfo->mMatchedImage.copyTo(ttttddddebug(cv::Rect(debugMatch.cols, 0, debugMatch.cols, debugMatch.rows)));
				std::stringstream sstdir;
				sstdir << mpSystem->GetDirPath(0) << "/testmatching/mapping_test2_" << mpTargetFrame->GetKeyFrameID() << "_" << mpPrevKeyFrame->GetKeyFrameID() << ".jpg";
				imwrite(sstdir.str(), ttttddddebug);

				std::cout << "LM?????????????????????????????????" << std::endl;
			}

			//std::cout << "mapping::1::" << mpPrevKeyFrame->mpMatchInfo->nPrevNumCPs << ", " << mpPrevKeyFrame->mpMatchInfo->mvTempPts.size() << std::endl;
			cv::Mat debugMatch, testDebugMatch;
			std::vector<cv::Point2f> vMatchPPrevPts, vMatchPrevPts, vMatchCurrPts;
			std::vector<cv::Point2f> vMappingPPrevPts, vMappingPrevPts, vMappingCurrPts;
			std::vector<bool> vbInliers;
			std::vector<int> vnIDXs;
			std::vector<int> vnPrevOctaves;
			auto vPrevCPs = mpPrevKeyFrame->mpMatchInfo->GetMatchingCPs();
			auto vPrevCandidatePoints = mpPrevKeyFrame->mpMatchInfo->GetMatchingPts(vnPrevOctaves);

			//mpMatcher->OpticalMatchingForMapping(mpTargetFrame, mpPrevKeyFrame, vPrevCandidatePoints, vMatchPrevPts, vMatchCurrPts, vnIDXs, vbInliers, debugMatch);
			
			/*cv::Mat testMatchingImg;
			mpMatcher->TestOpticalMatchingForMapping2(mpTargetFrame, mpPrevKeyFrame, mpPPrevKeyFrame, testMatchingImg);
			cv::Mat ttttddddebug = cv::Mat::zeros(testMatchingImg.rows, testMatchingImg.cols * 2, testMatchingImg.type());
			mpTargetFrame->mpMatchInfo->mMatchedImage.copyTo(ttttddddebug(cv::Rect(0, 0, testMatchingImg.cols, testMatchingImg.rows)));
			testMatchingImg.copyTo(ttttddddebug(cv::Rect(testMatchingImg.cols, 0, testMatchingImg.cols, testMatchingImg.rows)));
			std::stringstream sstdir;
			sstdir << mpSystem->GetDirPath(0) << "/kfmatching/mapping_test2_" << mpTargetFrame->GetKeyFrameID() << "_" << mpPrevKeyFrame->GetKeyFrameID() << ".jpg";
			imwrite(sstdir.str(), ttttddddebug);*/

			/*
			vMatchPPrevPts.clear();
			vMatchPrevPts.clear();
			vMatchCurrPts.clear();
			vnIDXs.clear();
			vbInliers.clear();
			mpMatcher->OpticalMatchingForMapping(mpTargetFrame, mpPrevKeyFrame, mpPPrevKeyFrame, vMatchPPrevPts, vMatchPrevPts, vMatchCurrPts, vnIDXs, vbInliers, debugMatch);
			*/
			//CP와 새로운 포인트 나누기
			std::vector<int> vnMappingIDXs;
			std::vector<bool> vbMappingInliers;
			std::vector<CandidatePoint*> vpDelayedCPs;
			std::vector<cv::Point2f> vDelayedPts;
			if (vMatchPrevPts.size() >= 10) {
				
			}
			else {
				std::cout << "????????????????::" << vMatchPrevPts.size() << std::endl;
			}
			int nTarget = mpPrevKeyFrame->mpMatchInfo->nPrevNumCPs;
			for (int i = 0; i < vMatchPPrevPts.size(); i++) {
				int idx = vnIDXs[i];
				if (idx >= nTarget) {
					vnMappingIDXs.push_back(idx);
					vMappingPPrevPts.push_back(vMatchPPrevPts[i]);
					vMappingPrevPts.push_back(vMatchPrevPts[i]);
					vMappingCurrPts.push_back(vMatchCurrPts[i]);
					vbMappingInliers.push_back(true);
				}
				else {
					vpDelayedCPs.push_back(mpPrevKeyFrame->mpMatchInfo->GetCP(idx));
					vDelayedPts.push_back(vMatchCurrPts[i]);
				}
			}
			std::vector<bool> vbCPs(vbMappingInliers.size(), false);
			
			///////////////중간 시각화
			auto mvpTargetMPs = mpTargetFrame->mpMatchInfo->GetMatchingMPs();
			cv::Point2f ptLeft1 = cv::Point2f(mnWidth, 0);
			cv::Point2f ptLeft2 = cv::Point2f(mnWidth * 2, 0);
			cv::Mat K = mpTargetFrame->mK.clone();
			cv::Mat Rpprev, Tpprev, Rprev, Tprev, Rcurr, Tcurr;
			mpTargetFrame->GetPose(Rcurr, Tcurr);
			mpPrevKeyFrame->GetPose(Rprev, Tprev);
			mpPPrevKeyFrame->GetPose(Rpprev, Tpprev);
			for (int i = 0; i < mvpTargetMPs.size(); i++) {
				auto pMPi = mvpTargetMPs[i];
				if (!pMPi || pMPi->isDeleted())
					continue;
				cv::Mat X3D = pMPi->GetWorldPos();
				cv::Scalar color(255,0,0);
				cv::Mat proj1 = Rcurr*X3D + Tcurr;
				cv::Mat proj2 = Rprev*X3D + Tprev;
				cv::Mat proj3 = Rpprev*X3D + Tpprev;
				proj1 = K*proj1;
				proj2 = K*proj2;
				proj3 = K*proj3;
				cv::Point2f projected1(proj1.at<float>(0) / proj1.at<float>(2), proj1.at<float>(1) / proj1.at<float>(2));
				cv::Point2f projected2(proj2.at<float>(0) / proj2.at<float>(2), proj2.at<float>(1) / proj2.at<float>(2));
				cv::Point2f projected3(proj3.at<float>(0) / proj3.at<float>(2), proj3.at<float>(1) / proj3.at<float>(2));
				projected1 += ptLeft2;
				projected2 += ptLeft1;
				circle(debugMatch, projected1, 5, color);
				circle(debugMatch, projected2, 5, color);
				circle(debugMatch, projected3, 5, color);

				cv::Point2f pppt = cv::Point2f(0, 0);
				cv::Point2f ppt = cv::Point2f(0, 0);
				cv::Point2f cpt = cv::Point2f(0, 0);

				int pppidx = pMPi->GetPointIndexInFrame(mpPPrevKeyFrame->mpMatchInfo);
				if (pppidx >= 0) {
					pppt = mpPPrevKeyFrame->mpMatchInfo->GetMatchingPt(pppidx);
					cv::line(debugMatch, projected3, pppt, color);
				}
				int ppidx = pMPi->GetPointIndexInFrame(mpPrevKeyFrame->mpMatchInfo);
				if (ppidx >= 0) {
					ppt = mpPrevKeyFrame->mpMatchInfo->GetMatchingPt(ppidx) + ptLeft1;
					cv::line(debugMatch, projected2, ppt, color);
				}
				int cpidx = pMPi->GetPointIndexInFrame(mpTargetFrame->mpMatchInfo);
				if (cpidx >= 0) {
					cpt = mpTargetFrame->mpMatchInfo->GetMatchingPt(cpidx) + ptLeft2;
					cv::line(debugMatch, projected1, cpt, color);
				}

			}
			//////////////중간 시각화
			
			//int nCreated = CreateMapPoints(mpTargetFrame, mpPrevKeyFrame, mpPPrevKeyFrame, vMappingPPrevPts, vMappingPrevPts, vMappingCurrPts, vbCPs, debugMatch, ddddbug);

			////parallax 체크 못한 포인트들 생성
			/*for (int i = 0; i < vbCPs.size(); i++) {
				if (vbCPs[i]) {
					auto pCP = new CandidatePoint();
					pCP->AddFrame(mpTargetFrame->mpMatchInfo, vMappingCurrPts[i]);
					pCP->AddFrame(mpPrevKeyFrame->mpMatchInfo, vMappingPrevPts[i]);
					pCP->AddFrame(mpPPrevKeyFrame->mpMatchInfo, vMappingPPrevPts[i]);
					mpTargetFrame->mpMatchInfo->mvTempPts.push_back(vMappingCurrPts[i]);
				}
			}*/
			
			////지연된 삼각화 실행
			//for (int i = 0; i < vpDelayedCPs.size(); i++) {
			//	circle(ddddbug, vDelayedPts[i] + ptLeft2, 2, cv::Scalar(0, 0, 0), -1);
			//	if (!vpDelayedCPs[i]->DelayedTriangulate(mpMap, mpTargetFrame->mpMatchInfo, vDelayedPts[i], mpPPrevKeyFrame->mpMatchInfo, mpPrevKeyFrame->mpMatchInfo, mK, mInvK, ddddbug)) {
			//		vpDelayedCPs[i]->AddFrame(mpTargetFrame->mpMatchInfo, vDelayedPts[i]);
			//		mpTargetFrame->mpMatchInfo->mvTempPts.push_back(vDelayedPts[i]);
			//		//circle(ddddbug, vDelayedPts[i]+ ptLeft2, 2, cv::Scalar(0, 0, 0),-1);
			//	}
			//}
			//std::cout << "Mapping::" << nCreated << "::" << vMatchPrevPts.size() << ", " << vnMappingIDXs.size() << ", " << vpDelayedCPs.size() << "::" << mpTargetFrame->mpMatchInfo->mvTempPts.size() << std::endl;

			////삼각화 통과 못한 애들 다시 추가
			targetMatchingMPs = matchInfo->GetMatchingMPs();
			std::stringstream ssdir;
			ssdir << mpSystem->GetDirPath(0) << "/kfmatching/mapping_test_" << mpTargetFrame->GetKeyFrameID() << "_" << mpPPrevKeyFrame->GetKeyFrameID() << ".jpg";
			imwrite(ssdir.str(), ddddbug);
			/*ssdir.str("");
			ssdir << mpSystem->GetDirPath(0) << "/kfmatching/matching_test1_" << mpTargetFrame->GetKeyFrameID() << "_" << mpPPrevKeyFrame->GetKeyFrameID() << ".jpg";
			imwrite(ssdir.str(), debugMatch);
			ssdir.str("");
			ssdir << mpSystem->GetDirPath(0) << "/kfmatching/matching_test2_" << mpTargetFrame->GetKeyFrameID() << "_" << mpPPrevKeyFrame->GetKeyFrameID() << ".jpg";
			imwrite(ssdir.str(), testDebugMatch);*/
			/////Delayed Triangulation
			///////////////////////////////////////////////////////
			
			/*if (mpTargetFrame->GetKeyFrameID() > 6) {
				auto pKF1 = mpTargetFrame->mpMatchInfo->mpTargetFrame->mpMatchInfo->mpTargetFrame;
				auto pKF2 = pKF1->mpMatchInfo->mpTargetFrame->mpMatchInfo->mpTargetFrame;
				std::vector<cv::Point2f> va, vb, vc;
				std::vector<int> vn;
				std::vector<bool> vi;
				cv::Mat dddddd;
				mpMatcher->OpticalMatchingForMapping(mpTargetFrame, pKF1, pKF2, va, vb, vc, vn, vi, dddddd);
				std::stringstream ssdir;
				ssdir << mpSystem->GetDirPath(0) << "/kfmatching/mapping_test_" << mpTargetFrame->GetKeyFrameID() <<"_"<<pKF2->GetKeyFrameID()<< ".jpg";
				imwrite(ssdir.str(), dddddd);

			}*/
			//매핑 테스트

			//////////////업데이트 키프레임
			//이건 단순히 해당 키프레임에서 추적되는 맵포인트가 연결되어 있는 프레임들의 리스트에 불과함.
			std::map<UVR_SLAM::Frame*, int> mmpCandidateKFs;
			//int nTargetID = mpTargetFrame->GetFrameID();
			int nMaxKF = 0;
			int nMinKF = INT_MAX;
			for (int i = 0; i <  targetMatchingMPs.size(); i++) {
				UVR_SLAM::MapPoint* pMP = targetMatchingMPs[i];
				if (!pMP || pMP->isDeleted())
					continue;
				auto mmpMP = pMP->GetConnedtedFrames();
				if (mmpMP.size() > nMaxKF)
					nMaxKF = mmpMP.size();
				for (auto biter = mmpMP.begin(), eiter = mmpMP.end(); biter != eiter; biter++) {
					auto pMatchInfo = biter->first;
					auto pkF = pMatchInfo->mpRefFrame;
					//UVR_SLAM::Frame* pCandidateKF = biter->first;
					if (nTargetID == pkF->GetFrameID())
						continue;
					mmpCandidateKFs[pkF]++;
				}
			}

			nMaxKF = 0;
			for (auto biter = mmpCandidateKFs.begin(), eiter = mmpCandidateKFs.end(); biter != eiter; biter++) {
				UVR_SLAM::Frame* pKF = biter->first;
				int nCount = biter->second;
				if (nMinKF > nCount)
					nMinKF = nCount;
				if (nMaxKF < nCount)
					nMaxKF = nCount;
				if (nCount > 10) {
					//mpTargetFrame->AddKF(pKF);
					//vPairs.push_back(std::make_pair(nCount, pKF));
					mpTargetFrame->AddKF(pKF, nCount);
					pKF->AddKF(mpTargetFrame, nCount);
				}
			}
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

			

			/////////////VoW 매칭
			/*auto vpNeighKFs = mpTargetFrame->GetConnectedKFs();
			for (int i = 0; i < vpNeighKFs.size(); i++) {
				auto pKFi = vpNeighKFs[i];
				if (mpTargetFrame->Score(pKFi) < 0.01) {
					imshow("Loop!!", pKFi->GetOriginalImage());
					cv::waitKey(1);
				}
			}*/
			/////////////VoW 매칭

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
			ssa << "LocalMapping : " << mpTargetFrame->GetKeyFrameID() << "::" << t_test1 <<"::"<< targetMatchingMPs.size()<<", "<< nCreated<< "::" << mpTargetFrame->GetConnectedKFs().size() << ", " << nMinKF << ", " << nMaxKF;
			mpSystem->SetLocalMapperString(ssa.str());

			std::cout << "lm::end::" <<mpTargetFrame->GetFrameID()<<"::"<<nCreated<< std::endl;
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


////////////200722 수정 필요
int UVR_SLAM::LocalMapper::CreateMapPoints(Frame* pCurrKF, Frame* pPrevKF, std::vector<cv::Point2f> vMatchPrevPts, std::vector<cv::Point2f> vMatchCurrPts, std::vector<CandidatePoint*> vPrevCPs, std::vector<int> vnIDXs, cv::Mat& debugMatch){
	
	cv::Point2f ptBottom = cv::Point2f(0, mnHeight);
	cv::Mat Rprev, Tprev, Rcurr, Tcurr;
	pCurrKF->GetPose(Rcurr, Tcurr);
	pPrevKF->GetPose(Rprev, Tprev);

	cv::Mat Pcurr, Pprev;
	cv::hconcat(Rcurr, Tcurr, Pcurr);
	cv::hconcat(Rprev, Tprev, Pprev);

	cv::Mat Map;
	cv::triangulatePoints(mK*Pprev, mK*Pcurr, vMatchPrevPts, vMatchCurrPts, Map);

	///////데이터 전처리
	cv::Mat Rcfromc = Rcurr.t();
	cv::Mat Rpfromc = Rprev.t();
	cv::Mat invT, invK, invP;
	invK = mK.inv();

	int nRes = 0;
	for (int i = 0; i < Map.cols; i++) {

		cv::Mat X3D = Map.col(i);
		auto pt1 = vMatchCurrPts[i];
		auto pt2 = vMatchPrevPts[i];

		if (abs(X3D.at<float>(3)) < 0.0001) {
			std::cout << "test::" << X3D.at<float>(3) << std::endl;
			cv::circle(debugMatch, pt1+ ptBottom, 2, cv::Scalar(0, 0, 0), -1);
			cv::circle(debugMatch, pt2, 2, cv::Scalar(0, 0, 0), -1);
			continue;
		}
		X3D /= X3D.at<float>(3);
		X3D = X3D.rowRange(0, 3);

		cv::Mat proj1 = Rcurr*X3D + Tcurr;
		cv::Mat proj2 = Rprev*X3D + Tprev;

		////depth test
		if (proj1.at<float>(2) < 0.0 || proj2.at<float>(2) < 0.0) {
			cv::circle(debugMatch, pt1+ ptBottom, 2, cv::Scalar(0, 255, 0), -1);
			cv::circle(debugMatch, pt2, 2, cv::Scalar(0, 255, 0), -1);
			continue;
		}
		////depth test

		////reprojection error
		proj1 = mK*proj1;
		proj2 = mK*proj2;
		cv::Point2f projected1(proj1.at<float>(0) / proj1.at<float>(2), proj1.at<float>(1) / proj1.at<float>(2));
		cv::Point2f projected2(proj2.at<float>(0) / proj2.at<float>(2), proj2.at<float>(1) / proj2.at<float>(2));

		auto diffPt1 = projected1 - pt1;
		auto diffPt2 = projected2 - pt2;
		float err1 = (diffPt1.dot(diffPt1));
		float err2 = (diffPt2.dot(diffPt2));
		if (err1 > 4.0 || err2 > 4.0) {
			cv::circle(debugMatch, pt1+ ptBottom, 2, cv::Scalar(255, 0, 0), -1);
			cv::circle(debugMatch, pt2, 2, cv::Scalar(255, 0, 0), -1);
			continue;
		}
		////reprojection error
		////CP 연결하기
		int nCPidx = vnIDXs[i];
		auto pCPi = vPrevCPs[nCPidx];
		pCPi->AddFrame(pCurrKF->mpMatchInfo, pt1);
		int nConnCP = pCPi->GetNumSize();

		////연결 확인
		if (nConnCP > 2 && pCPi->bCreated) {
			cv::circle(debugMatch, pt1 + ptBottom, 3, cv::Scalar(0, 255, 255));
			cv::circle(debugMatch, pt2, 3, cv::Scalar(0, 255, 255));
		}
		else if (nConnCP > 2 && !pCPi->bCreated) {
			cv::circle(debugMatch, pt1 + ptBottom, 3, cv::Scalar(255, 255, 0));
			cv::circle(debugMatch, pt2, 3, cv::Scalar(255, 255, 0));
		}
		else {
			cv::circle(debugMatch, pt1 + ptBottom, 3, cv::Scalar(0, 255, 0));
			cv::circle(debugMatch, pt2, 3, cv::Scalar(0, 255, 0));
		}

		//////parallax check
		//targettarget과 current만
		cv::Mat xn1 = (cv::Mat_<float>(3, 1) << pt1.x, pt1.y, 1.0);
		cv::Mat xn2 = (cv::Mat_<float>(3, 1) << pt2.x, pt2.y, 1.0);
		//std::cout << xn1.t() << xn3.t();
		cv::Mat ray1 = Rcfromc*invK*xn1;
		cv::Mat ray2 = Rpfromc*invK*xn2;
		//std::cout <<"\t"<< ray1.t() << ray3.t();
		float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1)*cv::norm(ray2));
		//std::cout << cosParallaxRays << std::endl;
		if (cosParallaxRays >= 0.9998) { //9999 : 위안홀까지 가능, 99999 : 비전홀, N5
			cv::circle(debugMatch, pt1+ptBottom, 2, cv::Scalar(0, 0, 255), -1);
			cv::circle(debugMatch, pt2, 2, cv::Scalar(0, 0, 255), -1);
			continue;
		}
		//////parallax check
		//패럴랙스 체크도 CP에서 가장 처음과 현재 것만??

		if (nConnCP > 2 && !pCPi->bCreated) {
			int label = 0;
			pCPi->bCreated = true;
			auto pMP = new UVR_SLAM::MapPoint(mpMap, mpTargetFrame, X3D, cv::Mat(), label, pCPi->octave);
			//여기서 모든 CP 다 연결하기?
			pMP->AddFrame(pCurrKF->mpMatchInfo, pt1);
			pMP->AddFrame(pPrevKF->mpMatchInfo, pt2);
		}


		///////////평면 정보로 맵생성
		//if (bPlaneMap && vLabels3[i] == 150) {
		//	cv::Mat a;
		//	if (UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(pt3, invP, invT, invK, a)) {
		//		X3D = a;
		//		//b = true;
		//	}
		//}
		//////평면과의 거리를 확인하였음. 이들 대부분 평면에 포함이 안됨. 포함하게 하면 오히려 에러가 남...
		//if (bPlaneMap && vLabels3[i] == 150)
		//{
		//	cv::Mat n;
		//	float d;
		//	targettargetFrame->mpPlaneInformation->GetPlane(1)->GetParam(n, d);
		//	//std::cout << "new mp ::" << abs(X3D.dot(n) + d) << std::endl;;
		//}
		///////////평면 정보로 맵생성

		/////CP 콜레스폰던스가 2이상이고, 패럴랙스 통과하면 MP로

		nRes++;
		//int label = pPrevKF->mpMatchInfo->mvObjectLabels[vnIDXs[i]];
		//int octave = pPrevKF->mpMatchInfo->mvnOctaves[vnIDXs[i]];
		{
			//////pMP
			//int label = 0;
			//int octave = 0;
			//auto pMP = new UVR_SLAM::MapPoint(mpMap, mpTargetFrame, X3D, cv::Mat(), label, octave);
			//if (label == 150) {
			//	pMP->SetPlaneID(1);
			//}
			//else if (label == 100) {
			//	pMP->SetPlaneID(2);
			//}
			///*if (vLabels3[i] > 0)
			//pMP->SetPlaneID(1);*/
		
			//pMP->AddFrame(pCurrKF->mpMatchInfo, pt1);
			//pMP->AddFrame(pPrevKF->mpMatchInfo, pt2);

			//auto pt3D = mpMap->ProjectMapPoint(pMP, mpMap->mfMapGridSize);
			//auto pMG = mpMap->GetGrid(pt3D);
			//if (!pMG) {
			//	pMG = mpMap->InsertGrid(pt3D);
			//}
			//mpMap->InsertMapPoint(pMP, pMG);

			////pMP->UpdateNormalAndDepth();
			////////visualize
			//cv::circle(debugMatch, pt1 + ptBottom, 2, cv::Scalar(255, 0, 255), -1);
			//cv::circle(debugMatch, pt2, 2, cv::Scalar(255, 0, 255), -1);
			////////visualize
		}
	}
	return nRes;
}
//매칭 포인트 결과와 실제 결과도 반영하도록
int UVR_SLAM::LocalMapper::CreateMapPoints(Frame* pCurrKF, Frame* pPrevKF, Frame* pPPrevKF, std::vector<cv::Point2f> vMatchPPrevPts, std::vector<cv::Point2f> vMatchPrevPts, std::vector<cv::Point2f> vMatchCurrPts, std::vector < bool>& vbCPs, cv::Mat& debugMatch, cv::Mat& debug) {
		
	cv::Point2f ptLeft1 = cv::Point2f(mnWidth, 0);
	cv::Point2f ptLeft2 = cv::Point2f(mnWidth * 2, 0);
	cv::Mat Rpprev, Tpprev, Rprev, Tprev, Rcurr, Tcurr;
	pCurrKF->GetPose(Rcurr, Tcurr);
	pPrevKF->GetPose(Rprev, Tprev);
	pPPrevKF->GetPose(Rpprev, Tpprev);
	cv::Mat mK = pCurrKF->mK.clone();

	cv::Mat Pcurr, Pprev, Ppprev;
	cv::hconcat(Rcurr, Tcurr, Pcurr);
	cv::hconcat(Rprev, Tprev, Pprev);
	cv::hconcat(Rpprev, Tpprev, Ppprev);

	cv::Mat K = mpTargetFrame->mK.clone();
	cv::Mat Map;
	cv::triangulatePoints(K*Ppprev, K*Pcurr, vMatchPPrevPts, vMatchCurrPts, Map);

	///////데이터 전처리
	cv::Mat Rcfromc = Rcurr.t();
	cv::Mat Rppfromc = Rpprev.t();
	cv::Mat invT, invK, invP;
	invK = K.inv();
	bool bPlaneMap = false;
	/*auto targetTargetPlaneInfo = targettargetFrame->mpPlaneInformation;
	if (targetTargetPlaneInfo) {
	bPlaneMap = true;
	targetTargetPlaneInfo->Calculate();
	targetTargetPlaneInfo->GetInformation(invP, invT, invK);
	}*/
	///////데이터 전처리

	//////LOCK
	//std::cout << "CreateMP::1" << std::endl;
	//std::unique_lock<std::mutex> lock(mpSystem->mMutexUseLocalMap);
	//while (!mpSystem->mbTrackingEnd) {
	//	mpSystem->cvUseLocalMap.wait(lock);
	//}
	//mpSystem->mbLocalMapUpdateEnd = false;
	//std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	//////LOCK

	/////////////에러 확인용
	cv::Mat usedCurr = pCurrKF->mpMatchInfo->used.clone();
	usedCurr.convertTo(usedCurr, debugMatch.type());
	cv::cvtColor(usedCurr, usedCurr, CV_GRAY2BGR);
	cv::Mat usedPrev = pPrevKF->mpMatchInfo->used.clone();
	usedPrev.convertTo(usedPrev, debugMatch.type());
	cv::cvtColor(usedPrev, usedPrev, CV_GRAY2BGR);
	cv::Mat usedPPrev = pPPrevKF->mpMatchInfo->used.clone();
	usedPPrev.convertTo(usedPPrev, debugMatch.type());
	cv::cvtColor(usedPPrev, usedPPrev, CV_GRAY2BGR);
	/////////////에러 확인용

	int nRes = 0;
	for (int i = 0; i < Map.cols; i++) {

		cv::Mat X3D = Map.col(i);
		auto pt1 = vMatchCurrPts[i];
		auto pt2 = vMatchPrevPts[i];
		auto pt3 = vMatchPPrevPts[i];

		if (abs(X3D.at<float>(3)) < 0.0001){
			std::cout << "test::" << X3D.at<float>(3) << std::endl;
			cv::circle(debugMatch, pt1 + ptLeft2, 2, cv::Scalar(0, 0, 0), -1);
			cv::circle(debugMatch, pt2 + ptLeft1, 2, cv::Scalar(0, 0, 0), -1);
			cv::circle(debugMatch, pt3, 2, cv::Scalar(0, 0, 0), -1);
			continue;
		}
		X3D /= X3D.at<float>(3);
		X3D = X3D.rowRange(0, 3);

		

		cv::Mat proj1 = Rcurr*X3D + Tcurr;
		cv::Mat proj2 = Rprev*X3D + Tprev;
		cv::Mat proj3 = Rpprev*X3D + Tpprev;

		////depth test
		if (proj1.at<float>(2) < 0.0 || proj2.at<float>(2) < 0.0 || proj3.at<float>(2) < 0.0) {
			cv::circle(debugMatch, pt1 + ptLeft2, 2, cv::Scalar(0, 255, 0), -1);
			cv::circle(debugMatch, pt2 + ptLeft1, 2, cv::Scalar(0, 255, 0), -1);
			cv::circle(debugMatch, pt3, 2, cv::Scalar(0, 255, 0), -1);
			cv::circle(usedCurr, pt1, 2, cv::Scalar(0, 255, 0), -1);
			cv::circle(usedPrev, pt2, 2, cv::Scalar(0, 255, 0), -1);
			cv::circle(usedPPrev, pt3, 2, cv::Scalar(0, 255, 0), -1);
			continue;
		}
		////depth test

		////reprojection error
		proj1 = K*proj1;
		proj2 = K*proj2;
		proj3 = K*proj3;
		cv::Point2f projected1(proj1.at<float>(0) / proj1.at<float>(2), proj1.at<float>(1) / proj1.at<float>(2));
		cv::Point2f projected2(proj2.at<float>(0) / proj2.at<float>(2), proj2.at<float>(1) / proj2.at<float>(2));
		cv::Point2f projected3(proj3.at<float>(0) / proj3.at<float>(2), proj3.at<float>(1) / proj3.at<float>(2));
		
		auto diffPt1 = projected1 - pt1;
		auto diffPt2 = projected2 - pt2;
		auto diffPt3 = projected3 - pt3;
		float err1 = (diffPt1.dot(diffPt1));
		float err2 = (diffPt2.dot(diffPt2));
		float err3 = (diffPt3.dot(diffPt3));
		if (err1 > 4.0 || err2 > 4.0 || err3 > 4.0) {
			cv::circle(debugMatch, pt1 + ptLeft2, 2, cv::Scalar(255, 0, 0), -1);
			cv::circle(debugMatch, pt2 + ptLeft1, 2, cv::Scalar(255, 0, 0), -1);
			cv::circle(debugMatch, pt3, 2, cv::Scalar(255, 0, 0), -1);
			cv::circle(usedCurr, pt1, 2, cv::Scalar(255, 0, 0), -1);
			cv::circle(usedPrev, pt2, 2, cv::Scalar(255, 0, 0), -1);
			cv::circle(usedPPrev, pt3, 2, cv::Scalar(255, 0, 0), -1);
			continue;
		}
		////reprojection error

		//////parallax check
		//targettarget과 current만
		cv::Mat xn1 = (cv::Mat_<float>(3, 1) << pt1.x, pt1.y, 1.0);
		cv::Mat xn3 = (cv::Mat_<float>(3, 1) << pt3.x, pt3.y, 1.0);
		//std::cout << xn1.t() << xn3.t();
		cv::Mat ray1 = Rcfromc*invK*xn1;
		cv::Mat ray3 = Rppfromc*invK*xn3;
		//std::cout <<"\t"<< ray1.t() << ray3.t();
		float cosParallaxRays = ray1.dot(ray3) / (cv::norm(ray1)*cv::norm(ray3));
		//std::cout << cosParallaxRays << std::endl;
		if (cosParallaxRays >= 0.9998) { //9999 : 위안홀까지 가능, 99999 : 비전홀, N5
			cv::circle(debugMatch, pt1 + ptLeft2, 2, cv::Scalar(0, 0, 255), -1);
			cv::circle(debugMatch, pt2 + ptLeft1, 2, cv::Scalar(0, 0, 255), -1);
			cv::circle(debugMatch, pt3, 2, cv::Scalar(0, 0, 255), -1);
			cv::circle(usedCurr, pt1, 2, cv::Scalar(0, 0, 255), -1);
			cv::circle(usedPrev, pt2, 2, cv::Scalar(0, 0, 255), -1);
			cv::circle(usedPPrev, pt3, 2, cv::Scalar(0, 0, 255), -1);
			
			//cp들 추가.
			//UVR_SLAM::CandidatePoint* pCP =  new UVR_SLAM::CandidatePoint()
			//
			vbCPs[i] = true;
			continue;
		}
		//////parallax check

		///////////평면 정보로 맵생성
		//if (bPlaneMap && vLabels3[i] == 150) {
		//	cv::Mat a;
		//	if (UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(pt3, invP, invT, invK, a)) {
		//		X3D = a;
		//		//b = true;
		//	}
		//}
		//////평면과의 거리를 확인하였음. 이들 대부분 평면에 포함이 안됨. 포함하게 하면 오히려 에러가 남...
		//if (bPlaneMap && vLabels3[i] == 150)
		//{
		//	cv::Mat n;
		//	float d;
		//	targettargetFrame->mpPlaneInformation->GetPlane(1)->GetParam(n, d);
		//	//std::cout << "new mp ::" << abs(X3D.dot(n) + d) << std::endl;;
		//}
		///////////평면 정보로 맵생성

		nRes++;
		//int label = pPrevKF->mpMatchInfo->mvObjectLabels[vnIDXs[i]];
		//int octave = pPrevKF->mpMatchInfo->mvnOctaves[vnIDXs[i]];
		int label = 0;
		int octave = 0;
		auto pMP = new UVR_SLAM::MapPoint(mpMap, mpTargetFrame, X3D, cv::Mat(), label, octave);
		if (label == 150) {
			pMP->SetPlaneID(1);
		}
		else if (label == 100) {
			pMP->SetPlaneID(2);
		}
		/*if (vLabels3[i] > 0)
		pMP->SetPlaneID(1);*/

		/*mvpMatchingMPs[vIDXs1[i]] = pMP;
		targetInfo->mvpMatchingMPs[vIDXs2[i]] = pMP;
		targetTargetInfo->mvpMatchingMPs[vIDXs3[i]] = pMP;
		auto pt1 = mvMatchingPts[vIDXs1[i]];
		auto pt2 = targetInfo->mvMatchingPts[vIDXs2[i]];
		auto pt3 = targetTargetInfo->mvMatchingPts[vIDXs3[i]];*/

		pMP->AddFrame(pCurrKF->mpMatchInfo,pt1);
		pMP->AddFrame(pPrevKF->mpMatchInfo, pt2);
		pMP->AddFrame(pPPrevKF->mpMatchInfo, pt3);

		auto pt3D = mpMap->ProjectMapPoint(pMP, mpMap->mfMapGridSize);
		auto pMG = mpMap->GetGrid(pt3D);
		if (!pMG) {
			pMG = mpMap->InsertGrid(pt3D);
		}
		mpMap->InsertMapPoint(pMP, pMG);

		//pMP->UpdateNormalAndDepth();
		//////visualize
		cv::circle(debugMatch, pt1+ptLeft2, 2, cv::Scalar(255, 0, 255), -1);
		cv::circle(debugMatch, pt2+ptLeft1, 2, cv::Scalar(255, 0, 255), -1);
		cv::circle(debugMatch, pt3, 2, cv::Scalar(255, 0, 255), -1);
		cv::circle(usedCurr, pt1, 2, cv::Scalar(255, 0, 255), -1);
		cv::circle(usedPrev, pt2, 2, cv::Scalar(255, 0, 255), -1);
		cv::circle(usedPPrev, pt3, 2, cv::Scalar(255, 0, 255), -1);
		//////visualize
	}
	
	cv::Mat debugUsed = cv::Mat::zeros(mnHeight, mnWidth * 3, CV_8UC3);
	std::stringstream suc;
	suc << "ID : "<<pCurrKF->GetFrameID();
	cv::rectangle(usedCurr, cv::Point2f(0, 0), cv::Point2f(usedCurr.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(usedCurr, suc.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	usedCurr.copyTo(debugUsed(cv::Rect(mnWidth * 2, 0, mnWidth, mnHeight)));
	
	suc.str("");
	suc << "ID : " << pPrevKF->GetFrameID();
	cv::rectangle(usedPrev, cv::Point2f(0, 0), cv::Point2f(usedCurr.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(usedPrev, suc.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	usedPrev.copyTo(debugUsed(cv::Rect(mnWidth, 0, mnWidth, mnHeight)));
	
	suc.str("");
	suc << "ID : " << pPPrevKF->GetFrameID();
	cv::rectangle(usedPPrev, cv::Point2f(0, 0), cv::Point2f(usedCurr.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(usedPPrev, suc.str(), cv::Point2f(0, 20), 2, 0.6, cv::Scalar::all(255));
	usedPPrev.copyTo(debugUsed(cv::Rect(mnWidth * 0, 0, mnWidth, mnHeight)));


	debug = cv::Mat::zeros(debugUsed.rows * 2, debugUsed.cols, debugMatch.type());
	debugMatch.copyTo(debug(cv::Rect(0, 0, debugMatch.cols, debugMatch.rows)));
	debugUsed.copyTo(debug(cv::Rect(0, debugMatch.rows, debugMatch.cols, debugMatch.rows)));
	//////LOCK
	//mpSystem->mbLocalMapUpdateEnd = true;
	//std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	//auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	//double tttt1 = duration1 / 1000.0;
	//std::cout << "CreateMP::2::" <<tttt1<< std::endl;
	//////LOCK
	return nRes;
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
