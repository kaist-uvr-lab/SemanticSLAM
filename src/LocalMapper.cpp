#include <LocalMapper.h>
#include <CandidatePoint.h>
#include <Frame.h>
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

#include <MapGrid.h>

#include <opencv2/core/mat.hpp>
#include <ctime>
#include <direct.h>

UVR_SLAM::LocalMapper::LocalMapper(){}
UVR_SLAM::LocalMapper::LocalMapper(System* pSystem, std::string strPath, int w, int h):mnWidth(w), mnHeight(h), mbStopBA(false), mbDoingProcess(false), mbStopLocalMapping(false), mpTargetFrame(nullptr), mpPrevKeyFrame(nullptr), mpPPrevKeyFrame(nullptr){
	mpSystem = pSystem;

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

void UVR_SLAM::LocalMapper::Init() {
	mpMap = mpSystem->mpMap;
	mpMatcher = mpSystem->mpMatcher;
	mpMapOptimizer = mpSystem->mpMapOptimizer;

	mpVisualizer = mpSystem->mpVisualizer;
	mpSegmentator = mpSystem->mpSegmentator;
	mpPlaneEstimator = mpSystem->mpPlaneEstimator;
	mpLoopCloser = mpSystem->mpLoopCloser;
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
	mpPPrevKeyFrame = mpPrevKeyFrame;
	mpPrevKeyFrame = mpTargetFrame;
	{
		std::unique_lock<std::mutex> lock(mMutexNewKFs);
		mpTargetFrame = mKFQueue.front();
		mKFQueue.pop();
		mbStopBA = false;
	}
	mpTargetFrame->Init(mpSystem->mpORBExtractor, mpSystem->mK, mpSystem->mD);

	if (mpPrevKeyFrame->mpMatchInfo->GetNumCPs() < 600) {
		double time5 = 0.0;
		mpPrevKeyFrame->DetectFeature();
		mpPrevKeyFrame->DetectEdge();
		mpPrevKeyFrame->mpMatchInfo->SetMatchingPoints();
		mpPrevKeyFrame->SetBowVec(mpSystem->fvoc);
	}
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
			
			double time1 = 0.0;
			double time2 = 0.0;

			ProcessNewKeyFrame();
			
			//std::cout << "Local Mapping :: ID = " << mpTargetFrame->GetFrameID() << "::Start::"<<mpTargetFrame->mpMatchInfo->GetNumCPs() <<"::"<<mpTargetFrame->mpMatchInfo->mfLowQualityRatio<< std::endl;
			int nTargetID = mpTargetFrame->GetFrameID();
			
			int nCreated = 0;
			////////New Matching & Create & Delayed CP test
			cv::Mat debugMatch;
			
			/////프레임 퀄리티 계산
			
			bool bLowQualityFrame = mpTargetFrame->mpMatchInfo->UpdateFrameQuality();
			
			/////프레임 퀄리티 계산
			/////////KF-KF 매칭
			////이미지 생성
			cv::Mat prevImg = mpPrevKeyFrame->GetOriginalImage();
			cv::Mat currImg = mpTargetFrame->GetOriginalImage();
			cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
			cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
			cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
			debugMatch = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
			prevImg.copyTo(debugMatch(mergeRect1));
			currImg.copyTo(debugMatch(mergeRect2));
			////이미지 생성
			std::vector<cv::Point2f> vOpticalMatchPPrevPts, vOpticalMatchPrevPts, vOpticalMatchCurrPts;
			std::vector<CandidatePoint*> vOpticalMatchCPs;
			//int nMatch = mpMatcher->OpticalMatchingForMapping(mpMap, mpTargetFrame, mpPrevKeyFrame, vOpticalMatchPrevPts, vOpticalMatchCurrPts, vOpticalMatchCPs, mK, mInvK, time1, debugMatch);
			int nMatch = mpMatcher->OpticalMatchingForMapping(mpMap, mpTargetFrame, mpPrevKeyFrame, mpPPrevKeyFrame, vOpticalMatchPPrevPts, vOpticalMatchPrevPts, vOpticalMatchCurrPts, vOpticalMatchCPs, mK, mInvK, time1, debugMatch);
			mpTargetFrame->mpMatchInfo->ConnectAll();
			//NewMapPointMarginalization();

			std::vector<cv::Point2f> vMappingPPrevPts, vMappingPrevPts, vMappingCurrPts;
			std::vector<CandidatePoint*> vMappingCPs;
			int nMapping = MappingProcess2(mpMap, mpTargetFrame, mpPrevKeyFrame, vMappingPrevPts, vMappingCurrPts, vMappingCPs, vOpticalMatchPrevPts, vOpticalMatchCurrPts, vOpticalMatchCPs, time2, debugMatch);
			////////////New Map Point Creation Test
			//{
			//	auto lastKF = mpMap->GetReverseWindowFrame(0);
			//	auto llastKF = mpMap->GetReverseWindowFrame(1);
			//	auto lastMatch = lastKF->mpMatchInfo;
			//	auto llastMatch = llastKF->mpMatchInfo;

			//	cv::Mat Rpprev, Tpprev,Rprev, Tprev, Rcurr, Tcurr;
			//	mpTargetFrame->GetPose(Rcurr, Tcurr);
			//	lastKF->GetPose(Rprev, Tprev);
			//	llastKF->GetPose(Rpprev, Tpprev);

			//	cv::Mat Pcurr, Pprev, Ppprev;
			//	cv::hconcat(Rcurr, Tcurr, Pcurr);
			//	cv::hconcat(Rprev, Tprev, Pprev);
			//	cv::hconcat(Rpprev, Tpprev, Ppprev);
			//	
			//	std::vector<cv::Point2f> vTempPTs1, vTempPTs2, vTempPTs3; //curr, prev, pprev
			//	std::vector<CandidatePoint*> vTempCPs;
			//	for (int i = 0; i < vMatchPrevCPs.size(); i++) {
			//		auto pCPi = vMatchPrevCPs[i];
			//		int pidx = pCPi->GetPointIndexInFrame(lastMatch);
			//		int ppidx = pCPi->GetPointIndexInFrame(llastMatch);
			//		if (ppidx < 0 || pidx < 0){
			//			continue;
			//		}
			//		if (!pCPi->GetQuality())
			//			pCPi->ResetMapPoint();
			//		auto pMPi = pCPi->GetMP();
			//		if (pMPi)
			//			continue;

			//		auto pt1 = vMatchCurrPts[i];
			//		auto pt2 = lastMatch->GetPt(pidx);
			//		auto pt3 = llastMatch->GetPt(ppidx);
			//		vTempCPs.push_back(pCPi);
			//		vTempPTs1.push_back(pt1);
			//		vTempPTs2.push_back(pt2);
			//		vTempPTs3.push_back(pt3);
			//	}
			//	if (vTempCPs.size() > 20) {
			//		cv::Mat Map;
			//		cv::triangulatePoints(mK*Ppprev, mK*Pcurr, vTempPTs3, vTempPTs1, Map);

			//		std::vector<cv::Point2f> vMapPTs1, vMapPTs2, vMapPTs3; //curr, prev, pprev
			//		std::vector<CandidatePoint*> vMapCPs;
			//		std::vector<cv::Mat> vMaps;

			//		for (int i = 0; i < Map.cols; i++) {

			//			cv::Mat X3D = Map.col(i);
			//			auto currPt = vTempPTs1[i];
			//			auto prevPt = vTempPTs3[i];
			//			if (abs(X3D.at<float>(3)) < 0.0001) {
			//				continue;
			//			}
			//			X3D /= X3D.at<float>(3);
			//			X3D = X3D.rowRange(0, 3);

			//			cv::Mat proj1 = Rcurr*X3D + Tcurr;
			//			cv::Mat proj2 = Rpprev*X3D + Tpprev;

			//			//depth test
			//			if (proj1.at<float>(2) < 0.0 || proj2.at<float>(2) < 0.0) {
			//				continue;
			//			}
			//			//depth test
			//			vMapPTs1.push_back(vTempPTs1[i]);
			//			vMapPTs2.push_back(vTempPTs2[i]);
			//			vMapPTs3.push_back(vTempPTs3[i]);
			//			vMapCPs.push_back(vTempCPs[i]);
			//			vMaps.push_back(X3D);
			//		}
			//		//이부분 수정 필요
			//		UVR_SLAM::Optimization::PoseRecoveryOptimization(mpTargetFrame, lastKF, llastKF, vMapPTs1, vMapPTs2, vMapPTs3, vMaps);

			//		mpMap->ClearReinit();
			//		auto spWindowKFs = mpMap->GetWindowFramesSet(1);
			//		/////시각화 확인
			//		for (int i = 0; i < vMaps.size(); i++) {
			//			cv::Mat X3D = vMaps[i];
			//			mpMap->AddReinit(X3D);
			//			auto pCPi = vMapCPs[i];
			//			int label = pCPi->GetLabel();
			//			auto pMP = new UVR_SLAM::MapPoint(mpMap, mpTargetFrame, pCPi, X3D, cv::Mat(), label, pCPi->octave);
			//			auto mmpFrames = pCPi->GetFrames();
			//			for (auto iter = mmpFrames.begin(); iter != mmpFrames.end(); iter++) {
			//				auto pMatch = iter->first;
			//				int idx = iter->second;
			//				auto pKF = pMatch->mpRefFrame;
			//				if (spWindowKFs.find(pKF) != spWindowKFs.end()) {
			//					pMatch->AddMP();
			//					pMP->ConnectFrame(pMatch, idx);
			//				}
			//			}
			//		}
			//	}
			//}
			////////////New Map Point Creation Test
			
			//////Pose Recovery
			//if (bLowQualityFrame) {
			//	auto llastKF = mpMap->GetReverseWindowFrame(1);
			//	if(llastKF){
			//		auto lastKF = mpMap->GetLastWindowFrame();
			//		auto lastMatch = lastKF->mpMatchInfo;
			//		auto llastMatch = llastKF->mpMatchInfo;
			//		int n = 0;
			//		std::vector<bool> vbTempInliers;
			//		std::vector<cv::Point2f> vTempPTs1, vTempPTs2, vTempPTs3; //curr, prev, pprev
			//		std::vector<CandidatePoint*> vTempCPs;
			//		for (int i = 0; i < vMatchPrevCPs.size(); i++) {
			//			auto pCPi = vMatchPrevCPs[i];
			//			int pidx = pCPi->GetPointIndexInFrame(lastMatch);
			//			int ppidx = pCPi->GetPointIndexInFrame(llastMatch);
			//			if (ppidx < 0 || pidx < 0 || pCPi->GetNumSize() < 2){
			//				vbTempInliers.push_back(false);
			//				continue;
			//			}
			//			auto pt1 = vMatchCurrPts[i];
			//			auto pt2 = lastMatch->GetPt(pidx);
			//			auto pt3 = llastMatch->GetPt(ppidx);
			//			vTempCPs.push_back(pCPi);
			//			vTempPTs1.push_back(pt1);
			//			vTempPTs2.push_back(pt2);
			//			vTempPTs3.push_back(pt3);

			//			auto pMPi = pCPi->GetMP();
			//			if (!pCPi->GetQuality())
			//				pCPi->ResetMapPoint();
			//			if (!pMPi || pMPi->isDeleted()) {
			//				n++;
			//				vbTempInliers.push_back(true);
			//			}
			//			else
			//				vbTempInliers.push_back(false);
			//		}
			//		double d3 = 0.0;
			//		cv::Mat R, T;
			//		RecoverPose(mpTargetFrame, lastKF, llastKF, vTempPTs1, vTempPTs2, vTempPTs3, vTempCPs, vbTempInliers, R, T, d3, mpTargetFrame->GetOriginalImage(), lastKF->GetOriginalImage(), llastKF->GetOriginalImage());
			//		std::cout << "recover test::" << lastKF->GetFrameID() << "::" << n <<", "<<vTempCPs.size()<< std::endl;
			//	}
			//}
			//////Pose Recovery
			/////Create Map Points
			//nCreated = CreateMapPoints(mpTargetFrame, vMappingCurrPts, vMappingCPs, time3, debugMatch); //왜인지는 모르겟으나 잘 동작함
			
			//std::cout << "LM::MappingProcess::" << nMapping <<", new = "<< nMapping << " , optical = " << nMatch << std::endl;
			cv::Mat resized;
			cv::resize(debugMatch, resized, cv::Size(debugMatch.cols / 2, debugMatch.rows / 2));
			mpVisualizer->SetOutputImage(resized, 3);
			/////Create Map Points
			/////////KF-KF 매칭
			auto pTarget = mpMap->AddWindowFrame(mpTargetFrame);
			if (pTarget) {
				mpSegmentator->InsertKeyFrame(pTarget);
				mpPlaneEstimator->InsertKeyFrame(pTarget);
				//mpLoopCloser->InsertKeyFrame(pTarget);
			}
			if (mpMapOptimizer->isDoingProcess()) {
				//std::cout << "lm::ba::busy" << std::endl;
				mpMapOptimizer->StopBA(true);
			}
			else {
				mpMapOptimizer->InsertKeyFrame(mpTargetFrame);
			}
			std::chrono::high_resolution_clock::time_point lm_end = std::chrono::high_resolution_clock::now();
			auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(lm_end - lm_start).count();
			float t_test1 = du_test1 / 1000.0;
			
			std::stringstream ssa;
			ssa << "LocalMapping : " << mpTargetFrame->GetKeyFrameID() << "::" << t_test1 << "::" << "::" << nMapping <<", "<< time1 << ", " << time2 << std::endl;;// << ", " << nMinKF << ", " << nMaxKF;
			mpSystem->SetLocalMapperString(ssa.str());

			//std::cout << "lm::end::" << mpPrevKeyFrame->mpMatchInfo->GetNumCPs()<<"::"<< t_test1 <<"="<<time0<<", "<< time1 << "||" << time2 << "||" << time3 << "||" << time4 <<", "<<time5<< std::endl;

			//std::cout << "lm::end::" <<mpTargetFrame->GetFrameID()<<"::"<<nCreated<< std::endl;
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
		auto pMPi = pCPi->GetMP();
		if (pMPi) {
			pMPi->SetWorldPos(X3D);
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
				pMatch->AddMP();
				pMP->ConnectFrame(pMatch, idx);
			}
			/*pMP->AddFrame(pCurrKF->mpMatchInfo, pt1);
			pMP->AddFrame(pPrevKF->mpMatchInfo, pt2);*/
		}
	}

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
				if (pMatch->mpRefFrame->GetKeyFrameID() % 3 != 0) {
					continue;
				}
				int idx = iter->second;
				pMatch->AddMP();
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
int UVR_SLAM::LocalMapper::MappingProcess(Map* pMap, Frame* pCurrKF, Frame* pPrevKF,
	std::vector<cv::Point2f>& vMappingPrevPts, std::vector<cv::Point2f>& vMappingCurrPts, std::vector<CandidatePoint*>& vMappingCPs,
	std::vector<cv::Point2f>  vMatchedPrevPts, std::vector<cv::Point2f>  vMatchedCurrPts, std::vector<CandidatePoint*>  vMatchedCPs,
	double& dtime, cv::Mat& debugging) {

	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();

	if (vMatchedPrevPts.size() < 10) {
		std::cout << "LM::Matching::error" << std::endl;
		return -1;
	}
	int nCurrKeyFrameID = pCurrKF->GetKeyFrameID();
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
	int nTargetID = pPrevKF->GetFrameID();
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
	if (vX3Ds.size() < 20)
		return -1;

	/////////Scale 계산
	std::nth_element(vfScales.begin(), vfScales.begin() + vfScales.size() / 2, vfScales.end());
	float medianPrevScale = vfScales[vfScales.size() / 2];
	/////////Scale 계산

	mpMap->ClearReinit();
	std::vector<bool> vbInliers(vX3Ds.size(), true);
	Optimization::LocalOptimization(mpSystem, mpMap, pCurrKF, vX3Ds, vMappingCPs, vbInliers, medianPrevScale);

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

int UVR_SLAM::LocalMapper::MappingProcess2(Map* pMap, Frame* pCurrKF, Frame* pPrevKF,
	std::vector<cv::Point2f>& vMappingPrevPts, std::vector<cv::Point2f>& vMappingCurrPts, std::vector<CandidatePoint*>& vMappingCPs,
	std::vector<cv::Point2f>  vMatchedPrevPts, std::vector<cv::Point2f>  vMatchedCurrPts, std::vector<CandidatePoint*>  vMatchedCPs,
	double& dtime, cv::Mat& debugging) {

	std::chrono::high_resolution_clock::time_point tracking_start = std::chrono::high_resolution_clock::now();

	if (vMatchedPrevPts.size() < 10) {
		std::cout << "LM::Matching::error" << std::endl;
		return -1;
	}
	int nCurrKeyFrameID = pCurrKF->GetKeyFrameID();
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
	int nTargetID = pPrevKF->GetFrameID();
	std::vector<float> vfScales;

	int N = vMatchedCurrPts.size();
	for (int i = 0; i < N; i++) {
		auto currPt = std::move(vMatchedCurrPts[i]);
		auto pCPi = std::move(vMatchedCPs[i]);
		auto pMPi = pCPi->GetMP();
		cv::Mat X3D;
		bool bMP = false;
		bool b1, b2;
		b1 = false;
		pCPi->CreateMapPoint(X3D, mK, mInvK, Pcurr, Rcurr, Tcurr, currPt, b1, b2);
		if (!b1)
			continue;
		if (pMPi && pMPi->GetQuality() && pMPi->isDeleted()) {
			X3D = std::move(pMPi->GetWorldPos());
			pMPi->SetLastVisibleFrame(nCurrKeyFrameID);
			//////scale
			//cv::Mat proj3 = Rprev*X3D + Tprev;
			//float depth3 = proj3.at<float>(2);
			//float scale = depth3 / depth2;
			//vfScales.push_back(scale);
			//////scale
			bMP = true;
		}
	
		//////////////////////////////////
		////이미 CP에 연결된 것들은 연결하지 않음.
		////현재 프레임에 CP 연결하기
		int idxa = pCPi->GetPointIndexInFrame(pCurrKF->mpMatchInfo);
		if (idxa == -1) {
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
		if (pCPi->GetNumSize() >= 3) {
			//nRes++;
			vMappingCurrPts.push_back(std::move(currPt));
			vMappingCPs.push_back(std::move(pCPi));
			vX3Ds.push_back(std::move(X3D));
		}
		////커넥트가 최소 3개인 CP들은 전부 참여

		//////시각화
		if (pCPi->mnFirstID == nTargetID) {
			cv::circle(debugging, currPt + ptBottom, 2, cv::Scalar(0, 0, 255), -1);
		}
		else {
			cv::circle(debugging, currPt + ptBottom, 2, cv::Scalar(255, 0, 0), -1);
		}
		if (pCPi->GetMP()) {
			cv::circle(debugging, currPt + ptBottom, 3, cv::Scalar(0, 255, 0));
		}
		//////시각화

	}

	////////////////////////////////////////최적화 진행
	if (vX3Ds.size() < 20)
		return -1;

	/////////Scale 계산
	/*std::nth_element(vfScales.begin(), vfScales.begin() + vfScales.size() / 2, vfScales.end());
	float medianPrevScale = vfScales[vfScales.size() / 2];*/
	/////////Scale 계산

	mpMap->ClearReinit();
	std::vector<bool> vbInliers(vX3Ds.size(), true);
	Optimization::LocalOptimization(mpSystem, mpMap, pCurrKF, vX3Ds, vMappingCPs, vbInliers, 1.0);//medianPrevScale

	///////////////////New Mp Creation
	////기존 MP도 여기 결과에 따라서 커넥션이 가능해야 할 듯함.

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
						pMatch->AddMP();
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
