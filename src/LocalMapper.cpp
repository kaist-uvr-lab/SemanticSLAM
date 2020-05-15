#include <LocalMapper.h>
#include <Frame.h>
#include <System.h>
#include <Map.h>
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
UVR_SLAM::LocalMapper::LocalMapper(Map* pMap, int w, int h):mnWidth(w), mnHeight(h), mbStopBA(false), mbDoingProcess(false), mbStopLocalMapping(false), mpTargetFrame(nullptr), mpPrevKeyFrame(nullptr), mpPPrevKeyFrame(nullptr){
	mpMap = pMap;
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
	mpTargetFrame->mpMatchInfo->SetKeyFrame();
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
			std::cout << "lm::start" << std::endl;
			ProcessNewKeyFrame();
			int nTargetID = mpTargetFrame->GetFrameID();

			//////////////업데이트 맵포인트
			auto matchInfo = mpTargetFrame->mpMatchInfo;
			for (int i = 0; i < matchInfo->mvMatchingPts.size(); i++) {
				auto pMPi = matchInfo->mvpMatchingMPs[i];
				auto pt = matchInfo->mvMatchingPts[i];
				if (pMPi && !pMPi->isDeleted()) {
					pMPi->AddFrame(matchInfo, i);
					//pMPi->AddDenseFrame(mpTargetFrame, pt);
				}
			}
			//mpMap->AddFrame(mpTargetFrame);
			//////////////업데이트 맵포인트

			//////////////업데이트 키프레임
			std::map<UVR_SLAM::Frame*, int> mmpCandidateKFs;
			//int nTargetID = mpTargetFrame->GetFrameID();
			int nMaxKF = 0;
			int nMinKF = INT_MAX;
			for (int i = 0; i < matchInfo->mvpMatchingMPs.size(); i++) {
				UVR_SLAM::MapPoint* pMP = matchInfo->mvpMatchingMPs[i];
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
			std::cout << "lm::updatekf::" << mmpCandidateKFs.size() <<", "<< nMaxKF << std::endl;
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
			std::cout << "lm::updatekf::" <<mpTargetFrame->GetKeyFrameID()<<"::"<< mmpCandidateKFs.size() <<", "<<mpTargetFrame->GetConnectedKFs().size()<<", "<< nMinKF << ", " << nMaxKF << std::endl;
			//////////////업데이트 키프레임

			/////////////////
			////맵포인트 생성
			cv::Mat ddebug;
			CreateMapPoints(mpTargetFrame->mpMatchInfo, ddebug);
			////맵포인트 생성
			/////////////////
			
			///////////////////Fuse Map Points
			
			///////////////////Fuse Map Points

			std::chrono::high_resolution_clock::time_point ba_start = std::chrono::high_resolution_clock::now();
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
			auto du_test2 = std::chrono::duration_cast<std::chrono::milliseconds>(lm_end - ba_start).count();
			float t_test2 = du_test2 / 1000.0;

			std::stringstream ssa;
			ssa << "LocalMapping : " << mpTargetFrame->GetKeyFrameID() <<"::"<<mpTargetFrame->GetConnectedKFs().size()<< "::" << t_test1<<"::"<< matchInfo->mvnTargetMatchingPtIDXs.size()<<", "<< matchInfo->mvnMatchingPtIDXs.size();
			mpSystem->SetLocalMapperString(ssa.str());

			std::cout << "lm::end" << std::endl;
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

void UVR_SLAM::LocalMapper::CreateMapPoints(MatchInfo* pCurrMatchInfo, cv::Mat& debug) {

	std::vector<cv::Point2f> vPts1, vPts2, vPts3;
	std::vector<int> vIDXs1, vIDXs2, vIDXs3;
	std::vector<int> vLabels3;

	auto currFrame = pCurrMatchInfo->mpRefFrame;
	auto targetFrame = pCurrMatchInfo->mpTargetFrame;//현재 매칭정보의 이전 프레임을 의미함.
	auto targettargetFrame = targetFrame->mpMatchInfo->mpTargetFrame;

	/////////debug
	//cv::Mat targettargetImg = targettargetFrame->GetOriginalImage();
	//cv::Mat currImg = currFrame->GetOriginalImage();
	//cv::Point2f ptBottom = cv::Point2f(0, currImg.rows);
	//cv::Rect mergeRect1 = cv::Rect(0, 0, currImg.cols, currImg.rows);
	//cv::Rect mergeRect2 = cv::Rect(0, currImg.rows, currImg.cols, currImg.rows);
	//debug = cv::Mat::zeros(currImg.rows * 2, currImg.cols, currImg.type());
	//targettargetImg.copyTo(debug(mergeRect1));
	//currImg.copyTo(debug(mergeRect2));
	/////////debug

	cv::Mat Pcurr, Ptarget, Ptargettarget;
	cv::Mat Rcurr, Tcurr, Rtarget, Ttarget, Rtargettarget, Ttargettarget;
	cv::Mat K = mpTargetFrame->mK.clone();
	currFrame->GetPose(Rcurr, Tcurr);
	targetFrame->GetPose(Rtarget, Ttarget);
	targettargetFrame->GetPose(Rtargettarget, Ttargettarget);

	cv::hconcat(Rcurr, Tcurr, Pcurr);
	cv::hconcat(Rtarget, Ttarget, Ptarget);
	cv::hconcat(Rtargettarget, Ttargettarget, Ptargettarget);

	auto targetInfo = targetFrame->mpMatchInfo;
	auto targetTargetInfo = targettargetFrame->mpMatchInfo;

	int nTargetMatch = pCurrMatchInfo->mnTargetMatch;
	int nTargetTargetSize = targetTargetInfo->mvMatchingPts.size();

	for (int i = 0; i < nTargetMatch; i++) {
		if (pCurrMatchInfo->mvpMatchingMPs[i])
			continue;
		int tidx1 = pCurrMatchInfo->mvnTargetMatchingPtIDXs[i];
		if (tidx1 >= targetInfo->mnTargetMatch) {
			continue;
		}
		int tidx2 = targetInfo->mvnTargetMatchingPtIDXs[tidx1];
		if (tidx2 >= nTargetTargetSize)
			continue;
		int idx2 = targetInfo->mvnMatchingPtIDXs[tidx1];
		int idx3 = targetTargetInfo->mvnMatchingPtIDXs[tidx2];
		
		if (targetInfo->mvpMatchingMPs[idx2]) {
			continue;
		}
		if (targetTargetInfo->mvpMatchingMPs[idx3]) {
			continue;
		}

		if (!targettargetFrame->isInImage(targetTargetInfo->mvMatchingPts[idx3].x, targetTargetInfo->mvMatchingPts[idx3].y, 10.0))
			continue;
		vIDXs1.push_back(i);
		vIDXs2.push_back(idx2);
		vIDXs3.push_back(idx3);

		vPts1.push_back(pCurrMatchInfo->mvMatchingPts[i]);
		vPts2.push_back(targetInfo->mvMatchingPts[idx2]);
		vPts3.push_back(targetTargetInfo->mvMatchingPts[idx3]);

		int label3 = targetTargetInfo->mvObjectLabels[idx3];
		vLabels3.push_back(label3);

		//////visualize
		//cv::circle(debug, targetTargetInfo->mvMatchingPts[idx3], 2, cv::Scalar(255, 0, 255), -1);
		//cv::circle(debug, mvMatchingPts[i] + ptBottom, 2, cv::Scalar(255, 0, 255), -1);
	}
	////////삼각화로 맵생성
	//std::cout << "before triangle::" << vPts3.size() << std::endl;
	if (vPts3.size() < 10) {
		//std::cout << "after triangle" << std::endl;
		return;
	}
	cv::Mat Map;
	cv::triangulatePoints(K*Ptargettarget, K*Pcurr, vPts3, vPts1, Map);
	//std::cout << "after triangle::" << Map.size() << std::endl;
	////////삼각화로 맵생성

	///////데이터 전처리
	cv::Mat Rcfromc = Rcurr.t();
	cv::Mat Rttfromc = Rtargettarget.t();
	cv::Mat invT, invK, invP;
	bool bPlaneMap = false;
	auto targetTargetPlaneInfo = targettargetFrame->mpPlaneInformation;
	if (targetTargetPlaneInfo) {
		bPlaneMap = true;
		targetTargetPlaneInfo->Calculate();
		targetTargetPlaneInfo->GetInformation(invP, invT, invK);
	}
	///////데이터 전처리

	int nRes = 0;
	for (int i = 0; i < Map.cols; i++) {

		cv::Mat X3D = Map.col(i);
		//X3D.convertTo(X3D, CV_32FC1);
		X3D /= X3D.at<float>(3);
		X3D = X3D.rowRange(0, 3);

		cv::Mat proj1 = Rcurr*X3D + Tcurr;
		cv::Mat proj2 = Rtarget*X3D + Ttarget;
		cv::Mat proj3 = Rtargettarget*X3D + Ttargettarget;

		////depth test
		if (proj1.at<float>(2) < 0.0 || proj2.at<float>(2) < 0.0 || proj3.at<float>(2) < 0.0) {
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
		auto pt1 = pCurrMatchInfo->mvMatchingPts[vIDXs1[i]];
		auto pt2 = targetInfo->mvMatchingPts[vIDXs2[i]];
		auto pt3 = targetTargetInfo->mvMatchingPts[vIDXs3[i]];
		auto diffPt1 = projected1 - pt1;
		auto diffPt2 = projected2 - pt2;
		auto diffPt3 = projected3 - pt3;
		float err1 = sqrt(diffPt1.dot(diffPt1));
		float err2 = sqrt(diffPt2.dot(diffPt2));
		float err3 = sqrt(diffPt3.dot(diffPt3));
		if (err1 > 4.0 || err2 > 4.0 || err3 > 4.0)
			continue;
		////reprojection error

		//////parallax check
		////targettarget과 current만
		//cv::Mat xn1 = (cv::Mat_<float>(3, 1) << pt1.x, pt1.y, 1.0);
		//cv::Mat xn3 = (cv::Mat_<float>(3, 1) << pt3.x, pt3.y, 1.0);
		////std::cout << xn1.t() << xn3.t();
		//cv::Mat ray1 = Rcfromc*invK*xn1;
		//cv::Mat ray3 = Rttfromc*invK*xn3;
		////std::cout <<"\t"<< ray1.t() << ray3.t();
		//float cosParallaxRays = ray1.dot(ray3) / (cv::norm(ray1)*cv::norm(ray3));
		////std::cout << cosParallaxRays << std::endl;
		//if (cosParallaxRays > 0.99999) //9999 : 위안홀까지 가능, 99999 : 비전홀, N5
		//	continue;
		//////parallax check

		///////////평면 정보로 맵생성
		//if (bPlaneMap && vLabels3[i] == 150) {
		//	cv::Mat a;
		//	if (UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(pt3, invP, invT, invK, a)) {
		//		X3D = a;
		//		//b = true;
		//	}
		//}
		///////////평면 정보로 맵생성

		nRes++;
		auto pMP = new UVR_SLAM::MapPoint(mpMap, currFrame, X3D, cv::Mat(), vLabels3[i]);
		if (vLabels3[i] > 0)
			pMP->SetPlaneID(1);

		/*mvpMatchingMPs[vIDXs1[i]] = pMP;
		targetInfo->mvpMatchingMPs[vIDXs2[i]] = pMP;
		targetTargetInfo->mvpMatchingMPs[vIDXs3[i]] = pMP;
		auto pt1 = mvMatchingPts[vIDXs1[i]];
		auto pt2 = targetInfo->mvMatchingPts[vIDXs2[i]];
		auto pt3 = targetTargetInfo->mvMatchingPts[vIDXs3[i]];*/

		pMP->AddFrame(currFrame->mpMatchInfo, vIDXs1[i]);
		pMP->AddFrame(targetFrame->mpMatchInfo, vIDXs2[i]);
		pMP->AddFrame(targettargetFrame->mpMatchInfo, vIDXs3[i]);
	}
	//std::cout << "lm::create new mp ::" <<vPts1.size()<<", "<< nRes <<" "<<Map.type()<< std::endl;
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
