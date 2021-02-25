#include <LocalMapper.h>
#include <CandidatePoint.h>
#include <Frame.h>
#include <FrameGrid.h>
#include <MapGrid.h>
#include <System.h>
#include <ORBextractor.h>
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
#include <future>

#include <WebAPI.h>

UVR_SLAM::LocalMapper::LocalMapper(){}
UVR_SLAM::LocalMapper::LocalMapper(System* pSystem):mbStopBA(false), mbDoingProcess(false), mbStopLocalMapping(false), mpTempFrame(nullptr),mpTargetFrame(nullptr), mpPrevKeyFrame(nullptr), mpPPrevKeyFrame(nullptr){
	mpSystem = pSystem;
	mnThreshMinKF = mpSystem->mnThreshMinKF;
}
UVR_SLAM::LocalMapper::~LocalMapper() {}

void UVR_SLAM::LocalMapper::Init() {

	mK = mpSystem->mK.clone();
	mInvK = mpSystem->mInvK.clone();
	mnWidth = mpSystem->mnWidth;
	mnHeight = mpSystem->mnHeight;

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
	UVR_SLAM::System::nKeyFrameID = 0;
	pKF1->mnKeyFrameID = UVR_SLAM::System::nKeyFrameID++;
	pKF2->mnKeyFrameID = UVR_SLAM::System::nKeyFrameID++;
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
	//mpTempFrame->Init(mpSystem->mpORBExtractor, mpSystem->mK, mpSystem->mD);
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

	auto mvpMPs = mpTargetFrame->GetMapPoints();
	for (size_t i = 0, iend = mvpMPs.size(); i < iend; i++) {
		auto pMP = mvpMPs[i];
		if (!pMP || pMP->isDeleted())
			continue;
		pMP->AddObservation(mpTargetFrame, i);
	}

	mpMap->AddFrame(mpTargetFrame);
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
	mpTargetFrame = nullptr;
	mpPrevKeyFrame = nullptr;
	mpPPrevKeyFrame = nullptr;
}

namespace UVR_SLAM {
	auto lambda_api_kf_match = [](std::string ip, int port, std::string map, int id1, int id2, int n) {

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		WebAPI* mpAPI = new WebAPI(ip, port);
		std::stringstream ss;
		ss << "/featurematch?map=" << map << "&id1=" << id1 << "&id2=" << id2;
		auto res = mpAPI->Send(ss.str(),"");
		cv::Mat matches;
		WebAPIDataConverter::ConvertStringToMatches(res.c_str(), n, matches);
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float t_test1 = du_test1 / 1000.0;
		//std::cout << "api::featurematch=" << t_test1 << std::endl;
		return matches;
	};

	auto lambda_api_create_mp = []
	(cv::Point2f prevPt, cv::Point2f currPt, 
		cv::Mat Rprev, cv::Mat Tprev, cv::Mat Rtprev, cv::Mat Pprev, 
		cv::Mat Rcurr, cv::Mat Tcurr, cv::Mat Rtcurr, cv::Mat Pcurr,
		Map* pMap, Frame* pCurr, cv::Mat desc, cv::Mat mK, cv::Mat mInvK) {
		cv::Mat xn1 = (cv::Mat_<float>(3, 1) << prevPt.x, prevPt.y, 1.0);
		cv::Mat xn2 = (cv::Mat_<float>(3, 1) << currPt.x, currPt.y, 1.0);
		cv::Mat ray1 = Rtprev*mInvK*xn1;
		cv::Mat ray2 = Rtcurr*mInvK*xn2;
		float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1)*cv::norm(ray2));

		bool bParallax = cosParallaxRays < 0.9998;

		cv::Mat A(4, 4, CV_32F);
		A.row(0) = prevPt.x*Pprev.row(2) - Pprev.row(0);
		A.row(1) = prevPt.y*Pprev.row(2) - Pprev.row(1);
		A.row(2) = currPt.x*Pcurr.row(2) - Pcurr.row(0);
		A.row(3) = currPt.y*Pcurr.row(2) - Pcurr.row(1);

		cv::Mat u, w, vt;
		cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
		cv::Mat x3D = vt.row(3).t();
		x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
		cv::Point2f pt1, pt2;
		bool bDepth1, bDepth2;
		bool bDist1, bDist2;

		float thresh = 9.0;
		{
			cv::Mat Xcam = Rprev * x3D + Tprev;
			cv::Mat Ximg = mK*Xcam;
			float fDepth = Ximg.at < float>(2);
			bDepth1 = fDepth > 0.0;
			pt1 = cv::Point2f(Ximg.at<float>(0) / fDepth, Ximg.at<float>(1) / fDepth);
			auto diffPt = pt1 - prevPt;
			float dist = diffPt.dot(diffPt);
			bDist1 = dist < thresh;
		}
		{
			cv::Mat Xcam = Rcurr * x3D + Tcurr;
			cv::Mat Ximg = mK*Xcam;
			float fDepth = Ximg.at < float>(2);
			bDepth2 = fDepth > 0.0;
			pt2 = cv::Point2f(Ximg.at<float>(0) / fDepth, Ximg.at<float>(1) / fDepth);
			auto diffPt = pt2 - currPt;
			float dist = diffPt.dot(diffPt);
			bDist2 = dist < thresh;
		}
		MapPoint* res = nullptr;
		if (bDist1 && bDepth1 && bDist2 && bDepth2) {
			pMap->AddTempMP(x3D);
			res = new UVR_SLAM::MapPoint(pMap, pCurr, x3D, desc, 0);
		}
		return res;
	};

}

void UVR_SLAM::LocalMapper::RunWithMappingServer() {
	int CODE_MATCH_ERROR = 10000;
	std::cout << "MappingServer::LocalMapper::Start" << std::endl;
	while (true) {
		////이전 키프레임과 연결
		////맵포인트 마지날라이제이션
		////새 맵포인트 생성
		////퓨즈
		if (CheckNewKeyFrames()) {
			std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
			SetDoingProcess(true);
			AcquireFrame();
#ifdef DEBUG_LOCAL_MAPPING_LEVEL_1
			std::cout << "MappingServer::LocalMapper::"<<mpTempFrame->mnFrameID<<"::Start" << std::endl;
#endif
			ProcessNewKeyFrame();
#ifdef DEBUG_LOCAL_MAPPING_LEVEL_2
			//std::cout << "MappingServer::LocalMapper::1" << std::endl;
#endif
			mpLoopCloser->InsertKeyFrame(mpTargetFrame);
#ifdef DEBUG_LOCAL_MAPPING_LEVEL_2
			//std::cout << "MappingServer::LocalMapper::2" << std::endl;
#endif
			NewMapPointMarginalization();
#ifdef DEBUG_LOCAL_MAPPING_LEVEL_2
			//std::cout << "MappingServer::LocalMapper::3" << std::endl;
#endif
			ComputeNeighborKFs(mpTargetFrame);
#ifdef DEBUG_LOCAL_MAPPING_LEVEL_2
			//std::cout << "MappingServer::LocalMapper::4" << std::endl;
#endif
			ConnectNeighborKFs(mpTargetFrame, mpTargetFrame->mmKeyFrameCount, 20);
#ifdef DEBUG_LOCAL_MAPPING_LEVEL_2
			std::cout << "MappingServer::LocalMapper::5" << std::endl;
#endif
			std::chrono::high_resolution_clock::time_point temp1 = std::chrono::high_resolution_clock::now();
			auto vpNeighKFs = mpPrevKeyFrame->GetConnectedKFs(5);
			vpNeighKFs.push_back(mpPrevKeyFrame);
			std::vector<Frame*> vpKFs;
			for (size_t i = 0, iend = vpNeighKFs.size(); i < iend; i++) {
				auto pKF = vpNeighKFs[i];
				if (pKF->mnFrameID == mpTargetFrame->mnFrameID)
					continue;
				vpKFs.push_back(pKF);
			}
			cv::Mat mMatches = cv::Mat::zeros(0, mpTargetFrame->mvPts.size(), CV_32SC1);
			
			std::vector<cv::Mat> Rs, Ts, Ps, Rts;
			for (size_t i = 0, iend = vpKFs.size(); i < iend; i++) {
				auto pKF = vpKFs[i];
				/*auto ftest = std::async(std::launch::async, UVR_SLAM::lambda_api_kf_match, "143.248.96.81", 35005, mpTargetFrame->mnFrameID, pKF->mnFrameID, mpTargetFrame->mvPts.size());
				cv::Mat temp = ftest.get();*/
				cv::Mat temp = UVR_SLAM::lambda_api_kf_match(mpSystem->ip, mpSystem->port, mpTargetFrame->mstrMapName, mpTargetFrame->mnFrameID, pKF->mnFrameID, mpTargetFrame->mvPts.size());
				if (mpTargetFrame->mvPts.size() != temp.cols) {
					std::cout << "Error::Matching::Invalid Matching Size::" << temp.cols << ", " << mpTargetFrame->mvPts.size() << std::endl;
				}
				std::vector<bool> vecBoolOverlap(pKF->mvPts.size(), false);
				for (size_t j = 0, jend = temp.cols; j < jend; j++) {
					int idx1 = j;
					int idx2 = temp.at<int>(idx1);
					if (idx2 == CODE_MATCH_ERROR)
						continue;
					if (idx2 > CODE_MATCH_ERROR || idx2 < -1) {
						temp.at<int>(idx1) = CODE_MATCH_ERROR;
						std::cout << "Error::Matching::Invalid Frame2 Indexs = " <<idx2 <<", "<<pKF->mvPts.size()<<"::"<<j<< std::endl;
						continue;
					}
					if (vecBoolOverlap[idx2])
					{
						temp.at<int>(idx1) = CODE_MATCH_ERROR;
						continue;
					}
					vecBoolOverlap[idx2] = true;
				}

				mMatches.push_back(temp);
			
				cv::Mat Rprev, Tprev, Pprev;
				pKF->GetPose(Rprev, Tprev);
				cv::hconcat(Rprev, Tprev, Pprev);
				Pprev = mK*Pprev;
				Rs.push_back(Rprev);
				Ts.push_back(Tprev);
				Ps.push_back(Pprev);
				Rts.push_back(Rprev.t());
			}
			std::chrono::high_resolution_clock::time_point temp2 = std::chrono::high_resolution_clock::now();
#ifdef DEBUG_LOCAL_MAPPING_LEVEL_2
			std::cout << "MappingServer::LocalMapper::6" << std::endl;
#endif		
			cv::Mat Rcurr, Tcurr, Pcurr;
			mpTargetFrame->GetPose(Rcurr, Tcurr);
			cv::hconcat(Rcurr, Tcurr, Pcurr);
			Pcurr = mK*Pcurr;
			cv::Mat Rtcurr = Rcurr.t();
			
			////포즈 리파인먼트
			std::vector < cv::Point2f> vTempPts;
			std::vector<MapPoint*> vpTempMPs;
			std::vector<bool> vbTempInliers;
			////포즈 리파인먼트

			int nNewMP = 0;
			int nFailNewMP = 0;
			int nFuse = 0;
			int nConnect = 0;
			mpMap->ClearTempMPs();
			for (int c = 0, cols = mMatches.cols; c < cols; c++) {
				std::map<MapPoint*, int> mpMPs;
				std::vector<std::pair<int, MapPoint*>> vPairMPs; //fuse용.
				std::vector<std::pair<int, int>> vPairMatches; //idx, pkf
				for (int r = 0, rows = mMatches.rows; r < rows; r++) {
					int idx = mMatches.at<int>(r, c);
					if (idx == CODE_MATCH_ERROR)
						continue;
					auto pKF = vpKFs[r];
					auto pMP = pKF->GetMapPoint(idx);
					if (pMP && !pMP->isDeleted() && !mpMPs.count(pMP)) {
						mpMPs[pMP] = pMP->GetObservations().size();
						vPairMPs.push_back(std::make_pair(pMP->GetObservations().size(), pMP));
					}else
						vPairMatches.push_back(std::make_pair(idx, r));
				}
				int nTempMatch = vPairMPs.size() + vPairMatches.size();
				if (nTempMatch == 0)
					continue;
				//MP 커넥트, && 생성
				if (mpMPs.size() == 0) {
					//생성 & connect
					int nSize = vPairMatches.size();
					/*if (nSize == 0)
						 continue;*/
					auto pair = vPairMatches[nSize - 1];
					auto pKF = vpKFs[pair.second];
					int kID = pair.second;
					auto prevPt = pKF->mvPts[pair.first];
					auto currPt = mpTargetFrame->mvPts[c];

					//mpTargetFrame->matDescriptor.row(c)
					auto pNewMP = lambda_api_create_mp(prevPt, currPt, Rs[kID], Ts[kID], Rts[kID], Ps[kID], Rcurr, Tcurr, Rtcurr, Pcurr, mpMap, mpTargetFrame, cv::Mat(), mK, mInvK);
					if (pNewMP) {

						nNewMP++;
						mpSystem->mlpNewMPs.push_back(pNewMP);
						mpTargetFrame->AddMapPoint(pNewMP, c);
						pNewMP->AddObservation(mpTargetFrame, c);
						pNewMP->IncreaseFound();
						pNewMP->IncreaseVisible();

						for (size_t i = 0, iend = vPairMatches.size(); i < iend; i++) {
							int idx = vPairMatches[i].first;
							auto pKF = vpKFs[vPairMatches[i].second];
							pNewMP->AddObservation(pKF, idx);
							pKF->AddMapPoint(pNewMP, idx);
							pNewMP->IncreaseFound();
							pNewMP->IncreaseVisible();
						}//pair
						
					}
					else
						nFailNewMP++;
				}
				else{
					nConnect++;
					if (mpMPs.size() > 1) {
						//error case
						if(mpMPs.size() != vPairMPs.size()){
#ifdef PRINT_ERROR
							std::cout << "Fuse Error!!!!!" << std::endl;
#endif
						}
						std::sort(vPairMPs.begin(), vPairMPs.end(), std::greater<>());
						auto pTargetMP = vPairMPs[0].second;
						for (size_t i = 1, iend = vPairMPs.size(); i < iend; i++) {
							auto pMP = vPairMPs[i].second;
							pMP->Fuse(pTargetMP);
						}
						/*for(auto iter = mpMPs.begin(), iend = mpMPs.end(); iter != iend; iter++) {
							std::cout << "Fuse::" << iter->first->mnMapPointID << ", " << iter->second <<"::"<<iter->first->mnFirstKeyFrameID<< std::endl;
						}*/
						nFuse++;
					}
					auto pMP = vPairMPs[0].second;

					mpTargetFrame->AddMapPoint(pMP, c);
					pMP->AddObservation(mpTargetFrame, c);
					pMP->IncreaseFound();
					pMP->IncreaseVisible();

					////포즈 리파인먼트
					vpTempMPs.push_back(pMP);
					vTempPts.push_back(mpTargetFrame->mvPts[c]);
					vbTempInliers.push_back(true);
					////포즈 리파인먼트

					for (size_t i = 0, iend = vPairMatches.size(); i < iend; i++) {
						
						int idx  = vPairMatches[i].first;
						auto pKF = vpKFs[vPairMatches[i].second];
						pMP->AddObservation(pKF, idx);
						pKF->AddMapPoint(pMP, idx);
						pMP->IncreaseFound();
						pMP->IncreaseVisible();
					}//pair
				}//if mps
			}//for match
			//포즈를 수정한 후 해당 프레임 아이디를 서버에 전송하기.
			int nPoseRecovery = Optimization::PoseOptimization(mpSystem->mpMap, mpTargetFrame, vpTempMPs, vTempPts, vbTempInliers);
			WebAPI* mpAPI = new WebAPI(mpSystem->ip, mpSystem->port);
			std::stringstream ss;
			ss << "/SetLastFrameID?map=" << mpTargetFrame->mstrMapName << "&id=" << mpTargetFrame->mnFrameID << "&key=reference";
			mpAPI->Send(ss.str(),"");
			
#ifdef DEBUG_LOCAL_MAPPING_LEVEL_1
			std::cout << "MappingServer::PoseRefinement::" << nPoseRecovery << ", " << vpTempMPs.size() << std::endl;
#endif


#ifdef DEBUG_LOCAL_MAPPING_LEVEL_2
			std::cout << "MappingServer::LocalMapper::7" << std::endl;
#endif
			std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
			auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			float t_test1 = du_test1 / 1000.0;
			auto du_test2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - temp2).count();
			float t_test2 = du_test2 / 1000.0;
			auto du_test3 = std::chrono::duration_cast<std::chrono::milliseconds>(temp2 - temp1).count();
			float t_test3 = du_test3 / 1000.0;
			auto du_test4 = std::chrono::duration_cast<std::chrono::milliseconds>(temp1 - start).count();
			float t_test4 = du_test4 / 1000.0;
#ifdef DEBUG_LOCAL_MAPPING_LEVEL_1
			std::cout << "MappingServer::LocalMapping::" << mpTempFrame->mnFrameID <<"::"<<mpTargetFrame->GetConnectedKFs().size()<<"::"<< nConnect <<", "<<nNewMP<<", "<< nFailNewMP <<", "<< nFuse <<"::"<< t_test1<<", "<< t_test2<<", "<< t_test3 <<", "<<t_test4<< "::End" << std::endl;
#endif

//			int nTargetID = mpTargetFrame->mnFrameID;
//			std::vector<UVR_SLAM::Frame*> vpOptKFs, vpTempKFs;
//			std::vector<UVR_SLAM::Frame*> vpFixedKFs;
//			std::vector<UVR_SLAM::MapPoint*> vpOptMPs, vpTempMPs;// , vpMPs2;
//			std::map<Frame*, int> mpKeyFrameCounts, mpGraphFrameCounts;
//
//			auto vpMPs = mpTargetFrame->GetMapPoints();
//			for (size_t i = 0, iend = mpTargetFrame->mvPts.size(); i < iend; i++) {
//				auto pMPi = vpMPs[i];
//				if (!pMPi || pMPi->isDeleted())
//					continue;
//				if (pMPi->mnLocalBAID == nTargetID) {
//					continue;
//				}
//				pMPi->mnLocalBAID = nTargetID;
//				vpOptMPs.push_back(pMPi);
//			}
//
//			mpTargetFrame->mnLocalBAID = nTargetID;
//			vpOptKFs.push_back(mpTargetFrame);
//
//			auto vpNeighKFs = mpTargetFrame->GetConnectedKFs(15);
//			for (auto iter = vpNeighKFs.begin(), iend = vpNeighKFs.end(); iter != iend; iter++) {
//				auto pKFi = *iter;
//				if (pKFi->isDeleted())
//					continue;
//				auto vpMPs = pKFi->GetMapPoints();
//				auto vPTs = pKFi->mvPts;
//				int N1 = 0;
//				for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
//					auto pMPi = vpMPs[i];
//					if (!pMPi || pMPi->isDeleted() || pMPi->mnLocalBAID == nTargetID)
//						continue;
//					pMPi->mnLocalBAID = nTargetID;
//					vpOptMPs.push_back(pMPi);
//				}
//				if (pKFi->mnLocalBAID == nTargetID) {
//					std::cout << "Error::pKF::" << pKFi->mnFrameID << ", " << pKFi->mnKeyFrameID << std::endl;
//					continue;
//				}
//				pKFi->mnLocalBAID = nTargetID;
//				vpOptKFs.push_back(pKFi);
//
//
//			}//for vpmps, vpkfs
//
//			 //Fixed KFs
//			for (size_t i = 0, iend = vpOptMPs.size(); i < iend; i++) {
//				auto pMPi = vpOptMPs[i];
//				auto mmpFrames = pMPi->GetObservations();
//				for (auto iter = mmpFrames.begin(), iter_end = mmpFrames.end(); iter != iter_end; iter++) {
//					auto pKFi = (iter->first);
//					if (pKFi->isDeleted())
//						continue;
//					if (pKFi->mnLocalBAID == nTargetID)
//						continue;
//					mpGraphFrameCounts[pKFi]++;
//					/*pKFi->mnLocalBAID = nTargetID;
//					vpFixedKFs.push_back(pKFi);*/
//				}
//			}//for fixed iter
//
//			 ////fixed kf를 정렬
//			std::vector<std::pair<int, Frame*>> vPairFixedKFs;
//			for (auto iter = mpGraphFrameCounts.begin(), iend = mpGraphFrameCounts.end(); iter != iend; iter++) {
//				auto pKFi = iter->first;
//				auto count = iter->second;
//				if (count < 10)
//					continue;
//				vPairFixedKFs.push_back(std::make_pair(count, pKFi));
//			}
//			std::sort(vPairFixedKFs.begin(), vPairFixedKFs.end(), std::greater<>());
//
//			////상위 N개의 Fixed KF만 추가
//			for (size_t i = 0, iend = vPairFixedKFs.size(); i < iend; i++) {
//				auto pair = vPairFixedKFs[i];
//				if (vpFixedKFs.size() == 20)
//					break;
//				auto pKFi = pair.second;
//				pKFi->mnFixedBAID = nTargetID;
//				pKFi->mnLocalBAID = nTargetID;
//				vpFixedKFs.push_back(pKFi);
//			}
//#ifdef DEBUG_MAP_OPTIMIZER_LEVEL_2
//			std::cout << "MappingServer::MapOptimizer::" << mpTargetFrame->mnFrameID << "::TEST::Start" << std::endl;
//			std::map<int, int> testMPs;
//			std::map<int, int> testKFs, testFixedKFs;
//			for (size_t i = 0, iend = vpOptMPs.size(); i < iend; i++) {
//				auto pMP = vpOptMPs[i];
//				testMPs[pMP->mnMapPointID]++;
//				if (testMPs[pMP->mnMapPointID] > 1) {
//					std::cout << "BA::MP::Error::" << pMP->mnMapPointID << "::" << testMPs[pMP->mnMapPointID] << std::endl;
//				}
//			}
//			for (size_t i = 0, iend = vpOptKFs.size(); i < iend; i++) {
//				auto pKF = vpOptKFs[i];
//				testKFs[pKF->mnKeyFrameID]++;
//				if (testKFs[pKF->mnKeyFrameID] > 1) {
//					std::cout << "BA::KF::Error::" << pKF->mnKeyFrameID << std::endl;
//				}
//			}
//			for (size_t i = 0, iend = vpFixedKFs.size(); i < iend; i++) {
//				auto pKF = vpFixedKFs[i];
//				testKFs[pKF->mnKeyFrameID]++;
//				if (testKFs[pKF->mnKeyFrameID] > 1) {
//					std::cout << "BA::Fixed::Error::" << pKF->mnKeyFrameID << std::endl;
//				}
//			}
//			std::cout << "MappingServer::MapOptimizer::" << mpTargetFrame->mnFrameID << "::TEST::END" << std::endl;
//#endif
//			Optimization::OpticalLocalBundleAdjustment(mpMap, mpMapOptimizer, vpOptMPs, vpOptKFs, vpFixedKFs);
//			std::chrono::high_resolution_clock::time_point end2 = std::chrono::high_resolution_clock::now();
//			auto du_test2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end).count();
//			float t_test2 = du_test2 / 1000.0;
//
//#ifdef DEBUG_LOCAL_MAPPING_LEVEL_1
//			std::cout << "MappingServer::LocalMapping::BA::" << mpTempFrame->mnFrameID << "::" << vpOptMPs.size() << " ," << vpOptKFs.size() << ", " << vpFixedKFs.size() << "::" << t_test2 << std::endl;
//#endif

			mpMapOptimizer->InsertKeyFrame(mpTargetFrame);
			SetDoingProcess(false);
		}
	}
}
void UVR_SLAM::LocalMapper::Run() {
	
	int nMinMapPoints = 1000;

	int numActive = 0;
	float totalActive = 0.f;

	int numLM = 0;
	float totalLM = 0.f;

	std::string ip = mpSystem->ip;
	int port = mpSystem->port;

	//FeatureMatchingWebAPI::Reset(ip, port);

	auto lambda_api_detect = [](std::string ip, int port, Frame* pCurr) {
		//std::cout << "api::superpoint=" <<pCurr->mnFrameID << "=start" << std::endl;
		/*std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		WebAPI* mpAPI = new WebAPI(ip, port);
		mpAPI->Send("receiveimage", WebAPIDataConverter::ConvertImageToString(pCurr->GetOriginalImage(), pCurr->mnFrameID));
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		std::vector<cv::Point2f> aa;
		WebAPIDataConverter::ConvertStringToPoints(mpAPI->Send("detectWithDesc", WebAPIDataConverter::ConvertNumberToString(pCurr->mnFrameID)).c_str(), aa, pCurr->matDescriptor);
		std::chrono::high_resolution_clock::time_point end2 = std::chrono::high_resolution_clock::now();
		pCurr->SetMapPoints(pCurr->mvPts.size());
		
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float t_test1 = du_test1 / 1000.0;
		auto du_test2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end).count();
		float t_test2 = du_test2 / 1000.0;
		std::cout << "api::superpoint=" << t_test1<<", "<< t_test2<< std::endl;*/
	};

	auto lambda_api_DetectAndMatch = [](std::string ip, int port, Frame* pCurr, int id2) {
		//std::cout << "api::superpoint=" <<pCurr->mnFrameID << "=start" << std::endl;
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		WebAPI* mpAPI = new WebAPI(ip, port);
		mpAPI->Send("receiveimage", WebAPIDataConverter::ConvertImageToString(pCurr->GetOriginalImage(), pCurr->mnFrameID));
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		int nRes;
		auto strID = WebAPIDataConverter::ConvertNumberToString(pCurr->mnFrameID);
		WebAPIDataConverter::ConvertStringToNumber(mpAPI->Send("detect", strID).c_str(), nRes);
		WebAPIDataConverter::ConvertStringToPoints(mpAPI->Send("getPts", strID).c_str(), pCurr->mvPts);
		std::chrono::high_resolution_clock::time_point endtemp = std::chrono::high_resolution_clock::now();
		WebAPIDataConverter::ConvertStringToDesc(mpAPI->Send("getDesc", strID).c_str(), nRes, pCurr->matDescriptor);
		std::chrono::high_resolution_clock::time_point end2 = std::chrono::high_resolution_clock::now();
		
		std::vector<int> matches;
		WebAPIDataConverter::ConvertStringToMatches(mpAPI->Send("featurematch", WebAPIDataConverter::ConvertNumberToString(pCurr->mnFrameID, id2)).c_str(), nRes, matches);
		std::chrono::high_resolution_clock::time_point end3 = std::chrono::high_resolution_clock::now();
		
		pCurr->SetMapPoints(nRes);
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float t_test1 = du_test1 / 1000.0;
		auto du_test2 = std::chrono::duration_cast<std::chrono::milliseconds>(endtemp - end).count();
		float t_test2 = du_test2 / 1000.0;
		auto du_test3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - end2).count();
		float t_test3 = du_test3 / 1000.0;
		auto du_test4 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - endtemp).count();
		float t_test4 = du_test4 / 1000.0;
		std::cout << "api::superpoint and match=" << t_test1 << ", " << t_test2 <<", "<<t_test4<<", "<<t_test3<<"::"<< matches .size()<< std::endl;

		return matches;
	};

	auto lambda_api_match = [](std::string ip, int port, int id1, int id2, int n) {
		
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		WebAPI* mpAPI = new WebAPI(ip, port);
		auto res = mpAPI->Send("featurematch", WebAPIDataConverter::ConvertNumberToString(id1, id2));
		std::vector<int> matches;
		WebAPIDataConverter::ConvertStringToMatches(res.c_str(), n, matches);
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float t_test1 = du_test1 / 1000.0;
		std::cout << "api::featurematch=" << t_test1 << std::endl;
		return matches;
	};
	auto lambda_api_get_desc = [](std::string ip, int port, Frame* pF) {

		//std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		//WebAPI* mpAPI = new WebAPI(ip, port);
		//auto res = mpAPI->Send("getPts", WebAPIDataConverter::ConvertNumberToString(pF->mnFrameID));
		//std::vector<cv::Point2f> test; //추후 삭제 
		//WebAPIDataConverter::ConvertStringToPoints(res.c_str(), test);
		////WebAPIDataConverter::ConvertStringToDesc(res.c_str(), pF->mvPts.size(), pF->matDescriptor);
		//std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		//auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		//float t_test1 = du_test1 / 1000.0;
		//std::cout << "api::get data=" << t_test1<<", "<< test.size() << std::endl;
	};
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

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

			////포인트 검출 전에는 키프레임이 아님
			//auto f = std::async(std::launch::async, lambda_api_detect, ip, port, mpTempFrame);
			auto ftest = std::async(std::launch::async, lambda_api_DetectAndMatch, ip, port, mpTempFrame, mpTargetFrame->mnFrameID);

			auto matches = ftest.get();
			lambda_api_get_desc(ip, port, mpTempFrame);

			cv::Mat debugMatch;
			cv::Mat prevImg = mpTempFrame->GetOriginalImage().clone();
			cv::Mat currImg = mpTargetFrame->GetOriginalImage().clone();
			cv::Point2f ptBottom = cv::Point2f(prevImg.cols, 0);
			cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
			cv::Rect mergeRect2 = cv::Rect(prevImg.cols, 0, prevImg.cols, prevImg.rows);
			debugMatch = cv::Mat::zeros(prevImg.rows, prevImg.cols * 2, prevImg.type());
			prevImg.copyTo(debugMatch(mergeRect1));
			currImg.copyTo(debugMatch(mergeRect2));

			for (size_t i = 0, iend = matches.size(); i < iend; i++) {
				int idx1 = i;
				int idx2 = matches[i];
				if (idx2 == -1)
					continue;
				auto pt1 = mpTempFrame->mvPts[idx1];
				auto pt2 = mpTargetFrame->mvPts[idx2]+ptBottom;
				cv::line(debugMatch, pt1, pt2, cv::Scalar(255, 0, 255), 1);
			}
			imshow("testeteststset", debugMatch); cv::waitKey(1);

			//mpMap->ClearTempMPs();
			if (bNeedNewKF) {
				////교체 신호를 줘야 교체
				////포인트 검출 과정
				
				//ProcessNewKeyFrame();
				int nFrameID = mpTargetFrame->mnFrameID;

				std::chrono::high_resolution_clock::time_point lm_start = std::chrono::high_resolution_clock::now();
				/*WebAPI* mpAPI = new WebAPI(mpSystem->ip, mpSystem->port);
				mpAPI->Send("receiveimage", WebAPIDataConverter::ConvertImageToString(mpTargetFrame->GetOriginalImage(), mpTargetFrame->mnFrameID));
				WebAPIDataConverter::ConvertStringToPoints(mpAPI->Send("detect", WebAPIDataConverter::ConvertNumberToString(mpTargetFrame->mnFrameID)).c_str(), mpTargetFrame->mvPts, mpTargetFrame->matDescriptor);
				mpTargetFrame->SetMapPoints(mpTargetFrame->mvPts.size());*/

				{

					int nParallax = 0;
					int nCreate = 0;
					int nDepth = 0;
					int nProjection = 0;
					////KFcurr-1의 포즈 리커버리
					if (mpPrevKeyFrame) {

						/*std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateCP);
						mpSystem->mbCreateCP = false;*/
						
						std::vector < cv::Point2f> vTempPts;
						std::vector<MapPoint*> vpTempMPs;
						std::vector<bool> vbTempInliers;

						////매핑X, 오직 KFcurr-1과 KFcurr-2을 비교하여 정보 갱신하는 것
						std::vector< std::vector<cv::DMatch> > knn_matches;
						matcher->knnMatch(mpTargetFrame->matDescriptor, mpPrevKeyFrame->matDescriptor, knn_matches, 2);
						const float ratio_thresh = 0.7f;
						std::vector<cv::DMatch> good_matches;
						std::vector<bool> vecBoolOverlap(mpPrevKeyFrame->matDescriptor.rows, false);
						for (size_t i = 0; i < knn_matches.size(); i++)
						{
							if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
							{
								int idx1 = knn_matches[i][0].queryIdx;
								int idx2 = knn_matches[i][0].trainIdx;
								auto currPt = mpTargetFrame->mvPts[idx1];
								auto prevPt = mpPrevKeyFrame->mvPts[idx2];
								if (vecBoolOverlap[idx2])
								{
									continue;
								}
								vecBoolOverlap[idx2] = true;

								auto pMPi1 = mpPrevKeyFrame->GetMapPoint(idx2);
								bool bMP1 = pMPi1 && !pMPi1->isDeleted();

								if (bMP1 && pMPi1->mnLocalMapID != nFrameID) {
									pMPi1->mnLocalMapID = nFrameID;
									mpTargetFrame->AddMapPoint(pMPi1, idx1);
									pMPi1->AddObservation(mpTargetFrame, idx1);

									vpTempMPs.push_back(pMPi1);
									vTempPts.push_back(mpTargetFrame->mvPts[idx1]);
									vbTempInliers.push_back(true);
								}
							}//if knn match
						}//for
						int nPoseRecovery = Optimization::PoseOptimization(mpMap, mpTargetFrame, vpTempMPs, vTempPts, vbTempInliers, mpSystem->mpORBExtractor->GetInverseScaleSigmaSquares());
						mpMap->AddWindowFrame(mpTargetFrame);
						for (size_t i = 0; i < vbTempInliers.size(); i++)
						{
							vpTempMPs[i]->IncreaseVisible();
							if (vbTempInliers[i])
								vpTempMPs[i]->IncreaseFound();
						}
						std::cout << "Poser Recovery = " << mpTargetFrame->mnFrameID << "=" << nPoseRecovery << ", " << vpTempMPs.size() << std::endl;
						/*mpSystem->mbCreateCP = true;
						lock.unlock();
						mpSystem->cvUseCreateCP.notify_all();*/

						////키프레임 설정. 이 후에 이것들은 매핑에 포함하지 않기 위함.
						mpTargetFrame->mnLocalMapFrameID = mpTargetFrame->mnKeyFrameID;
						//mpPPrevKeyFrame->mnLocalMapFrameID = mpPrevKeyFrame->mnKeyFrameID;
					
						auto vpKFs = mpMap->GetWindowFramesVector(1);
						auto pKFtarget = mpTargetFrame;
						for (int k = 0, kend = vpKFs.size()-2; k < kend; k++) {
						//for (int k = vpKFs.size() - 1, kend = k - 6; k > kend && k >= 0; k--) {
							auto pKF = vpKFs[k];

							/*if (pKF->mnLocalMapFrameID == pKFtarget->mnKeyFrameID)
							continue;
							pKF->mnLocalMapFrameID = pKFtarget->mnKeyFrameID;*/

							////생성 테스트 관련
							cv::Mat Rcurr, Tcurr, Pcurr;
							pKFtarget->GetPose(Rcurr, Tcurr);
							cv::hconcat(Rcurr, Tcurr, Pcurr);
							cv::Mat Rprev, Tprev, Pprev;
							pKF->GetPose(Rprev, Tprev);
							cv::hconcat(Rprev, Tprev, Pprev);
							Pcurr = mK*Pcurr;
							Pprev = mK*Pprev;
							cv::Mat Rtcurr = Rcurr.t();
							cv::Mat Rtprev = Rprev.t();

							auto f2 = std::async(std::launch::async, lambda_api_match, ip, port, pKFtarget->mnFrameID, pKF->mnFrameID, pKFtarget->mvPts.size());


							std::vector< std::vector<cv::DMatch> > knn_matches;
							matcher->knnMatch(pKFtarget->matDescriptor, pKF->matDescriptor, knn_matches, 2);

							const float ratio_thresh = 0.7f;
							std::vector<cv::DMatch> good_matches;

							//mpMap->ClearTempMPs();
							std::vector<bool> vecBoolOverlap(pKF->matDescriptor.rows, false);
							
							for (size_t i = 0; i < knn_matches.size(); i++)
							{
								if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
								{
									int idx1 = knn_matches[i][0].queryIdx;
									int idx2 = knn_matches[i][0].trainIdx;
									auto currPt = pKFtarget->mvPts[idx1];
									auto prevPt = pKF->mvPts[idx2];
									if (vecBoolOverlap[idx2])
									{
										continue;
									}
									vecBoolOverlap[idx2] = true;

									auto pMPi1 = pKF->GetMapPoint(idx2);
									bool bMP1 = pMPi1 && !pMPi1->isDeleted();
									auto pMPi2 = pKFtarget->GetMapPoint(idx1);
									bool bMP2 = pMPi2 && !pMPi2->isDeleted();

									if (bMP1 && !bMP2 && pMPi1->mnLocalMapID != nFrameID) {
										pMPi1->mnLocalMapID = nFrameID;
										pKFtarget->AddMapPoint(pMPi1, idx1);
										pMPi1->AddObservation(pKFtarget, idx1);
										pMPi1->IncreaseFound();
										pMPi1->IncreaseVisible();
									}
									if (!bMP1 && bMP2) {
										pKF->AddMapPoint(pMPi2, idx2);
										pMPi2->AddObservation(pKF, idx2);
										pMPi2->IncreaseFound();
										pMPi2->IncreaseVisible();
									}
									if (!bMP1 && !bMP2) {
										////parallax check
										cv::Mat xn1 = (cv::Mat_<float>(3, 1) << prevPt.x, prevPt.y, 1.0);
										cv::Mat xn2 = (cv::Mat_<float>(3, 1) << currPt.x, currPt.y, 1.0);
										cv::Mat ray1 = Rtprev*mInvK*xn1;
										cv::Mat ray2 = Rtcurr*mInvK*xn2;
										float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1)*cv::norm(ray2));

										////projection
										bool bParallax = cosParallaxRays < 0.9998;
										/*if (!bParallax)
										std::cout << "parallax!!!!!!" << std::endl;*/

										cv::Mat A(4, 4, CV_32F);
										A.row(0) = prevPt.x*Pprev.row(2) - Pprev.row(0);
										A.row(1) = prevPt.y*Pprev.row(2) - Pprev.row(1);
										A.row(2) = currPt.x*Pcurr.row(2) - Pcurr.row(0);
										A.row(3) = currPt.y*Pcurr.row(2) - Pcurr.row(1);

										cv::Mat u, w, vt;
										cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
										cv::Mat x3D = vt.row(3).t();
										x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
										cv::Point2f pt1, pt2;
										bool bDepth1, bDepth2;
										bool bDist1, bDist2;
										float thresh = 9.0;
										{
											cv::Mat Xcam = Rprev * x3D + Tprev;
											cv::Mat Ximg = mK*Xcam;
											float fDepth = Ximg.at < float>(2);
											bDepth1 = fDepth > 0.0;
											pt1 = cv::Point2f(Ximg.at<float>(0) / fDepth, Ximg.at<float>(1) / fDepth);
											auto diffPt = pt1 - prevPt;
											float dist = diffPt.dot(diffPt);
											bDist1 = dist < thresh;
										}
										{
											cv::Mat Xcam = Rcurr * x3D + Tcurr;
											cv::Mat Ximg = mK*Xcam;
											float fDepth = Ximg.at < float>(2);
											bDepth2 = fDepth > 0.0;
											pt2 = cv::Point2f(Ximg.at<float>(0) / fDepth, Ximg.at<float>(1) / fDepth);
											auto diffPt = pt2 - currPt;
											float dist = diffPt.dot(diffPt);
											bDist2 = dist < thresh;
										}
										if (!bParallax) {
											//cv::circle(debug_glue, currPt, 3, cv::Scalar(255, 0, 255), -1);
											//cv::circle(debug_glue, prevPt + ptBottom, 3, cv::Scalar(255, 0, 255), -1);
											nParallax++;
										}
										else if (!bDepth1 || !bDepth2){
											nDepth++;
										}
										else if (!bDist1 || !bDist2) {
											nProjection++;
										}
										else if (bDist1 && bDepth1 && bDist2 && bDepth2) { //bParallax && 
											//cv::circle(debug_glue, currPt, 3, cv::Scalar(0, 255, 255), -1);
											//cv::circle(debug_glue, prevPt + ptBottom, 3, cv::Scalar(0, 255, 255), -1);
											//mpMap->AddTempMP(x3D);

											auto pNewMP = new UVR_SLAM::MapPoint(mpMap, pKFtarget, x3D, cv::Mat(), 0);
											pNewMP->SetOptimization(true);
											mpSystem->mlpNewMPs.push_back(pNewMP);

											pKFtarget->AddMapPoint(pNewMP, idx1);
											pNewMP->AddObservation(pKFtarget, idx1);

											pKF->AddMapPoint(pNewMP, idx2);
											pNewMP->AddObservation(pKF, idx2);

											pNewMP->IncreaseFound(2);
											pNewMP->IncreaseVisible(2);
											//mpMap->AddTempMP(x3D);
											nCreate++;
										}
										else {
										}
									}//if !bMP1 && !bMP2


									 /*if (testaaa.count(knn_matches[i][0].trainIdx)) {
									 std::cout << "181818118181811818" << std::endl;
									 }
									 testaaa.insert(knn_matches[i][0].trainIdx);*/
									 /*cv::circle(debug_glue, currPt, 3, cv::Scalar(0, 0, 255), -1);
									 cv::circle(debug_glue, prevPt + ptBottom, 3, cv::Scalar(0, 0, 255), -1);
									 cv::line(debug_glue, currPt, prevPt + ptBottom, cv::Scalar(0, 0, 255), 1);*/
								}//if match
							}//for match

							 //if(k == vpKFs.size() - 1){
							 //	
							 //	////포즈 다시 최적화
							 //	std::vector < cv::Point2f> vTempPts;
							 //	std::vector<MapPoint*> vpTempMPs;
							 //	std::vector<bool> vbTempInliers;
							 //	for (size_t i = 0, iend = pKFtarget->mvPts.size(); i < iend; i++) {
							 //		auto pMPi = pKFtarget->GetMapPoint(i);
							 //		if (pMPi && !pMPi->isDeleted()) {
							 //			vpTempMPs.push_back(pMPi);
							 //			vTempPts.push_back(pKFtarget->mvPts[i]);
							 //			vbTempInliers.push_back(true);
							 //		}
							 //	}

							 //	Optimization::PoseOptimization(mpMap, pKFtarget, vpTempMPs, vTempPts, vbTempInliers, mpSystem->mpORBExtractor->GetInverseScaleSigmaSquares());
							 //	mpMap->AddWindowFrame(pKFtarget);
							 //	pKFtarget->GetPose(Rcurr, Tcurr);
							 //	cv::hconcat(Rcurr, Tcurr, Pcurr);
							 //	////포즈 다시 최적화
							 //}
						}//for window keyframes

						
						NewMapPointMarginalization();
						ComputeNeighborKFs(mpTargetFrame);
						ConnectNeighborKFs(mpTargetFrame, mpTargetFrame->mmKeyFrameCount, 20);
						std::cout << "LM::" << mpTargetFrame->mnKeyFrameID << "=" << mpTargetFrame->GetConnectedKFs().size() <<"="<< nCreate <<", "<< nProjection <<", "<<nDepth<<", "<< nParallax << std::endl;// << "::" << nNew << ", " << nConnect << ", " << nFuse << "::" << nFail << std::endl;

						mpLoopCloser->InsertKeyFrame(mpTargetFrame);
						mpSegmentator->InsertKeyFrame(mpTargetFrame);

						//mpMap->AddWindowFrame(mpTargetFrame);
						if (mpMapOptimizer->isDoingProcess()) {
							//std::cout << "lm::ba::busy" << std::endl;
							mpMapOptimizer->StopBA(true);
						}
						else {
							mpMapOptimizer->InsertKeyFrame(mpTargetFrame);
						}
					}//if prev keyframe
				}////mapping
				
				//f.get();
				ProcessNewKeyFrame();

				std::chrono::high_resolution_clock::time_point lm_end = std::chrono::high_resolution_clock::now();

				
				//std::cout << "api test = " << mpTargetFrame->mnFrameID << "=" << mpTargetFrame->matDescriptor.rows << std::endl;

				//////기존 MP 연결 과정 & 생성
				//auto vpKFs = mpMap->GetWindowFramesVector(1);
				//int nNew = 0;
				//int nConnect = 0;
				//int nFuse = 0;
				//int nFail = 0;
				////for (int k = vpKFs.size()-1, kend = k-3; k > kend && k >= 0; k--) {
				//int kend = vpKFs.size();
				//int k = kend - 3;;
				//if (k < 0)
				//	k = 0;
				//for (;k < kend; k++) {
				//	auto pKF = vpKFs[k];
				//	std::cout << "LM::" << pKF->mnFrameID << std::endl;
				//	std::vector<int> vMatches;
				//	FeatureMatchingWebAPI::RequestMatch(ip, port, mpTargetFrame->mnFrameID, pKF->mnFrameID, vMatches);

				//	//입력 프레임에서 현재 프레임으로 프로젝션되는 매트릭스
				//	cv::Mat Rrel, Trel;
				//	mpTargetFrame->GetRelativePoseFromTargetFrame(pKF, Rrel, Trel);

				//	////생성 테스트 관련
				//	cv::Mat Rcurr, Tcurr, Pcurr;
				//	mpTargetFrame->GetPose(Rcurr, Tcurr);
				//	cv::hconcat(Rcurr, Tcurr, Pcurr);
				//	cv::Mat Rprev, Tprev, Pprev;
				//	pKF->GetPose(Rprev, Tprev);
				//	cv::hconcat(Rprev, Tprev, Pprev);
				//	Pcurr = mK*Pcurr;
				//	Pprev = mK*Pprev;
				//	cv::Mat Rtcurr = Rcurr.t();
				//	cv::Mat Rtprev = Rprev.t();

				//	////디버깅
				//	cv::Mat aimg = mpTargetFrame->GetOriginalImage().clone();
				//	cv::Mat bimg = pKF->GetOriginalImage().clone();
				//	cv::Point2f ptBottom = cv::Point2f(aimg.cols, 0);
				//	cv::Rect mergeRect1 = cv::Rect(0, 0, aimg.cols, aimg.rows);
				//	cv::Rect mergeRect2 = cv::Rect(aimg.cols, 0, aimg.cols, aimg.rows);
				//	cv::Mat debug_glue = cv::Mat::zeros(aimg.rows, aimg.cols * 2, aimg.type());
				//	aimg.copyTo(debug_glue(mergeRect1));
				//	bimg.copyTo(debug_glue(mergeRect2));
				//	////디버깅

				//	auto vpMPs = pKF->GetMapPoints();

				//	for (size_t i = 0, iend = vMatches.size(); i < iend; i++) {
				//		if (vMatches[i] == -1)
				//			continue;
				//		int idx = vMatches[i];
				//		auto pMPi = vpMPs[idx];
				//		bool bMP1 = pMPi && !pMPi->isDeleted();
				//		auto pMPi2 = mpTargetFrame->GetMapPoint(i);
				//		bool bMP2 = pMPi2 && !pMPi2->isDeleted();

				//		////constraints
				//		auto prevPt = pKF->mvPts[idx];
				//		auto currPt = mpTargetFrame->mvPts[i];
				//		cv::Mat ray = mpSystem->mInvK*(cv::Mat_<float>(3, 1) << prevPt.x, prevPt.y, 1.0);
				//		float z_min, z_max;
				//		z_min = 0.01f;
				//		z_max = 1.0f;
				//		cv::Point2f XimgMin, XimgMax;
				//		mpMatcher->ComputeEpiLinePoint(XimgMin, XimgMax, ray, z_min, z_max, Rrel, Trel, mK); //ray,, Rrel, Trel
				//		cv::Mat lineEqu = mpMatcher->ComputeLineEquation(XimgMin, XimgMax);
				//		bool bEpiConstraints = mpMatcher->CheckLineDistance(lineEqu, currPt, 1.2);
				//		if (!bEpiConstraints) {
				//			cv::circle(debug_glue, currPt, 3, cv::Scalar(0, 0, 255), -1);
				//			cv::circle(debug_glue, prevPt + ptBottom, 3, cv::Scalar(0, 0, 255), -1);
				//			cv::line(debug_glue, currPt, prevPt + ptBottom, cv::Scalar(0, 0, 255), 1);
				//			//continue;
				//		}
				//		////constraints

				//		if (bMP1 && !bMP2 && pMPi->mnLocalMapID != nFrameID) {
				//			pMPi->mnLocalMapID = nFrameID;
				//			mpTargetFrame->AddMapPoint(pMPi, i);
				//			pMPi->AddObservation(mpTargetFrame, i);
				//		}
				//		if (!bMP1 && bMP2) {
				//			nConnect++;
				//			pKF->AddMapPoint(pMPi2, idx);
				//			pMPi2->AddObservation(pKF, idx);
				//		}
				//		if (bMP1 && bMP2 && pMPi != pMPi2) {
				//			nFuse++;
				//			/*if (pMPi->GetObservations().size() > pMPi2->GetObservations().size()) {
				//				pMPi2->Fuse(pMPi);
				//			}
				//			else {
				//				pMPi->Fuse(pMPi2);
				//			}*/
				//			//pMPi->DeleteMapPoint();
				//		}
				//		if (!bMP1 && !bMP2) {
				//			////parallax check
				//			cv::Mat xn1 = (cv::Mat_<float>(3, 1) << prevPt.x, prevPt.y, 1.0);
				//			cv::Mat xn2 = (cv::Mat_<float>(3, 1) << currPt.x, currPt.y, 1.0);
				//			cv::Mat ray1 = Rtprev*mInvK*xn1;
				//			cv::Mat ray2 = Rtcurr*mInvK*xn2;
				//			float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1)*cv::norm(ray2));

				//			////projection
				//			bool bParallax = cosParallaxRays < 0.999;
				//			/*if (!bParallax)
				//			std::cout << "parallax!!!!!!" << std::endl;*/

				//			cv::Mat A(4, 4, CV_32F);
				//			A.row(0) = prevPt.x*Pprev.row(2) - Pprev.row(0);
				//			A.row(1) = prevPt.y*Pprev.row(2) - Pprev.row(1);
				//			A.row(2) = currPt.x*Pcurr.row(2) - Pcurr.row(0);
				//			A.row(3) = currPt.y*Pcurr.row(2) - Pcurr.row(1);

				//			cv::Mat u, w, vt;
				//			cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
				//			cv::Mat x3D = vt.row(3).t();
				//			x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
				//			cv::Point2f pt1, pt2;
				//			bool bDepth1, bDepth2;
				//			bool bDist1, bDist2;
				//			float thresh = 4.0;
				//			{
				//				cv::Mat Xcam = Rprev * x3D + Tprev;
				//				cv::Mat Ximg = mK*Xcam;
				//				float fDepth = Ximg.at < float>(2);
				//				bDepth1 = fDepth > 0.0;
				//				pt1 = cv::Point2f(Ximg.at<float>(0) / fDepth, Ximg.at<float>(1) / fDepth);
				//				auto diffPt = pt1 - prevPt;
				//				float dist = diffPt.dot(diffPt);
				//				bDist1 = dist < thresh;
				//			}
				//			{
				//				cv::Mat Xcam = Rcurr * x3D + Tcurr;
				//				cv::Mat Ximg = mK*Xcam;
				//				float fDepth = Ximg.at < float>(2);
				//				bDepth2 = fDepth > 0.0;
				//				pt2 = cv::Point2f(Ximg.at<float>(0) / fDepth, Ximg.at<float>(1) / fDepth);
				//				auto diffPt = pt2 - currPt;
				//				float dist = diffPt.dot(diffPt);
				//				bDist2 = dist < thresh;
				//			}
				//			if (!bParallax) {
				//				//cv::circle(debug_glue, currPt, 3, cv::Scalar(255, 0, 255), -1);
				//				//cv::circle(debug_glue, prevPt + ptBottom, 3, cv::Scalar(255, 0, 255), -1);
				//			}
				//			else if (bParallax && bDist1 && bDepth1 && bDist2 && bDepth2) {
				//				//cv::circle(debug_glue, currPt, 3, cv::Scalar(0, 255, 255), -1);
				//				//cv::circle(debug_glue, prevPt + ptBottom, 3, cv::Scalar(0, 255, 255), -1);
				//				//mpMap->AddTempMP(x3D);

				//				auto pNewMP = new UVR_SLAM::MapPoint(mpMap, mpTargetFrame, x3D, cv::Mat(), 0);
				//				pNewMP->SetOptimization(true);
				//				mpSystem->mlpNewMPs.push_back(pNewMP);
				//				mpTargetFrame->AddMapPoint(pNewMP, i);
				//				pNewMP->AddObservation(mpTargetFrame, i);

				//				pKF->AddMapPoint(pNewMP, idx);
				//				pNewMP->AddObservation(pKF, idx);

				//				pNewMP->IncreaseFound(2);
				//				pNewMP->IncreaseVisible(2);
				//				nNew++;
				//				mpMap->AddTempMP(x3D);

				//			}
				//			else {
				//				//cv::circle(debug_glue, currPt, 3, cv::Scalar(0, 255, 0), -1);
				//				//cv::circle(debug_glue, prevPt + ptBottom, 3, cv::Scalar(0, 255, 0), -1);
				//				nFail++;
				//			}
				//		}//if !bMP1 && !bMP2
				//		cv::circle(debug_glue, currPt, 3, cv::Scalar(0, 255, 0), -1);
				//		cv::circle(debug_glue, prevPt + ptBottom, 3, cv::Scalar(0, 255, 0), -1);
				//		cv::line(debug_glue, currPt, prevPt + ptBottom, cv::Scalar(255, 255, 0), 1);
				//	}//for match
				//	if(k == vpKFs.size() - 1){
				//		mpMap->AddWindowFrame(mpTargetFrame);
				//		cv::imshow("SuperPoint::SuperGlue", debug_glue);
				//		cv::waitKey(1);
				//	}
				//}//for k
				//////기존 MP 연결 과정 & 생성

				

				auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(lm_end - lm_start).count();
				float t_test1 = du_test1 / 1000.0;

				std::stringstream ssa;
				ssa << "LocalMapping : " << mpTargetFrame->mnKeyFrameID << "::" << t_test1 << std::endl;;// << ", " << nMinKF << ", " << nMaxKF;
				mpSystem->SetLocalMapperString(ssa.str());
			}
			SetDoingProcess(false);
			continue;
































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
					//mpSegmentator->InsertKeyFrame(mpTempFrame);

					///////CP 테스트
					//{
					//	auto pFloorParam = mpPlaneEstimator->GetPlaneParam();
					//	if (pFloorParam->mbInit) {
					//		int nSize = mpSystem->mnRadius * 2;
					//		int nGridSize = mpSystem->mnRadius * 2;
					//		
					//		int maxLvl = 0;
					//		int searchSize = 5;

					//		std::vector<cv::Mat> vpTempPlaneVisPTs;
					//		std::vector<FrameGrid*> vpTempGrids;
					//		Frame* pSegFrame = nullptr;
					//		mpPlaneEstimator->GetTempPTs(pSegFrame, vpTempGrids, vpTempPlaneVisPTs);

					//		if (pSegFrame)
					//		{
					//			cv::Mat R, t;
					//			mpTempFrame->GetPose(R, t);
					//			cv::Scalar tempPlaneColor1(255, 0, 255);
					//			cv::Scalar tempPlaneColor2(255, 255, 0);
					//			cv::Scalar tempPlaneColor3(0, 255, 255);

					//			cv::Mat prevImg = pSegFrame->GetOriginalImage();
					//			cv::Mat currImg = mpTempFrame->GetOriginalImage();

					//			///////debug
					//			cv::Point2f ptBottom = cv::Point2f(0, prevImg.rows);
					//			cv::Rect mergeRect1 = cv::Rect(0, 0, prevImg.cols, prevImg.rows);
					//			cv::Rect mergeRect2 = cv::Rect(0, prevImg.rows, prevImg.cols, prevImg.rows);
					//			cv::Mat debugging = cv::Mat::zeros(prevImg.rows * 2, prevImg.cols, prevImg.type());
					//			prevImg.copyTo(debugging(mergeRect1));
					//			currImg.copyTo(debugging(mergeRect2));
					//			///////debug

					//			for (size_t i = 0, iend = vpTempPlaneVisPTs.size(); i < iend; i++) {
					//				cv::Mat x3D = vpTempPlaneVisPTs[i];
					//				cv::Mat temp = mK*(R*x3D + t);
					//				float depth = temp.at<float>(2);
					//				cv::Point2f pt(temp.at<float>(0) / depth, temp.at<float>(1) / depth);
					//				if (!mpTempFrame->isInImage(pt.x, pt.y, 5.0))
					//					continue;

					//				auto ptLeft = mpTempFrame->GetGridBasePt(pt, nGridSize);
					//				cv::Point2f ptRight(ptLeft.x + nSize, ptLeft.y + nSize);
					//				if (ptRight.x >= mnWidth || ptRight.y >= mnHeight) {
					//					continue;
					//				}
					//				auto prevGrid = vpTempGrids[i];
					//				auto pCPi = prevGrid->mpCP;

					//				if (pCPi->mnTrackingFrameID >= mpTempFrame->mnFrameID)
					//					continue;

					//				cv::Mat img1 = pSegFrame->GetOriginalImage()(prevGrid->rect);
					//				cv::Rect rect1(ptLeft, ptRight);
					//				cv::Mat img2 = mpTempFrame->GetOriginalImage()(rect1);

					//				std::vector<uchar> status;
					//				std::vector<float> err;
					//				std::vector<cv::Point2f> prevPts, currPts;
					//				prevPts.push_back(prevGrid->pt - prevGrid->basePt);

					//				/*if(pMPi->mLastMatchPatch.empty())
					//				std::cout << "??????" << std::endl;*/
					//				/*if (pMPi->mLastMatchPatch.channels() != 3 || pMPi->mLastMatchPatch.rows != nGridSize * 2)
					//				std::cout << "??????" << std::endl;
					//				if (img1.size() != pMPi->mLastMatchPatch.size())
					//				std::cout << "??????" << img1.size << "::" << pMPi->mLastMatchPatch.size() << std::endl;*/

					//				cv::circle(debugging, prevGrid->pt, 2, cv::Scalar(0, 255, 0), -1);

					//				cv::calcOpticalFlowPyrLK(img1, img2, prevPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);
					//				if (!status[0])
					//					continue;
					//				cv::Point2f currPt = currPts[0] + ptLeft;

					//				/*cv::Point right2(gridPt.x + nGridSize, gridPt.y + nGridSize);
					//				if (right2.x >= mnWidth || right2.y >= mnHeight)
					//				continue;
					//				auto rect = cv::Rect(gridPt, right2);*/

					//				if (mpTempFrame->mmpFrameGrids.count(ptLeft))
					//					continue;

					//				mpTempFrame->mmbFrameGrids[ptLeft] = true;
					//				auto currGrid = new FrameGrid(ptLeft, rect1, 0);
					//				mpTempFrame->mmpFrameGrids[ptLeft] = currGrid;
					//				mpTempFrame->mmpFrameGrids[ptLeft]->mpCP = pCPi;
					//				mpTempFrame->mmpFrameGrids[ptLeft]->pt = currPt;
					//				mpTempFrame->mpMatchInfo->AddCP(pCPi, currPt);
					//				cv::line(debugging, prevGrid->pt, currPt + ptBottom, cv::Scalar(255), 2);

					//				/*if (!prevGrid->mpCP) {
					//				circle(testImg, pt, 3, tempPlaneColor3);
					//				}
					//				else if (mpTargetFrame->mmpFrameGrids.count(ptLeft)) {
					//				circle(testImg, pt, 3, tempPlaneColor1);
					//				}
					//				else {
					//				circle(testImg, pt, 3, tempPlaneColor2);
					//				}*/

					//				/*
					//				auto pGrid = mpTargetFrame->mmpFrameGrids[gridPt];
					//				if (!pGrid) {
					//				pGrid = new UVR_SLAM::FrameGrid(gridPt, nGridSize);
					//				mpTargetFrame->mmpFrameGrids[gridPt] = pGrid;
					//				mpTargetFrame->mmbFrameGrids[gridPt] = false;
					//				}
					//				auto pPrevGrid = vpTempGrids[i];
					//				pPrevGrid->mObjCount.at<int>(mnLabel_floor)++;
					//				pGrid->mObjCount = pPrevGrid->mObjCount.clone();                                   
					//				pGrid->mObjArea = pPrevGrid->mObjArea.clone();*/
					//			}
					//			cv::imshow("plane test projection", debugging); cv::waitKey(1);
					//		}	
					//	}
					//}
					///////CP 테스트

					//FuseKeyFrame(mpTempFrame, mpPrevKeyFrame, mpSystem->mnRadius * 2);
					std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateCP);
					std::chrono::high_resolution_clock::time_point lm_start = std::chrono::high_resolution_clock::now();
					/*FeatureMatchingWebAPI::SendImage(ip, port, mpTempFrame->matFrame, mpTempFrame->mnFrameID);
					FeatureMatchingWebAPI::RequestDetect(ip, port, mpTempFrame->mnFrameID, mpTempFrame->mvEdgePts, mpTempFrame->matDescriptor);*/
					//FeatureMatchingWebAPI::RequestDetect(ipaa, 35005, mpTempFrame->matFrame, mpTempFrame->mnFrameID, mpTempFrame->mvEdgePts);
					//mpTempFrame->mvEdgePts = std::vector<cv::Point2f>(aaaa.begin(), aaaa.end());

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

					///////////매칭 테스트
					//cv::Mat desc1, desc2;
					//std::vector<cv::Point2f> pts1, pts2;
					//{
					//	auto matchInfo1 = mpPrevKeyFrame->mpMatchInfo;
					//	auto matchInfo2 = mpTempFrame->mpMatchInfo;
					//	std::cout << mpPrevKeyFrame->matDescriptor.rows << ", " << matchInfo1->mvpMatchingCPs.size() << std::endl;
					//	/*for (size_t i = 0, iend = matchInfo1->mvpMatchingCPs.size(); i < iend; i++) {
					//		auto pCPi = matchInfo1->mvpMatchingCPs[i];
					//		auto pMPi = pCPi->GetMP();
					//		if (pMPi && !pMPi->isDeleted())
					//			continue;
					//		pts1.push_back(matchInfo1->mvMatchingPts[i]);
					//		desc1.push_back(mpPrevKeyFrame->matDescriptor.row(i));
					//	}
					//	for (size_t i = 0, iend = matchInfo2->mvpMatchingCPs.size(); i < iend; i++) {
					//		auto pCPi = matchInfo2->mvpMatchingCPs[i];
					//		auto pMPi = pCPi->GetMP();
					//		if (pMPi && !pMPi->isDeleted())
					//			continue;
					//		pts2.push_back(matchInfo2->mvMatchingPts[i]);
					//		desc2.push_back(mpTempFrame->matDescriptor.row(i));
					//	}*/
					//	//if (desc1.rows != 0 && desc2.rows != 0)
					//	{
					//		std::cout << desc1.size() << " " << desc2.size() << std::endl;
					//		cv::Mat img1 = mpPrevKeyFrame->GetOriginalImage().clone();
					//		cv::Mat img2 = mpTempFrame->GetOriginalImage().clone();
					//		std::vector< std::vector<cv::DMatch> > matches;
					//		std::vector<cv::DMatch> vMatches;
					//		auto matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
					//		matcher->knnMatch(mpPrevKeyFrame->matDescriptor, mpTempFrame->matDescriptor, matches, 2);

					//		for (unsigned long i = 0; i < matches.size(); i++) {
					//			if (matches[i][0].distance < 0.8f * matches[i][1].distance) {
					//				vMatches.push_back(matches[i][0]);
					//				cv::circle(img1, mpPrevKeyFrame->mvKeyPoints[matches[i][0].trainIdx].pt, 3, cv::Scalar(255, 0, 255), -1);
					//				cv::circle(img2, mpTempFrame->mvKeyPoints[matches[i][0].queryIdx].pt, 3, cv::Scalar(255, 0, 255), -1);
					//			}
					//		}
					//		imshow("a", img1); imshow("b", img2);
					//		cv::waitKey(1);
					//	}
					//	
					//}
					///////////매칭 테스트
				}

			}
			//else {
			//	////여기에 락이 왜 필요한거였지?
			//	std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateCP);
			//	mpSystem->mbCreateCP = true;
			//	////tempolar point reprojection error check
			//	////해당 그리드 정보도 필요함.
			//	{
			//		auto pFloorParam = mpPlaneEstimator->GetPlaneParam();
			//		if (pFloorParam->mbInit) {
			//			int nSize = mpSystem->mnRadius * 2;
			//			int nGridSize = mpSystem->mnRadius * 2;
			//			cv::Mat testImg = mpTargetFrame->GetOriginalImage().clone();
			//			std::vector<cv::Mat> vpTempPlaneVisPTs;
			//			std::vector<FrameGrid*> vpTempGrids;
			//			mpPlaneEstimator->GetTempPTs(vpTempGrids, vpTempPlaneVisPTs);
			//			cv::Mat R, t;
			//			mpTargetFrame->GetPose(R, t);
			//			cv::Scalar tempPlaneColor(255, 0, 255);
			//			for (size_t i = 0, iend = vpTempPlaneVisPTs.size(); i < iend; i++) {
			//				cv::Mat x3D = vpTempPlaneVisPTs[i];
			//				cv::Mat temp = mK*(R*x3D + t);
			//				float depth = temp.at<float>(2);
			//				cv::Point2f pt(temp.at<float>(0) / depth, temp.at<float>(1) / depth);
			//				if (!mpTargetFrame->isInImage(pt.x, pt.y, 5.0))
			//					continue;
			//				cv::Point2f ptRight(pt.x + nSize, pt.y + nSize);
			//				if (ptRight.x > mnWidth || ptRight.y > mnHeight) {
			//					continue;
			//				}
			//				auto gridPt = mpTargetFrame->GetGridBasePt(pt, nGridSize);
			//				auto pGrid = mpTargetFrame->mmpFrameGrids[gridPt];
			//				if (!pGrid) {
			//					pGrid = new UVR_SLAM::FrameGrid(gridPt, nGridSize);
			//					mpTargetFrame->mmpFrameGrids[gridPt] = pGrid;
			//					mpTargetFrame->mmbFrameGrids[gridPt] = false;
			//				}
			//				auto pPrevGrid = vpTempGrids[i];
			//				pPrevGrid->mObjCount.at<int>(mnLabel_floor)++;
			//				pGrid->mObjCount = pPrevGrid->mObjCount.clone();
			//				pGrid->mObjArea = pPrevGrid->mObjArea.clone();
			//				circle(testImg, pt, 3, tempPlaneColor);
			//			}
			//			cv::imshow("plane test projection", testImg); cv::waitKey(1);
			//		}
			//	}
			//	////tempolar point reprojection error check
			//	lock.unlock();
			//	mpSystem->cvUseCreateCP.notify_all();
			//}
			
			
			/*
			bool bTempMP = false;
			if (Nkf < 15 && mpPPrevKeyFrame) {
				bTempMP = true;
				std::cout << "연결 프레임 부족!!"<<Nkf<<"= 새 맵포인트 생성" << std::endl;
			}*/
			
			if (bNeedNewKF) {
				ProcessNewKeyFrame();
				mpTargetFrame->mpMatchInfo->UpdateKeyFrame();
				NewMapPointMarginalization();
				ComputeNeighborKFs(mpTempFrame);
				
				//int Nkf = 0;
				//for (auto biter = mpCandiateKFs.begin(), eiter = mpCandiateKFs.end(); biter != eiter; biter++) {
				//	auto count = biter->second;
				//	auto pKFi = biter->first;
				//	//std::cout << pKFi->mnKeyFrameID << std::endl;
				//	if (count > 20) {
				//		Nkf++;
				//	}
				//}
				//std::cout << "LM::NeighborFrame::" << Nkf << "::" << mpCandiateKFs.size() << std::endl;
			}

			if (bNeedMP) {

				////////RefKeyFrame 관련 작업
				auto mpRefKeyFrame = mpPPrevKeyFrame;
				std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateMP);
				//nCreated = MappingProcess(mpMap, mpTargetFrame, mpPPrevKeyFrame, time2, currImg);
				int nCreated = MappingProcess(mpMap, mpTempFrame, mpPrevKeyFrame, mpRefKeyFrame->mfMedianDepth, mpRefKeyFrame->mfMeanDepth, mpRefKeyFrame->mfStdDev, time2);
				mpSystem->mbCreateMP = true;
				lock.unlock();
				mpSystem->cvUseCreateMP.notify_all();

				////Planar Refinement
				{
					////이왕이면 현재 포인트가 속하는 프레임 정보를 미리 알고, 현재 프레임셋에서 3개 이하이면 없애기.
					////각 프레임별로 인버스 매트릭스 미리 계산해놓기

					std::vector<UVR_SLAM::MapPoint*> vpTempFloorMPs, vpTempObjectMPs;
					std::map<CandidatePoint*, cv::Mat> mpKeyPointLabels;
					std::map<CandidatePoint*, int> mpKeyPointCounts;
					std::map<MapPoint*, cv::Mat> mpMapPointLabels;
					std::map<MapPoint*, int> mpMapPointCounts;
					std::map<MapPoint*, std::set<Frame*>> mpMapPointKFs;
					std::map<Frame*, std::tuple<cv::Mat, cv::Mat, cv::Mat>> mInverseInformation;
					auto pFloorParam = mpPlaneEstimator->GetPlaneParam();
					if (pFloorParam->mbInit) {
						auto spGraphKFs = mpMap->GetWindowFramesSet();
						auto vpGraphKFs = mpMap->GetWindowFramesVector();
						cv::Mat invK = mpSystem->mInvK;
						////미리 인버스 플레인값 획득하기

						cv::Mat RcurrInv, TcurrInv, PlaneCurrInv, TempCurr;
						mpTargetFrame->GetInversePose(RcurrInv, TcurrInv);
						auto pCurrPlaneInformation = new UVR_SLAM::PlaneProcessInformation(mpTargetFrame, pFloorParam);
						pCurrPlaneInformation->Calculate(pFloorParam);
						pCurrPlaneInformation->GetInformation(PlaneCurrInv, TempCurr, invK);
						mInverseInformation[mpTargetFrame] = std::make_tuple(PlaneCurrInv, RcurrInv, TcurrInv);

						for (auto iter = spGraphKFs.begin(), iend = spGraphKFs.end(); iter != iend; iter++) {
							auto pKFi = *iter;

							auto vpGrids = pKFi->mmpFrameGrids;

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
								if (pGrid->mvpCPs.size() == 0)
									continue;
								auto pCPi = pGrid->mvpCPs[0];
								if (!pCPi)
									continue;
								auto pMPi = pCPi->GetMP();
								bool bMP = !pMPi || pMPi->isDeleted() || !pMPi->GetQuality();
								if (bMP) {
									if (!mpKeyPointCounts.count(pCPi)) {
										cv::Mat temp = cv::Mat::zeros(1, ObjectColors::mvObjectLabelColors.size(), CV_32SC1);
										mpKeyPointLabels.insert(std::make_pair(pCPi, temp));
									}
									mpKeyPointCounts[pCPi]++;
									mpKeyPointLabels[pCPi] += pGrid->mObjCount;
								}
								else {
									if (!mpMapPointCounts.count(pMPi)) {
										cv::Mat temp = cv::Mat::zeros(1, ObjectColors::mvObjectLabelColors.size(), CV_32SC1);
										mpMapPointLabels.insert(std::make_pair(pMPi, temp));
									}
									mpMapPointCounts[pMPi]++;
									mpMapPointKFs[pMPi].insert(pKFi);
									mpMapPointLabels[pMPi] += pGrid->mObjCount;
								}

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
						int nCP = 0;
						int nSucceessCP = 0;
						int nFailCP = 0;
						for (auto iter = mpKeyPointCounts.begin(), iend = mpKeyPointCounts.end(); iter != iend; iter++) {
							auto pCPi = iter->first;
							int count = iter->second;
							if (count < mpSystem->mnThreshMinKF)
								continue;
							nCP++;
							int idx = pCPi->GetPointIndexInFrame(mpTargetFrame->mpMatchInfo);

							cv::Mat label = mpKeyPointLabels[pCPi];
							////wall, ceil, obj 분류 추가 예정
							int nCountFloor = label.at<int>(mnLabel_floor);

							cv::Point maxIdx;
							double maxVal;
							cv::minMaxLoc(label, NULL, &maxVal, NULL, &maxIdx);
							//std::cout << maxIdx <<"::"<<maxVal<<"::"<< nCountFloor << std::endl;
							if (maxIdx.x != mnLabel_floor) {
								continue;
							}

							auto data = mInverseInformation[mpTargetFrame];
							cv::Mat Pinv = std::get<0>(data);
							cv::Mat Rinv = std::get<1>(data);
							cv::Mat Tinv = std::get<2>(data);
							float fMaxDepth = mpTargetFrame->mfMedianDepth + mpTargetFrame->mfRange;

							if (idx >= 0) {
								cv::Mat Xw;
								auto pt = mpTargetFrame->mpMatchInfo->mvMatchingPts[idx];
								bool b = PlaneInformation::CreatePlanarMapPoint(Xw, pt, Pinv, invK, Rinv, Tinv, fMaxDepth);
								if (b) {
									auto observations = pCPi->GetFrames();
									//for(auto citer = observations.b)
									//pMPi->SetWorldPos(Xw);
									//vpTempFloorMPs.push_back(pMPi);
									//std::cout << "???????????????????22222222222222222222222" << std::endl;
									//cv::circle(testImg, pt, 3, cv::Scalar(255));
									nSucceessCP++;
								}
								else
									nFailCP++;
							}
						}
						mpMap->ClearReinit();
						for (auto iter = mpMapPointCounts.begin(), iend = mpMapPointCounts.end(); iter != iend; iter++) {
							auto pMPi = iter->first;
							int count = iter->second;
							
							cv::Mat label = mpMapPointLabels[pMPi];
							////wall, ceil, obj 분류 추가 예정
							int nCountFloor = label.at<int>(mnLabel_floor);

							cv::Point maxIdx;
							double maxVal;
							cv::minMaxLoc(label, NULL, &maxVal, NULL, &maxIdx);
							
							if (maxIdx.x != mnLabel_floor) {
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
							if (idx >= 0) {
								cv::Mat Xw;
								auto pt = pKF->mpMatchInfo->mvMatchingPts[idx];
								bool b = PlaneInformation::CreatePlanarMapPoint(Xw, pt, Pinv, invK, Rinv, Tinv, fMaxDepth);
								if (b) {
									/*auto observations = pMPi->GetConnedtedFrames();
									for (auto miter = observations.begin(), miend = observations.end(); miter != miend; miter++) {
										auto tempKF = miter->first;
										int tempIdx = miter->second;
										auto tempPt = tempKF->mvMatchingPts[tempIdx];
										cv::Mat Rtemp, Ttemp;
										tempKF->mpRefFrame->GetPose(Rtemp, Ttemp);
										cv::Mat proj = mK*(Rtemp*Xw + Ttemp);
										float tempDepth = proj.at<float>(2);
										cv::Point2f projPt(proj.at<float>(0) / tempDepth, proj.at<float>(1) / tempDepth);
										auto diffPt = projPt - tempPt;
										float tempDist = sqrt(diffPt.dot(diffPt));
										std::cout << "dist::" << tempDist << std::endl;
									}*/
									mpMap->AddReinit(Xw);
									pMPi->SetWorldPos(Xw);
									vpTempFloorMPs.push_back(pMPi);
									//std::cout << "???????????????????22222222222222222222222" << std::endl;
									//cv::circle(testImg, pt, 3, cv::Scalar(255));
								}else {
									nFail++;
								}
							}
							
						}//for mp

						////포즈 보정
						//pNormal, pDist 삭제
						int nObjFail = 0;
						Optimization::PlanarPoseRefinement(mpMap, vpTempFloorMPs, vpGraphKFs);
						float thHuber = sqrt(5.991);
						for (size_t i = 0, iend = vpTempObjectMPs.size(); i < iend; i++) {
							if (!Optimization::ObjectPointRefinement(mpMap, vpTempObjectMPs[i], vpGraphKFs, spGraphKFs, mpSystem->mnThreshMinKF, thHuber)) {
								nObjFail++;
							}
						}
						std::cout << "Refinement::" << nSucceessCP << ", " << nFailCP << ". " << nCP << "=" << vpTempFloorMPs.size() <<", "<<nFail<< "::" << vpTempObjectMPs.size() << ", " << nObjFail << "::" << mpMapPointCounts.size() << std::endl;
						//Optimization::ObjectPointRefinement(mpMap, vpTempObjectMPs, vpGraphKFs);
						////오브젝트 포인트 보정(포즈 고정)

					}//if plane
				}
				////Planar Refinement


				//FuseKeyFrame(mpTempFrame, mpPPrevKeyFrame, mpSystem->mnRadius*4);
			
				
			}

			if (bNeedNewKF) {
			
				//키프레임 컬링
				//KeyFrameMarginalization(mpTargetFrame, 0.92);

				//키프레임 연결
				ConnectNeighborKFs(mpTargetFrame, mpTargetFrame->mmKeyFrameCount, 20);
				
				//{
				//	auto vpNeighKFs = mpTargetFrame->GetConnectedKFs(10);
				//	if (vpNeighKFs.size() > 4) {
				//		std::vector<cv::Mat> currPyr, prevPyr;
				//		std::vector<uchar> status;
				//		std::vector<float> err;
				//		int nSize = vpNeighKFs.size() - 1;
				//		auto pKF = vpNeighKFs[nSize];
				//		cv::Mat prevImg = pKF->mvPyramidImages[2].clone();
				//		cv::Mat currImg = mpTargetFrame->mvPyramidImages[2].clone();

				//		int maxLvl = 0;
				//		int searchSize = 5;
				//		std::vector<cv::Point2f> prevPts, currPts;
				//		prevPts = pKF->mvPyramidPts;
				//		if (prevPts.size() > 10) {
				//			cv::calcOpticalFlowPyrLK(prevImg, currImg, prevPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);
				//			for (size_t i = 0, iend = prevPts.size(); i < iend; i++) {
				//				if (status[i] == 0) {
				//					continue;
				//				}
				//				if (currPts[i] == prevPts[i])
				//				{
				//					std::cout << "??????????????aaa" << std::endl;
				//					continue;
				//				}
				//				if (!pKF->isInImage(currPts[i].x, currPts[i].y, 20))
				//					continue;
				//				cv::circle(prevImg, prevPts[i], 2, cv::Scalar(255), -1);
				//				cv::circle(currImg, currPts[i], 2, cv::Scalar(255), -1);
				//			}
				//			imshow("level::curr", currImg);
				//			imshow("level::prev", prevImg); cv::waitKey(1);
				//		}
				//	}//if neigh
				//}

				//////평면 관련 프로세스
				////평면 레이블링
				{
					int mnLabel_floor = 4;
					int mnLabel_ceil = 6;
					int mnLabel_wall = 1;
					auto pFloorParam = mpPlaneEstimator->GetPlaneParam();
				}

				auto pTarget = mpMap->AddWindowFrame(mpTargetFrame);
			
				if (mpMapOptimizer->isDoingProcess()) {
					//std::cout << "lm::ba::busy" << std::endl;
					mpMapOptimizer->StopBA(true);
				}
				else {
					mpMapOptimizer->InsertKeyFrame(mpTargetFrame);

					std::chrono::high_resolution_clock::time_point feature_start = std::chrono::high_resolution_clock::now();
					std::vector<int> vMatches, vMatches2;
					
					//FeatureMatchingWebAPI::RequestDetect(ipaa, 35005, mpTargetFrame->matFrame, mpTargetFrame->mnFrameID, 0, vSuperPoitns);

					auto vNeighKFs = mpTargetFrame->GetConnectedKFs(15);
					int nLast = vNeighKFs.size() - 1;
					int nID = 2;
					int nMinKF = 3;
					if(nLast > nID){
						//FeatureMatchingWebAPI::RequestDetect(ipaa, 35005, vNeighKFs[4]->matFrame, vNeighKFs[4]->mnFrameID, 1, vSuperPoitns2);
						//FeatureMatchingWebAPI::RequestMatch(ip, port, mpTargetFrame->mnFrameID, vNeighKFs[nID]->mnFrameID, vMatches);
						//FeatureMatchingWebAPI::RequestMatch(ip, port, mpTargetFrame->mnFrameID, vNeighKFs[8]->mnFrameID, vMatches2);
					}
					
					std::chrono::high_resolution_clock::time_point feature_end = std::chrono::high_resolution_clock::now();
					auto du_feature = std::chrono::duration_cast<std::chrono::milliseconds>(feature_end - feature_start).count();
					float t_feature = du_feature / 1000.0;
					std::cout << "as;dlfj;asdlkfjaasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfsdf::" << t_feature << std::endl;

					if (vMatches.size() > 0) {
						auto pKF = vNeighKFs[nID];
						auto pMatch = vMatches;

						//입력 프레임에서 현재 프레임으로 프로젝션되는 매트릭스
						cv::Mat Rrel, Trel;
						mpTargetFrame->GetRelativePoseFromTargetFrame(pKF, Rrel, Trel);

						////생성 테스트 관련
						cv::Mat Rcurr, Tcurr, Pcurr;
						mpTargetFrame->GetPose(Rcurr, Tcurr);
						cv::hconcat(Rcurr, Tcurr, Pcurr);
						cv::Mat Rprev, Tprev, Pprev;
						pKF->GetPose(Rprev, Tprev);
						cv::hconcat(Rprev, Tprev, Pprev);
						Pcurr = mK*Pcurr;
						Pprev = mK*Pprev;
						cv::Mat Rtcurr = Rcurr.t();
						cv::Mat Rtprev = Rprev.t();

						cv::Mat aimg = mpTargetFrame->GetOriginalImage().clone();
						cv::Mat bimg = pKF->GetOriginalImage().clone();
						cv::Rect mergeRect1 = cv::Rect(0, 0, aimg.cols, aimg.rows);
						cv::Rect mergeRect2 = cv::Rect(aimg.cols, 0, aimg.cols, aimg.rows);
						cv::Mat debug_glue = cv::Mat::zeros(aimg.rows, aimg.cols * 2, aimg.type());
						cv::Point2f ptBottom = cv::Point2f(aimg.cols, 0);
						aimg.copyTo(debug_glue(mergeRect1));
						bimg.copyTo(debug_glue(mergeRect2));
						//mpMap->ClearTempMPs();
						for (size_t i = 0, iend = pMatch.size(); i < iend; i++) {
							if (pMatch[i] == -1)
								continue;
							int idx = pMatch[i];

							auto prevPt = pKF->mvEdgePts[idx];
							auto currPt = mpTargetFrame->mvEdgePts[i];

							cv::Mat ray = mpSystem->mInvK*(cv::Mat_<float>(3, 1) << prevPt.x, prevPt.y, 1.0);
							float z_min, z_max;
							z_min = 0.01f;
							z_max = 1.0f;
							cv::Point2f XimgMin, XimgMax;
							mpMatcher->ComputeEpiLinePoint(XimgMin, XimgMax, ray, z_min, z_max, Rrel, Trel, mK); //ray,, Rrel, Trel
							cv::Mat lineEqu = mpMatcher->ComputeLineEquation(XimgMin, XimgMax);
							bool bEpiConstraints = mpMatcher->CheckLineDistance(lineEqu, currPt, 1.0);
							if (!bEpiConstraints) {
								continue;
							}

							////parallax check
							cv::Mat xn1 = (cv::Mat_<float>(3, 1) << prevPt.x, prevPt.y, 1.0);
							cv::Mat xn2 = (cv::Mat_<float>(3, 1) << currPt.x, currPt.y, 1.0);
							cv::Mat ray1 = Rtprev*mInvK*xn1;
							cv::Mat ray2 = Rtcurr*mInvK*xn2;
							float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1)*cv::norm(ray2));

							////projection
							bool bParallax = cosParallaxRays < 0.999;
							/*if (!bParallax)
								std::cout << "parallax!!!!!!" << std::endl;*/

							cv::Mat A(4, 4, CV_32F);
							A.row(0) = prevPt.x*Pprev.row(2) - Pprev.row(0);
							A.row(1) = prevPt.y*Pprev.row(2) - Pprev.row(1);
							A.row(2) = currPt.x*Pcurr.row(2) - Pcurr.row(0);
							A.row(3) = currPt.y*Pcurr.row(2) - Pcurr.row(1);

							cv::Mat u, w, vt;
							cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
							cv::Mat x3D = vt.row(3).t();
							x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
							cv::Point2f pt1, pt2;
							bool bDepth1, bDepth2;
							bool bDist1, bDist2;
							float thresh = 4.0;
							{
								cv::Mat Xcam = Rprev * x3D + Tprev;
								cv::Mat Ximg = mK*Xcam;
								float fDepth = Ximg.at < float>(2);
								bDepth1 = fDepth > 0.0;
								pt1 = cv::Point2f(Ximg.at<float>(0) / fDepth, Ximg.at<float>(1) / fDepth);
								auto diffPt = pt1 - prevPt;
								float dist = diffPt.dot(diffPt);
								bDist1 = dist < thresh;
							}
							{
								cv::Mat Xcam = Rcurr * x3D + Tcurr;
								cv::Mat Ximg = mK*Xcam;
								float fDepth = Ximg.at < float>(2);
								bDepth2 = fDepth > 0.0;
								pt2 = cv::Point2f(Ximg.at<float>(0) / fDepth, Ximg.at<float>(1) / fDepth);
								auto diffPt = pt2 - currPt;
								float dist = diffPt.dot(diffPt);
								bDist2 = dist < thresh;
							}
							if (!bParallax) {
								cv::circle(debug_glue, currPt, 3, cv::Scalar(255, 0, 255), -1);
								cv::circle(debug_glue, prevPt + ptBottom, 3, cv::Scalar(255, 0, 255), -1);
							}else if (bParallax && bDist1 && bDepth1 && bDist2 && bDepth2) {
								cv::circle(debug_glue, currPt, 3, cv::Scalar(0, 255, 255), -1);
								cv::circle(debug_glue, prevPt + ptBottom, 3, cv::Scalar(0, 255, 255), -1);
								//mpMap->AddTempMP(x3D);
							}
							else {
								cv::circle(debug_glue, currPt, 3, cv::Scalar(0, 255, 0), -1);
								cv::circle(debug_glue, prevPt + ptBottom, 3, cv::Scalar(0, 255, 0), -1);
							}
							cv::line(debug_glue, mpTargetFrame->mvEdgePts[i], pKF->mvEdgePts[idx] + ptBottom, cv::Scalar(255, 255, 0), 1);
						}
						imshow("SuperPoint::SuperGlue", debug_glue);
						cv::waitKey(1);
					}
					
					
				}
			}
			
			std::chrono::high_resolution_clock::time_point lm_end = std::chrono::high_resolution_clock::now();
			auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(lm_end - lm_start).count();
			float t_test1 = du_test1 / 1000.0;
			numLM++;
			totalLM += t_test1;
			
			std::stringstream ssa;
			ssa << "LocalMapping : " << mpTargetFrame->mnKeyFrameID << "::" << (totalLM / numLM) <<"::"<< "::" << 0 <<", "<< time1 << ", " << time2 << std::endl;;// << ", " << nMinKF << ", " << nMaxKF;
			mpSystem->SetLocalMapperString(ssa.str());
			
			SetDoingProcess(false);
			continue;
			//////200412
		}
	}//while
}

void UVR_SLAM::LocalMapper::ComputeNeighborKFs(Frame* pKF) {
	
	int nTargetID = pKF->mnFrameID;
	auto vpMPs = pKF->GetMapPoints();
	for (size_t i = 0, iend = pKF->mvPts.size(); i < iend; i++) {
		auto pMPi = vpMPs[i];
		if (!pMPi || pMPi->isDeleted()) {
			continue;
		}
		auto mmpMP = pMPi->GetObservations();//GetConnedtedFrames
		for (auto biter = mmpMP.begin(), eiter = mmpMP.end(); biter != eiter; biter++) {
			UVR_SLAM::Frame* pCandidateKF = biter->first;
			if (nTargetID == pCandidateKF->mnFrameID)
				continue;
			pKF->mmKeyFrameCount[pCandidateKF]++;
		}
	}
}
void UVR_SLAM::LocalMapper::ConnectNeighborKFs(Frame* pKF, std::map<UVR_SLAM::Frame*, int> mpCandiateKFs, int thresh) {
	std::vector<std::pair<int, UVR_SLAM::Frame*>> vPairs;
	for (auto biter = mpCandiateKFs.begin(), eiter = mpCandiateKFs.end(); biter != eiter; biter++) {
		UVR_SLAM::Frame* pTempKF = biter->first;
		
		int nCount = biter->second;

		if (nCount > thresh) {
			pKF->AddKF(pTempKF, nCount);
			pTempKF->AddKF(pKF, nCount);
		}
	}
}

//MP가 아닌 CP로 할 경우에는?
void UVR_SLAM::LocalMapper::KeyFrameMarginalization(Frame* pKF, float thresh, int thresh2){
	for (auto iter = mpTargetFrame->mmKeyFrameCount.begin(), iend = mpTargetFrame->mmKeyFrameCount.end(); iter != iend; iter++) {
		auto pKFi = iter->first;

		//std::vector<MapPoint*> vpNeighMPs;
		//auto matchInfo = pKFi->mpMatchInfo;
		//auto vpCPs = matchInfo->mvpMatchingCPs;
		//auto vPTs = matchInfo->mvMatchingPts;
		//int N1 = vpCPs.size();
		//int N2 = 0;
		//for (size_t i = 0; i < N1; i++) {
		//	auto pCPi = vpCPs[i];
		//	if (pCPi->GetNumSize() > mnThreshMinKF) {
		//		N2++;
		//	}
		//}

		std::vector<MapPoint*> vpNeighMPs;
		auto matchInfo = pKFi->mpMatchInfo;
		auto vpCPs = matchInfo->mvpMatchingCPs;
		auto vPTs = matchInfo->mvMatchingPts;
		int N1 = 0;
		int N2 = 0;
		for (size_t i = 0, iend = vpCPs.size(); i < iend; i++) {
			auto pCPi = vpCPs[i];
			auto pMPi = pCPi->GetMP();
			if (!pMPi || pMPi->isDeleted() || pMPi->GetNumConnectedFrames() < mnThreshMinKF || !pMPi->GetQuality())
				continue;
			N1++;
			if (pMPi->GetNumConnectedFrames() > mnThreshMinKF)
				N2++;
		}

		if (N2 > N1*thresh && N1 > thresh2) {
			std::cout << "Delete::KF::" <<pKFi->mnKeyFrameID<<"::"<< N1*thresh << ", " << N2 << std::endl;
			pKFi->Delete();
			std::cout << "Delete::KF::End" << std::endl;
			iter->second = 0;
		}
		/*else {
			std::cout << "Delete::KF::" << pKFi->mnKeyFrameID << "::" << N1*thresh << ", " << N2 << std::endl;
		}*/
	}//for kf iter
}

////pKF1이 현재 = mpTempFrame
////pKF2가 이전 = mppprev 등
void UVR_SLAM::LocalMapper::FuseKeyFrame(Frame* pKF1, Frame* pKF2, int nGridSize) {
	//////Ref-Curr 매칭
	{
		//int nGridSize = mpSystem->mnRadius * 2 * 2;
		int nOriGridSize = mpSystem->mnRadius * 2;
		std::map < cv::Point2f, FrameGrid*, Point2fLess> mapRefGrids, mapCurrGrids;
		std::map<FrameGridKey, std::vector<cv::Point2f>> mapFrameGridAndKeyPoints;
		std::map<FrameGridKey, std::vector<CandidatePoint*>>   mapFrameGridAndMapPoints;
		cv::Mat R, t;
		pKF1->GetPose(R, t);
		for (int x = 0; x < mnWidth; x += nGridSize) {
			for (int y = 0; y < mnHeight; y += nGridSize) {
				cv::Point2f ptLeft(x, y);
				cv::Point2f ptRight(x + nGridSize, y + nGridSize);

				if (ptRight.x > mnWidth || ptRight.y > mnHeight) {
					//cv::circle(testImg, ptLeft, 3, cv::Scalar(255, 255, 0), -1);
					continue;
				}
				cv::Rect rect(ptLeft, ptRight);
				auto pGrid1 = new FrameGrid((ptLeft), (rect), 0);
				auto pGrid2 = new FrameGrid((ptLeft), (rect), 0);
				mapRefGrids.insert(std::make_pair(ptLeft, pGrid1));
				mapCurrGrids.insert(std::make_pair(ptLeft, pGrid2));
			}
		}
		for (size_t i = 0, iend = pKF2->mpMatchInfo->mvpMatchingCPs.size(); i < iend; i++) {
			auto pCP = pKF2->mpMatchInfo->mvpMatchingCPs[i];
			if (pCP->mnTrackingFrameID == pKF1->mnFrameID)
				continue;
			auto pMP = pCP->GetMP();
			if (!pMP || pMP->isDeleted())
				continue;
			if (pMP->isInFrame(pKF1->mpMatchInfo))
				continue;
			auto pt = pKF2->mpMatchInfo->mvMatchingPts[i];
			auto refBasePt = pKF2->GetGridBasePt(pt, nGridSize);
			if (refBasePt.x > mnWidth || refBasePt.y > mnHeight) {
				//cv::circle(testImg, ptLeft, 3, cv::Scalar(255, 255, 0), -1);
				continue;
			}

			//뎁스, 프로젝션 포인트 위치, 그리드 위치 확인하기
			cv::Mat Xw = pMP->GetWorldPos();
			cv::Mat temp = mK*(R*Xw + t);
			float depth = temp.at<float>(2);
			cv::Point2f projPt(temp.at<float>(0) / depth, temp.at<float>(1) / depth);
			auto currBasePt = pKF1->GetGridBasePt(projPt, nGridSize);

			auto refGrid = mapRefGrids[refBasePt];
			if (!refGrid)
				continue;
			auto currGrid = mapCurrGrids[currBasePt];
			if (!currGrid)
				continue;

			FrameGridKey key(refGrid, currGrid);
			mapFrameGridAndKeyPoints[key].push_back(pt - refBasePt);
			mapFrameGridAndMapPoints[key].push_back(pCP);
		}
		//매칭
		cv::Mat refImg = pKF2->GetOriginalImage().clone();
		cv::Mat currImg = pKF1->GetOriginalImage().clone();

		cv::Mat Rrel, Trel;
		pKF1->GetRelativePoseFromTargetFrame(pKF2, Rrel, Trel);

		int nCurrFrameID = pKF1->mnFrameID;
		int maxLvl = 3;
		int searchSize = 10;
		std::vector<cv::Point2f> matchPrevPTs, matchCurrPTs;
		for (auto iter = mapFrameGridAndKeyPoints.begin(), iend = mapFrameGridAndKeyPoints.end(); iter != iend; iter++) {
			auto key = iter->first;
			auto refGrid = key.mpKey1;
			auto currGrid = key.mpKey2;
			auto refRect = refGrid->rect;
			auto currRect = currGrid->rect;
			auto refBasePt = refGrid->basePt;
			auto currBasePt = currGrid->basePt;
			auto refPts = iter->second;
			auto refMPs = mapFrameGridAndMapPoints[key];

			if (refPts.size() == 0)
				continue;

			cv::Mat refRectImg = pKF2->GetOriginalImage()(refRect);
			cv::Mat currRectImg = pKF1->GetOriginalImage()(currRect);

			std::vector<uchar> status;
			std::vector<float> err;
			std::vector<cv::Point2f> currPts;
			cv::calcOpticalFlowPyrLK(refRectImg, currRectImg, refPts, currPts, status, err, cv::Size(searchSize, searchSize), maxLvl);

			for (size_t i = 0, iend2 = refPts.size(); i < iend2; i++) {
				if (status[i] == 0) {
					continue;
				}
				cv::Point2f refPt = refPts[i] + refBasePt;
				cv::Point2f currPt = currPts[i] + currBasePt;
				if (!pKF2->isInImage(currPt.x, currPt.y, 20))
					continue;
				auto pCP = refMPs[i];
				auto pMP = pCP->GetMP();

				cv::Mat ray = mpSystem->mInvK*(cv::Mat_<float>(3, 1) << refPt.x, refPt.y, 1.0);
				float z_min, z_max;
				z_min = 0.01f;
				z_max = 1.0f;
				cv::Point2f XimgMin, XimgMax;
				cv::Mat Rcr, Tcr;
				mpMatcher->ComputeEpiLinePoint(XimgMin, XimgMax, ray, z_min, z_max, Rrel, Trel, mK); //ray,, Rrel, Trel
				cv::Mat lineEqu = mpMatcher->ComputeLineEquation(XimgMin, XimgMax);
				bool bEpiConstraints = mpMatcher->CheckLineDistance(lineEqu, currPt, 1.0);
				if (!bEpiConstraints) {
					//cv::circle(refImg, refPt, 3, cv::Scalar(255, 0, 255), -1);
					//cv::circle(currImg, currPt, 3, cv::Scalar(255, 0, 255), -1);
				}
				else {
					auto currGridBasePt = pKF1->GetGridBasePt(currPt, nOriGridSize);
					if (!pKF1->mmpFrameGrids.count(currGridBasePt)) {

						/*cv::Mat Xw = pMP->GetWorldPos();
						cv::Mat temp = mK*(R*Xw + t);
						float depth = temp.at<float>(2);
						cv::Point2f projPt(temp.at<float>(0) / depth, temp.at<float>(1) / depth);

						auto diffPt = projPt - currPt;
						float dist = sqrt(diffPt.dot(diffPt));
						std::cout << "dist::" << dist << std::endl;*/

						//중복문제 해결하기
						int idxi = pKF1->mpMatchInfo->AddCP(pCP, currPt);
						//pCP->ConnectFrame(pKF1->mpMatchInfo, idxi);

						//////grid 추가
						////추후 이 함수 다시 이용할 때 수정이 필요함
						/*auto rect = cv::Rect(currGridBasePt, std::move(cv::Point2f(currGridBasePt.x + nOriGridSize, currGridBasePt.y + nOriGridSize)));
						pKF1->mmbFrameGrids[currGridBasePt] = true;
						auto currGrid = new FrameGrid(currGridBasePt, rect, 0);
						pKF1->mmpFrameGrids[currGridBasePt] = currGrid;
						pKF1->mmpFrameGrids[currGridBasePt]->mpCP = pCP;
						pKF1->mmpFrameGrids[currGridBasePt]->pt = currPt;*/
						//////grid 추가

						cv::circle(refImg, refPt, 3, cv::Scalar(255, 255, 0), -1);
						cv::circle(currImg, currPt, 3, cv::Scalar(255, 255, 0), -1);
					}
					//auto currGrid2 = pKF1->mmpFrameGrids[currGridBasePt];
					
					/*if (currGrid2) {
						cv::circle(currImg, currGrid2->pt, 3, cv::Scalar(255, 0, 0), -1);
						cv::line(currImg, currGrid2->pt, currPt, cv::Scalar(255), 2);
					}
					*/
				}

				/*auto pCP = pPrev->mpMatchInfo->mvpMatchingCPs[idx];
				if (pCP->mnTrackingFrameID == nCurrFrameID)
				continue;
				pCP->mnTrackingFrameID = nCurrFrameID;
				vpTempCPs.push_back(pCP);
				vTempPrevPts.push_back(prevPt);
				vTempCurrPts.push_back(currPt);
				vbTempInliers.push_back(true);*/
			}
		}
		cv::imshow("KF-KF::ref", refImg);
		cv::imshow("KF-KF::curr", currImg);
		cv::waitKey(1);
	}
	//////Ref-Curr 매칭
}

//맵포인트가 삭제 되면 현재 프레임에서도 해당 맵포인트를 삭제 해야 하며, 
//이게 수행되기 전에는 트래킹이 동작하지 않도록 막아야 함.
//
void UVR_SLAM::LocalMapper::NewMapPointMarginalization() {
	//std::cout << "Maginalization::Start" << std::endl;
	//mvpDeletedMPs.clear();
	int nMarginalized = 0;
	int nNumRequireKF = 3;
	float mfRatio = 0.25f;

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
		else if (nDiffKF >= 2 && pMP->GetObservations().size()<=2) {
			bBad = true;
			lit = mpSystem->mlpNewMPs.erase(lit);
		}
		else if (nDiffKF >= nNumRequireKF) {
			lit = mpSystem->mlpNewMPs.erase(lit);
			pMP->SetNewMP(false);

			////맵그리드에 추가
			/*auto key = MapGrid::ComputeKey(pMP->GetWorldPos());
			auto pMapGrid = mpMap->GetMapGrid(key);
			if(!pMapGrid){
				pMapGrid = mpMap->AddMapGrid(key);
			}
			pMapGrid->AddMapPoint(pMP);*/
		}
		else
			lit++;
		if (bBad) {
			pMP->DeleteMapPoint();
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
	//mpMap->ClearReinit();
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
	//mpMap->ClearReinit();
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
int UVR_SLAM::LocalMapper::MappingProcess(Map* pMap, Frame* pCurrKF, Frame* pPrevKF, float fMedianDepth, float fMeanDepth, float fStdDev, double& dtime) {
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
	int NNN = 0;
	int nNumMatchingSize = pCurrKF->mpMatchInfo->mvPrevMatchingIdxs.size();
	for (size_t i = 0, iend = vpTempPTsCurr.size(); i < iend; i++) {
		auto pCPi = vpTempCPs[i];

		int prevIDX = pCurrKF->mpMatchInfo->mvPrevMatchingIdxs[i];//pCPi->GetPointIndexInFrame(pPrevMatch);
		if (prevIDX == -1)
			continue;
		if (pCPi->GetNumSize() < 2)//mnThreshMinKF
			continue;
		auto pMPi = pCPi->GetMP();
		if (pMPi && !pMPi->isDeleted())
			continue;
		auto currPt = vpTempPTsCurr[i];
		auto prevPt = vpTempPTsPrev[prevIDX];
		vMatchPrevPTs.push_back(prevPt);
		vMatchCurrPTs.push_back(currPt);
		vMatchCPs.push_back(pCPi);
	}
	
	if (vMatchCPs.size() < 3) {
		std::cout << "LM::NewMP::포인트부족="<< vMatchCPs.size()<< std::endl;
		return -1;
	}
	cv::Mat TempMap;
	cv::triangulatePoints(mK*Pprev, mK*Pcurr, vMatchPrevPTs, vMatchCurrPTs, TempMap);
	cv::Scalar color1(255, 0, 0);
	cv::Scalar color2(255, 255, 0);
	cv::Scalar color3(0, 0, 255);
	cv::Scalar color4(0, 255, 255);
	cv::Scalar color5(0, 255, 0);

	float fMaxTH = pCurrKF->mfMedianDepth + pCurrKF->mfRange;// fMeanDepth + fRange * fStdDev;

	std::vector<CandidatePoint*> vMappingCPs;
	std::vector<cv::Point2f> vPTs;
	std::vector<cv::Mat> vX3Ds;

	std::vector<float> vfScales;
	float thresh = 16.0;
	float thHuber = sqrt(5.991);
	auto spKFs = mpMap->GetWindowFramesSet();
	mpMap->ClearReinit();
	int N2 = 0;
	int N1 = 0;
	int N3 = 0;
	int N4 = 0;
	int N5 = 0;
	for (size_t i = 0, iend = TempMap.cols; i < iend; i++) {

		////Seed 확인이 필요. 초기화 & 업데이트를 여기서 수행하고 확인도 여기서 하기.

		auto currPt = std::move(vMatchCurrPTs[i]);
		auto prevPt = std::move(vMatchPrevPTs[i]);
		auto pCPi = std::move(vMatchCPs[i]);
		auto pMPi = pCPi->GetMP();
		
		cv::Mat X3D;
		float depth;

		bool bNewMP = true;

		bool bOldMP = pMPi && pMPi->GetQuality() && !pMPi->isDeleted();
		if (bOldMP) {
			X3D = std::move(pMPi->GetWorldPos());
			bNewMP = false;
			mpMap->AddReinit(X3D);
		}
		float depth1, depth2;
		if(bNewMP)
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

			if (depth1  <= 0.0 || depth2 <= 0.0) {
				N4++;
				bNewMP = false;
			}
			if (depth1 > fMaxTH) {
				N5++;
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
					N3++;
					bNewMP = false;
				}

				////CReateMP test
				/*cv::Mat X3D2;
				float depthRef;
				auto bNewMP2 = pCPi->CreateMapPoint(X3D2, depthRef, mK, mInvK, Pcurr, Rcurr, Tcurr, currPt);*/

			}//reproj
		}
		
		////수정 후 이것만 쓸 예정
		if (bNewMP) {
			N1++;
			bool bSuccess = Optimization::PointRefinement(mpMap, pCurrKF, pCPi, X3D, pCPi->GetFrames(), spKFs, mnThreshMinKF, thHuber);
			if(bSuccess){
				N2++;
				mpSystem->mlpNewMPs.push_back(pCPi->GetMP());
				//mpMap->AddReinit(pCPi->GetMP()->GetWorldPos());
			}
		}
		////수정 후 이것만 쓸 예정

		/////여기는 이제 버려질 것
		/*if (!bNewMP && !bOldMP) {
			continue;
		}
		vMappingCPs.push_back(pCPi);
		vX3Ds.push_back(X3D);
		vPTs.push_back(currPt);*/
		/////여기는 이제 버려질 것
	}
	std::cout << "LM::New::" << N2 <<"::"<<N1<<"="<< N3<<", "<<N4<<", "<<N5 <<"::"<<NNN <<", "<<TempMap.cols<<std::endl;
	return 0;
	////이 아래는 버려질 예정
	////////////////////////////////////////최적화 진행
	if (vX3Ds.size() < 10) {
		std::cout << "포인트 부족11" << std::endl;
		return -1;
	}

	mpMap->ClearReinit();
	std::vector<bool> vbInliers(vX3Ds.size(), true);
	std::vector<bool> vbInliers2(vX3Ds.size(), true);
	Optimization::LocalOptimization(mpSystem, mpMap, pCurrKF, vX3Ds, vMappingCPs, vbInliers, vbInliers2, 1.0, fMedianDepth, fMeanDepth, fStdDev);
	int nFail = vX3Ds.size();
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
