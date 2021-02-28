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
////매칭인포 삭제 관련 수정 필요. 210228
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
		/*auto matchInfo = pKFi->mpMatchInfo;
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
		}*/
		
	}//for kf iter
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
