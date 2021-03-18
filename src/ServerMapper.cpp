#include <ServerMapper.h>
#include <System.h>
#include <Matcher.h>
#include <Frame.h>
#include <MapPoint.h>
#include <LoopCloser.h>
#include <ServerMapOptimizer.h>
#include <User.h>
#include <ServerMap.h>
#include <WebApi.h>
#include <future>
#include <Map.h>

class System;
class MapPoint;
class Frame;
class LoopCloser;
class MapOptimizer;

class User;
class ServerMap;

namespace UVR_SLAM {
	ServerMapper::ServerMapper(System* pSystem){
		mpSystem = pSystem;
	}
	ServerMapper::~ServerMapper(){
	
	}
	void ServerMapper::Init() {
		mpLoopCloser = mpSystem->mpLoopCloser;
		mpMapOptimizer = mpSystem->mpServerMapOptimizer;
	}
	void ServerMapper::InsertKeyFrame(std::pair<Frame*, std::string> pairInfo){
		std::unique_lock<std::mutex> lock(mMutexQueue);
		mQueue.push(pairInfo);
	}
	bool ServerMapper::CheckNewKeyFrames(){
		std::unique_lock<std::mutex> lock(mMutexQueue);
		return(!mQueue.empty());
	}
	int ServerMapper::KeyframesInQueue() {
		std::unique_lock<std::mutex> lock(mMutexQueue);
		return mQueue.size();
	}
	void ServerMapper::AcquireFrame(){
		std::unique_lock<std::mutex> lock(mMutexQueue);
		mPairFrameInfo = mQueue.front();
		mQueue.pop();
		std::string user = mPairFrameInfo.second;
		mpTargetFrame = mPairFrameInfo.first;
		mpTargetUser = mpSystem->GetUser(user);
		if(mpTargetUser)
			mpTargetMap = mpSystem->GetMap(mpTargetUser->mapName);
	}
	void ServerMapper::ProcessNewKeyFrame(){

		mpTargetMap->AddFrame(mpTargetFrame);
		auto mvpMPs = mpTargetFrame->GetMapPoints();
		for (size_t i = 0, iend = mvpMPs.size(); i < iend; i++) {
			auto pMP = mvpMPs[i];
			if (!pMP || pMP->isDeleted())
				continue;
			pMP->AddObservation(mpTargetFrame, i);
		}
		mpTargetFrame->ComputeSceneDepth();
	}
	bool ServerMapper::isDoingProcess(){
		std::unique_lock<std::mutex> lock(mMutexDoingProcess);
		return mbDoingProcess;
	}
	void ServerMapper::SetDoingProcess(bool flag){
		std::unique_lock<std::mutex> lock(mMutexDoingProcess);
		mbDoingProcess = flag;
	}
	void ServerMapper::RunWithMappingServer(){
		int CODE_MATCH_ERROR = 10000;
		std::cout << "MappingServer::ServerMapper::Start" << std::endl;
		while (true) {
			
			if (CheckNewKeyFrames()) {
				SetDoingProcess(true);
				AcquireFrame();
				if (!mpTargetUser || !mpTargetMap)
				{
					SetDoingProcess(false);
					continue;
				}
				ProcessNewKeyFrame();
				ComputeNeighborKFs(mpTargetFrame);
				ConnectNeighborKFs(mpTargetFrame, mpTargetFrame->mmKeyFrameCount, 20);
				NewMapPointMarginalization();
				////수정 필요함.
				////로컬맵 매칭이 필요함.
				CreateMapPoints();
				SendData(mpTargetFrame, mPairFrameInfo.second, mpTargetUser->mapName);
				mpMapOptimizer->InsertKeyFrame(mPairFrameInfo);
				mpLoopCloser->InsertKeyFrame(mpTargetFrame);
				SetDoingProcess(false);
			}
		}
	}
	void ServerMapper::NewMapPointMarginalization(){
		int nMarginalized = 0;
		int nNumRequireKF = 3;
		float mfRatio = 0.25f;

		std::list<UVR_SLAM::MapPoint*>::iterator lit = mpTargetMap->mlpNewMapPoints.begin();
		while (lit != mpTargetMap->mlpNewMapPoints.end()) {
			UVR_SLAM::MapPoint* pMP = *lit;
			int nDiffKF = mpTargetFrame->mnKeyFrameID - pMP->mnFirstKeyFrameID;
			bool bBad = false;
			if (pMP->isDeleted()) {
				//already deleted
				lit = mpTargetMap->mlpNewMapPoints.erase(lit);
			}
			else if (pMP->GetFVRatio() < mfRatio) {
				bBad = true;
				lit = mpTargetMap->mlpNewMapPoints.erase(lit);
			}
			else if (nDiffKF >= 2 && pMP->GetObservations().size() <= 2) {
				bBad = true;
				lit = mpTargetMap->mlpNewMapPoints.erase(lit);
			}
			else if (nDiffKF >= nNumRequireKF) {
				lit = mpTargetMap->mlpNewMapPoints.erase(lit);
				pMP->SetNewMP(false);
			}
			else
				lit++;
			if (bBad) {
				pMP->DeleteMapPoint();
			}
		}
		return;
	}
	
	void ServerMapper::ComputeNeighborKFs(Frame* pKF){
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
	void ServerMapper::ConnectNeighborKFs(Frame* pKF, std::map<UVR_SLAM::Frame*, int> mpCandiateKFs, int thresh){
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

	auto server_kf_match = [](std::string ip, int port, std::string map, int id1, int id2) {
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		WebAPI* mpAPI = new WebAPI(ip, port);
		std::stringstream ss;
		ss << "/featurematch?map=" << map << "&id1=" << id1 << "&id2=" << id2;
		auto res = mpAPI->Send(ss.str(), "");
		cv::Mat matches = cv::Mat::zeros(res.size() / sizeof(int), 1, CV_32SC1);
		std::memcpy(matches.data, res.data(), matches.rows * sizeof(int));
		//WebAPIDataConverter::ConvertStringToMatches(res.c_str(), n, matches);
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float t_test1 = du_test1 / 1000.0;
		//std::cout << "api::featurematch=" << t_test1 << std::endl;
		return matches;
	};
	auto server_projection_test = [](cv::Mat Xw, cv::Mat R, cv::Mat t, cv::Mat K, cv::Point2f pt, float thresh) {
		cv::Mat Xcam = R * Xw + t;
		cv::Mat Ximg = K*Xcam;
		float fDepth = Ximg.at < float>(2);
		if (fDepth <= 0.0)
			return false;
		cv::Point2f pt1 = cv::Point2f(Ximg.at<float>(0) / fDepth, Ximg.at<float>(1) / fDepth);
		auto diffPt = pt1 - pt;
		float dist = diffPt.dot(diffPt);
		bool bDist1 = dist < thresh;
		return bDist1;
	};
	auto server_create_mp = []
	(ServerMap* pMap, Frame* pCurr, Frame* pPrev, int idx1, int idx2, cv::Mat desc) {
		
		auto prevPt = pPrev->mvPts[idx2];
		auto currPt = pCurr->mvPts[idx1];
		cv::Mat Rcurr, Tcurr, Pcurr;
		pCurr->GetPose(Rcurr, Tcurr);
		cv::hconcat(Rcurr, Tcurr, Pcurr);
		Pcurr = pCurr->mK*Pcurr;
		
		cv::Mat Rprev, Tprev, Pprev;
		pPrev->GetPose(Rprev, Tprev);
		cv::hconcat(Rprev, Tprev, Pprev);
		Pprev = pPrev->mK*Pprev;

		cv::Mat Rtprev = Rprev.t();
		cv::Mat Rtcurr = Rcurr.t();

		cv::Mat xn1 = (cv::Mat_<float>(3, 1) << prevPt.x, prevPt.y, 1.0);
		cv::Mat xn2 = (cv::Mat_<float>(3, 1) << currPt.x, currPt.y, 1.0);
		cv::Mat ray1 = Rtprev*pPrev->mInvK*xn1;
		cv::Mat ray2 = Rtcurr*pCurr->mInvK*xn2;
		float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1)*cv::norm(ray2));

		bool bParallax = cosParallaxRays < 0.9998;
		MapPoint* res = nullptr;
		if (!bParallax)
			return res;

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
		float dist1, dist2;
		float thresh = 9.0;
		{
			cv::Mat Xcam = Rprev * x3D + Tprev;
			cv::Mat Ximg = pPrev->mK*Xcam;
			float fDepth = Ximg.at < float>(2);
			bDepth1 = fDepth > 0.0;
			pt1 = cv::Point2f(Ximg.at<float>(0) / fDepth, Ximg.at<float>(1) / fDepth);
			auto diffPt = pt1 - prevPt;
			dist1 = diffPt.dot(diffPt);
			bDist1 = dist1 < thresh;
		}
		bool bMaxDepth;
		{
			cv::Mat Xcam = Rcurr * x3D + Tcurr;
			cv::Mat Ximg = pCurr->mK*Xcam;
			float fDepth = Ximg.at < float>(2);
			bDepth2 = fDepth > 0.0;
			bMaxDepth = fDepth < pCurr->mfMaxDepth;
			pt2 = cv::Point2f(Ximg.at<float>(0) / fDepth, Ximg.at<float>(1) / fDepth);
			auto diffPt = pt2 - currPt;
			dist2 = diffPt.dot(diffPt);
			bDist2 = dist2 < thresh;
		}
		//std::cout << "test::" << dist2 << ", " << dist1 << ", " << std::endl;
		if (bDist1 && bDepth1 && bDist2 && bDepth2) {
			//pMap->AddTempMP(x3D);
			res = new UVR_SLAM::MapPoint(pMap, pCurr, x3D, desc, 0);
		}
		return res;
	};

	void ServerMapper::CreateMapPoints() {

		auto vpKFs = mpTargetFrame->GetConnectedKFs(15);
		mpSystem->mpMap->ClearReinit();
		mpSystem->mpMap->ClearTempMPs();
		auto pMatcher = mpSystem->mpMatcher;

		int ncreate = 0;
		int nkf = 0;
		for (size_t k = 0, kend = vpKFs.size(); k < kend; k++) {
			if (k > 0 && CheckNewKeyFrames())
				break;
			nkf++;
			auto pKF = vpKFs[k];
			auto matches = server_kf_match(mpSystem->ip, mpSystem->port, mpTargetUser->mapName, mpTargetFrame->mnFrameID, pKF->mnFrameID);

			cv::Mat Rrel, Trel;
			mpTargetFrame->GetRelativePoseFromTargetFrame(pKF, Rrel, Trel);

			std::vector<bool> vecBoolOverlap(pKF->mvPts.size(), false);
			for (size_t i = 0, iend = matches.rows; i < iend; i++) {
				int idx1 = i;
				int idx2 = matches.at<int>(idx1);
				if (idx2 == 10000)
					continue;
				if (vecBoolOverlap[idx2])
				{
					matches.at<int>(idx1) = 10000;
					continue; 
				}
				vecBoolOverlap[idx2] = true;

				auto pCurrMP = mpTargetFrame->GetMapPoint(idx1);
				auto pPrevMP = pKF->GetMapPoint(idx2);

				bool bCurrMP = pCurrMP && !pCurrMP->isDeleted();
				bool bPrevMP = pPrevMP && !pPrevMP->isDeleted();
				
				auto prevPt = pKF->mvPts[idx2];
				auto currPt = mpTargetFrame->mvPts[idx1];

				/////////check epipolar constraints
				cv::Mat ray = pKF->mInvK*(cv::Mat_<float>(3, 1) << prevPt.x, prevPt.y, 1.0);
				float z_min, z_max;
				z_min = 0.01f;
				z_max = 1.0f;
				cv::Point2f XimgMin, XimgMax;
				pMatcher->ComputeEpiLinePoint(XimgMin, XimgMax, ray, z_min, z_max, Rrel, Trel, mpTargetFrame->mK); //ray,, Rrel, Trel
				cv::Mat lineEqu = pMatcher->ComputeLineEquation(XimgMin, XimgMax);
				bool bEpiConstraints = pMatcher->CheckLineDistance(lineEqu, currPt, 1.0);
				if (!bEpiConstraints) {
					continue;
				}
				/////////check epipolar constraints

				if (!bCurrMP && !bPrevMP) {
					auto pNewMP = server_create_mp(mpTargetMap, mpTargetFrame, pKF, idx1, idx2, mpTargetFrame->matDescriptor.row(idx1));
					if (pNewMP) {
						ncreate++;
						mpTargetFrame->AddMapPoint(pNewMP, idx1);
						pNewMP->AddObservation(mpTargetFrame, idx1);
						pKF->AddMapPoint(pNewMP, idx2);
						pNewMP->AddObservation(pKF, idx2);
						pNewMP->IncreaseFound(2);
						pNewMP->IncreaseVisible(2);
						mpTargetMap->mlpNewMapPoints.push_back(pNewMP);
						mpSystem->mpMap->AddReinit(pNewMP->GetWorldPos());
					}
				}
				else if (bCurrMP && !bPrevMP) {
					//projection test : pCurrMP in prev frame    pPrevMP in curr frame
					/*cv::Mat R, t;
					pKF->GetPose(R, t);
					if (server_projection_test(pCurrMP->GetWorldPos(), R, t, pKF->mK, pKF->mvPts[idx2], 9.0)) {
						
					}*/
					//if (pKF->isInFrustum(pCurrMP, 0.6f)) {
						pKF->AddMapPoint(pCurrMP, idx2);
						pCurrMP->AddObservation(pKF, idx2);
						pCurrMP->IncreaseFound();
						pCurrMP->IncreaseVisible();
						mpSystem->mpMap->AddTempMP(pCurrMP->GetWorldPos());
					//}
					
				}
				else if (!bCurrMP && bPrevMP) {
					//projection test : pPrevMP in curr frame
					/*cv::Mat R, t;
					mpTargetFrame->GetPose(R, t);
					if (server_projection_test(pPrevMP->GetWorldPos(), R, t, mpTargetFrame->mK, mpTargetFrame->mvPts[idx1], 9.0)) {
						
					}*/

					//if (mpTargetFrame->isInFrustum(pPrevMP, 0.6f)) {
						mpTargetFrame->AddMapPoint(pPrevMP, idx1);
						pPrevMP->AddObservation(mpTargetFrame, idx1);
						pPrevMP->IncreaseFound();
						pPrevMP->IncreaseVisible();
						mpSystem->mpMap->AddTempMP(pPrevMP->GetWorldPos());
					//}
					
				}
				else{
					if (pCurrMP->mnMapPointID != pPrevMP->mnMapPointID) {
						////fuse case
						//projection test : pPrevMP in curr frame
						cv::Mat R, t;
						mpTargetFrame->GetPose(R, t);
						if (!server_projection_test(pPrevMP->GetWorldPos(), R, t, mpTargetFrame->mK, mpTargetFrame->mvPts[idx1], 4.0)) {
							continue;
						}
						if (pCurrMP->GetNumObservations() > pPrevMP->GetNumObservations()) {
							pPrevMP->Fuse(pCurrMP);
						}
						else {
							pCurrMP->Fuse(pPrevMP);
						}
					}//fuse
				}

			}//for i

		}// for k
		std::cout << "ID = "<<mpTargetFrame->mnFrameID<<"::"<<"CreateMP=" <<ncreate<<", "<< nkf <<"::"<< vpKFs.size() << ", " << KeyframesInQueue() << std::endl;

	}

	void ServerMapper::SendData(Frame* pF, std::string user, std::string map) {
		//포즈: 프레임과 포즈는
		//로컬맵
		//프레임데이터

		/////데이터 전송
		auto fres = std::async(std::launch::async, [](std::string ip, int port, std::string mapName, Frame* pF) {
			cv::Mat data = cv::Mat::zeros(0, 1, CV_32FC1);
			auto vpMPs = pF->GetMapPoints();
			for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
				auto pMPi = vpMPs[i];
				if (!pMPi || pMPi->isDeleted())
					continue;
				auto pt = pF->mvPts[i];
				cv::Mat Xw = pMPi->GetWorldPos();
				data.push_back(pt.x);
				data.push_back(pt.y);
				data.push_back(Xw.at<float>(0));
				data.push_back(Xw.at<float>(1));
				data.push_back(Xw.at<float>(2));
			}
			WebAPI* mpAPI = new WebAPI(ip, port);
			std::stringstream ss;
			ss << "/ReceiveData?map=" << mapName << "&id=" << pF->mnFrameID << "&key=btracking";
			mpAPI->Send(ss.str(), data.data, data.rows*sizeof(float));
			
		}, mpSystem->ip, mpSystem->port, map, pF);

		auto fpose = std::async(std::launch::async, [](std::string ip, int port, std::string mapName, Frame* pF) {
			cv::Mat poses = cv::Mat::zeros(0, 3, CV_32FC1);
			cv::Mat R, t;
			pF->GetPose(R, t);
			poses.push_back(R);
			poses.push_back(t.t());
			WebAPI* mpAPI = new WebAPI(ip, port);
			std::stringstream ss;
			ss << "/ReceiveData?map=" << mapName << "&id=" << pF->mnFrameID << "&key=bpose";
			auto res = mpAPI->Send(ss.str(), poses.data, sizeof(float)*12);
		}, mpSystem->ip, mpSystem->port, map, pF);
	
		auto fframeID = std::async(std::launch::async, [](std::string ip, int port, std::string userName, int id) {
			WebAPI* mpAPI = new WebAPI(ip, port);
			std::stringstream ss;
			ss << "/ReceiveFrameID?user=" << userName << "&id=" << id;
			auto res = mpAPI->Send(ss.str(), "");
			
		}, mpSystem->ip, mpSystem->port, user, pF->mnFrameID);

	}
}

