#include <ServerMapOptimizer.h>
#include <System.h>
#include <Frame.h>
#include <MapPoint.h>
#include <User.h>
#include <ServerMap.h>
#include <Optimization.h>
#include <Map.h>

namespace UVR_SLAM {
	ServerMapOptimizer::ServerMapOptimizer(System* pSystem):mpSystem(pSystem){
	}
	ServerMapOptimizer::~ServerMapOptimizer(){
	
	}
	void ServerMapOptimizer::InsertKeyFrame(std::pair<Frame*, std::string> pairInfo){
		std::unique_lock<std::mutex> lock(mMutexQueue);
		mQueue.push(pairInfo);
	}
	int ServerMapOptimizer::KeyframesInQueue(){
		std::unique_lock<std::mutex> lock(mMutexQueue);
		return(!mQueue.empty());
	}
	bool ServerMapOptimizer::CheckNewKeyFrames(){
		std::unique_lock<std::mutex> lock(mMutexQueue);
		return mQueue.size();
	}
	void ServerMapOptimizer::AcquireFrame(){
		std::unique_lock<std::mutex> lock(mMutexQueue);
		mPairFrameInfo = mQueue.front();
		mQueue.pop();
		std::string user = mPairFrameInfo.second;
		mpTargetFrame = mPairFrameInfo.first;
		mpTargetUser = mpSystem->GetUser(user);
		if (mpTargetUser)
			mpTargetMap = mpSystem->GetMap(mpTargetUser->mapName);
	}
	void ServerMapOptimizer::ProcessNewKeyFrame(){
	
	}
	bool ServerMapOptimizer::isDoingProcess(){
		std::unique_lock<std::mutex> lock(mMutexDoingProcess);
		return mbDoingProcess;
	}
	void ServerMapOptimizer::SetDoingProcess(bool flag){
		std::unique_lock<std::mutex> lock(mMutexDoingProcess);
		mbDoingProcess = flag;
	}
	void ServerMapOptimizer::RunWithMappingServer(){
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
				
				int nTargetID = mpTargetFrame->mnFrameID;

				std::vector<UVR_SLAM::Frame*> vpOptKFs, vpTempKFs;
				std::vector<UVR_SLAM::Frame*> vpFixedKFs;
				std::vector<UVR_SLAM::MapPoint*> vpOptMPs, vpTempMPs;// , vpMPs2;
				std::map<Frame*, int> mpKeyFrameCounts, mpGraphFrameCounts;

				auto vpMPs = mpTargetFrame->GetMapPoints();
				for (size_t i = 0, iend = mpTargetFrame->mvPts.size(); i < iend; i++) {
					auto pMPi = vpMPs[i];
					if (!pMPi || pMPi->isDeleted())
						continue;
					if (pMPi->mnLocalBAID == nTargetID) {
						continue;
					}
					pMPi->mnLocalBAID = nTargetID;
					vpOptMPs.push_back(pMPi);
				}

				mpTargetFrame->mnLocalBAID = nTargetID;
				vpOptKFs.push_back(mpTargetFrame);

				auto vpNeighKFs = mpTargetFrame->GetConnectedKFs(15);
				for (auto iter = vpNeighKFs.begin(), iend = vpNeighKFs.end(); iter != iend; iter++) {
					auto pKFi = *iter;
					if (pKFi->isDeleted())
						continue;
					auto vpMPs = pKFi->GetMapPoints();
					auto vPTs = pKFi->mvPts;
					int N1 = 0;
					for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
						auto pMPi = vpMPs[i];
						if (!pMPi || pMPi->isDeleted() || pMPi->mnLocalBAID == nTargetID)
							continue;
						pMPi->mnLocalBAID = nTargetID;
						vpOptMPs.push_back(pMPi);
					}
					if (pKFi->mnLocalBAID == nTargetID) {
						std::cout << "Error::pKF::" << pKFi->mnFrameID << ", " << pKFi->mnKeyFrameID << std::endl;
						continue;
					}
					pKFi->mnLocalBAID = nTargetID;
					vpOptKFs.push_back(pKFi);


				}//for vpmps, vpkfs

				 //Fixed KFs
				for (size_t i = 0, iend = vpOptMPs.size(); i < iend; i++) {
					auto pMPi = vpOptMPs[i];
					auto mmpFrames = pMPi->GetObservations();
					for (auto iter = mmpFrames.begin(), iter_end = mmpFrames.end(); iter != iter_end; iter++) {
						auto pKFi = (iter->first);
						if (pKFi->isDeleted())
							continue;
						if (pKFi->mnLocalBAID == nTargetID)
							continue;
						mpGraphFrameCounts[pKFi]++;
						/*pKFi->mnLocalBAID = nTargetID;
						vpFixedKFs.push_back(pKFi);*/
					}
				}//for fixed iter

				 ////fixed kf를 정렬
				std::vector<std::pair<int, Frame*>> vPairFixedKFs;
				for (auto iter = mpGraphFrameCounts.begin(), iend = mpGraphFrameCounts.end(); iter != iend; iter++) {
					auto pKFi = iter->first;
					auto count = iter->second;
					if (count < 10)
						continue;
					vPairFixedKFs.push_back(std::make_pair(count, pKFi));
				}
				std::sort(vPairFixedKFs.begin(), vPairFixedKFs.end(), std::greater<>());

				////상위 N개의 Fixed KF만 추가
				for (size_t i = 0, iend = vPairFixedKFs.size(); i < iend; i++) {
					auto pair = vPairFixedKFs[i];
					if (vpFixedKFs.size() == 40)
						break;
					auto pKFi = pair.second;
					pKFi->mnFixedBAID = nTargetID;
					pKFi->mnLocalBAID = nTargetID;
					vpFixedKFs.push_back(pKFi);
				}

				Optimization::OpticalLocalBundleAdjustment(mpTargetMap, this, vpOptMPs, vpOptKFs, vpFixedKFs);
				
				SetDoingProcess(false);
			}
		}
	}
}