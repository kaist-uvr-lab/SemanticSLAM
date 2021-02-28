#include <MapOptimizer.h>
#include <System.h>
#include <Frame.h>
#include <MapPoint.h>
#include <CandidatePoint.h>

#include <Optimization.h>
#include <PlanarOptimization.h>

#include <Plane.h>
#include <Map.h>
#include <MapGrid.h>

#include <SemanticSegmentator.h>
#include <PlaneEstimator.h>
#include <Visualizer.h>
#include <LoopCloser.h>

UVR_SLAM::MapOptimizer::MapOptimizer(System* pSystem) : mpTargetFrame(nullptr), mbStopBA(false), mbDoingProcess(false)
{
	mpSystem = pSystem;
}
UVR_SLAM::MapOptimizer::~MapOptimizer() {}

void UVR_SLAM::MapOptimizer::Init() {

	mK = mpSystem->mK.clone();
	mnWidth = mpSystem->mnWidth;
	mnHeight = mpSystem->mnHeight;

	mpVisualizer = mpSystem->mpVisualizer;
	mpLoopCloser = mpSystem->mpLoopCloser;
	mpPlaneEstimator = mpSystem->mpPlaneEstimator;
	mpSegmentator = mpSystem->mpSegmentator;
	mpMap = mpSystem->mpMap;
}

void UVR_SLAM::MapOptimizer::SetDoingProcess(bool b) {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	mbDoingProcess = b;
}
bool UVR_SLAM::MapOptimizer::isDoingProcess() {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	return mbDoingProcess;
}

void UVR_SLAM::MapOptimizer::InsertKeyFrame(UVR_SLAM::Frame *pKF)
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	//mbStopBA = true;
	mKFQueue.push(pKF);
}

bool UVR_SLAM::MapOptimizer::isStopBA() {
	std::unique_lock<std::mutex> lock(mMutexStopBA);
	return mbStopBA;
}
void UVR_SLAM::MapOptimizer::StopBA(bool b)
{
	std::unique_lock<std::mutex> lock(mMutexStopBA);
	mbStopBA = b;
}

bool UVR_SLAM::MapOptimizer::CheckNewKeyFrames()
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	return(!mKFQueue.empty());
}

void UVR_SLAM::MapOptimizer::ProcessNewKeyFrame()
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	mpTargetFrame = mKFQueue.front();
	mKFQueue.pop();
}
void UVR_SLAM::MapOptimizer::RunWithMappingServer() {
	std::cout << "MappingServer::MapOptimizer::Start" << std::endl;
	while (true) {
		if (CheckNewKeyFrames()) {
			SetDoingProcess(true);
			ProcessNewKeyFrame();
			std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
#ifdef DEBUG_MAP_OPTIMIZER_LEVEL_1
			std::cout << "MappingServer::MapOptimizer::" << mpTargetFrame->mnFrameID << "::Start" << std::endl;
#endif
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
				if (vpFixedKFs.size() == 20)
					break;
				auto pKFi = pair.second;
				pKFi->mnFixedBAID = nTargetID;
				pKFi->mnLocalBAID = nTargetID;
				vpFixedKFs.push_back(pKFi);
			}
#ifdef DEBUG_MAP_OPTIMIZER_LEVEL_2
			std::cout << "MappingServer::MapOptimizer::" << mpTargetFrame->mnFrameID << "::TEST::Start" << std::endl;
			std::map<int, int> testMPs;
			std::map<int, int> testKFs, testFixedKFs;
			for (size_t i = 0, iend = vpOptMPs.size(); i < iend; i++) {
				auto pMP = vpOptMPs[i];
				testMPs[pMP->mnMapPointID]++;
				if (testMPs[pMP->mnMapPointID] > 1) {
					std::cout << "BA::MP::Error::" << pMP->mnMapPointID <<"::"<<testMPs[pMP->mnMapPointID ]<< std::endl;
				}
			}
			for (size_t i = 0, iend = vpOptKFs.size(); i < iend; i++) {
				auto pKF = vpOptKFs[i];
				testKFs[pKF->mnKeyFrameID]++;
				if (testKFs[pKF->mnKeyFrameID] > 1) {
					std::cout << "BA::KF::Error::" << pKF->mnKeyFrameID << std::endl;
				}
			}
			for (size_t i = 0, iend = vpFixedKFs.size(); i < iend; i++) {
				auto pKF = vpFixedKFs[i];
				testKFs[pKF->mnKeyFrameID]++;
				if (testKFs[pKF->mnKeyFrameID] > 1) {
					std::cout << "BA::Fixed::Error::" << pKF->mnKeyFrameID << std::endl;
				}
			}
			std::cout << "MappingServer::MapOptimizer::" << mpTargetFrame->mnFrameID << "::TEST::END" << std::endl;
#endif
			Optimization::OpticalLocalBundleAdjustment(mpMap, this, vpOptMPs, vpOptKFs, vpFixedKFs);
			std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
			auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			float t_test1 = du_test1 / 1000.0;
#ifdef DEBUG_MAP_OPTIMIZER_LEVEL_1
			std::cout << "MapOptimization::BA::"<<mpTargetFrame->mnFrameID<<"::END::" << vpOptMPs.size() << " ," << vpOptKFs.size() << ", " << vpFixedKFs.size() <<"::"<< t_test1 << std::endl;
#endif
			SetDoingProcess(false);
		}
	}
}
void UVR_SLAM::MapOptimizer::Run() {
	std::string mStrPath;

	int mnThreshMinKF = mpSystem->mnThreshMinKF;

	while (1) {
		if (CheckNewKeyFrames()) {
			SetDoingProcess(true);
			std::chrono::high_resolution_clock::time_point s_start = std::chrono::high_resolution_clock::now();
			ProcessNewKeyFrame();
			mStrPath = mpSystem->GetDirPath(mpTargetFrame->mnKeyFrameID);
			StopBA(false);
			
			auto targetFrame = mpTargetFrame;
			///////////////////////////////////////////////////////////////
			////preprocessing
			//std::cout << "ba::processing::start" << std::endl;
			int nTargetID = mpTargetFrame->mnFrameID;
			int nTargetKeyID = mpTargetFrame->mnKeyFrameID;
			mpTargetFrame->mnLocalBAID = nTargetID;
			//std::cout << "BA::preprocessing::start" << std::endl;
			std::chrono::high_resolution_clock::time_point temp_1 = std::chrono::high_resolution_clock::now();
			
			std::vector<UVR_SLAM::MapPoint*> vpOptMPs, vpTempMPs;// , vpMPs2;
			std::set<Frame*> spTempKFs, spGraphFrames;
			std::vector<UVR_SLAM::Frame*> vpOptKFs, vpTempKFs;
			std::vector<UVR_SLAM::Frame*> vpFixedKFs;
			std::vector<MapGrid*> vpLocalMapGrid;
			//auto vpGraphKFs = mpMap->GetGraphFrames();
						
			std::map<MapPoint*, int> mpMapPointCounts;
			std::map<Frame*, int> mpKeyFrameCounts, mpGraphFrameCounts;

			std::chrono::high_resolution_clock::time_point s_start2 = std::chrono::high_resolution_clock::now();
			//////기존의 방식
			{
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
				
				////현재 타겟 프레임과 연결된 프레임들을 이용해 로컬맵 확장 및 키프레임, 픽스드 프레임 확장
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
					vpOptKFs.push_back(pKFi);
					pKFi->mnLocalBAID = nTargetID;
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
					if (vpFixedKFs.size() == 20)
						break;
					auto pKFi = pair.second;
					pKFi->mnFixedBAID = nTargetID;
					pKFi->mnLocalBAID = nTargetID;
					vpFixedKFs.push_back(pKFi);
				}

			}
			//////기존의 방식

			auto vpPlaneInfos = mpMap->GetPlaneInfos();
			int n = vpPlaneInfos.size() - 1;
			if(n < 0)
				Optimization::OpticalLocalBundleAdjustment(mpMap, this, vpOptMPs, vpOptKFs, vpFixedKFs);
			else
			{
				std::cout << "Plane optimization" << std::endl;
				//PlanarOptimization::OpticalLocalBundleAdjustmentWithPlane(this, vpPlaneInfos[n], vpOptMPs, vpOptKFs, vpFixedKFs);
			}
			
			/*std::cout << "BA::Delete::Start" << std::endl;
			mpMap->DeleteMPs();
			std::cout << "BA::Delete::End" << std::endl;*/
			std::chrono::high_resolution_clock::time_point s_end = std::chrono::high_resolution_clock::now();
			auto leduration = std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_start).count();
			float letime = leduration / 1000.0;
			std::stringstream ss;
			ss << "Map Optimizer::" << mpTargetFrame->mnKeyFrameID <<", "<< vpOptKFs[0]->mnKeyFrameID <<"::"<<letime<<"::" <<"||"<< vpOptKFs.size()<<", "<< vpFixedKFs.size()<<", "<<vpOptMPs.size();
			mpSystem->SetMapOptimizerString(ss.str());
			//종료
			SetDoingProcess(false);
		}
	}
}