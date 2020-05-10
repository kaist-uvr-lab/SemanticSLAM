#include <MapOptimizer.h>
#include <System.h>
#include <Frame.h>
#include <Visualizer.h>
#include <FrameWindow.h>
#include <MapPoint.h>
#include <Optimization.h>
#include <PlaneEstimator.h>
#include <Plane.h>
#include <Map.h>

UVR_SLAM::MapOptimizer::MapOptimizer(std::string strPath, Map* pMap) : mpTargetFrame(nullptr), mbStopBA(false)
{
	cv::FileStorage fs(strPath, cv::FileStorage::READ);
	float fx = fs["Camera.fx"];
	float fy = fs["Camera.fy"];
	float cx = fs["Camera.cx"];
	float cy = fs["Camera.cy"];

	mK = cv::Mat::eye(3, 3, CV_32F);
	mK.at<float>(0, 0) = fx;
	mK.at<float>(1, 1) = fy;
	mK.at<float>(0, 2) = cx;
	mK.at<float>(1, 2) = cy;

	mnWidth = fs["Image.width"];
	mnHeight = fs["Image.height"];

	fs.release();

	mpMap = pMap;
}
UVR_SLAM::MapOptimizer::~MapOptimizer() {}

void UVR_SLAM::MapOptimizer::SetSystem(System* pSystem) {
	mpSystem = pSystem;
}
void UVR_SLAM::MapOptimizer::SetVisualizer(Visualizer* pVis) {
	mpVisualizer = pVis;
}
void UVR_SLAM::MapOptimizer::SetFrameWindow(UVR_SLAM::FrameWindow* pWindow) {
	mpFrameWindow = pWindow;
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
	mbStopBA = true;
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

void UVR_SLAM::MapOptimizer::Run() {
	std::string mStrPath;
	while (1) {
		if (CheckNewKeyFrames()) {
			SetDoingProcess(true);
			std::chrono::high_resolution_clock::time_point s_start = std::chrono::high_resolution_clock::now();
			ProcessNewKeyFrame();
			std::cout << "ba::start::" << mpTargetFrame->GetFrameID() << std::endl;
			mStrPath = mpSystem->GetDirPath(mpTargetFrame->GetKeyFrameID());
			StopBA(false);
			auto currMatchInfo = mpTargetFrame->mpMatchInfo;
			auto targetFrame = mpTargetFrame;
			///////////////////////////////////////////////////////////////
			////preprocessing
			//std::cout << "ba::processing::start" << std::endl;
			int nTargetID = mpTargetFrame->GetFrameID();
			mpTargetFrame->mnLocalBAID = nTargetID;
			//std::cout << "BA::preprocessing::start" << std::endl;
			std::chrono::high_resolution_clock::time_point temp_1 = std::chrono::high_resolution_clock::now();
			std::vector<UVR_SLAM::MapPoint*> vpMPs;// , vpMPs2;
			std::vector<UVR_SLAM::Frame*> vpKFs;
			std::vector<UVR_SLAM::Frame*> vpFixedKFs;
			
			auto tempKFs1 = mpTargetFrame->GetConnectedKFs(15);
			auto tempKFs2 = mpTargetFrame->GetConnectedKFs();
			vpKFs.push_back(mpTargetFrame);
			
			for (int i = 0; i < tempKFs1.size(); i++){
				tempKFs1[i]->mnLocalBAID = nTargetID;
				vpKFs.push_back(tempKFs1[i]);
			}
			for (int i = 0; i < tempKFs2.size(); i++) {
				if (tempKFs2[i]->mnLocalBAID != nTargetID && tempKFs2[i]->mnFixedBAID != nTargetID)
				{
					tempKFs2[i]->mnFixedBAID = nTargetID;
					vpFixedKFs.push_back(tempKFs2[i]);
				}
			}

			for (int k = 0; k < vpKFs.size(); k++){
				auto pKFi = vpKFs[k];
				auto matchInfo = pKFi->mpMatchInfo;
				for (int i = 0; i < matchInfo->mvpMatchingMPs.size(); i++) {
					auto pMPi = matchInfo->mvpMatchingMPs[i];
					if (!pMPi || pMPi->isDeleted() || pMPi->mnLocalBAID == nTargetID) {
						continue;
					}
					pMPi->mnLocalBAID = nTargetID;
					vpMPs.push_back(pMPi);
				}
				if (vpMPs.size() > 1500)
					break;
			}
			for (int i = 0; i < vpMPs.size(); i++)
			{
				UVR_SLAM::MapPoint* pMP = vpMPs[i];
				auto observations = pMP->GetConnedtedFrames();
				for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
				{
					auto pMatch = mit->first;
					auto pKFi = pMatch->mpRefFrame;

					if (pKFi->mnLocalBAID != nTargetID && pKFi->mnFixedBAID != nTargetID)
					{
						pKFi->mnFixedBAID = nTargetID;
						vpFixedKFs.push_back(pKFi);
					}
				}
			}
			
			////for (int k = 0; k < 30; k++) {
			////	if (!targetFrame)
			////		break;
			////	vpKFs.push_back(targetFrame);
			////	if(targetFrame->mnLocalBAID != nTargetID){
			////		targetFrame->mnLocalBAID = nTargetID;
			////		auto matchInfo = targetFrame->mpMatchInfo;
			////		for (int i = 0; i < matchInfo->mvpMatchingMPs.size(); i++) {
			////			auto pMPi = matchInfo->mvpMatchingMPs[i];
			////			if (!pMPi || pMPi->isDeleted() || pMPi->mnLocalBAID == nTargetID) {
			////				continue;
			////			}
			////			/*if(vpMPs.size() < 2500){
			////				vpMPs.push_back(pMPi);
			////				pMPi->mnLocalBAID = nTargetID;
			////			}
			////			vpMPs2.push_back(pMPi);*/
			////			vpMPs.push_back(pMPi);
			////			pMPi->mnLocalBAID = nTargetID;
			////		}

			////	}
			////	if (vpMPs.size() > 2000)
			////		break;
			////	//타겟 프레임 변경
			////	targetFrame = targetFrame->mpMatchInfo->mpTargetFrame;
			////}
			//for (int k = 0; k < vpKFs.size(); k++){
			//	auto pKFi = vpKFs[k];
			//	if (pKFi->mnLocalBAID != nTargetID) {
			//		pKFi->mnLocalBAID = nTargetID;
			//		vpKFs2.push_back(pKFi);
			//		auto matchInfo = pKFi->mpMatchInfo;
			//		for (int i = 0; i < matchInfo->mvpMatchingMPs.size(); i++) {
			//			auto pMPi = matchInfo->mvpMatchingMPs[i];
			//			if (!pMPi || pMPi->isDeleted() || pMPi->mnLocalBAID == nTargetID) {
			//				continue;
			//			}
			//			vpMPs.push_back(pMPi);
			//			pMPi->mnLocalBAID = nTargetID;
			//		}

			//	}
			//	if (vpMPs.size() > 1500)
			//		break;
			//	
			//}

			//// Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
			//
			//for (int i = 0; i < vpMPs.size(); i++)
			//{
			//	UVR_SLAM::MapPoint* pMP = vpMPs[i];
			//	auto observations = pMP->GetConnedtedFrames();
			//	//map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
			//	for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			//	{
			//		auto pMatch = mit->first;
			//		auto pKFi = pMatch->mpRefFrame;

			//		if (pKFi->mnLocalBAID != nTargetID && pKFi->mnFixedBAID != nTargetID)
			//		{
			//			pKFi->mnFixedBAID = nTargetID;
			//			vpFixedKFs.push_back(pKFi);
			//		}
			//	}
			//}
			//std::cout << "ba::processing::end" << std::endl;
			//std::cout << "BA::preprocessing::end" << std::endl;
			///////////////////////////////////////////////////////////////////////////////////////////
			////Optimization
			if (mpTargetFrame->mpPlaneInformation) {
				auto pPlaneInfo = mpTargetFrame->mpPlaneInformation->GetFloorPlane();
				//float a = pPlaneInfo->GetParam().at<float>(1);
				Optimization::OpticalLocalBundleAdjustmentWithPlane(this, pPlaneInfo, vpMPs, vpKFs, vpFixedKFs);
				//float b = pPlaneInfo->GetParam().at<float>(1);
			}
			else {
				Optimization::OpticalLocalBundleAdjustment(this, vpMPs, vpKFs, vpFixedKFs);
			}
			//
			////Optimization
			///////////////////////////////////////////////////////////////////////////////////////////
			

			//			
			//for(int j = 0; j < 1000; j++)
			//for (int i = 0; i < vpMPs.size(); i++) {
			//	auto pMPi = vpMPs[i];
			//}
			//std::chrono::high_resolution_clock::time_point temp_2 = std::chrono::high_resolution_clock::now();
			//std::list<UVR_SLAM::MapPoint*> lpMPs;
			//std::list<UVR_SLAM::Frame*> lpKFs;
			//targetFrame = mpTargetFrame;
			//for (int i = 0; i < 10; i++) {
			//	if (!targetFrame)
			//		continue;
			//	lpKFs.push_back(targetFrame);
			//	auto matchInfo = targetFrame->mpMatchInfo;
			//	for (int i = 0; i < matchInfo->mvpMatchingMPs.size(); i++) {
			//		auto pMPi = matchInfo->mvpMatchingMPs[i];
			//		if (!pMPi || pMPi->isDeleted()) {
			//			continue;
			//		}
			//		lpMPs.push_back(pMPi);
			//	}
			//	//프레임 변경
			//	targetFrame = targetFrame->mpMatchInfo->mpTargetFrame;
			//}
			//for (int j = 0; j < 1000; j++)
			//for (auto iter = lpMPs.begin(); iter != lpMPs.end(); iter++) {
			//	auto pMPi = *iter;
			//}
			/*std::chrono::high_resolution_clock::time_point temp_3 = std::chrono::high_resolution_clock::now();
			auto du1 = std::chrono::duration_cast<std::chrono::milliseconds>(temp_2 - temp_1).count();
			float t1 = du1 / 1000.0;
			auto du2 = std::chrono::duration_cast<std::chrono::milliseconds>(temp_3 - temp_2).count();
			float t2 = du2 / 1000.0;*/
			////Visualizer
			//mpVisualizer->SetMPs(vpMPs2);
			////preprocessing
			///////////////////////////////////////////////////////////////
			
			/*if(mpMap->isFloorPlaneInitialized())
				Optimization::LocalBundleAdjustmentWithPlane(mpMap,mpTargetFrame, mpFrameWindow, &mbStopBA);
			else*/
			//Optimization::OpticalLocalBundleAdjustment(this, mpTargetFrame, mpFrameWindow);
			
			std::chrono::high_resolution_clock::time_point s_end = std::chrono::high_resolution_clock::now();
			auto leduration = std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_start).count();
			float letime = leduration / 1000.0;
			std::stringstream ss;
			ss << "Map Optimizer::" << mpTargetFrame->GetKeyFrameID() <<"::"<<letime<<"||"<< vpKFs.size()<<", "<< vpFixedKFs.size()<<", "<<vpMPs.size();
			mpSystem->SetMapOptimizerString(ss.str());
			std::cout << "ba::end::" << mpTargetFrame->GetKeyFrameID() << std::endl;
			//종료
			SetDoingProcess(false);
		}
	}
}