#include <MapOptimizer.h>
#include <System.h>
#include <Frame.h>
#include <MapPoint.h>
#include <CandidatePoint.h>

#include <Optimization.h>
#include <PlanarOptimization.h>

#include <Plane.h>
#include <Map.h>

#include <SemanticSegmentator.h>
#include <PlaneEstimator.h>
#include <Visualizer.h>
#include <LoopCloser.h>

UVR_SLAM::MapOptimizer::MapOptimizer(std::string strPath, System* pSystem) : mpTargetFrame(nullptr), mbStopBA(false), mbDoingProcess(false)
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

	mpSystem = pSystem;
	
}
UVR_SLAM::MapOptimizer::~MapOptimizer() {}

void UVR_SLAM::MapOptimizer::Init() {
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

	int mnThreshMinKF = mpSystem->mnThreshMinKF;

	while (1) {
		if (CheckNewKeyFrames()) {
			SetDoingProcess(true);
			std::chrono::high_resolution_clock::time_point s_start = std::chrono::high_resolution_clock::now();
			ProcessNewKeyFrame();
			mStrPath = mpSystem->GetDirPath(mpTargetFrame->mnKeyFrameID);
			StopBA(false);
			auto currMatchInfo = mpTargetFrame->mpMatchInfo;
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
			auto vpGraphKFs = mpMap->GetGraphFrames();

			
			std::map<MapPoint*, int> mpMapPointCounts;
			std::map<Frame*, int> mpKeyFrameCounts, mpGraphFrameCounts;

			std::chrono::high_resolution_clock::time_point s_start2 = std::chrono::high_resolution_clock::now();
			{

				////기본 MP획득
				for (size_t i = 0, iend = mpTargetFrame->mpMatchInfo->mvpMatchingCPs.size(); i < iend; i++) {
					auto pCPi = mpTargetFrame->mpMatchInfo->mvpMatchingCPs[i];
					auto pMPi = pCPi->GetMP();
					if (!pMPi || pMPi->isDeleted() || !pMPi->GetQuality() || pMPi->GetNumConnectedFrames() < mnThreshMinKF)
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
					auto matchInfo = pKFi->mpMatchInfo;
					auto vpCPs = matchInfo->mvpMatchingCPs;
					auto vPTs = matchInfo->mvMatchingPts;
					int N1 = 0;
					for (size_t i = 0, iend = vpCPs.size(); i < iend; i++) {
						auto pCPi = vpCPs[i];
						auto pMPi = pCPi->GetMP();
						if (!pMPi || pMPi->isDeleted() || pMPi->GetNumConnectedFrames() < mnThreshMinKF || pMPi->mnLocalBAID == nTargetID || !pMPi->GetQuality())
							continue;
						N1++;
						pMPi->mnLocalBAID = nTargetID;
						vpOptMPs.push_back(pMPi);
					}
					vpOptKFs.push_back(pKFi);
					pKFi->mnLocalBAID = nTargetID;
				}//for vpmps, vpkfs

				//Fixed KFs
				for (size_t i = 0, iend = vpOptMPs.size(); i < iend; i++) {
					auto pMPi = vpOptMPs[i];
					auto mmpFrames = pMPi->GetConnedtedFrames();
					for (auto iter = mmpFrames.begin(), iter_end = mmpFrames.end(); iter != iter_end; iter++) {
						auto pKFi = (iter->first)->mpRefFrame;
						if (pKFi->mnLocalBAID == nTargetID)
							continue;
						pKFi->mnLocalBAID = nTargetID;
						vpFixedKFs.push_back(pKFi);
					}
				}//for fixed iter
			}

			//{
			//	////기본 MP획득
			//	for (size_t i = 0, iend = mpTargetFrame->mpMatchInfo->mvpMatchingCPs.size(); i < iend; i++) {
			//		auto pCPi = mpTargetFrame->mpMatchInfo->mvpMatchingCPs[i];
			//		auto pMPi = pCPi->GetMP();
			//		if (!pMPi || pMPi->isDeleted() || !pMPi->GetQuality() || pMPi->GetNumConnectedFrames() < mnThreshMinKF)
			//			continue;
			//		if (pMPi->mnLocalBAID == nTargetID) {
			//			continue;
			//		}
			//		pMPi->mnLocalBAID = nTargetID;
			//		vpOptMPs.push_back(pMPi);
			//	}
			//	mpTargetFrame->mnLocalBAID = nTargetID;
			//	vpOptKFs.push_back(mpTargetFrame);

			//	////MP 획득 및 redundant keyframe 찾기
			//	auto mKeyframeCount = mpTargetFrame->mmKeyFrameCount;
			//	if (mKeyframeCount.count(mpTargetFrame))
			//		std::cout << "BA::error::keyframe" << std::endl;
			//	for (auto iter = mKeyframeCount.begin(), iend = mKeyframeCount.end(); iter != iend; iter++) {
			//		auto pKFi = iter->first;
			//		
			//		std::vector<MapPoint*> vpNeighMPs;
			//		auto matchInfo = pKFi->mpMatchInfo;
			//		auto vpCPs = matchInfo->mvpMatchingCPs;
			//		auto vPTs = matchInfo->mvMatchingPts;
			//		int N1 = 0;
			//		int N2 = 0;
			//		for (size_t i = 0, iend = vpCPs.size(); i < iend; i++) {
			//			auto pCPi = vpCPs[i];
			//			auto pMPi = pCPi->GetMP();
			//			if (!pMPi || pMPi->isDeleted() || pMPi->GetNumConnectedFrames() < mnThreshMinKF || !pMPi->GetQuality())
			//				continue;
			//			N1++;
			//			if (pMPi->GetNumConnectedFrames() > mnThreshMinKF)
			//				N2++;

			//			if (pMPi->mnLocalBAID == nTargetID ) {
			//				continue;
			//			}
			//			pMPi->mnLocalBAID = nTargetID;
			//			vpOptMPs.push_back(pMPi);
			//		}
			//		bool bErase = false;
			//		if (N2 > N1*0.9){
			//			std::cout << "Delete::KF::" << N1 << ", " << N2 << std::endl;
			//			pKFi->Delete();
			//			bErase = true;
			//			continue;
			//		}
			//		/*else {
			//			std::cout << "BA::ratio = " << N1 << ", " << N2 << std::endl;
			//		}*/
			//		
			//		auto count = iter->second;
			//		if (count < 15)
			//			continue;
			//		vpOptKFs.push_back(pKFi);
			//		pKFi->mnLocalBAID = nTargetID;

			//	}//for kf iter
			//	
			//	//Fixed KFs
			//	for (size_t i = 0, iend = vpOptMPs.size(); i < iend; i++) {
			//		auto pMPi = vpOptMPs[i];
			//		auto mmpFrames = pMPi->GetConnedtedFrames();
			//		for (auto iter = mmpFrames.begin(), iter_end = mmpFrames.end(); iter != iter_end; iter++) {
			//			auto pKFi = (iter->first)->mpRefFrame;
			//			if (pKFi->mnLocalBAID == nTargetID)
			//				continue;
			//			pKFi->mnLocalBAID = nTargetID;
			//			vpFixedKFs.push_back(pKFi);
			//		}
			//	}//for fixed iter
			//	std::cout << "BA::" <<mpTargetFrame->mnKeyFrameID<<"="<<vpOptMPs.size()<<", "<<vpOptKFs.size()<<", "<<vpFixedKFs.size()<< std::endl;
			//}


			//auto lpKFs = mpMap->GetWindowFramesVector();
			//for (auto iter = lpKFs.begin(); iter != lpKFs.end(); iter++) {
			//	auto pKF = *iter;
			//	
			//	spTempKFs.insert(pKF);
			//	vpTempKFs.push_back(pKF);
			//}
			//for (auto iter = vpGraphKFs.begin(); iter != vpGraphKFs.end(); iter++) {
			//	auto pKF = *iter;
			//	spGraphFrames.insert(pKF);
			//}

			//for (size_t k = 0, kend= vpTempKFs.size(); k < kend; k++){
			//	auto pKFi = vpTempKFs[k];
			//	auto matchInfo = pKFi->mpMatchInfo;
			//	auto vpCPs = matchInfo->mvpMatchingCPs;
			//	auto vPTs = matchInfo->mvMatchingPts;
	
			//	for (size_t i = 0, iend = vpCPs.size(); i < iend; i++){
			//		auto pCPi = vpCPs[i];
			//		auto pMPi = pCPi->GetMP();
			//		if (!pMPi || pMPi->isDeleted() || pMPi->mnLocalBAID == nTargetID || !pMPi->GetQuality() || pMPi->GetNumConnectedFrames() < mnThreshMinKF) {
			//			continue;
			//		}
			//		////퀄리티 체크
			//		/*pMPi->ComputeQuality();
			//		if (!pMPi->GetQuality()){
			//			pMPi->Delete();
			//			continue;
			//		}*/
			//		////퀄리티 체크
			//		pMPi->mnLocalBAID = nTargetID;
			//		vpTempMPs.push_back(pMPi);
			//	}
			//}
			//
			///////////////////////////////////////////////////////////
			//////데이터 취득 과정
			//for (size_t i = 0, iend = vpTempMPs.size(); i < iend; i++) {
			//	auto pMPi = vpTempMPs[i];
			//	auto mmpFrames = pMPi->GetConnedtedFrames();

			//	if (mpMapPointCounts[pMPi]) {
			//		std::cout << "Map::error" << std::endl;
			//	}
			//	for (auto iter = mmpFrames.begin(), iter_end = mmpFrames.end(); iter != iter_end; iter++) {
			//		auto pKFi = (iter->first)->mpRefFrame;
			//		mpMapPointCounts[pMPi]++;
			//		if (spGraphFrames.count(pKFi)) {
			//			mpGraphFrameCounts[pKFi]++;
			//		}
			//		if (spTempKFs.count(pKFi)){
			//			mpKeyFrameCounts[pKFi]++;
			//		}
			//	}
			//}

			//for (auto iter = mpMapPointCounts.begin(), iend = mpMapPointCounts.end(); iter != iend; iter++) {
			//	auto pMPi = iter->first;
			//	auto count = iter->second;
			//	if (count < mnThreshMinKF)
			//		continue;
			//	vpOptMPs.push_back(pMPi);
			//}

			//for (auto iter = mpKeyFrameCounts.begin(), iend = mpKeyFrameCounts.end(); iter != iend; iter++) {
			//	auto pKFi = iter->first;
			//	auto count = iter->second;
			//	if (count < 10)
			//		continue;
			//	vpOptKFs.push_back(pKFi);
			//	pKFi->mnLocalBAID = nTargetID;
			//}

			//for (auto iter = mpGraphFrameCounts.begin(), iend = mpGraphFrameCounts.end(); iter != iend; iter++) {
			//	auto pKFi = iter->first;
			//	auto count = iter->second;
			//	if (count < 10)
			//		continue;
			//	pKFi->mnFixedBAID = nTargetID;
			//	pKFi->mnLocalBAID = nTargetID;
			//	vpFixedKFs.push_back(pKFi);
			//}

			//std::chrono::high_resolution_clock::time_point s_end2 = std::chrono::high_resolution_clock::now();
			//auto leduration2 = std::chrono::duration_cast<std::chrono::milliseconds>(s_end2 - s_start2).count();
			//float letime2 = leduration2 / 1000.0;
			//////데이터 취득 과정
			///////////////////////////////////////////////////////////

			auto vpPlaneInfos = mpMap->GetPlaneInfos();
			int n = vpPlaneInfos.size() - 1;
			if(n < 0)
				Optimization::OpticalLocalBundleAdjustment(mpMap, this, vpOptMPs, vpOptKFs, vpFixedKFs);
			else
			{
				std::cout << "Plane optimization" << std::endl;
				PlanarOptimization::OpticalLocalBundleAdjustmentWithPlane(this, vpPlaneInfos[n], vpOptMPs, vpOptKFs, vpFixedKFs);
			}
			///////KF 이미지 시각화1
			{
				
				//int nRows = mpVisualizer->mnWindowImgRows;
				//int nCols = mpVisualizer->mnWindowImgCols;
				//cv::Mat ba_img = cv::Mat::zeros(mnHeight*nRows, mnWidth*nCols, CV_8UC3);
				//int nidx = 0;

				//int nKF = lpKFs.size();
				//auto lastKF = vpOptKFs[nKF - 1];
				//auto lastMatch = lastKF->mpMatchInfo;
				//
				//cv::Scalar color1(0, 0, 255);
				//cv::Scalar color2(0, 255, 0);
				//cv::Scalar color3(255, 0, 0);
				//cv::Scalar color4(255, 255, 0);
				//cv::Scalar color5(0, 255, 255);
				//cv::Scalar color6(255, 0, 255);

				//for (int i = 0; i < nKF; i++) {
				//	auto pKFi = vpOptKFs[i];
				//	auto pMatch = pKFi->mpMatchInfo;
				//	cv::Mat img = pKFi->GetOriginalImage().clone();

				//	cv::Mat R, t;
				//	pKFi->GetPose(R, t);
				//	
				//	for (size_t j = 0, jend = pMatch->mvpMatchingCPs.size(); j < jend; j++){
				//		auto pCPi = pMatch->mvpMatchingCPs[j];
				//		auto pt = pMatch->mvMatchingPts[j];
				//		auto pMPi = pCPi->GetMP();
				//		if (!pMPi || pMPi->isDeleted() || !pMPi->GetQuality())
				//			continue;
				//		if (!pMPi->isInFrame(pMatch)) {
				//			continue;
				//		}
				//		cv::Point2f pt3;
				//		pMPi->Projection(pt3, pKFi, mnWidth, mnHeight);
				//		cv::Point2f diffPt = pt3 - pt;
				//		float dist = diffPt.dot(diffPt);
				//		if (dist > 9.0) {
				//			cv::circle(img, pt, 6, color6, 1);
				//			pMPi->DisconnectFrame(pMatch);
				//		}
				//		if(pMPi->mnFirstKeyFrameID == nTargetKeyID)
				//			cv::circle(img, pt, 4, color3, 2);
				//		
				//		if (pMPi->isInFrame(lastMatch)) {
				//			cv::line(img, pt, pt3, color1, 2);
				//			cv::circle(img, pt, 3, color4, -1);
				//		}
				//		else {
				//			cv::line(img, pt, pt3, color2, 2);
				//			cv::circle(img, pt, 3, color5, -1);
				//		}
				//	}
				//	/*for (int j = 0; j < vpMPs.size(); j++) {
				//		auto pMPj = vpMPs[j];
				//		if (!pMPj || pMPj->isDeleted())
				//			continue;
				//		int idx = pMPj->GetPointIndexInFrame(pMatch);
				//		if (idx < 0)
				//			continue;
				//		auto pt = pMatch->GetPt(idx);
				//		cv::circle(img, pt, 2, cv::Scalar(0, 0, 255), -1);
				//	}*/

				//	int h = nidx / nCols;
				//	int w = nidx % nCols;
				//	
				//	cv::Rect tmpRect(mnWidth*w, mnHeight*h, mnWidth, mnHeight);
				//	img.copyTo(ba_img(tmpRect));
				//	nidx++;

				//}
				//cv::resize(ba_img, ba_img, ba_img.size() / 2);
				//mpVisualizer->SetOutputImage(ba_img, 5);
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