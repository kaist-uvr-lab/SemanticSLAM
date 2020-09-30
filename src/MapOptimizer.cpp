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
	while (1) {
		if (CheckNewKeyFrames()) {
			SetDoingProcess(true);
			std::chrono::high_resolution_clock::time_point s_start = std::chrono::high_resolution_clock::now();
			ProcessNewKeyFrame();
			//std::cout << "ba::start::" << mpTargetFrame->GetFrameID() << std::endl;
			mStrPath = mpSystem->GetDirPath(mpTargetFrame->GetKeyFrameID());
			StopBA(false);
			auto currMatchInfo = mpTargetFrame->mpMatchInfo;
			auto targetFrame = mpTargetFrame;
			///////////////////////////////////////////////////////////////
			////preprocessing
			//std::cout << "ba::processing::start" << std::endl;
			int nTargetID = mpTargetFrame->GetFrameID();
			int nTargetKeyID = mpTargetFrame->GetKeyFrameID();
			mpTargetFrame->mnLocalBAID = nTargetID;
			//std::cout << "BA::preprocessing::start" << std::endl;
			std::chrono::high_resolution_clock::time_point temp_1 = std::chrono::high_resolution_clock::now();
			std::vector<UVR_SLAM::MapPoint*> vpMPs;// , vpMPs2;
			std::vector<UVR_SLAM::CandidatePoint*> vpCPs;
			std::vector<UVR_SLAM::Frame*> vpKFs;
			std::vector<UVR_SLAM::Frame*> vpFixedKFs;
			
			auto lpKFs = mpMap->GetWindowFramesVector();
			for (auto iter = lpKFs.begin(); iter != lpKFs.end(); iter++) {
				auto pKF = *iter;
				pKF->mnLocalBAID = nTargetID;
				vpKFs.push_back(pKF);
			}

			//auto tempKFs1 = mpTargetFrame->GetConnectedKFs();
			////auto tempKFs2 = mpTargetFrame->GetConnectedKFs();
			//vpKFs.push_back(mpTargetFrame);
			//
			//for (int i = 0; i < tempKFs1.size(); i++){
			//	tempKFs1[i]->mnLocalBAID = nTargetID;
			//	vpKFs.push_back(tempKFs1[i]);
			//}
			
			for (int k = 0; k < vpKFs.size(); k++){
				auto pKFi = vpKFs[k];
				auto matchInfo = pKFi->mpMatchInfo;
				std::vector<MapPoint*> mvpMatchingMPs;
				std::vector<CandidatePoint*> mvpMatchingCPs;
				auto mvpMatchingPTs = matchInfo->GetMatchingPtsOptimization(mvpMatchingCPs, mvpMatchingMPs);
				for (int i = 0; i < mvpMatchingMPs.size(); i++) {
					auto pMPi = mvpMatchingMPs[i];
					if (!pMPi || pMPi->isDeleted() || pMPi->mnLocalBAID == nTargetID) {
						continue;
					}
					if (pMPi->GetNumConnectedFrames() < 3) {
						continue;
					}
					auto pCPi = mvpMatchingCPs[i];
					pMPi->mnLocalBAID = nTargetID;
					vpMPs.push_back(pMPi);
					vpCPs.push_back(pCPi);
				}
			}

			auto spGraphKFs = mpMap->GetGraphFrames();
			for (auto iter = spGraphKFs.begin(); iter != spGraphKFs.end(); iter++) {
				auto pKFi = *iter;
				pKFi->mnFixedBAID = nTargetID;
				pKFi->mnLocalBAID = nTargetID;
				vpFixedKFs.push_back(pKFi);
			}

			auto vpPlaneInfos = mpMap->GetPlaneInfos();
			int n = vpPlaneInfos.size() - 1;
			if(n < 0)
				Optimization::OpticalLocalBundleAdjustment(this, vpMPs, vpKFs, vpFixedKFs);
			else
			{
				std::cout << "Plane optimization" << std::endl;
				PlanarOptimization::OpticalLocalBundleAdjustmentWithPlane(this, vpPlaneInfos[n], vpMPs, vpKFs, vpFixedKFs);
			}

			for (int i = 0; i < vpCPs.size(); i++)
			{
				vpCPs[i]->SetOptimization(true);
			}

			///////KF 이미지 시각화
			{
				
				int nRows = mpVisualizer->mnWindowImgRows;
				int nCols = mpVisualizer->mnWindowImgCols;
				cv::Mat ba_img = cv::Mat::zeros(mnHeight*nRows, mnWidth*nCols, CV_8UC3);
				int nidx = 0;

				int nKF = lpKFs.size();
				auto lastKF = vpKFs[nKF - 1];
				auto lastMatch = lastKF->mpMatchInfo;
				//std::vector<MapPoint*> vpMPs;
				//auto vPTs = lastKF->mpMatchInfo->GetMatchingPts(vpMPs);

				cv::Scalar color1(0, 0, 255);
				cv::Scalar color2(0, 255, 0);
				cv::Scalar color3(255, 0, 0);

				for (int i = 0; i < nKF; i++) {
					auto pKFi = vpKFs[i];
					auto pMatch = pKFi->mpMatchInfo;
					cv::Mat img = pKFi->GetOriginalImage();

					cv::Mat R, t;
					pKFi->GetPose(R, t);

					int nCP = pMatch->GetNumCPs();
					for (int j = 0; j < nCP; j++) {
						auto pCPi = pMatch->mvpMatchingCPs[j];
						auto pt = pMatch->mvMatchingPts[j];
						auto pMPi = pCPi->GetMP();
						if (!pMPi || pMPi->isDeleted() || !pCPi->GetQuality())
							continue;
						cv::Point2f pt3;
						pMPi->Projection(pt3, pKFi, mnWidth, mnHeight);
						if(pMPi->mnFirstKeyFrameID == nTargetKeyID)
							cv::circle(img, pt, 4, color3, 2);
						if (pMPi->isInFrame(lastMatch)) {
							cv::circle(img, pt, 2, color1, -1);
							cv::line(img, pt, pt3, color1, 2);
						}
						else {
							cv::circle(img, pt, 2, color2, -1);
							cv::line(img, pt, pt3, color2, 2);
						}
					}
					/*for (int j = 0; j < vpMPs.size(); j++) {
						auto pMPj = vpMPs[j];
						if (!pMPj || pMPj->isDeleted())
							continue;
						int idx = pMPj->GetPointIndexInFrame(pMatch);
						if (idx < 0)
							continue;
						auto pt = pMatch->GetPt(idx);
						cv::circle(img, pt, 2, cv::Scalar(0, 0, 255), -1);
					}*/

					int h = nidx / nCols;
					int w = nidx % nCols;
					
					cv::Rect tmpRect(mnWidth*w, mnHeight*h, mnWidth, mnHeight);
					img.copyTo(ba_img(tmpRect));
					nidx++;

				}
				cv::resize(ba_img, ba_img, ba_img.size() / 2);
				mpVisualizer->SetOutputImage(ba_img, 1);
			}
			
			/*std::cout << "BA::Delete::Start" << std::endl;
			mpMap->DeleteMPs();
			std::cout << "BA::Delete::End" << std::endl;*/

			std::chrono::high_resolution_clock::time_point s_end = std::chrono::high_resolution_clock::now();
			auto leduration = std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_start).count();
			float letime = leduration / 1000.0;
			std::stringstream ss;
			ss << "Map Optimizer::" << mpTargetFrame->GetKeyFrameID() <<", "<<vpKFs[0]->GetKeyFrameID()<<"::"<<letime<<"||"<< vpKFs.size()<<", "<< vpFixedKFs.size()<<", "<<vpMPs.size();
			mpSystem->SetMapOptimizerString(ss.str());
			//종료
			SetDoingProcess(false);
		}
	}
}