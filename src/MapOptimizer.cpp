#include <MapOptimizer.h>
#include <System.h>
#include <Frame.h>
#include <FrameWindow.h>
#include <MapPoint.h>
#include <Optimization.h>
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
	mpSystem->SetMapOptimizerID(mpTargetFrame->GetKeyFrameID());
	mKFQueue.pop();
}

void UVR_SLAM::MapOptimizer::Run() {
	std::string mStrPath;
	while (1) {
		if (CheckNewKeyFrames()) {
			SetDoingProcess(true);
			std::chrono::high_resolution_clock::time_point s_start = std::chrono::high_resolution_clock::now();
			ProcessNewKeyFrame();

			//local map
			//local map 과 연결된 키프레임
			//해당 키프레임에 연결된 mp와 fixed frame설정

			std::cout << "ba::start::" << mpTargetFrame->GetFrameID() << std::endl;
			mStrPath = mpSystem->GetDirPath(mpTargetFrame->GetKeyFrameID());
			StopBA(false);
			std::cout << "ba::1" << std::endl;
			/*if(mpMap->isFloorPlaneInitialized())
				Optimization::LocalBundleAdjustmentWithPlane(mpMap,mpTargetFrame, mpFrameWindow, &mbStopBA);
			else*/
			Optimization::OpticalLocalBundleAdjustment(this, mpTargetFrame, mpFrameWindow);
			std::cout << "ba::2" << std::endl;
			std::chrono::high_resolution_clock::time_point s_end = std::chrono::high_resolution_clock::now();
			auto leduration = std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_start).count();
			float letime = leduration / 1000.0;
			mpSystem->SetMapOptimizerTime(letime);
			std::cout << "ba::end::" << mpTargetFrame->GetKeyFrameID() << std::endl;
			//종료
			SetDoingProcess(false);
		}
	}
}