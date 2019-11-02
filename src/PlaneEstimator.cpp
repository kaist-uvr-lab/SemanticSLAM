#include <PlaneEstimator.h>
#include <System.h>
#include <Frame.h>
#include <FrameWindow.h>
#include <SegmentationData.h>

UVR_SLAM::PlaneEstimator::PlaneEstimator() :mbDoingProcess(false) {
}
UVR_SLAM::PlaneEstimator::PlaneEstimator(int w, int h) : mbDoingProcess(false), mnWidth(w), mnHeight(h) {
}
UVR_SLAM::PlaneEstimator::~PlaneEstimator() {}

void UVR_SLAM::PlaneEstimator::Run() {

	while (1) {

		if (isDoingProcess()) {
			std::cout << "PlaneEstimator::RUN::Start" << std::endl;

			cv::waitKey(10);
			SetBoolDoingProcess(false);
			std::cout << "PlaneEstimator::RUN::End" << std::endl;
		}
	}
}
void UVR_SLAM::PlaneEstimator::SetSystem(System* pSystem) {
	mpSystem = pSystem;
}
void UVR_SLAM::PlaneEstimator::SetFrameWindow(UVR_SLAM::FrameWindow* pWindow) {
	mpFrameWindow = pWindow;
}
void UVR_SLAM::PlaneEstimator::SetTargetFrame(Frame* pFrame) {
	mpTargetFrame = pFrame;
}
void UVR_SLAM::PlaneEstimator::SetBoolDoingProcess(bool b) {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	mbDoingProcess = b;
}
bool UVR_SLAM::PlaneEstimator::isDoingProcess() {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	return mbDoingProcess;
}