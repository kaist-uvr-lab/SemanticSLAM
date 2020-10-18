#include "LoopCloser.h"
#include "Frame.h"
#include "System.h"
#include "Map.h"

namespace UVR_SLAM {
	LoopCloser::LoopCloser() {}
	LoopCloser::LoopCloser(System* pSys, int w, int h, cv::Mat K):mpSystem(pSys), mnWidth(w), mnHeight(h), mK(K), mbProcessing(false){
	}
	LoopCloser::~LoopCloser() {}
	void LoopCloser::Init() {
		mpMap = mpSystem->mpMap;
	}
	void LoopCloser::Run() {

		while (1) {
			if (CheckNewKeyFrames()) {
				SetBoolProcessing(true);
				ProcessNewKeyFrame();

				/////////////VoW ¸ÅÄª
				auto vpGrahWindows = mpMap->GetGraphFrames();
				for (int i = 0; i < vpGrahWindows.size(); i++) {
					auto pKFi = vpGrahWindows[i];
					auto score = mpTargetFrame->Score(pKFi);
					
					if (score > 0.01) {
						std::cout << "Loop::Score::" << score << std::endl;
						imshow("Loop::1", mpTargetFrame->GetOriginalImage());
						imshow("Loop::2", pKFi->GetOriginalImage());
						cv::waitKey(500);
					}
				}
				/////////////VoW ¸ÅÄª

				SetBoolProcessing(false);
			}//visualize
		}
	}
	void LoopCloser::InsertKeyFrame(UVR_SLAM::Frame *pKF)
	{
		std::unique_lock<std::mutex> lock(mMutexNewKFs);
		mKFQueue.push(pKF);
	}

	bool LoopCloser::CheckNewKeyFrames()
	{
		std::unique_lock<std::mutex> lock(mMutexNewKFs);
		return(!mKFQueue.empty());
	}

	void LoopCloser::ProcessNewKeyFrame()
	{
		std::unique_lock<std::mutex> lock(mMutexNewKFs);
		mpTargetFrame = mKFQueue.front();
		mKFQueue.pop();
	}
	bool LoopCloser::isProcessing() {
		std::unique_lock<std::mutex> lock(mMutexProcessing);
		return mbProcessing;
	}
	void LoopCloser::SetBoolProcessing(bool b) {
		std::unique_lock<std::mutex> lock(mMutexProcessing);
		mbProcessing = b;
	}
}