#include "LoopCloser.h"
#include "Frame.h"
#include "System.h"
#include "Map.h"

namespace UVR_SLAM {
	LoopCloser::LoopCloser() {}
	LoopCloser::LoopCloser(int w, int h, cv::Mat K, Map* pMap) :mnWidth(w), mnHeight(h), mK(K), mbProcessing(false){
		mpMap = pMap;
	}
	LoopCloser::~LoopCloser() {}
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
					std::cout << "Loop::" << score << std::endl;
					if (score < 0.01) {
						imshow("Loop!!", pKFi->GetOriginalImage());
						cv::waitKey(1);
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
	void LoopCloser::SetSystem(UVR_SLAM::System* pSystem) {
		mpSystem = pSystem;
	}
	void LoopCloser::SetBoolProcessing(bool b) {
		std::unique_lock<std::mutex> lock(mMutexProcessing);
		mbProcessing = b;
	}
}