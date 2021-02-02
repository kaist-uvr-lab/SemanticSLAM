#include "LoopCloser.h"
#include "Frame.h"
#include "System.h"
#include <Matcher.h>
#include <KeyframeDatabase.h>
#include "Map.h"

namespace UVR_SLAM {
	LoopCloser::LoopCloser() {}
	LoopCloser::LoopCloser(System* pSys, int w, int h, cv::Mat K):mpSystem(pSys), mnWidth(w), mnHeight(h), mK(K), mbProcessing(false){
	}
	LoopCloser::~LoopCloser() {}
	void LoopCloser::Init() {
		mpMap = mpSystem->mpMap;
		mpKeyFrameDatabase = mpSystem->mpKeyframeDatabase;
		mpMatcher = mpSystem->mpMatcher;
		mK = mpSystem->mK.clone();
		mInvK = mpSystem->mInvK.clone();
	}
	void LoopCloser::Run() {

		while (false) {
			if (CheckNewKeyFrames()) {
				SetBoolProcessing(true);
				std::cout << "LOOP::start" << std::endl;
				ProcessNewKeyFrame();

				///////////////VoW ¸ÅÄª
				//auto vpGrahWindows = mpMap->GetGraphFrames();
				//for (int i = 0; i < vpGrahWindows.size(); i++) {
				//	auto pKFi = vpGrahWindows[i];
				//	auto score = mpTargetFrame->Score(pKFi);
				//	
				//	if (score > 0.01) {
				//		std::cout << "Loop::Score::" << score << std::endl;
				//		imshow("Loop::1", mpTargetFrame->GetOriginalImage());
				//		imshow("Loop::2", pKFi->GetOriginalImage());
				//		cv::waitKey(500);
				//	}
				//}
				///////////////VoW ¸ÅÄª
				std::cout << "LOOP::Add" << std::endl;
				mpKeyFrameDatabase->Add(mpTargetFrame);
				std::cout << "LOOP::end" << std::endl;
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
		mpTargetFrame->SetBowVec(mpSystem->fvoc);
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