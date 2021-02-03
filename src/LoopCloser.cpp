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
		mpVoc = mpSystem->mpDBoWVoc;
		mpMatcher = mpSystem->mpMatcher;
		mK = mpSystem->mK.clone();
		mInvK = mpSystem->mInvK.clone();
	}
	void LoopCloser::Run() {

		while (true) {
			if (CheckNewKeyFrames()) {
				SetBoolProcessing(true);
				ProcessNewKeyFrame();

				if (DetectLoop()) {
					if (ComputeSim3())
					{
						CorrectLoop();
					}
				}

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
				mpKeyFrameDatabase->Add(mpTargetFrame);
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
		mpTargetFrame->ComputeBoW();
	}
	bool LoopCloser::isProcessing() {
		std::unique_lock<std::mutex> lock(mMutexProcessing);
		return mbProcessing;
	}
	void LoopCloser::SetBoolProcessing(bool b) {
		std::unique_lock<std::mutex> lock(mMutexProcessing);
		mbProcessing = b;
	}

	bool LoopCloser::DetectLoop() {

		std::vector<Frame*> vpKFs = mpTargetFrame->GetConnectedKFs();
		float minScore = 1;
		const DBoW3::BowVector &CurrentBowVec = mpTargetFrame->mBowVec;
		for (auto iter = vpKFs.begin(), iter2 = vpKFs.end(); iter != iter2; iter++) {
			auto pKF = *iter;
			if (mpTargetFrame == pKF)
				continue;
			const DBoW3::BowVector &BowVec = pKF->mBowVec;
			float score = mpVoc->score(CurrentBowVec, BowVec);
			if (score < minScore)
				minScore = score;
		}

		auto vpCandidateKFs = mpKeyFrameDatabase->DetectLoopCandidates(mpTargetFrame, minScore);
		
		
		return false;
	}
	bool LoopCloser::ComputeSim3() {
		return false;
	}
	void LoopCloser::CorrectLoop() {
		
	}
	void LoopCloser::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap) {

	}
}