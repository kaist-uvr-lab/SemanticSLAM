#include "LoopCloser.h"
#include "Frame.h"
#include "System.h"
#include "SemanticSegmentator.h"
#include "PlaneEstimator.h"
#include <KeyframeDatabase.h>
#include "Map.h"
#include <Sim3Solver.h>
#include <Matcher.h>

namespace UVR_SLAM {
	LoopCloser::LoopCloser() {}
	LoopCloser::LoopCloser(System* pSys, int w, int h, cv::Mat K) :mnWidth(w), mnHeight(h), mK(K), mbProcessing(false), mnThreshConsistency(3), mnLastLoopClosingID(0){
		mpSystem = pSys;
	}
	LoopCloser::~LoopCloser() {}
	void LoopCloser::Run() {

		while (1) {
			if (CheckNewKeyFrames()) {
				SetBoolProcessing(true);
				std::chrono::high_resolution_clock::time_point loop_start = std::chrono::high_resolution_clock::now();
				//std::cout << "LC::0" << std::endl;
				ProcessNewKeyFrame();
				
				if (mpMap->isAddedGraph()) {
					mpMap->UpdateGraphConnection();
				}
				double time_matching = 0.0;
				int nTargetID = mpTargetFrame->GetKeyFrameID();
				if (nTargetID < mnLastLoopClosingID + 15) {

					//////Local Loop Closing
					//auto vpKFs = mpMap->GetWindowFramesVector();
					//std::vector<cv::Point2f> vOpticalMatchPrevPts, vOpticalMatchCurrPts;
					//std::vector<CandidatePoint*> vOpticalMatchCPs;
					//if(vpKFs.size() > 10){
					//cv::Mat debugMatch;
					//mpMatcher->OpticalMatchingForLocalLoopClosing(mpTargetFrame, vpKFs[0], vOpticalMatchPrevPts, vOpticalMatchCurrPts, vOpticalMatchCPs, mK, mInvK, 21, time_matching, debugMatch);
					//}
					/////Local Loop Closing
				}else{
				mnLastLoopClosingID = nTargetID;

					/*if (mpMap->isAddedGraph()) {
						std::chrono::high_resolution_clock::time_point s1 = std::chrono::high_resolution_clock::now();
						mpMap->UpdateGraphConnection();
						std::chrono::high_resolution_clock::time_point s2 = std::chrono::high_resolution_clock::now();
						auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(s2 - s1).count();
						float tttt = duration / 1000.0;
						std::cout << "connection::time::" << tttt << std::endl;
					}*/
				
					//std::cout << "LC::1" << std::endl;
				
					if (DetectLoopFrame()) {
						if (ComputeSim3())
							CorrectLoop();
					}
				
				}
				//std::cout << "LC::2" << std::endl;
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
				//std::cout << "LC::3" << std::endl;
				
				std::chrono::high_resolution_clock::time_point loop_end = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();
				float time = duration / 1000.0;

				std::stringstream ss;
				ss << "Loop Closer::" << mpTargetFrame->GetKeyFrameID()<<"::"<<time<<"="<< time_matching;
				mpSystem->SetLoopCloserString(ss.str());

				SetBoolProcessing(false);
			}//visualize
		}
	}

	bool LoopCloser::DetectLoopFrame() {
		//std::cout << "Loop Closer::DetectLoop::Start" << std::endl;
		auto vpGraphKFs = mpMap->GetGraphFrames();
		if (vpGraphKFs.size() < 4)
			return false;
		
		auto vpKFs = mpMap->GetWindowFramesVector();

		float minScore = 1;
		for (auto iter = vpKFs.begin(), iter2 = vpKFs.end(); iter != iter2; iter++) {
			auto pKF = *iter;
			if (mpTargetFrame == pKF)
				continue;
			float score = mpTargetFrame->Score(pKF);
			if (score < minScore)
				minScore = score;
		}

		auto vpCandidateKFs = mpKeyFrameDatabase->DetectLoopCandidates(mpTargetFrame, minScore);

		if (vpCandidateKFs.empty()){
			mvConsistentGroups.clear();
			return false;
		}
		mvpEnoughConsistentCandidates.clear();
		std::vector<ConsistentGroup> vCurrentConsistentGroups;
		std::vector<bool> vbConsistentGroup(mvConsistentGroups.size(), false);
		for (size_t i = 0, iend = vpCandidateKFs.size(); i < iend; i++) {
			auto pCandidateKF = vpCandidateKFs[i];
			auto spCandidateGroup = pCandidateKF->GetConnectedKFsSet();
			spCandidateGroup.insert(pCandidateKF);

			bool bEnoughConsistent = false;
			bool bConsistentForSomeGroup = false;
			for (size_t iG = 0, iendG = mvConsistentGroups.size(); iG < iendG; iG++) {
				auto sPreviousGroup = mvConsistentGroups[iG].first;

				bool bConsistent = false;
				for (auto sit = spCandidateGroup.begin(), send = spCandidateGroup.end(); sit != send; sit++)
				{
					if (sPreviousGroup.count(*sit))
					{
						bConsistent = true;
						bConsistentForSomeGroup = true;
						break;
					}
				}

				if (bConsistent)
				{
					int nPreviousConsistency = mvConsistentGroups[iG].second;
					int nCurrentConsistency = nPreviousConsistency + 1;
					if (!vbConsistentGroup[iG])
					{
						ConsistentGroup cg = make_pair(spCandidateGroup, nCurrentConsistency);
						vCurrentConsistentGroups.push_back(cg);
						vbConsistentGroup[iG] = true; //this avoid to include the same group more than once
					}
					if (nCurrentConsistency >= mnThreshConsistency && !bEnoughConsistent)
					{
						mvpEnoughConsistentCandidates.push_back(pCandidateKF);
						bEnoughConsistent = true; //this avoid to insert the same candidate more than once
					}
				}
			}//for ig

			if (!bConsistentForSomeGroup)
			{
				ConsistentGroup cg = make_pair(spCandidateGroup, 0);
				vCurrentConsistentGroups.push_back(cg);
			}

		}//for vpcandiate

		mvConsistentGroups = vCurrentConsistentGroups;
		if (mvpEnoughConsistentCandidates.empty())
		{
			return false;
		}
		/*cv::imshow("Loop Frame", mvpEnoughConsistentCandidates[0]->GetOriginalImage()); cv::waitKey(1);
		std::cout << "Detect Loop Frame ::" << mvpEnoughConsistentCandidates.size() << std::endl;*/
		return true;
	}

	bool LoopCloser::ComputeSim3() {

		int nInitialCandidates = mvpEnoughConsistentCandidates.size();

		std::vector<Sim3Solver*> vpSim3Solvers;
		vpSim3Solvers.resize(nInitialCandidates);

		std::vector<std::vector<MapPoint*> > vvpMapPointMatches;
		vvpMapPointMatches.resize(nInitialCandidates);

		std::vector<bool> vbDiscarded;
		vbDiscarded.resize(nInitialCandidates);

		int nCandidates = 0; //candidates with enough matches

		for (int i = 0; i < nInitialCandidates; i++)
		{
			auto pKF = mvpEnoughConsistentCandidates[i];
			//match
			std::vector<cv::Point2f> vOpticalMatchPrevPts, vOpticalMatchCurrPts;
			std::vector<CandidatePoint*> vOpticalMatchCPs;
			double time1 = 0.0;
			cv::Mat debugMatch;
			//int nMatch = mpMatcher->OpticalMatchingForLoopClosing(mpTargetFrame, pKF, vOpticalMatchPrevPts, vOpticalMatchCurrPts, vOpticalMatchCPs, mK, mInvK, 21, time1, debugMatch);
			mpMatcher->OpticalMatchingForLocalLoopClosing(mpTargetFrame, pKF, vOpticalMatchPrevPts, vOpticalMatchCurrPts, vOpticalMatchCPs, mK, mInvK, 21, time1, debugMatch);
			if(i == 0){
				cv::imshow("Loop::Match", debugMatch); cv::waitKey(1);
				break;
			}

		}

		return false;
	}
	void LoopCloser::CorrectLoop(){
		std::cout << "Correct Loop!!" << std::endl;
	}

	void LoopCloser::Init() {
		mpMap = mpSystem->mpMap;
		mpKeyFrameDatabase = mpSystem->mpKeyframeDatabase;
		mpSegmentator = mpSystem->mpSegmentator;
		mpPlaneEstimator = mpSystem->mpPlaneEstimator;
		mpMatcher = mpSystem->mpMatcher;
		mK = mpSystem->mK.clone();
		mInvK = mpSystem->mInvK.clone();
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
		if (mpTargetFrame->matDescriptor.rows == 0) {
			//std::cout << "Feature::Start" << std::endl;
			mpTargetFrame->DetectFeature();
			//std::cout << "Feature::1" << std::endl;
			mpTargetFrame->SetBowVec(mpSystem->fvoc);
			//std::cout << "Feature::End" << std::endl;
		}
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