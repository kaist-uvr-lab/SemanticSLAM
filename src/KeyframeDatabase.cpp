#include <KeyframeDatabase.h>
#include <System.h>
#include <Map.h>
#include <Frame.h>

namespace UVR_SLAM {
	int nFail = 0;
	int nSuccess = 0;
	KeyframeDatabase::KeyframeDatabase(System* pSys, DBoW3::Vocabulary* voc) :mpVocabulary(voc){
		mpSystem = pSys;
		mvInvertedFile.resize(mpVocabulary->size());
	}
	KeyframeDatabase::~KeyframeDatabase() {}

	void KeyframeDatabase::Init() {
		mpMap = mpSystem->mpMap;
	}

	void KeyframeDatabase::Add(Frame* pKF) {
		std::unique_lock<std::mutex> lock(mMutex);
		for (auto iter = pKF->mBowVec.begin(); iter != pKF->mBowVec.end(); iter++) {
			if (iter->first >= mvInvertedFile.size()) {
				std::cout << "KFDB::Add::Fail::" << ++nFail <<", "<< nSuccess <<":"<<iter->first<<", "<<mvInvertedFile.size()<< std::endl;
				continue;
			}
			//std::cout << "adddata" << std::endl;
			nSuccess++;
			mvInvertedFile[iter->first].push_back(pKF);
		}
	}
	void KeyframeDatabase::Remove(Frame* pKF) {
		std::unique_lock<std::mutex> lock(mMutex);

		for (auto vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end(); vit != vend; vit++)
		{

			std::list<Frame*> &lKFs = mvInvertedFile[vit->first];
			for (auto lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
			{
				if (pKF == *lit)
				{
					lKFs.erase(lit);
					break;
				}
			}
		}
	}
	void KeyframeDatabase::Reset() {
		mvInvertedFile.clear();
		mvInvertedFile.resize(mpVocabulary->size());
	}
	std::vector<Frame*> KeyframeDatabase::DetectPlaceCandidates(Frame* pKF) {
		std::map<Frame*, int> KeyFrameCount;
		int nTargetID = pKF->mnFrameID;
		////키프레임 카운트
		////lock : mvInvertedFiles
		{
			std::unique_lock<std::mutex>(mMutex);
			auto mBowVec = pKF->mBowVec; 
			
			for (auto iter = mBowVec.begin(), iend = mBowVec.end(); iter != iend; iter++) {
				if (iter->first >= mvInvertedFile.size()) {
					std::cout << "KFDB::DetectLoopCandidates::BowVec::Fail::" << iter->first << ", " << mvInvertedFile.size() << std::endl;
					continue;
				}
				std::list<Frame*> &lKFs = mvInvertedFile[iter->first];
				for (auto liter = lKFs.begin(), lend = lKFs.end(); liter != lend; liter++)
				{
					auto pKFi = *liter;
					if (pKFi->mnFrameID == nTargetID)
						continue;
					KeyFrameCount[pKFi]++;
				}
			}
		}
		////lock
		////sort keyframecount
		std::vector<Frame*> res(0);
		std::vector<std::pair<int, Frame*>> vPairKFs;
		for (auto iter = KeyFrameCount.begin(), iend = KeyFrameCount.end(); iter != iend; iter++) {
			auto pKFi = iter->first;
			auto count = iter->second;
			if (count < 10)
				continue;
			vPairKFs.push_back(std::make_pair(count, pKFi));
		}
		if (vPairKFs.size() == 0) {
#ifdef PRINT_ERROR
			std::cout << "Place Recognition::No candidate KFs!!!" << std::endl;
#endif
			return res;
		}
		std::sort(vPairKFs.begin(), vPairKFs.end(), std::greater<>());
		
		for (size_t i = 0, iend = vPairKFs.size(); i < iend; i++) {
			
			auto pair = vPairKFs[i];
#ifdef DEBUG_LOOP_CLOSING_LEVEL_3
			std::cout << "KF :: " << i << ", " << pair.first <<"::"<<nTargetID<<", "<<pair.second->mnFrameID<< std::endl;
#endif
			res.push_back(pair.second);
			if (res.size() == 4)
				break;
		}
		return res;
	}
	std::vector<Frame*> KeyframeDatabase::DetectLoopCandidates(Frame* pKF, float minScore) {
		//std::cout << "DetectLoopCandidates::Start" << std::endl;
		auto spKFs = pKF->GetConnectedKeyFrameSet();
		std::list<Frame*> lpWordKFs;
		int mnLoopID = pKF->mnKeyFrameID;
		////lock
		{
			std::unique_lock<std::mutex>(mMutex);
			auto mBowVec = pKF->mBowVec;

			for (auto iter = mBowVec.begin(), iend = mBowVec.end(); iter != iend; iter++) {
				if (iter->first >= mvInvertedFile.size()) {
					std::cout << "KFDB::DetectLoopCandidates::BowVec::Fail::" << iter->first << ", " << mvInvertedFile.size() << std::endl;
					continue;
				}
				std::list<Frame*> &lKFs = mvInvertedFile[iter->first];
				for (auto liter = lKFs.begin(), lend = lKFs.end(); liter != lend; liter++)
				{
					auto pKFi = *liter;
					if (pKFi->mnLoopClosingID != mnLoopID)
					{
						pKFi->mnLoopBowWords = 0;
						if (!spKFs.count(pKFi))
						{
							pKFi->mnLoopClosingID = mnLoopID;
							lpWordKFs.push_back(pKFi);
						}
					}
					pKFi->mnLoopBowWords++;
				}

			}
		}
		////lock

		if (lpWordKFs.empty()) {
			//std::cout << "DetectLoopCandidates::End" << std::endl;
			return std::vector<Frame*>();
		}


		////가장 많은 워드가 겹치는 프레임을 찾기
		int nMaxWords = 0;
		for (auto iter = lpWordKFs.begin(), iend = lpWordKFs.end(); iter != iend; iter++) {
			auto pKFi = *iter;
			if (pKFi->mnLoopBowWords < nMaxWords)
				nMaxWords = pKFi->mnLoopBowWords;
		}

		int nMinRequireWords = nMaxWords*0.8f;
		std::list<std::pair<float, Frame*> > lPairTempKFs;
		for (auto iter = lpWordKFs.begin(), iend = lpWordKFs.end(); iter != iend; iter++) {
			auto pKFi = *iter;
			if (pKFi->mnLoopBowWords < nMinRequireWords)
				continue;
			float score = mpVocabulary->score(pKF->mBowVec, pKFi->mBowVec);
			pKFi->mfLoopScore = score;
			if (score >= minScore) {
				lPairTempKFs.push_back(std::make_pair(score, pKFi));
			}
		}

		if (lPairTempKFs.empty()) {
			//std::cout << "DetectLoopCandidates::End" << std::endl;
			return std::vector<Frame*>();
		}

		////연결된 프레임 정보까지 합해서 선택함.
		//단일 프레임 고려가 아닌 여러 프레임을 누적해서 진행함.
		std::list<std::pair<float, Frame*> > lAccScoreAndMatch;
		float bestAccScore = minScore;
		for (auto it = lPairTempKFs.begin(), eit = lPairTempKFs.end(); it != eit; it++)
		{
			auto pKFi = it->second;
			auto pBestKF = pKFi;
			auto vpNeighsKFs = pKFi->GetConnectedKFs(10);
			float bestScore = it->first;
			float accScore = it->first;

			for (auto vit = vpNeighsKFs.begin(), vend = vpNeighsKFs.end(); vit != vend; vit++)
			{
				auto pKF2 = *vit;
				if (pKF2->mnLoopClosingID == mnLoopID&& pKF2->mnLoopBowWords>nMinRequireWords)
				{
					accScore += pKF2->mfLoopScore;
					if (pKF2->mfLoopScore>bestScore)
					{
						pBestKF = pKF2;
						bestScore = pKF2->mfLoopScore;
					}
				}
			}

			lAccScoreAndMatch.push_back(std::make_pair(accScore, pBestKF));
			if (accScore>bestAccScore)
				bestAccScore = accScore;
		}


		std::set<Frame*> spLoopCandidateKFs;
		std::vector<Frame*> vpLoopCandidateKFs;
		vpLoopCandidateKFs.reserve(lPairTempKFs.size());
		float fMinScore = bestAccScore*0.75f;
		for (auto iter = lAccScoreAndMatch.begin(), iend = lAccScoreAndMatch.end(); iter != iend; iter++) {
			auto score = iter->first;
			if (score > fMinScore) {
				auto pKFi = iter->second;
				if (!spLoopCandidateKFs.count(pKFi)) {
					spLoopCandidateKFs.insert(pKFi);
					vpLoopCandidateKFs.push_back(pKFi);
				}
			}
		}
		//std::cout << "DetectLoopCandidates::End" << std::endl;
		return vpLoopCandidateKFs;
	}

}