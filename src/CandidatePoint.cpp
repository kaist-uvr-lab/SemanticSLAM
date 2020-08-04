#include "CandidatePoint.h"
#include "Frame.h"

namespace  UVR_SLAM{
	CandidatePoint::CandidatePoint(){}
	CandidatePoint::CandidatePoint(cv::Point2f apt, int aoct):pt(apt), octave(aoct), bCreated(false){

	}
	CandidatePoint::~CandidatePoint(){}

	std::map<MatchInfo*, int> CandidatePoint::GetFrames() {
		std::unique_lock<std::mutex> lockMP(mMutexCP);
		return std::map<UVR_SLAM::MatchInfo*, int>(mmpFrames.begin(), mmpFrames.end());
	}
	void CandidatePoint::AddFrame(MatchInfo* pF, cv::Point2f pt) {
		std::unique_lock<std::mutex> lockMP(mMutexCP);
		auto res = mmpFrames.find(pF);
		if (res == mmpFrames.end()) {
			int idx = pF->AddCP(this, pt);
			mmpFrames.insert(std::pair<UVR_SLAM::MatchInfo*, int>(pF, idx));
		}
	}
}

