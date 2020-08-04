#ifndef UVR_SLAM_CANDIDATE_POINT_H
#define UVR_SLAM_CANDIDATE_POINT_H
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
#include <map>
namespace UVR_SLAM {
	class MatchInfo;
	class CandidatePoint {
		
	public:
		CandidatePoint();
		CandidatePoint(cv::Point2f apt, int aoct = 0);
		virtual ~CandidatePoint();
		cv::Point2f pt;
		int octave;
		bool bCreated;
		std::map<MatchInfo*, int> GetFrames();
		void AddFrame(UVR_SLAM::MatchInfo* pF, cv::Point2f pt); //index in frame
		void RemoveFrame(UVR_SLAM::MatchInfo* pKF);
	private:
		std::mutex mMutexCP;
		std::map<UVR_SLAM::MatchInfo*, int> mmpFrames;

	};
}
#endif
