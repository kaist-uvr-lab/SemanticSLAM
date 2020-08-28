#ifndef UVR_SLAM_CANDIDATE_POINT_H
#define UVR_SLAM_CANDIDATE_POINT_H
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
#include <map>
namespace UVR_SLAM {
	class MatchInfo;
	class Map;
	class MapPoint;
	class CandidatePoint {
		
	public:
		CandidatePoint();
		CandidatePoint(MatchInfo* pRefKF, int aoct = 0);
		virtual ~CandidatePoint();
		std::map<MatchInfo*, int> GetFrames();
		void AddFrame(UVR_SLAM::MatchInfo* pF, cv::Point2f pt); //index in frame
		void RemoveFrame(UVR_SLAM::MatchInfo* pKF);
		void Delete();
		int GetPointIndexInFrame(MatchInfo* pF);
		bool DelayedTriangulate(Map* pMap, MatchInfo* pMatch, cv::Point2f pt, MatchInfo* pPPrevMatch, MatchInfo* pPrevMatch, cv::Mat K, cv::Mat invK, cv::Mat& debug);
		float CalcParallax(cv::Mat Rkf1c, cv::Mat Rkf2c, cv::Point2f pt1, cv::Point2f pt2, cv::Mat invK);
		cv::Mat Triangulate(cv::Point2f pt1, cv::Point2f pt2, cv::Mat P1, cv::Mat P2, bool& bRank);
		cv::Point2f Projection(cv::Mat Xw, cv::Mat R, cv::Mat T, cv::Mat K, bool& bDepth);
		bool CheckDepth(float depth);
		bool CheckReprojectionError(cv::Mat x3D, cv::Mat K, cv::Point2f pt, float thresh);
		bool CheckReprojectionError(cv::Point2f pt1, cv::Point2f pt2, float thresh);
		int GetNumSize();
		
		void Test(cv::Mat& X3D, cv::Mat K,cv::Mat invK, cv::Mat Pcurr, cv::Mat Rcurr, cv::Mat Tcurr, cv::Point2f ptCurr, bool& bProjec, bool& bParallax, cv::Mat& debug);
		//�ﰢȭ
		//�ƿ����̾� üũ(���������� ����)
		//�����׽�Ʈ
	public:
		int octave;
		bool bCreated;
		MapPoint* mpMapPoint;
	private:
		MatchInfo* mpRefKF;
		bool mbDelete;
		std::mutex mMutexCP;
		std::map<UVR_SLAM::MatchInfo*, int> mmpFrames;
		int mnConnectedFrames;
	};
}
#endif