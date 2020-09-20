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
		int mnCandidatePointID;
		CandidatePoint();
		CandidatePoint(MatchInfo* pRefKF, int alabel = 0, int aoct = 0);
		virtual ~CandidatePoint();
		std::map<MatchInfo*, int> GetFrames();
		//void AddFrame(UVR_SLAM::MatchInfo* pF, cv::Point2f pt); //index in frame
		void ConnectFrame(UVR_SLAM::MatchInfo* pF, int idx); //index in frame
		void DisconnectFrame(UVR_SLAM::MatchInfo* pKF);
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
		
		void CreateMapPoint(cv::Mat& X3D, cv::Mat K,cv::Mat invK, cv::Mat Pcurr, cv::Mat Rcurr, cv::Mat Tcurr, cv::Point2f ptCurr, bool& bProjec, bool& bParallax, cv::Mat& debug);
		//삼각화
		//아웃라이어 체크(리프로젝션 에러)
		//뎁스테스트
	public:
		int mnFirstID; //처음 발견한 프레임
		int octave;
	private:
		MatchInfo* mpRefKF;
		bool mbDelete;
		std::mutex mMutexCP;
		std::map<UVR_SLAM::MatchInfo*, int> mmpFrames;
		int mnConnectedFrames;
	//////////////label
	public:
		int GetLabel();
		void SetLabel(int a);
	private:
		std::mutex mMutexLabel;
		int label;

	////////////////////////
	////////MP관리
	public:
		void DeleteMapPoint();
		void ResetMapPoint();
		void SetMapPoint(MapPoint* pMP, int id);
		MapPoint* GetMP();
	private:
		int mnMapPoint;
		bool bCreated;
		MapPoint* mpMapPoint;
	////////MP관리
	////////////////////////

	///////////////////
	////매칭 퀄리티 관련
	public:
		int mnVisibleFrameID;
		float GetRatio();
		void AddFail(int n = 1);
		int GetFail();
		void AddSuccess(int n = 1);
		int GetSuccess();
		void SetLastSuccessFrame(int id);
		int GetLastSuccessFrame();
		void SetLastVisibleFrame(int id);
		int GetLastVisibleFrame();
		void ComputeQuality();
		bool GetQuality();
		void SetOptimization(bool b);
		bool isOptimized();
	private:
		bool mbOptimized;
		int mnFail, mnSuccess, mnTotal;
		int mnLastFrameID;
		int mnFirstMapPointID;
		bool mbLowQuality;
	////매칭 퀄리티 관련
	///////////////////
	};


}
#endif
