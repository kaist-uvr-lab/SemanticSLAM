#ifndef UVR_SLAM_CANDIDATE_POINT_H
#define UVR_SLAM_CANDIDATE_POINT_H
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
#include <map>

namespace UVR_SLAM {
	class Frame;
	class Map;
	class MapPoint;
	class DepthFilter;
	class Seed;
	class CandidatePoint {
		
	public:
		
		CandidatePoint();
		CandidatePoint(Frame* pRefKF, int alabel = 0, int aoct = 0);
		virtual ~CandidatePoint();
				
		float CalcParallax(cv::Mat Rkf1c, cv::Mat Rkf2c, cv::Point2f pt1, cv::Point2f pt2, cv::Mat invK);
		cv::Mat Triangulate(cv::Point2f pt1, cv::Point2f pt2, cv::Mat P1, cv::Mat P2, bool& bRank);
		cv::Point2f Projection(cv::Mat Xw, cv::Mat R, cv::Mat T, cv::Mat K, float& fDepth, bool& bDepth);
		bool CheckDepth(float depth);
		bool CheckReprojectionError(cv::Mat x3D, cv::Mat K, cv::Point2f pt, float thresh);
		bool CheckReprojectionError(cv::Point2f pt1, cv::Point2f pt2, float thresh);
		int GetNumSize();

		//삼각화
		//아웃라이어 체크(리프로젝션 에러)
		//뎁스테스트
	public:
		int mnCandidatePointID;
		int mnFirstID; //처음 발견한 프레임
		Frame* mpRefKF;
		int octave;
	private:
		bool mbDelete;
		std::mutex mMutexCP;
		int mnConnectedFrames;
	//////////////label
	public:
		int GetLabel();
		void SetLabel(int a);
	private:
		std::mutex mMutexLabel;
		std::map<int, int> mmnObjectLabelHistory;
		int label;

	////////////////////////
	////////MP관리
	public:
		void ResetMapPoint();
		void SetMapPoint(MapPoint* pMP);
		//매핑에서만
		void SetLastVisibleFrame(int id);
		int GetLastVisibleFrame(); 
		void SetLastSuccessFrame(int id);
		int GetLastSuccessFrame();
		//매핑에서만
		//트래킹에서
		int mnTrackingFrameID;
		MapPoint* GetMP();
	private:
		bool bCreated;
		MapPoint* mpMapPoint;

		int mnLastVisibleFrameID;
		int mnLastMatchingFrameID;
	////////MP관리
	////////////////////////
	//////Depth Filer
	public:
		Seed* mpSeed;
	//////Depth Filer
	};


}
#endif
