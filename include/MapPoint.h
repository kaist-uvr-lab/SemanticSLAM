
#ifndef UVR_SLAM_MAP_POINT_H
#define UVR_SLAM_MAP_POINT_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
#include <SegmentationData.h>

namespace UVR_SLAM {

	const enum MapPointType {
		NORMAL_MP,
		PLANE_MP,
		PLANE_DENSE_MP
	};
	class CandidatePoint;
	class Map;
	class Frame;
	class MatchInfo;
	class FrameWindow;
	class MapPoint {
	public:
		//초기 포즈 만들 때는 double형으로 형변환
		MapPoint();
		MapPoint(Map* pMap, UVR_SLAM::Frame* pRefKF, cv::Mat _p3D, cv::Mat _desc, int alabel = -1, int octave = 0);
		MapPoint(Map* pMap, UVR_SLAM::Frame* pRefKF, CandidatePoint* pCP, cv::Mat _p3D, cv::Mat _desc, int label, int nOctave = 0);
		MapPoint(Map* pMap, UVR_SLAM::Frame* pRefKF, CandidatePoint* pCP, cv::Mat _p3D, cv::Mat _desc, MapPointType ntype, int label, int nOctave = 0);
		virtual ~MapPoint();
	public:

		std::map<Frame*, int> GetObservations();
		void AddObservation(Frame* pF, int idx);
		void EraseObservation(Frame* pF);
		void DeleteMapPoint();
		bool isInFrame(Frame* pF);

	private:

		std::mutex mMutexFeatures;
		std::map<UVR_SLAM::Frame*, int> mmpObservations;








		///////////////////////////////////////////////////////////////////////////////////////////////
	public:
		void SetWorldPos(cv::Mat X);
		cv::Mat GetWorldPos();
		void SetNewMP(bool _b);
		bool isNewMP();
		bool isInFrame(MatchInfo* pF);
		int GetPointIndexInFrame(MatchInfo* pF);

		void SetMapGridID(int id);
		int GetMapGridID();

		void UpdateNormalAndDepth();
		int PredictScale(const float &currentDist, Frame* pKF);
		
		void SetDescriptor(cv::Mat _desc);
		cv::Mat GetDescriptor();
		void IncreaseVisible(int n = 1);
		void IncreaseFound(int n = 1);
		float GetFVRatio();
		void Fuse(MapPoint* pMP);

		bool Projection(cv::Point2f& _P2D, cv::Mat& _Pcam, cv::Mat R, cv::Mat t, cv::Mat K, int w, int h);
		bool Projection(cv::Point2f& _P2D, cv::Mat& _Pcam, float& _depth, cv::Mat R, cv::Mat t, cv::Mat K, int w, int h);
		bool Projection(cv::Point2f& _P2D, Frame* pF, int w, int h);
		bool isSeen();

		MapPointType GetMapPointType();
		void SetMapPointType(MapPointType type);

		void SetPlaneID(int nid);
		int GetPlaneID();
		
		//void SetDelete(bool b);
		bool isDeleted();
		
		float GetMaxDistance();
		float GetMinDistance();
		cv::Mat GetNormal();
		
	public:
		int mnMapPointID;
		int mnFirstKeyFrameID;
		int mnLocalBAID, mnLocalMapID, mnTrackingID;
		
		int mnOctave;
		////마지막 트래킹 정보.
		Frame* mLastFrame;
		cv::Mat mLastMatchPatch;
		cv::Point2f mLastMatchBasePt;
		cv::Point2f mLastMatchPoint;
		bool mbLastMatch;
		CandidatePoint* mpCP; //현재 맵포인트를 만든 CP와 연결함.
	private:
		////map grid
		int mnMapGridID;
		std::mutex mMutexMapGrid;
	private:
		Map* mpMap;
		Frame* mpRefKF;
		std::mutex mMutexMP;
		bool mbDelete;
		int mnPlaneID;
		MapPointType mnType;
		
		float mfDepth;
		bool mbSeen;
		bool mbNewMP;
		cv::Mat p3D;
		int mnDenseFrames;
		
		cv::Mat desc;
		
		std::map<UVR_SLAM::Frame*, cv::Point2f> mmpDenseFrames;
		
		int mnVisible;
		int mnFound;

		float mfMaxDistance, mfMinDistance;
		cv::Mat mNormalVector;

	//local map 및 최근 트래킹 관련 index 관련
	public:
		int GetRecentLayoutFrameID();
		void SetRecentLayoutFrameID(int nFrameID);
	private:
		std::mutex mMutexRecentLayoutFrameID;
		int mnLayoutFrameID;//mnLocalMapID
	//Object Type
	public:
		void SetObjectType(ObjectType nType);
		ObjectType  GetObjectType();
	private:
		std::mutex mMutexObjectType;
		ObjectType mObjectType;

		//////////////////////프레임과 관련된 것들
	public:
		void ConnectFrame(UVR_SLAM::MatchInfo* pF, int idx); //index in frame
		void DisconnectFrame(UVR_SLAM::MatchInfo* pKF);
		std::map<MatchInfo*, int> GetConnedtedFrames();
		int GetNumConnectedFrames();
		void Delete();
	private:
		std::map<UVR_SLAM::MatchInfo*, int> mmpFrames;
		int mnConnectedFrames;
		//////////////////////프레임과 관련된 것들
	////label
	public:
		int GetLabel();
		void SetLabel(int a);
	private:
		std::mutex mMutexLabel;
		int label;
	////label

	///////////////////
	////매칭 퀄리티 관련
	public:
		int mnLastVisibleFrameID;
		int mnLastMatchingFrameID;
		/*void AddFail(float n = 1.0);
		float GetFail();
		void AddSuccess(float n = 1.0);
		float GetSuccess();*/
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
		int mnTotal;
		float mnSuccess;
		bool mbLowQuality;
		////매칭 퀄리티 관련
		///////////////////
	};
}

#endif