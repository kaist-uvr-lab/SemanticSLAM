#ifndef UVR_SLAM_PLANE_ESTIMATOR_H
#define UVR_SLAM_PLANE_ESTIMATOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <SegmentationData.h>
#include <mutex>

namespace UVR_SLAM {

	class System;
	class WallPlane;
	class PlaneProcessInformation;
	class Map;
	class Frame;
	class Line;
	class FrameWindow;
	class MapPoint;
	class Initializer;
	class Matcher;
	class LineProcessor {
	public:
		float static CalcSlope(cv::Point2f pt1, cv::Point2f pt2) {
			float dx = pt1.x - pt2.x;
			float dy = pt1.y - pt2.y;
			float slope;
			if (dx == 0.0)
				slope = 1000.0;
			else
				slope = dy / dx;
			return slope;
		}
	};
	class PlaneOperator {
	public:
		
	};
	
	class PlaneInformation {
	public:
		int mnPlaneID;
		int mnFrameID;
		int mnCount;
		ObjectType mnPlaneType; //바닥, 벽, 천장
		std::vector<MapPoint*> mvpMPs;
		std::vector<MapPoint*> tmpMPs; //이전에 만들어진 포인트 중에서 여러개 연결된 경우

		
		void SetParam(cv::Mat n, float d);
		void GetParam(cv::Mat& n, float& d);
		cv::Mat GetParam();
	private:
		std::mutex mMutexParam;
		cv::Mat normal;
		float distance;
		float norm;
		cv::Mat matPlaneParam;
	public:
		float CalcOverlapMPs(PlaneInformation* p, int nID);
		void Merge(PlaneInformation* p, int nID, float thresh);

		float CalcCosineSimilarity(PlaneInformation* p);
		float CalcCosineSimilarity(cv::Mat P);
		float CalcPlaneDistance(PlaneInformation* p);
		float CalcPlaneDistance(cv::Mat X);
		//플루커 라인 관련 함수들
		cv::Mat FlukerLineProjection(PlaneInformation* P, cv::Mat R, cv::Mat t, cv::Mat K, float& m);
		
		//테스트
		static float CalcCosineSimilarity(cv::Mat P1, cv::Mat P2);
		static float CalcPlaneDistance(cv::Mat X1, cv::Mat X2);
		static cv::Mat PlaneWallEstimator(cv::Mat s, cv::Mat e, cv::Mat normal1, cv::Mat invP, cv::Mat invT, cv::Mat invK);
		static cv::Mat PlaneWallEstimator(Line* line, cv::Mat normal1, cv::Mat invP, cv::Mat invT, cv::Mat invK);
		static cv::Mat PlaneWallEstimator(cv::Vec4i line, cv::Mat normal1, cv::Mat invP, cv::Mat invT, cv::Mat invK);
		static cv::Mat PlaneWallEstimator(UVR_SLAM::Frame* pCurrF, UVR_SLAM::Frame* pTargetF);
		static cv::Mat CreatePlanarMapPoint(cv::Point2f pt, cv::Mat invP, cv::Mat invT, cv::Mat invK);
		static cv::Mat FlukerLineProjection(cv::Mat P1, cv::Mat P2, cv::Mat R, cv::Mat t, cv::Mat K, float& m);
		static void CalcFlukerLinePoints(cv::Point2f& sPt, cv::Point2f& ePt, float f1, float f2, cv::Mat mLine);
		static cv::Point2f CalcLinePoint(float y, cv::Mat mLine);
		static cv::Mat PlaneLineEstimator(WallPlane* pWall, PlaneInformation* pFloor);

		static bool calcUnitNormalVector(cv::Mat& X);
		static bool PlaneInitialization(PlaneInformation* pPlane, std::vector<MapPoint*> spMPs, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio);

		static cv::Mat CalcPlaneRotationMatrix(cv::Mat P);
		static void CreatePlanarMapPoint(Frame* pTargetF, PlaneInformation* pFloor, std::vector<cv::Mat>& vPlanarMaps);
		static bool CreatePlanarMapPoint(cv::Point2f pt, cv::Mat invP, cv::Mat invT, cv::Mat invK, cv::Mat& X3D);
		static bool CreatePlanarMapPoint(cv::Point2f pt, cv::Mat invP, cv::Mat invT, cv::Mat invK, cv::Mat fNormal, float fDist,cv::Mat& X3D);
		static void CreatePlanarMapPoints(Frame* pF, System* pSystem);
		static void CreateWallMapPoints(Frame* pF, WallPlane* pWall, Line* pLine, std::vector<cv::Mat>& vPlanarMaps, System* pSystem);
	};

	class PlaneEstimator {
	public:
		PlaneEstimator();
		PlaneEstimator(Map* pMap,std::string strPath, cv::Mat K, cv::Mat K2,int w, int h);
		virtual ~PlaneEstimator();
	public:
		void InsertKeyFrame(UVR_SLAM::Frame *pKF);
		bool CheckNewKeyFrames();
		void ProcessNewKeyFrame();

		void CreatePlanarMapPoints(std::vector<UVR_SLAM::MapPoint*> mvpMPs, std::vector<UVR_SLAM::ObjectType> mvpOPs, UVR_SLAM::PlaneInformation* pPlane, cv::Mat invT);
		void CreateWallMapPoints(std::vector<UVR_SLAM::MapPoint*> mvpMPs, std::vector<UVR_SLAM::ObjectType> mvpOPs, UVR_SLAM::PlaneInformation* pPlane, cv::Mat invT, int wtype,int MinX, int MaxX, bool b1, bool b2);
		void Run();
		void SetSystem(System* pSystem);
		void SetFrameWindow(FrameWindow* pFrameWindow);
		void SetTargetFrame(Frame* pFrame);
		void SetMatcher(Matcher* pMatcher);
		void SetInitializer(Initializer* pInitializer);
		void SetBoolDoingProcess(bool b, int ptype);
		bool isDoingProcess();

	private:

		

		
		void reversePlaneSign(cv::Mat& param);
		void UpdatePlane(PlaneInformation* pPlane, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio);
		bool PlaneInitialization(UVR_SLAM::PlaneInformation* pPlane, UVR_SLAM::PlaneInformation* GroundPlane, int type, std::vector<UVR_SLAM::MapPoint*> spMPs, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio);
		bool ConnectedComponentLabeling(cv::Mat img, cv::Mat& dst, cv::Mat& stat);
	public:
		cv::Mat mK2;
	private:

		std::queue<UVR_SLAM::Frame*> mKFQueue;
		std::mutex mMutexNewKFs;

		int mnWidth, mnHeight;
		int mnRansacTrial;
		float mfThreshPlaneDistance;
		float mfThreshPlaneRatio;
		float mfThreshNormal;

		cv::Mat mK;//
		System* mpSystem;
		Map* mpMap;
		std::mutex mMutexDoingProcess;
		bool mbDoingProcess;
		int mnProcessType;
		Frame* mpLayoutFrame;
		Frame* mpTargetFrame, *mpPrevFrame;
		FrameWindow* mpFrameWindow;
		Initializer* mpInitializer;
		Matcher* mpMatcher;

		
		
	};
}
#endif