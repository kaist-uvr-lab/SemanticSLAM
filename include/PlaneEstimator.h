#ifndef UVR_SLAM_PLANE_ESTIMATOR_H
#define UVR_SLAM_PLANE_ESTIMATOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <SegmentationData.h>
#include <mutex>

namespace UVR_SLAM {

	class System;
	class Frame;
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
	class PlaneInformation {
	public:
		int mnPlaneID;
		int mnFrameID;
		int mnCount;
		ObjectType mnPlaneType; //바닥, 벽, 천장
		std::vector<MapPoint*> mvpMPs;
		std::vector<MapPoint*> tmpMPs; //이전에 만들어진 포인트 중에서 여러개 연결된 경우

		cv::Mat matPlaneParam;
		void SetParam(cv::Mat n, float d);
		void GetParam(cv::Mat& n, float& d);
	private:
		std::mutex mMutexParam;
		cv::Mat normal;
		float distance;
		float norm;
	public:
		float CalcOverlapMPs(PlaneInformation* p, int nID);
		void Merge(PlaneInformation* p, int nID, float thresh);

		float CalcCosineSimilarity(PlaneInformation* p);
		float CalcCosineSimilarity(cv::Mat P);
		float CalcPlaneDistance(PlaneInformation* p);
		float CalcPlaneDistance(cv::Mat X);
		//플루커 라인 관련 함수들
		cv::Mat FlukerLineProjection(PlaneInformation* P, cv::Mat R, cv::Mat t, cv::Mat K, float& m);
		void CalcFlukerLinePoints(cv::Point2f& sPt, cv::Point2f& ePt, float f1, float f2, cv::Mat mLine);
		cv::Point2f CalcLinePoint(float y, cv::Mat mLine);
	};

	class PlaneEstimator {
	public:
		PlaneEstimator();
		PlaneEstimator(std::string strPath, cv::Mat K, cv::Mat K2,int w, int h);
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

		bool isFloorPlaneInitialized();
		void SetFloorPlaneInitialization(bool b);

	private:

		cv::Mat CalcPlaneRotationMatrix(cv::Mat P);

		bool calcUnitNormalVector(cv::Mat& X);
		void reversePlaneSign(cv::Mat& param);
		void UpdatePlane(PlaneInformation* pPlane, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio);
		bool PlaneInitialization(PlaneInformation* pPlane, std::vector<MapPoint*> spMPs, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio);
		bool PlaneInitialization(UVR_SLAM::PlaneInformation* pPlane, UVR_SLAM::PlaneInformation* GroundPlane, int type, std::vector<UVR_SLAM::MapPoint*> spMPs, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio);
		bool ConnectedComponentLabeling(cv::Mat img, cv::Mat& dst, cv::Mat& stat);
	private:

		std::queue<UVR_SLAM::Frame*> mKFQueue;
		std::mutex mMutexNewKFs;

		int mnWidth, mnHeight;
		int mnRansacTrial;
		float mfThreshPlaneDistance;
		float mfThreshPlaneRatio;
		float mfThreshNormal;

		cv::Mat mK, mK2;//
		System* mpSystem;
		std::mutex mMutexDoingProcess;
		bool mbDoingProcess;
		int mnProcessType;
		Frame* mpLayoutFrame;
		Frame* mpTargetFrame, *mpPrevFrame;
		FrameWindow* mpFrameWindow;
		Initializer* mpInitializer;
		Matcher* mpMatcher;

		std::mutex mMutexInitFloorPlane;
		bool mbInitFloorPlane;
		
	};
}
#endif