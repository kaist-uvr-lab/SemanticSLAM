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
	class Initializer;
	class MapPoint;
	class PlaneInformation {
	public:
		cv::Mat matPlaneParam;
		int mnPlaneID;
		int mnFrameID;
		int mnCount;
		ObjectType mnPlaneType; //바닥, 벽, 천장
		std::vector<MapPoint*> mvpMPs;

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
		void Run();
		void SetSystem(System* pSystem);
		void SetFrameWindow(FrameWindow* pFrameWindow);
		void SetTargetFrame(Frame* pFrame);
		void SetInitializer(Initializer* pInitializer);
		void SetBoolDoingProcess(bool b, int ptype);
		bool isDoingProcess();
	private:
		bool calcUnitNormalVector(cv::Mat& X);
		void reversePlaneSign(cv::Mat& param);
		bool PlaneInitialization(PlaneInformation* pPlane, std::set<MapPoint*> spMPs, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio);
		bool PlaneInitialization(UVR_SLAM::PlaneInformation* pPlane, UVR_SLAM::PlaneInformation* GroundPlane, int type, std::set<UVR_SLAM::MapPoint*> spMPs, int nTargetID, int ransac_trial, float thresh_distance, float thresh_ratio);
		
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
		
	};
}
#endif