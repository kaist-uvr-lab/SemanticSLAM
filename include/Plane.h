#pragma once
#ifndef UVR_SLAM_PLANE_H
#define UVR_SLAM_PLANE_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {

	class Frame;
	class PlaneInformation;

	//Frame�� Plane ���̿� ó���� �� �ʿ��� ������ ���� & ����.
	class PlaneProcessInformation {
		public:
			PlaneProcessInformation();
			PlaneProcessInformation(Frame* p, PlaneInformation* plane);
			virtual ~PlaneProcessInformation();

			void Calculate();
			void GetInformation(cv::Mat& pInvP, cv::Mat& pInvT, cv::Mat& pInvK);
		private:
			std::mutex mMutexProessor;
			Frame* mpFrame;
			PlaneInformation* mpFloor;
			cv::Mat invP, invT, invK;
	};

	class Line {
	public:
		Line(Frame* p, cv::Point2f f, cv::Point2f t);
		virtual ~Line();
		void SetLinePts();
		cv::Mat GetLinePts();
	public:
		int mnPlaneID;
		UVR_SLAM::Frame* mpFrame;
		cv::Point2f from, to;
	private:
		std::mutex mMutexLinePts;
		cv::Mat mvLinePts;
	};

	class WallPlane {
	public:
		WallPlane();
		virtual ~WallPlane();
		
	public:
		void CreateWall();
		//add, remove line
		//getlines
		void AddLine(Line* line);
		size_t GetSize();
		std::vector<Line*> GetLines();

		//update param
		void SetParam(cv::Mat p);
		cv::Mat GetParam();

		void AddFrame(Frame* pF);
		bool isContain(Frame* pF);
		int  GetNumFrames();

		int GetRecentKeyFrameID();
		void SetRecentKeyFrameID(int id);

	public:
		int mnPlaneID;
		
	private:
		std::mutex mMutexParam;
		cv::Mat param;

		std::mutex mMutexLiens;
		std::vector<Line*> mvLines;

		std::mutex mMutexFrames;
		std::set<Frame*> mspFrames;

		std::mutex mMutexRecentKeyFrameID;
		int mnRecentKeyFrameID;
	};

	
}
#endif

