#pragma once
#ifndef UVR_SLAM_PLANE_H
#define UVR_SLAM_PLANE_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace UVR_SLAM {

	class Frame;
	class PlaneInformation;

	//Frame과 Plane 사이에 처리할 때 필요한 정보들 관리 & 저장.
	class PlaneProcessInformation {
		public:
			PlaneProcessInformation();
			PlaneProcessInformation(Frame* p, PlaneInformation* plane);
			virtual ~PlaneProcessInformation();

			void Calculate();
			void GetInformation(cv::Mat& pInvP, cv::Mat& pInvT, cv::Mat& pInvK);
			PlaneInformation* GetInformation();
			PlaneInformation* GetFloorPlane();
			Frame* GetReferenceFrame();
		private:
			std::mutex mMutexProessor;
			Frame* mpFrame;
			PlaneInformation* mpFloor;
			cv::Mat invP, invT, invK;
	};

	class Line {
	public:
		Line(Frame* p, int w, cv::Point2f f, cv::Point2f t);
		virtual ~Line();
		void SetLinePts();
		cv::Mat GetLinePts();
		float CalcLineDistance(cv::Point2f);
	public:
		int mnPlaneID;
		UVR_SLAM::Frame* mpFrame;
		cv::Point2f from, to;
		cv::Point2f fromExt, toExt;
		std::vector<Line*> mvpLines;
		cv::Mat mLineEqu;
		float mfSlope;
	private:
		cv::Point2f CalcLinePoint(float y);
		
		std::mutex mMutexLinePts;
		cv::Mat mvLinePts;
	};

	class WallPlane {
	public:
		WallPlane();
		virtual ~WallPlane();
		
	public:
		void CreateWall();
		
		//update param
		void SetParam(cv::Mat p);
		cv::Mat GetParam();

		int GetRecentKeyFrameID();
		void SetRecentKeyFrameID(int id);

		//add, remove line
		//getlines
		size_t GetSize();
		bool isContain(Frame* pF);

		void AddLine(Line* line, Frame* pF);
		std::vector<Line*> GetLines();
		std::pair<std::multimap<Frame*, Line*>::iterator, std::multimap<Frame*, Line*>::iterator> GetLines(Frame* pF);

		//int  GetNumFrames();
		//void AddFrame(Frame* pF);

		

	public:
		int mnPlaneID;
		
	private:
		std::mutex mMutexParam;
		cv::Mat param;

		std::mutex mMutexLines;
		std::vector<Line*> mvLines;
		std::multimap<Frame*, Line*> mmpLines;

		//std::mutex mMutexFrames;
		//std::set<Frame*> mspFrames;

		std::mutex mMutexRecentKeyFrameID;
		int mnRecentKeyFrameID;
	};

	
}
#endif

