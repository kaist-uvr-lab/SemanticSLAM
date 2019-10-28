//
// Created by UVR-KAIST on 2019-02-25.
//

#ifndef ANDROIDOPENCVPLUGINPROJECT_MANAGEMENT_H
#define ANDROIDOPENCVPLUGINPROJECT_MANAGEMENT_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <mutex>
#include <iterator>

/*
#include "MAP.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "CandidatePoint.h"
#include "IMUProcessor.h"
*/
//#include "Optimizer.h"

using namespace cv;
namespace UVR {
	class Optimizer;
	class Tracker;
	class PlaneThread;
	class IMUProcessor;

	class Management {
	public:
		/*
		Management() :thresh_intersection(20), mnNumKFs(15), mnNumMPs(2000), mnNumMPsInKF(30), mbInit(false), thresh_mp_live(4), thresh_mp_connected_kf(3), mbStopLocalBA(false), mbDoingNewKF(false) {}
		Management(int _live, int _ckf, int _minMPKF, int w, int h) :mnNumMPsInKF(_minMPKF), mbInit(false), thresh_mp_live(_live), thresh_mp_connected_kf(_ckf), mbDoingLocalBA(false), mbFinishedLocalBA(false), mbStopLocalBA(false), mbDoingNewKF(false)
		{
		}
		virtual ~Management() {
		}
		void Run();

		int thresh_intersection;
		int thresh_mp_live;
		int thresh_mp_connected_kf;
		int mnNumMPsInKF;
		int mnNumKFs;
		int mnNumMPs;
		bool mbInit;

		//local map
		//use optimization and next feature matching
		
		std::vector<UVR::MapPoint*> mvpPoseMPs;
		std::vector<bool> mvbPoseMPInliers;
		std::vector<int> mvnPoseMPIndexes;

		std::vector<UVR::MapPoint*> mvpMPs;
		std::vector<int> mvnMatchindIndexs;
		std::vector<bool> mvbMatchingInliers;
		cv::Mat mMatLocalMapDesc;

		std::set<UVR::MapPoint*> mspMPs;
		std::vector<UVR::MapPoint*> mvpNewMPs;
		std::vector<cv::KeyPoint> mvKPs, mvNewKPs;
		cv::Mat mDescNewMPs;
		



		////
		//LocalBA
		//mbDoLocalBA 이게 트루가 되면 로컬비에이 데이터 저장.
		//mbDoingLocalBA가 펄스일 때 외부에서 트루값이 오면 옵티마이즈를 트루로 함.
		//외부에서 트루값이 왔을 때 mbDoingLocalBA가 트루이면, 아직 옵티마이즈 중. 이때는 수행 안함.
		//외부에서 펄스값이 오면 항상 안함.
		//mbDoingLocalBA는 옵티마이저에서 끄기
		std::mutex mMutexDoingLocalBA, mMutexFinishLocalBA;
		bool mbDoingLocalBA, mbFinishedLocalBA, mbStopLocalBA;
		//얘는 최적화전에 맵의 포인트 벡터랑 동일함
		//최적화가 끝났고 로컬맵 생성전에 삭제할 것이기 때문에 문제 될 것은 없음.
		//최적화 중일 때는 동작 안하도록 하면 됨.
		std::mutex mMutexDeleteMPs, mMutexDeleteKFs;
		std::set<UVR::MapPoint*> mspDeleteMPs;
		std::set<UVR::KeyFrame*> mspDeleteKFs;
		std::vector<UVR::MapPoint*> mvpLocalBAMPs;
		std::map<UVR::KeyFrame*, int> mmpLocalBAKFs;
		std::vector<UVR::KeyFrame*> mvpLocalBAKFs;
		std::vector<UVR::CandidatePoint*> mvpCPs;
		cv::Mat mDescCPs;

		////190430 New Map Points Constraints
		bool CheckCosineParallax(float invfx, float invfy, float cx, float cy, cv::Point2f pt1, cv::Point2f pt2, cv::Mat Rinv1, cv::Mat Rinv2, float thresh);
		bool CheckReprojectionError(cv::Mat x3D, cv::Mat K, Point2f pt, float thresh);
		bool CheckReprojectionError(cv::Point2f pt1, Point2f pt2, float thresh);
		bool CheckDepth(float depth);
		bool CheckBaseLine(UVR::KeyFrame* pKF1, UVR::KeyFrame* pKF2);
		bool CheckScaleConsistency(cv::Mat x3D, cv::Mat Ow1, cv::Mat Ow2, float fRatioFactor, float fScaleFactor1, float fScaleFactor2);
		////190430 New Map Points Constraints

		////190711
		void CreateMapPoints(UVR::KeyFrame* pNewKF, UVR::KeyFrame* pLastKF);
		void CreateMapPoints(UVR::KeyFrame* pNewKF);
		int MatchingWithKF(UVR::KeyFrame* pNewKF, UVR::KeyFrame* pLastKF);
		int MatchingWithKF(UVR::KeyFrame* pNewKF, UVR::KeyFrame* pCurrKF, cv::Mat& res);

		////LOCALBA
		void OverlapTest(UVR::Frame* pF);

		bool SetLocalMap(UVR::KeyFrame* pNewKF, bool bFlagBA);
		void InitLocalMap(UVR::Frame* pLastF, UVR::Frame* pCurrF);
		void UpdateLocalMap(UVR::Frame* pCurrF);
		void SetLocalMap(UVR::Frame* pCurrF);
		int UpdatePoseOptimizationResults(UVR::Frame* pCurrF);
		int UpdatePoseOptimizationResults(UVR::Frame* pCurrF, UVR::KeyFrame* pCurrKF, bool& bRes);
		int UpdatePoseOptimizationResults2(UVR::Frame* pCurrF, UVR::KeyFrame* pCurrKF, bool& bRes);
		void SetLocalMap(UVR::Frame* pCurrF, UVR::KeyFrame* pCurrKF);
		void UpdateNewMP();
		void UpdateMP(UVR::KeyFrame* pInitKF1, UVR::KeyFrame* pInitKF2, UVR::Frame* pLast, UVR::Frame* pCurr);
		void UpdateMP(UVR::KeyFrame* pCurrKF, UVR::Frame* pCurrF);

		int CheckRealMPs(std::vector<UVR::MapPoint*> vpMPs1);
		int FindIntersection(std::vector<UVR::MapPoint*> vpMPs1, std::vector<UVR::MapPoint*> vpMPs2);

		//정렬은 트랙드 맵포인트를 만든 후, 맵포인트를 만든 후 이미 수행했다고 가정하자. 아니면 트랙드 맵포인트만 비교하면 나머지 구조는 건드리지 않으니까
		void RemoveKeyFrame(UVR::MAP* pMap);
		void RemoveKeyFrame(UVR::KeyFrame* pKF);
		void RemoveMapPoint(UVR::KeyFrame* pKF);
		void Remove();
		void RemoveCandidatePoint();
		UVR::KeyFrame* RemoveKeyFrame(UVR::KeyFrame* pRefKF, std::queue<UVR::KeyFrame*>& qpKFs);
		void RemoveMapPoint(UVR::KeyFrame* pKF, int nQueueSize);

		void AddMP(UVR::MapPoint* pMP); //전체 맵에 추가
		void AddMP(UVR::MapPoint* pMP, UVR::KeyFrame* pKF); //키프레임에 추가
		void AddMP(UVR::MapPoint* pMP, UVR::KeyFrame* pKF, cv::KeyPoint kp); //키프레임에 추가
		void AddCP(UVR::CandidatePoint* pCP); // 전체 캔디데이트 포인트에 추가
		void AddCP(UVR::CandidatePoint* pCP, UVR::MAP* pMap);
		void AddCP(UVR::CandidatePoint* pCP, UVR::KeyFrame* pKF, Point2f pt);

		void AddKF(UVR::KeyFrame* pKF); //전체 키프레임 벡터에 추가

		void CreateMP(UVR::KeyFrame* pNewKF, cv::Mat ori);
		void CreateMP(UVR::KeyFrame* pNewKF);
		void CreateMP(UVR::KeyFrame* pCurrKF, UVR::KeyFrame* pNewKF);
		void CreateCP(UVR::KeyFrame* pNewKF);
		void CreateCP(UVR::KeyFrame* pNewKF, UVR::KeyFrame* pCurrKF);
		void CreateCP(UVR::KeyFrame* pCurrKF, UVR::Frame* pCurrF);

		void CalcCovMP(UVR::MapPoint* pMP);
		void DeleteMP(UVR::MapPoint* pMP);
		void DeleteKF(UVR::KeyFrame* pKF);

		////DEBUGGING
		void DebugTracking(UVR::Frame* pCurrFrame);
		void DebugCreateKeyFrame(UVR::KeyFrame* pNewKF);
		////DEBUGGING

		//190326 새롭게 추가한 것
		bool CheckTrackingGrid(UVR::KeyFrame* pKF, UVR::Frame* pF);
		void ConnectKF(UVR::KeyFrame* pCurrKF, UVR::KeyFrame* pNewKF);
		void ConnectKF(UVR::KeyFrame* pNewKF);

		cv::Mat Triangulate(cv::Point2f pt1, cv::Point2f pt2, cv::Mat P1, cv::Mat P2);
		bool CheckEpipolarConstraints(UVR::KeyFrame* pKF, cv::Mat F12, cv::Point2f pt1, cv::Point2f pt2, int octave, float& res);
		cv::Mat CalcFundamentalMatrix(cv::Mat R1, cv::Mat t1, cv::Mat R2, cv::Mat t2, cv::Mat K);

		bool CalcDiffKeyFrames(UVR::Frame* pF, UVR::KeyFrame* pKF);
		int CalcDiffKeyframe(UVR::Frame* pF, UVR::KeyFrame* pKF);


		bool CalcDiffPose(UVR::Frame* pF, UVR::KeyFrame* pKF, float th_angle, float th_dist);
		bool CalcDiffPose(UVR::Frame* pF, float min_th_angle, float min_th_dist, float max_th_angle, float max_th_dist);


		void CalcDiffPose(UVR::KeyFrame* pKF, UVR::Frame* pF, float& dangle, float& ddist);
		void CalcDiffPose(cv::Mat R1, cv::Mat t1, cv::Mat R2, cv::Mat t2, cv::Mat &diffR, cv::Mat& diffT);
		void CalcDiffPose(cv::Mat DiffR, cv::Mat DiffT, float& dangle, float& ddist);
		float CalcAngularVelocity(cv::Mat DiffR);
		float CalcPositionDistance(cv::Mat DiffT);

		////190801
		//Keyframe Distribution
		void DitributeKeyFrame(UVR::KeyFrame* pNewKF, UVR::KeyFrame* pCurrKF);

		bool GetBoolDoingBA() {
			std::unique_lock<std::mutex> lockBA(mMutexDoingLocalBA);
			return mbDoingLocalBA;
		}
		void SetBoolDoingBA(bool bBA) {
			std::unique_lock<std::mutex> lockBA(mMutexDoingLocalBA);
			mbDoingLocalBA = bBA;
		}
		bool GetBoolStopBA() {
			std::unique_lock<std::mutex> lockBA(mMutexDoingLocalBA);
			return mbStopLocalBA;
		}
		void SetBoolStopBA(bool bBA) {
			std::unique_lock<std::mutex> lockBA(mMutexDoingLocalBA);
			mbStopLocalBA = bBA;
		}
		////Data
		//OpenMP로 접근하기 위해서 벡터 이용
		std::set<UVR::MapPoint*> mspTrackedMPs; //커넥티드 엠피에 사용?
		std::vector<cv::Point2f> mvLocalMapGoodFeatures;
		std::map<UVR::KeyFrame*, int> msTempConnectedKFs;
		std::vector<UVR::MapPoint*> mvpAllMps;
		std::vector<UVR::KeyFrame*> mvpAllKFs;
		std::vector<UVR::CandidatePoint*> mvpAllCPs;

		////Connection 정보
		//커넥션 정보에서 연결된 정보를 얻은 후 그 값을 이용해서 처리
		//맵포인트의 경우 맵이 가지고 있는 KF 정보까지 삭제 하게 해야 함.
		//std::set<UVR::MapPoint*, std::vector<UVR::KeyFrame*>> mspMPKFConnection;
		//std::set<UVR::CandidatePoint*, std::vector<UVR::KeyFrame*>> mspCPKFConnection;
		//std::map<UVR::KeyFrame*, std::vector<UVR::MapPoint*>> mspKFMPConnection;
		//std::map<UVR::KeyFrame*, std::vector<UVR::CandidatePoint*>> mspKFCPConnection;

		//맵포인트를 추가하는 경우 mvpAll에 저장
		//또한 맵포인트에는 키프레임의 어느 위치에 저장되는지, 그 때 어떠한 포인트와 연결되는지, 전체 벡터에 어느 위치에 저장되는 지를 기록함.
		//삭제 시 위의 정보 해제.
		//삭제시에는 자신이 가지고 있는 KF를 정보를 통해서 해제 함.
		//allMPS에서 자기 정보 삭제, 키프레임 내의 mvpMP에서 자기 정보 삭제

		//키프레임을 추가할 때는
		//KFMP에는 자신이 갖게 된 맵포인트를 추가, 그 맵포인트들도 MPKF에 정보 추가, 내부적으로 인덱스 정보, 키프레임 pt 정보를 추가 하게 됨
		//삭제시에는 KFMP를 돌면서 자신을 참조하는 MP들에게 KF와 연결을 해제하도록 함.


		UVR::Tracker* mpTracker;
		void SetTracker(UVR::Tracker* _tracker) {
			mpTracker = _tracker;
		}

		UVR::MAP* mpMap;
		void SetMap(UVR::MAP* _map) {
			mpMap = _map;
		}
		void SetNewKF(bool bFlag) {
			std::unique_lock<std::mutex> lockKF(mMutexDoingNewKF);
			mbDoingNewKF = bFlag;
		}
		bool isDoingNewKF() {
			std::unique_lock<std::mutex> lockKF(mMutexDoingNewKF);
			return mbDoingNewKF;
		}

		void SetNewKF(UVR::KeyFrame* pCurrKF, UVR::KeyFrame* pLastKF) {
			//std::unique_lock<std::mutex> lockKF(mMutexCreateKF);
			mpLastKF = pLastKF;
			mpCurrKF = pCurrKF;
		}

		void SetOptimizer(UVR::Optimizer* pOptimizer) {
			mpOptimizer = pOptimizer;
		}
		void SetPlaneManager(UVR::PlaneThread* pPlane) {
			mpPlaneManager = pPlane;
		}

		UVR::IMUProcessor* GetIMU() {
			return mpIMU;
		}
		void SetIMU(UVR::IMUProcessor* pIMU) {
			mpIMU = pIMU;
		}

		////Initial Matching
		std::vector<UVR::MapPoint*> mvpInitMPs;
		std::vector<bool> mvbInitInliers;
		cv::Mat mMatInitDescs;

	private:
		bool mbDoingNewKF;
		std::mutex mMutexDoingNewKF;

		UVR::KeyFrame* mpLastKF, *mpCurrKF;

		UVR::Optimizer* mpOptimizer;
		UVR::PlaneThread* mpPlaneManager;
		UVR::IMUProcessor* mpIMU;
		*/
	};
	
}


#endif //ANDROIDOPENCVPLUGINPROJECT_MANAGEMENT_H
