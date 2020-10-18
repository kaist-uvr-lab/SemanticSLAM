#include "Map.h"
#include "MapGrid.h"
#include "Frame.h"
#include "MapPoint.h"
#include "Plane.h"
#include "System.h"

namespace UVR_SLAM{
	Map::Map():mnMaxConnectedKFs(8), mnHalfConnectedKFs(4), mnQuarterConnectedKFs(2), mnMaxCandidateKFs(4), mnHalfCandidate(2), mbInitFloorPlane(false), mbInitWallPlane(false), mfMapGridSize(0.2){}
	Map::Map(System* pSystem, int nConnected, int nCandiate) :mpSystem(pSystem), mnMaxConnectedKFs(nConnected), mnHalfConnectedKFs(nConnected/2), mnQuarterConnectedKFs(nConnected/4), mnMaxCandidateKFs(nCandiate), mnHalfCandidate(nCandiate/2), mbInitFloorPlane(false), mbInitWallPlane(false), mfMapGridSize(0.2) {
		std::cout << "MAP::" << mnMaxConnectedKFs << ", " << mnMaxCandidateKFs << std::endl;
	}
	Map::~Map() {}
	Frame* Map::GetReverseWindowFrame(int idx) {
		std::unique_lock<std::mutex> lock(mMutexWindowFrames);
		int n = 0;
		for (auto iter = mQueueFrameWindows1.rbegin(); iter != mQueueFrameWindows1.rend(); iter++, n++) {
			if (n == idx) {
				return *iter;
			}
		}
		return nullptr;
	}
	Frame* Map::GetLastWindowFrame() {
		std::unique_lock<std::mutex> lock(mMutexWindowFrames);
		return mQueueFrameWindows1.back();
	}
	Frame* Map::AddWindowFrame(Frame* pF){
		std::unique_lock<std::mutex> lock(mMutexWindowFrames);
		Frame* res = nullptr;
		if (mQueueFrameWindows1.size() == mnMaxConnectedKFs) {
			auto pKF = mQueueFrameWindows1.front();
			if (pKF->GetKeyFrameID() % 2 == 0) {
				mvpTrajectoryKFs.push_back(pKF);
				mQueueFrameWindows2.push_back(pKF);
				res = pKF;
				//pKF->SetBowVec(mpSystem->fvoc); //키프레임 파트로 옮기기
			}
			else {
				//pKF->mpMatchInfo->DisconnectAll();
			}
			mQueueFrameWindows1.pop_front();
			if (mQueueFrameWindows2.size() > mnHalfConnectedKFs) {
				auto pKF = mQueueFrameWindows2.front();
				if (pKF->GetKeyFrameID() % 4 == 0) {
					mQueueFrameWindows3.push_back(pKF);
					//res = pKF;
				}
				else {
					//pKF->mpMatchInfo->DisconnectAll();
				}
				/*mQueueFrameWindows3.push_back(pKF);
				res = pKF;*/
				mQueueFrameWindows2.pop_front();
			}
			if (mQueueFrameWindows3.size() > mnQuarterConnectedKFs) {
				auto pKF = mQueueFrameWindows3.front();
				//if (pKF->GetKeyFrameID() % 8 == 0) {
				//	mspGraphFrames.insert(pKF);
				//}
				//else {
				//	//pKF->mpMatchInfo->DisconnectAll();
				//}
				mspGraphFrames.insert(pKF);
				mQueueFrameWindows3.pop_front();
			}
		}
		mQueueFrameWindows1.push_back(pF);
		
		return res;
	}

	std::vector<Frame*> Map::GetTrajectoryFrames() {
		return mvpTrajectoryKFs;
	}

	//level = 1이면 첫번째 레벨의큐, 2이면 두번째 레벨의 큐 접근, 3이면 세번째 레벨 큐 접근
	std::vector<Frame*> Map::GetWindowFramesVector(int level){
		std::unique_lock<std::mutex> lock(mMutexWindowFrames);
		std::vector<Frame*> res;

		if (level > 2)
			for (auto iter = mQueueFrameWindows3.begin(); iter != mQueueFrameWindows3.end(); iter++) {
				res.push_back(*iter);
			}
		if (level > 1)
			for (auto iter = mQueueFrameWindows2.begin(); iter != mQueueFrameWindows2.end(); iter++) {
				res.push_back(*iter);
			}
		if(level > 0)
			for (auto iter = mQueueFrameWindows1.begin(); iter != mQueueFrameWindows1.end(); iter++) {
				res.push_back(*iter);
			}
		
		return res;
	}
	std::set<Frame*> Map::GetWindowFramesSet(int level) {
		std::unique_lock<std::mutex> lock(mMutexWindowFrames);
		std::set<Frame*> res;

		if (level > 2)
			for (auto iter = mQueueFrameWindows3.begin(); iter != mQueueFrameWindows3.end(); iter++) {
				res.insert(*iter);
			}
		if (level > 1)
			for (auto iter = mQueueFrameWindows2.begin(); iter != mQueueFrameWindows2.end(); iter++) {
				res.insert(*iter);
			}
		if (level > 0)
			for (auto iter = mQueueFrameWindows1.begin(); iter != mQueueFrameWindows1.end(); iter++) {
				res.insert(*iter);
			}
		return res;
	}
	std::vector<Frame*> Map::GetGraphFrames() {
		std::unique_lock<std::mutex> lock(mMutexWindowFrames);
		return std::vector<Frame*>(mspGraphFrames.begin(), mspGraphFrames.end());
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
////맵포인트 관리
void UVR_SLAM::Map::AddMap(MapPoint* pMP, int label){
	std::unique_lock<std::mutex> lock(mMutexMap);
	if(mmpMapMPs.find(pMP) == mmpMapMPs.end())
		mmpMapMPs.insert(std::make_pair(pMP, label));
}
void UVR_SLAM::Map::RemoveMap(MapPoint* pMP){
	std::unique_lock<std::mutex> lock(mMutexMap);
	if (mmpMapMPs.find(pMP) != mmpMapMPs.end())
		mmpMapMPs.erase(pMP);
}
std::map<UVR_SLAM::MapPoint*, int> UVR_SLAM::Map::GetMap() {
	std::unique_lock<std::mutex> lock(mMutexMap);
	return std::map<UVR_SLAM::MapPoint*, int>(mmpMapMPs.begin(), mmpMapMPs.end());
}
void UVR_SLAM::Map::AddDeleteMP(MapPoint* pMP) {
	std::unique_lock<std::mutex> lock(mMutexDeleteMapPointSet);
	auto findres = mspDeleteMPs.find(pMP);
	if (findres == mspDeleteMPs.end()) {
		mspDeleteMPs.insert(pMP);
		mnDeleteMPs++;
	}
}

void UVR_SLAM::Map::DeleteMPs() {
	std::unique_lock<std::mutex> lock(mMutexDeleteMapPointSet);
	/*int f = mQueueNumDelete.front();
	if (f == 0)
		mQueueNumDelete.pop();*/
	if (mQueueNumDelete.size() < 4)
		return;
	int n = 0;
	//mnDeleteMPs = 5;
	auto begin = mspDeleteMPs.begin();
	int nDelete = mQueueNumDelete.front();
	mQueueNumDelete.pop();
	std::set<MapPoint*>::iterator end = begin;
	for (auto iter = mspDeleteMPs.begin(); iter != mspDeleteMPs.end() && n < nDelete; iter++, n++) {
		end = iter;
		auto pMP = *iter;
		delete pMP;
	}
	end++;
	std::cout << "Map::Delete::AAA::" << nDelete <<", "<< mspDeleteMPs .size()<< std::endl;
	mspDeleteMPs.erase(begin, end);
	std::cout << "Map::Delete::BBB::" << nDelete <<", "<< mspDeleteMPs .size()<< std::endl;
	/*
	auto end = mspDeleteMPs.begin() + 5;
	mspDeleteMPs.erase()*/
	//mspDeleteMPs.clear();
}
void UVR_SLAM::Map::SetNumDeleteMP() {
	std::unique_lock<std::mutex> lock(mMutexDeleteMapPointSet);
	if(mnDeleteMPs > 0){
		mQueueNumDelete.push(mnDeleteMPs);
		mnDeleteMPs = 0;
	}
	//mnDeleteMPs = mspDeleteMPs.size();
}
////맵포인트 관리
//////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////
/////플로우 관리
void UVR_SLAM::Map::AddFlow(int nFrameID, cv::Mat flow) {
	std::unique_lock<std::mutex> lock(mMutexFlows);
	mmFlows.insert(std::make_pair(nFrameID, flow));
}
cv::Mat UVR_SLAM::Map::GetFlow(int nFrameID){
	std::unique_lock<std::mutex> lock(mMutexFlows);
	return mmFlows[nFrameID];
}
std::vector<cv::Mat> UVR_SLAM::Map::GetFlows(int nStartID, int nEndID) {
	
	std::map<int, cv::Mat>::iterator sIter, eIter;
	std::vector<cv::Mat> res;
	{
		std::unique_lock<std::mutex> lock(mMutexFlows);
		sIter = mmFlows.find(nStartID);
		eIter = mmFlows.find(nEndID);
	}
	for (auto iter = sIter; iter != eIter; iter++) {
		res.push_back(iter->second);
	}
	return res;
}
/////플로우 관리
////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////평면 관리
void UVR_SLAM::Map::AddPlaneInfo(PlaneProcessInformation* pPlane) {
	std::unique_lock<std::mutex> lock(mMutexPlaneInfo);
	mvpPlaneInfos.push_back(pPlane);
}
std::vector<UVR_SLAM::PlaneProcessInformation*> UVR_SLAM::Map::GetPlaneInfos() {
	std::unique_lock<std::mutex> lock(mMutexPlaneInfo);
	return std::vector<PlaneProcessInformation*>(mvpPlaneInfos.begin(), mvpPlaneInfos.end());
}
/////평면 관리
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////














void UVR_SLAM::Map::ClearWalls() {
	std::unique_lock<std::mutex> lock(mMutexWallPlanes);
	mvpWallPlanes.clear();
}

bool UVR_SLAM::Map::isFloorPlaneInitialized() {
	std::unique_lock<std::mutex> lockTemp(mMutexInitFloorPlane);
	return mbInitFloorPlane;
}
void UVR_SLAM::Map::SetFloorPlaneInitialization(bool b) {
	std::unique_lock<std::mutex> lockTemp(mMutexInitFloorPlane);
	mbInitFloorPlane = b;
}
bool UVR_SLAM::Map::isWallPlaneInitialized() {
	std::unique_lock<std::mutex> lockTemp(mMutexInitWallPlane);
	return mbInitWallPlane;
}
void UVR_SLAM::Map::SetWallPlaneInitialization(bool b) {
	std::unique_lock<std::mutex> lockTemp(mMutexInitWallPlane);
	mbInitWallPlane = b;
}

std::vector<UVR_SLAM::WallPlane*> UVR_SLAM::Map::GetWallPlanes() {
	std::unique_lock<std::mutex> lockTemp(mMutexWallPlanes);
	return std::vector<UVR_SLAM::WallPlane*>(mvpWallPlanes.begin(), mvpWallPlanes.end());
}
void UVR_SLAM::Map::AddWallPlane(WallPlane* pWall){
	std::unique_lock<std::mutex> lockTemp(mMutexWallPlanes);
	mvpWallPlanes.push_back(pWall);
}


////////////////////////////////////////
/////Map Grid
cv::Point3f UVR_SLAM::Map::ProjectMapPoint(UVR_SLAM::MapPoint* pMP, float fSize) {
	cv::Mat x3D = pMP->GetWorldPos();
	int nx = x3D.at<float>(0) / fSize;
	int ny = x3D.at<float>(1) / fSize;
	int nz = x3D.at<float>(2) / fSize;
	float fx = nx*fSize;
	float fy = ny*fSize;
	float fz = nz*fSize;
	return cv::Point3f(fx, fy, fz);
	//cv::Point2f tpt = cv::Point2f(x3D.at<float>(0) s* mnViScale, -x3D.at<float>(2) * mnVisScale);
}

bool UVR_SLAM::Map::CheckGrid(cv::Point3f pt) {
	std::unique_lock<std::mutex> lock(mMutexMapGrid);
	auto findres = mmMapGrids.find(pt);
	if (findres == mmMapGrids.end())
		return true;
	return false;
}
UVR_SLAM::MapGrid* UVR_SLAM::Map::InsertGrid(cv::Point3f pt) {
	std::unique_lock<std::mutex> lock(mMutexMapGrid);
	MapGrid* pMG = new UVR_SLAM::MapGrid(pt, mfMapGridSize);
	//int idx = mvMapGrids.size();
	mmMapGrids.insert(std::make_pair(pt, pMG));
	return pMG;
	/*mvMapGrids.push_back(pMG);
	return idx;*/
}
std::vector<UVR_SLAM::MapGrid*> UVR_SLAM::Map::GetMapGrids() {
	
	std::map<cv::Point3f, UVR_SLAM::MapGrid*, Point3fLess> tempMap;
	{
		std::unique_lock<std::mutex> lock(mMutexMapGrid);
		tempMap = mmMapGrids;
	}
	std::vector<UVR_SLAM::MapGrid*> tempVec;
	for (auto iter = mmMapGrids.begin(); iter != mmMapGrids.end(); iter++) {
		tempVec.push_back(iter->second);
	}
	return std::vector<UVR_SLAM::MapGrid*>(tempVec.begin(), tempVec.end());
}

//////Map Grid & Map Points
bool UVR_SLAM::Map::CheckGrid(cv::Point3f pt1, cv::Point3f pt2){
	std::unique_lock<std::mutex> lock(mMutexMapGrid);
	if (pt1.x == pt2.x && pt1.y == pt2.y && pt1.z == pt2.z && pt1 != pt2)
		std::cout << "??????????????? check mp error" << std::endl;
	return pt1 == pt2;
}
UVR_SLAM::MapGrid* UVR_SLAM::Map::GetGrid(UVR_SLAM::MapPoint* pMP){
	std::unique_lock<std::mutex> lock(mMutexMapGrid);
	auto findres = mmMapPointAndMapGrids.find(pMP);
	if (findres == mmMapPointAndMapGrids.end()) {
		return nullptr;
	}
	return findres->second;
	//return mmMapPointAndMapGrids[pMP];
}
UVR_SLAM::MapGrid* UVR_SLAM::Map::GetGrid(cv::Point3f pt) {
	std::unique_lock<std::mutex> lock(mMutexMapGrid);
	auto findres = mmMapGrids.find(pt);
	if (findres == mmMapGrids.end()) {
		return nullptr;
	}
	return findres->second;
}
void UVR_SLAM::Map::InsertMapPoint(UVR_SLAM::MapPoint* pMP, UVR_SLAM::MapGrid* pMG){
	std::unique_lock<std::mutex> lock(mMutexMapGrid);
	mmMapPointAndMapGrids[pMP] = pMG;
	pMG->InsertMapPoint(pMP);
}
void UVR_SLAM::Map::DeleteMapPoint(UVR_SLAM::MapPoint* pMP){
	std::unique_lock<std::mutex> lock(mMutexMapGrid);
	auto findres = mmMapPointAndMapGrids.find(pMP);
	if (findres != mmMapPointAndMapGrids.end()) {
		auto pMG2 = findres->second;
		pMG2->RemoveMapPoint(pMP);
		mmMapPointAndMapGrids.erase(findres);
	}
}
void UVR_SLAM::Map::UpdateMapPoint(UVR_SLAM::MapPoint* pMP, UVR_SLAM::MapGrid* pMG) {
	std::unique_lock<std::mutex> lock(mMutexMapGrid);
	auto findres = mmMapPointAndMapGrids.find(pMP);
	if (findres != mmMapPointAndMapGrids.end()) {
		auto pMG2 = findres->second;
		pMG2->RemoveMapPoint(pMP);
		mmMapPointAndMapGrids.erase(findres);
	}
	mmMapPointAndMapGrids[pMP] = pMG;

}



//////Map Grid & Map Points
/////Map Grid
////////////////////////////////////////

//////////Reinit test code
std::vector<cv::Mat> UVR_SLAM::Map::GetReinit(){
	std::unique_lock<std::mutex> lock(mMutexReinit);
	return std::vector<cv::Mat>(mvReinit.begin(), mvReinit.end());
}
void UVR_SLAM::Map::ClearReinit(){
	std::unique_lock<std::mutex> lock(mMutexReinit);
	mvReinit.clear();
}
void UVR_SLAM::Map::AddReinit(cv::Mat m){
	mvReinit.push_back(m);
}
//////////Reinit test code