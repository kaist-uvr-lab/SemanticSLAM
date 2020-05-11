#include "Map.h"
#include "Frame.h"
#include "MapPoint.h"
#include "Plane.h"

UVR_SLAM::Map::Map():mbInitFloorPlane(false), mbInitWallPlane(false), mpCurrFrame(nullptr){}
UVR_SLAM::Map::~Map(){}

//////////////////////////////////////////////////////////////////////////////////////////////////////
////맵포인트 관리
void UVR_SLAM::Map::AddMap(MapPoint* pMP, int label){
	std::unique_lock<std::mutex> lock(mMutexMap);
	mmpMapMPs.insert(std::make_pair(pMP, label));
}
void UVR_SLAM::Map::RemoveMap(MapPoint* pMP){
	std::unique_lock<std::mutex> lock(mMutexMap);
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

void UVR_SLAM::Map::SetCurrFrame(Frame* pF) {
	std::unique_lock<std::mutex> lock(mMutexCurrFrame);
	mpCurrFrame = pF;
}
UVR_SLAM::Frame* UVR_SLAM::Map::GetCurrFrame() {
	std::unique_lock<std::mutex> lock(mMutexCurrFrame);
	return mpCurrFrame;
}

void UVR_SLAM::Map::AddFrame(Frame* pF) {
	std::unique_lock<std::mutex> lock(mMutexGlobalFrames);
	mvpGlobalFrames.push_back(pF);
	if (mvpGlobalFrames.size() % 5 == 0) {
		mvpLoopFrames.push_back(pF);
	}
}
std::vector<UVR_SLAM::Frame*> UVR_SLAM::Map::GetFrames() {
	std::unique_lock<std::mutex> lock(mMutexGlobalFrames);
	return std::vector<UVR_SLAM::Frame*>(mvpGlobalFrames.begin(), mvpGlobalFrames.end());
}
std::vector<UVR_SLAM::Frame*> UVR_SLAM::Map::GetLoopFrames() {
	std::unique_lock<std::mutex> lock(mMutexGlobalFrames);
	return std::vector<UVR_SLAM::Frame*>(mvpLoopFrames.begin(), mvpLoopFrames.end());
}
void UVR_SLAM::Map::ClearFrames() {
	std::unique_lock<std::mutex> lock(mMutexGlobalFrames);
	mvpGlobalFrames.clear();
	mvpLoopFrames.clear();
}

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