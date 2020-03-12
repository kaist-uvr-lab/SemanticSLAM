#include "Map.h"
#include "Frame.h"
#include "Plane.h"

UVR_SLAM::Map::Map():mbInitFloorPlane(false), mbInitWallPlane(false){}
UVR_SLAM::Map::~Map(){}

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