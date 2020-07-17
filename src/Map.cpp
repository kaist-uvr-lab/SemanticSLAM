#include "Map.h"
#include "MapGrid.h"
#include "Frame.h"
#include "MapPoint.h"
#include "Plane.h"

UVR_SLAM::Map::Map():mbInitFloorPlane(false), mbInitWallPlane(false), mpCurrFrame(nullptr), mfMapGridSize(0.2){}
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

void UVR_SLAM::Map::AddTraFrame(Frame* pF) {
	std::unique_lock<std::mutex> lock(mMutexAllFrames);
	mvpAllTrajectoryFrames.push_back(pF);
}
std::vector<UVR_SLAM::Frame*> UVR_SLAM::Map::GetAllTrajectoryFrames() {
	std::unique_lock<std::mutex> lock(mMutexAllFrames);
	return std::vector<UVR_SLAM::Frame*>(mvpAllTrajectoryFrames.begin(), mvpAllTrajectoryFrames.end());
}

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