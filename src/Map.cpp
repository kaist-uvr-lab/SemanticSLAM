#include "Map.h"
#include "Frame.h"
#include "MapPoint.h"
#include "Plane.h"
#include "MapGrid.h"
#include "System.h"
#include <WebAPI.h>
#include <Base64Encoder.h>
#include <rapidjson\document.h>
#include <future>

namespace UVR_SLAM{
	Map::Map():mnMaxConnectedKFs(8), mnHalfConnectedKFs(4), mnQuarterConnectedKFs(2), mnMaxCandidateKFs(4), mnHalfCandidate(2), mbInitFloorPlane(false), mbInitWallPlane(false){}
	Map::Map(System* pSystem, int nConnected, int nCandiate) :mpSystem(pSystem), mnMaxConnectedKFs(nConnected), mnHalfConnectedKFs(nConnected/2), mnQuarterConnectedKFs(nConnected/4), mnMaxCandidateKFs(nCandiate), mnHalfCandidate(nCandiate/2), mbInitFloorPlane(false), mbInitWallPlane(false){
		std::cout << "MAP::" << mnMaxConnectedKFs << ", " << mnMaxCandidateKFs << std::endl;
	}
	Map::~Map() {}
	void Map::Reset() {
		std::map<UVR_SLAM::MapPoint*, int> maps;
		{
			std::unique_lock<std::mutex> lock1(mMutexFrames);
			std::unique_lock<std::mutex> lock2(mMutexMap);
			//maps = mmpMapMPs;
			for (auto iter = mmpMapMPs.begin(), iend = mmpMapMPs.end(); iter != iend; iter++) {
				auto pMP = iter->first;
				pMP->DeleteMapPoint();
			}
			mmpMapMPs.clear();
			mspFrames.clear();
		}
		
	}

	void Map::LoadMapDataFromServer(std::string mapname, std::vector<Frame*>& vpMapFrames) {
		/*
		auto f1 = std::async(std::launch::async, [](std::string ip, int port, std::string mapname) {
			std::stringstream ss;
			ss << "/SendData?map="<< mapname<<"&attr=MapPoints&id=-1&key=bmpids";
			WebAPI* wapi = new WebAPI(ip, port);
			auto resdata = wapi->Send(ss.str(), "");
			////처리 후 전송
			int n = resdata.size() / 4;
			cv::Mat mpids = cv::Mat::zeros(1, resdata.size()/4, CV_32SC1);
			std::memcpy(mpids.data, resdata.data(), n* sizeof(int));
			return mpids;
		},mpSystem->ip, mpSystem->port, mapname);
		auto f2 = std::async(std::launch::async, [](std::string ip, int port, std::string mapname) {
			std::stringstream ss;
			ss << "/SendData?map=" << mapname << "&id=-1&key=bkfids";
			WebAPI* wapi = new WebAPI(ip, port);
			auto resdata = wapi->Send(ss.str(), "");
			////처리 후 전송
			int n = resdata.size() / 4;
			cv::Mat mpids = cv::Mat::zeros(1, resdata.size() / 4, CV_32SC1);
			std::memcpy(mpids.data, resdata.data(), n * sizeof(int));
			return mpids;
		}, mpSystem->ip, mpSystem->port, mapname);
		auto f3 = std::async(std::launch::async, [](std::string ip, int port, std::string mapname) {
			std::stringstream ss;
			ss << "/SendData?map=" << mapname << "&id=-1&attr=MapPoints&key=bx3ds";
			WebAPI* wapi = new WebAPI(ip, port);
			auto resdata = wapi->Send(ss.str(), "");
			////처리 후 전송
			int n = resdata.size() / 4;
			cv::Mat mpids = cv::Mat::zeros(resdata.size() / 4, 1, CV_32FC1);
			std::memcpy(mpids.data, resdata.data(), n * sizeof(int));
			return mpids;
		}, mpSystem->ip, mpSystem->port, mapname);
		auto f4 = std::async(std::launch::async, [](std::string ip, int port, std::string mapname) {
			std::stringstream ss;
			ss << "/SendData?map=" << mapname << "&id=-1&key=bmpidxs";
			WebAPI* wapi = new WebAPI(ip, port);
			auto resdata = wapi->Send(ss.str(), "");
			////처리 후 전송
			int n = resdata.size() / 4;
			cv::Mat mpids = cv::Mat::zeros(1, resdata.size() / 4, CV_32SC1);
			std::memcpy(mpids.data, resdata.data(), n * sizeof(int));
			return mpids;
		}, mpSystem->ip, mpSystem->port, mapname);
		auto f5 = std::async(std::launch::async, [](std::string ip, int port, std::string mapname) {
			std::stringstream ss;
			ss << "/SendData?map=" << mapname << "&id=-1&key=bposes";
			WebAPI* wapi = new WebAPI(ip, port);
			auto resdata = wapi->Send(ss.str(), "");
			////처리 후 전송
			int n = resdata.size() / 4;
			cv::Mat mpids = cv::Mat::zeros(1, resdata.size() / 4, CV_32FC1);
			std::memcpy(mpids.data, resdata.data(), n * sizeof(int));
			return mpids;
		}, mpSystem->ip, mpSystem->port, mapname);

		auto mpIDs = f1.get();
		auto kfIDs = f2.get();
		auto mpXws = f3.get();
		auto kfPoses = f5.get();
		auto kfMpIdxs = f4.get();

		std::cout << mpIDs.cols << ", " << kfIDs.cols << ", " << mpXws.rows << std::endl;
		std::map<int, MapPoint*> mmpMPs;
		for (int i = 0, iend = mpIDs.cols; i < iend; i++) {
			int id = mpIDs.at<int>(i);
			cv::Mat Xw = mpXws.rowRange(3 * i, 3 * (i + 1));
			auto pNewMP = new MapPoint(this, Xw, cv::Mat());
			mmpMPs[id] = pNewMP;
		}

		for (int i = 0, iend = kfIDs.cols, mpIndex = 0; i < iend; i++) {
			int id = kfIDs.at<int>(i);
			auto fimg = std::async(std::launch::async, [](std::string ip, int port, int id, std::string map, int w, int h) {
				WebAPI* mpAPI = new WebAPI(ip, port);
				std::stringstream ss;
				ss << "/SendData?map=" << map << "&id=" << id << "&key=bimage";
				auto res = mpAPI->Send(ss.str(), "");
				int n = res.size();
				cv::Mat temp = cv::Mat::zeros(n,1, CV_8UC1);
				std::memcpy(temp.data, res.data(), n * sizeof(uchar));
				cv::Mat img = cv::imdecode(temp, cv::IMREAD_COLOR);
				return img;
			}, mpSystem->ip, mpSystem->port, id, mapname, mpSystem->mnWidth, mpSystem->mnHeight);

			auto fpts = std::async(std::launch::async, [](std::string ip, int port, int id, std::string map) {
				WebAPI* mpAPI = new WebAPI(ip, port);
				std::stringstream ss;
				ss << "/SendData?map=" << map << "&id=" << id << "&key=bkpts";
				auto res = mpAPI->Send(ss.str(), "");
				int n = res.size() / 8;
				cv::Mat seg = cv::Mat::zeros(n, 2, CV_32FC1);
				std::memcpy(seg.data, res.data(), n * 2 * sizeof(float));
				std::vector<cv::Point2f> resPts;
				for (int j = 0; j < n; j++) {
					cv::Point2f pt(seg.at<float>(2 * j), seg.at<float>(2 * j + 1));
					resPts.push_back(pt);
				}
				return resPts;
			},mpSystem->ip, mpSystem->port, id, mapname);

			auto fdesc = std::async(std::launch::async, [](std::string ip, int port, int id, std::string map) {
				cv::Mat desc;
				WebAPI* mpAPI = new WebAPI(ip, port);
				std::stringstream ss;
				ss << "/SendData?map=" << map << "&id=" << id << "&key=bdesc";
				auto res = mpAPI->Send(ss.str(), "");
				int n = res.size() / (256 * 4);
				WebAPIDataConverter::ConvertBytesToDesc(res.c_str(), n, desc);
				return desc;

			}, mpSystem->ip, mpSystem->port, id, mapname);

			auto img = fimg.get();
			auto vPTs = fpts.get();
			auto desc = fdesc.get();

			Frame* newFrame = new UVR_SLAM::Frame(mpSystem, img, img.cols, img.rows, mpSystem->mK, 0.0);
			newFrame->SetMapPoints(vPTs.size());
			newFrame->mnFrameID = id;
			newFrame->mvPts = std::vector<cv::Point2f>(vPTs.begin(), vPTs.end());
			newFrame->matDescriptor = desc.clone();
			newFrame->ComputeBoW();
			vpMapFrames.push_back(newFrame);

			//mp와 pose
			cv::Mat R = kfPoses.colRange(12 * i, 12 * i + 9).reshape(1, 3);
			cv::Mat t = kfPoses.colRange(12 * i + 9, 12 * i + 12).t();
			newFrame->SetPose(R, t);

			int mpEndIndex = mpIndex + vPTs.size();
			cv::Mat idxs = kfMpIdxs.colRange(mpIndex, mpEndIndex);
			for (int j = 0, jend = idxs.cols; j < jend; j++) {
				int mpID = idxs.at<int>(j);
				if (mpID == -1) {
					newFrame->AddMapPoint(nullptr, j);
					continue;
				}
				auto pMP = mmpMPs[mpID];
				newFrame->AddMapPoint(pMP, j);
				pMP->AddObservation(newFrame, j);
			}
			mpIndex = mpEndIndex;
		}
		*/
	}

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

		/*if (mQueueFrameWindows1.size() == mnMaxConnectedKFs) {
			auto pKF1 = mQueueFrameWindows1.back();
			mQueueFrameWindows1.pop_back();
			
			auto pKFs = pF->GetConnectedKFs(10);
			Frame* pKF2 = nullptr;
			for(size_t i = 0, iend = pKFs.size(); i < iend; i++){
				auto pKF = pKFs[i];
				if (pKF->mnKeyFrameID != pKF1->mnKeyFrameID) {
					pKF2 = pKF;
					break;
				}
			}
			for (auto iter = mQueueFrameWindows1.begin(), iend = mQueueFrameWindows1.end(); iter != iend; iter++) {
				auto pKF = *iter;
				if (pKF->mnKeyFrameID != pKF1->mnKeyFrameID && pKF->mnKeyFrameID != pKF2->mnKeyFrameID) {
					mspGraphFrames.insert(pKF);
				}
			}
			mQueueFrameWindows1.clear();
			mQueueFrameWindows1.push_back(pKF1);
			mQueueFrameWindows1.push_back(pKF2);
		}*/

		if (mQueueFrameWindows1.size() == mnMaxConnectedKFs) {
			auto pKF = mQueueFrameWindows1.front();
			//mvpTrajectoryKFs.push_back(pKF);
			if (pKF->mnKeyFrameID % 2 == 0) {
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
				if (pKF->mnKeyFrameID % 4 == 0) {
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

///////////////////////////////////
////키프레임 관리
void UVR_SLAM::Map::AddFrame(Frame* pF) {
	std::unique_lock<std::mutex> lock(mMutexFrames);
	mspFrames.insert(pF);
}
void UVR_SLAM::Map::RemoveFrame(Frame* pF) {
	std::unique_lock<std::mutex> lock(mMutexFrames);
	mspFrames.erase(pF);
}
std::vector<UVR_SLAM::Frame*> UVR_SLAM::Map::GetFrames() {
	std::unique_lock<std::mutex> lock(mMutexFrames);
	return std::vector<Frame*>(mspFrames.begin(), mspFrames.end());
}
///////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////
////맵포인트 관리
void UVR_SLAM::Map::AddMap(MapPoint* pMP, int label){
	std::unique_lock<std::mutex> lock(mMutexMap);
	if(mmpMapMPs.find(pMP) == mmpMapMPs.end())
		mmpMapMPs.insert(std::make_pair(pMP, label));
}
void UVR_SLAM::Map::RemoveMap(MapPoint* pMP){
	std::unique_lock<std::mutex> lock(mMutexMap);
	if (mmpMapMPs.find(pMP) != mmpMapMPs.end()){
		mmpMapMPs.erase(pMP);
	}
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



///////////////////////////////
////



namespace UVR_SLAM {
	MapGrid* Map::AddMapGrid(cv::Point3f key) {
		auto newGrid = new MapGrid();
		std::unique_lock<std::mutex> lock(mMutexMapGrids);
		mmpMapGrids[key] = newGrid;
		return newGrid;
	}
	MapGrid* Map::GetMapGrid(cv::Point3f key) {
		std::unique_lock<std::mutex> lock(mMutexMapGrids);
		if (mmpMapGrids.count(key))
			return mmpMapGrids[key];
		else
			return nullptr;
	}
	std::vector<MapGrid*> Map::GetMapGrids() {
		std::map<cv::Point3f, MapGrid*, Point3fLess> temp;
		std::vector<MapGrid*> res;
		{
			std::unique_lock<std::mutex> lock(mMutexMapGrids);
			temp = mmpMapGrids;
		}
		for (auto iter = temp.begin(), iend = temp.end(); iter != iend; iter++) {
			auto pGrid = iter->second;
			res.push_back(pGrid);
		}
		return res;
	}
}

////MapGrid
///////////////////////////////









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

//////////Reinit test code
std::vector<cv::Point2f> UVR_SLAM::Map::GetTrackingPoints() {
	std::unique_lock<std::mutex> lock(mMutexTrackingPTs);
	return mvTrackingPTs;
}
void UVR_SLAM::Map::SetTrackingPoints(std::vector<cv::Point2f> vPTs) {
	std::unique_lock<std::mutex> lock(mMutexTrackingPTs);
	mvTrackingPTs = vPTs;
}
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
std::vector<cv::Mat> UVR_SLAM::Map::GetTempMPs() {
	std::unique_lock<std::mutex> lock(mMutexReinit);
	return std::vector<cv::Mat>(mvTempMPs.begin(), mvTempMPs.end());
}
void UVR_SLAM::Map::ClearTempMPs() {
	std::unique_lock<std::mutex> lock(mMutexReinit);
	mvTempMPs.clear();
}
void UVR_SLAM::Map::AddTempMP(cv::Mat m) {
	mvTempMPs.push_back(m);
}
//////////Reinit test code