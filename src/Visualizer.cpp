#include <Visualizer.h>
#include <Frame.h>
#include <MapPoint.h>
#include <System.h>
#include <CandidatePoint.h>
#include <PlaneEstimator.h>
#include <Map.h>
#include <MapGrid.h>
#include <plane.h>
#include <WebAPI.h>
#include <ServerMap.h>
#include <User.h>

UVR_SLAM::Visualizer::Visualizer() {}
UVR_SLAM::Visualizer::Visualizer(System* pSystem) :mpSystem(pSystem), mnFontFace(2), mfFontScale(0.6){
}
UVR_SLAM::Visualizer::~Visualizer() {}

void UVR_SLAM::Visualizer::AddUser(User* pUser) {
	std::unique_lock<std::mutex>lock(mMutexUserList);
	mspUserLists.insert(pUser);
}
void UVR_SLAM::Visualizer::RemoveUser(User* pUser) {
	std::unique_lock<std::mutex>lock(mMutexUserList);
	if (mspUserLists.count(pUser))
		mspUserLists.erase(pUser);
}
std::vector<UVR_SLAM::User*> UVR_SLAM::Visualizer::GetUsers() {
	std::unique_lock<std::mutex>lock(mMutexUserList);
	return std::vector<UVR_SLAM::User*>(mspUserLists.begin(), mspUserLists.end());
}

void CalcLineEquation(cv::Point2f pt1, cv::Point2f pt2, float& slope, float& dist) {
	float dx = pt2.x - pt1.x;	//a
	float dy = pt2.y - pt1.y;   //b
	slope;
	if (dx == 0.0)
		slope = 1000.0;
	else
		slope = dy / dx;
	if (dx == 0.0) {
		dist = 0.0;
	}
	else if (dy == 0.0) {
		dist = pt1.y;
	}else
		dist = pt1.y - slope*pt1.x;

}

////////////////////////////
int mnMode = 2;
int mnMaxMode = 3;
int mnAxis1 = 0;
int mnAxis2 = 2;

bool bSaveMap = false;
bool bLoadMap = false;
bool bShowOnlyTrajectory = true;


int UVR_SLAM::Visualizer::GetScale(){
	std::unique_lock<std::mutex>lock(mMutexScale);
	return mnVisScale;
}
void UVR_SLAM::Visualizer::SetScale(int s){
	std::unique_lock<std::mutex>lock(mMutexScale);
	mnVisScale = s;
}

void SetAxisMode() {
	switch (mnMode) {
	case 0:
		mnAxis1 = 0;
		mnAxis2 = 2;
		break;
	case 1:
		mnAxis1 = 1;
		mnAxis2 = 2;
		break;
	case 2:
		mnAxis1 = 0;
		mnAxis2 = 1;
		break;
	}
}
cv::Point2f rectPt;
void UVR_SLAM::Visualizer::CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	int* tempData = (int*)userdata;
	
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		//std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
		tempData[2] = x - tempData[0];
		tempData[3] = y;

		////button interface
		if (tempData[2] < 50 && y < 50) {
			bSaveMap = true;
		}else if (tempData[2] < 50 && (y >= 50 && y < 100)) {
			//bShowOnlyTrajectory = !bShowOnlyTrajectory;
			bLoadMap = true;
		}
		////button interface
	}
	else if (event == cv::EVENT_LBUTTONUP)
	{
		//std::cout << "Left button of the mouse is released - position (" << x << ", " << y << ")" << std::endl;
		tempData[4] = x - tempData[0];
		tempData[5] = y;
		tempData[6] = tempData[4] - tempData[2];
		tempData[7] = tempData[5] - tempData[3];
	}
	else if (event == cv::EVENT_RBUTTONDOWN) {
		mnMode++;
		mnMode %= mnMaxMode;
		SetAxisMode();
		/*switch (mnMode) {
		case 0:
			mnAxis1 = 0;
			mnAxis2 = 2;
			break;
		case 1:
			mnAxis1 = 1;
			mnAxis2 = 2;
			break;
		case 2:
			mnAxis1 = 0;
			mnAxis2 = 1;
			break;
		}*/

	}else if (event == cv::EVENT_MOUSEWHEEL) {
		//std::cout << "Wheel event detection" << std::endl;
		if (flags > 0) {
			//scroll up
			tempData[1] += 20;
		}
		else {
			//scroll down
			tempData[1] -= 20;
			if (tempData[1] <= 0){
				tempData[1] = 20;
			}
		}
	}
}

void UVR_SLAM::Visualizer::Init(int w, int h) {
	
	mpPlaneEstimator = mpSystem->mpPlaneEstimator;
	mpMap = mpSystem->mpMap;
	mnDisplayX = mpSystem->mnDisplayX;
	mnDisplayY = mpSystem->mnDisplayY;

	mnWidth = w;
	mnHeight = h;
	mnVisScale = mpSystem->mnVisScale;

	//Visualization
	mVisPoseGraph = cv::Mat(mnHeight * 2, mnWidth * 2, CV_8UC3, cv::Scalar(255, 255, 255));
	rectangle(mVisPoseGraph, cv::Rect(0, 0, 50, 50), cv::Scalar(255, 255, 0), -1);
	rectangle(mVisPoseGraph, cv::Rect(0, 50, 50, 50), cv::Scalar(0, 255, 255), -1);
	mVisMidPt = cv::Point2f(mnHeight, mnWidth);
	mVisPrevPt = mVisMidPt;

	////맵 옆의 4개의 이미지
	//tracking, segmentation, mapping, ??
	cv::Mat leftImg1 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);
	cv::Mat leftImg2 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);
	cv::Mat leftImg3 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);
	cv::Mat leftImg4 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);
	
	//맵
	cv::Mat mapImage = cv::Mat::zeros(mnHeight * 2, mnWidth * 2, CV_8UC3);

	//sliding window
	mnWindowImgRows = 4;
	int nWindowSize = mpMap->mnMaxConnectedKFs + mpMap->mnHalfConnectedKFs + mpMap->mnQuarterConnectedKFs;
	mnWindowImgCols = nWindowSize / mnWindowImgRows;
	if (nWindowSize % 4 != 0)
		mnWindowImgCols++;
	cv::Mat kfWindowImg = cv::Mat::zeros(mnWindowImgRows*mnHeight / 2, mnWindowImgCols * mnWidth / 2, CV_8UC3);
	
	//0 1 2 3
	mvOutputImgs.push_back((leftImg1));
	cv::Rect r1(0, 0, leftImg1.cols, leftImg1.rows);
	mvRects.push_back(r1);
	mvOutputImgs.push_back((leftImg2));
	cv::Rect r2(0, leftImg1.rows,	leftImg1.cols, leftImg1.rows);
	mvRects.push_back(r2);
	mvOutputImgs.push_back((leftImg3));
	cv::Rect r3(0, leftImg1.rows * 2, leftImg1.cols, leftImg1.rows);
	mvRects.push_back(r3);
	mvOutputImgs.push_back((leftImg4));
	cv::Rect r4(0, leftImg1.rows * 3, leftImg1.cols, leftImg1.rows);
	mvRects.push_back(r4);

	//4
	mvOutputImgs.push_back((mapImage));
	cv::Rect rMap(leftImg1.cols, 0, mapImage.cols, mapImage.rows);
	mvRects.push_back(rMap);

	//5 윈도우이미지
	mvOutputImgs.push_back((kfWindowImg));
	cv::Rect rWindow(leftImg1.cols + mapImage.cols, 0, kfWindowImg.cols, kfWindowImg.rows);
	mvRects.push_back(rWindow);

	mvOutputChanged = std::vector<bool>(mvRects.size(), false);
	//map
	
	rectPt = cv::Point2f(r3.x, r3.y);
	int nDisRows = mnHeight * 2;
	int nDisCols = leftImg1.cols + mapImage.cols + kfWindowImg.cols;
	mOutputImage = cv::Mat::zeros(nDisRows, nDisCols, CV_8UC3);
	
	/*std::cout << nDisRows << ", " << nDisCols <<"::"<<img1.cols<<", "<<img2.cols<<", "<<img3.cols<< std::endl;
	std::cout << r1.x << " " << r2.x << ", " << r3.x << "::" << r1.width << ", " << r2.width << ", " << r3.width << std::endl;*/

	//set image
	/*cv::namedWindow("Output::Display");*/
}

void UVR_SLAM::Visualizer::SetBoolDoingProcess(bool b) {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	mbDoingProcess = b;
}
bool UVR_SLAM::Visualizer::isDoingProcess() {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	return mbDoingProcess;
}
void UVR_SLAM::Visualizer::SetOutputImage(cv::Mat out, int type) {
	std::unique_lock<std::mutex> lockTemp(mMutexOutput);
	mvOutputImgs[type] = out.clone();
	mvOutputChanged[type] = true;
}
cv::Mat UVR_SLAM::Visualizer::GetOutputImage(int type){
	std::unique_lock<std::mutex> lockTemp(mMutexOutput);
	mvOutputChanged[type] = false;
	return mvOutputImgs[type].clone();
}
bool UVR_SLAM::Visualizer::isOutputTypeChanged(int type) {
	std::unique_lock<std::mutex> lockTemp(mMutexOutput);
	return mvOutputChanged[type];
}
void UVR_SLAM::Visualizer::RunWithMappingServer() {
	std::cout << "MappingServer::Visualizer::Start" << std::endl;
	SetAxisMode();
	int nFrame = 0;
	//현재 여기서 시각화가 안됨
	//원인 찾는 중
	cv::imshow("Output::Display", mOutputImage);
	cv::moveWindow("Output::Display", mnDisplayX, mnDisplayY);
	int mapControlData[8] = {0,};
	mapControlData[0] = mnWidth;
	mapControlData[1] = mnVisScale;
	cv::setMouseCallback("Output::Display", UVR_SLAM::Visualizer::CallBackFunc, (void*)mapControlData);

	while (true) {
		
		if (bSaveMap) {
			WebAPI* mpAPI = new WebAPI(mpSystem->ip, mpSystem->port);

			auto mmpMap = mpServerMap->GetMapPoints();
			cv::Mat ids = cv::Mat::zeros(0, 1, CV_32SC1);
			cv::Mat x3ds = cv::Mat::zeros(0, 1, CV_32FC1);
			for (auto iter = mmpMap.begin(); iter != mmpMap.end(); iter++) {
				auto pMPi = *iter;//->first;
				if (!pMPi || pMPi->isDeleted())
					continue;
				cv::Mat Xw = pMPi->GetWorldPos();
				cv::Mat temp = cv::Mat::zeros(1, 1, CV_32FC1);
				temp.at<int>(0) = pMPi->mnMapPointID;
				ids.push_back(temp);
				x3ds.push_back(Xw);
			}
			int nMP = ids.rows;
			auto mspKFs = mpServerMap->GetFrames();
			
			cv::Mat poses = cv::Mat::zeros(0, 3, CV_32FC1);
			cv::Mat mps = cv::Mat::zeros(0, 1, CV_32SC1);
			cv::Mat kfids = cv::Mat::zeros(mspKFs.size(), 1, CV_32SC1);
			int j = 0;
			for (auto iter = mspKFs.begin(); iter != mspKFs.end(); iter++, j++) {
				auto pKFi = *iter;
				kfids.at<int>(j) = pKFi->mnFrameID;
				cv::Mat R, t;
				pKFi->GetPose(R, t);
				poses.push_back(R);
				poses.push_back(t.t());
				
				auto vpMPs = pKFi->GetMapPoints();
				cv::Mat temp = cv::Mat::ones(vpMPs.size(), 1, CV_32SC1)*-1;
				for (int i = 0; i < vpMPs.size(); i++) {
					auto pMPi = vpMPs[i];
					if (!pMPi || pMPi->isDeleted()) {
						continue;
					}
					int id = pMPi->mnMapPointID;
					temp.at<int>(i) = id;
				}
				mps.push_back(temp);

				////키프레임 커넥션 데이터 전송
				auto vNeigKFs = pKFi->GetConnectedKFsWithWeight();
				cv::Mat kfData = cv::Mat::zeros(0, 1, CV_32SC1);
				for (auto jit = vNeigKFs.begin(), jitend = vNeigKFs.end(); jit != jitend; jit++) {
					auto weight = jit->first;
					auto id = jit->second->mnFrameID;
					cv::Mat temp = cv::Mat::zeros(2, 1, CV_32SC1);
					temp.at<int>(0) = id;
					temp.at<int>(1) = weight;
					kfData.push_back(temp);
				}
				std::stringstream ss;
				ss << "/ReceiveData?map=" << mspKFs[0]->mstrMapName << "&id=" << pKFi->mnFrameID << "&key=bconnectedkfs";
				mpAPI->Send(ss.str(), kfData.data, kfData.rows * sizeof(int));
			}
			std::cout << "test::" << kfids .rows<<" "<< poses.rows << " ," << mps.rows <<", "<< ids.rows <<" "<<x3ds.rows<<", "<< x3ds.at<float>(x3ds.rows-1)<< std::endl;
			
			mpAPI->Send("/SaveMap?map="+ mspKFs[0]->mstrMapName, WebAPIDataConverter::ConvertMapDataToJson(ids, x3ds, kfids, poses, mps));
			//일단 모든 데이터를 전송하기.
			//1)맵포인트 아이디와 좌표값으로 
			//2)모든 프레임에 대해 포즈와 맵포인트 아이디로
			bSaveMap = false;
		}
		if (bLoadMap) {
			////load map test
			/*WebAPI* mpAPI = new WebAPI(mpSystem->ip, mpSystem->port);
			mpAPI->Send("LoadMap", "");*/
			bLoadMap = false;
		}
		nFrame++;
		//////Update Visualizer 
		mVisMidPt += cv::Point2f(mapControlData[6], mapControlData[7]);
		mapControlData[6] = 0;
		mapControlData[7] = 0;
		mnVisScale = mapControlData[1];
		//////Update Visualizer 

		//if (isDoingProcess()) {
			cv::Mat tempVis = mVisPoseGraph.clone();
			auto mmpMap = mpServerMap->GetMapPoints();
			for (auto iter = mmpMap.begin(); iter != mmpMap.end(); iter++) {
				auto pMPi = *iter;// ->first;
				if (!pMPi || pMPi->isDeleted())
					continue;
				//int label = iter->second;
				cv::Mat x3D = pMPi->GetWorldPos();
				cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
				tpt += mVisMidPt;

				cv::Scalar color = cv::Scalar(0, 0, 0);
				cv::circle(tempVis, tpt, 2, color, -1);
			}

			auto lKFs = mpServerMap->GetFrames();
			int nMaxID = 0;
			UVR_SLAM::Frame* lastKF = nullptr;
			for (auto iter = lKFs.begin(); iter != lKFs.end(); iter++) {
				auto pKFi = *iter;
				cv::Mat t1 = pKFi->GetCameraCenter();
				cv::Point2f pt1 = cv::Point2f(t1.at<float>(mnAxis1)* mnVisScale, t1.at<float>(mnAxis2)* mnVisScale);
				pt1 += mVisMidPt;
				cv::circle(tempVis, pt1, 2, cv::Scalar(0, 155, 248), -1);
				if (nMaxID < pKFi->mnKeyFrameID) {
					nMaxID = pKFi->mnKeyFrameID;
					lastKF = pKFi;
				}
			}
			if (lKFs.size()>0)
			{
				auto currKF = lastKF;// lKFs[lKFs.size() - 1];
				cv::Mat t1 = currKF->GetCameraCenter();
				cv::Point2f pt1 = cv::Point2f(t1.at<float>(mnAxis1)* mnVisScale, t1.at<float>(mnAxis2)* mnVisScale);
				pt1 += mVisMidPt;
				cv::circle(tempVis, pt1, 3, cv::Scalar(0, 0, 255), -1);

				cv::Mat directionZ = currKF->GetRotation().row(2);
				cv::Point2f dirPtZ = cv::Point2f(directionZ.at<float>(mnAxis1)* mnVisScale / 10.0, directionZ.at<float>(mnAxis2)* mnVisScale / 10.0) + pt1;
				cv::line(tempVis, pt1, dirPtZ, cv::Scalar(255, 0, 0), 2);

				cv::Mat directionY = currKF->GetRotation().row(1);
				cv::Point2f dirPtY = cv::Point2f(directionY.at<float>(mnAxis1)* mnVisScale / 10.0, directionY.at<float>(mnAxis2)* mnVisScale / 10.0) + pt1;
				cv::line(tempVis, pt1, dirPtY, cv::Scalar(0, 255, 0), 2);

				cv::Mat directionX = currKF->GetRotation().row(0);
				cv::Point2f dirPtX1 = pt1 + cv::Point2f(directionX.at<float>(mnAxis1)* mnVisScale / 10.0, directionX.at<float>(mnAxis2)* mnVisScale / 10.0);
				cv::Point2f dirPtX2 = pt1 - cv::Point2f(directionX.at<float>(mnAxis1)* mnVisScale / 10.0, directionX.at<float>(mnAxis2)* mnVisScale / 10.0);
				cv::line(tempVis, dirPtX1, dirPtX2, cv::Scalar(0, 0, 255), 2);
			}

			{
				////User 위치 시각화
				//추후 유저별 정보로 변경
				auto vpUsers = GetUsers();
				for (size_t i = 0, iend = vpUsers.size(); i < iend; i++) {
					auto user = vpUsers[i];
					if (!user)
						continue;
					auto pos = user->GetCameraCenter();
					cv::Point2f pt1 = cv::Point2f(pos.at<float>(mnAxis1)* mnVisScale, pos.at<float>(mnAxis2)* mnVisScale);
					pt1 += mVisMidPt;
					cv::circle(tempVis, pt1, 5, cv::Scalar(0, 0, 255), -1);
				}
			}

			cv::Scalar color = cv::Scalar(0, 255, 255);
			auto vReinit = mpMap->GetTempMPs();
			for (int i = 0; i < vReinit.size(); i++) {
				cv::Mat x3D = vReinit[i];
				cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
				tpt += mVisMidPt;
				cv::circle(tempVis, tpt, 4, color, -1);
			}

			{
				cv::Scalar color = cv::Scalar(255, 255, 0);
				vReinit = mpMap->GetReinit();
				for (int i = 0; i < vReinit.size(); i++) {
					cv::Mat x3D = vReinit[i];
					cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
					tpt += mVisMidPt;
					cv::circle(tempVis, tpt, 4, color, -1);
				}
			}

			SetOutputImage(tempVis, 4);
			SetBoolDoingProcess(false);
		//}//doing process

		////////Update Map Visualizer
		if (isOutputTypeChanged(0)) {
			cv::Mat mTrackImg = GetOutputImage(0);
			mTrackImg.copyTo(mOutputImage(mvRects[0]));
		}
		if (isOutputTypeChanged(1)) {
			cv::Mat mWinImg = GetOutputImage(1);
			mWinImg.copyTo(mOutputImage(mvRects[1]));
		}
		if (isOutputTypeChanged(2)) {
			cv::Mat mMapImg = GetOutputImage(2);
			mMapImg.copyTo(mOutputImage(mvRects[2]));
		}
		if (isOutputTypeChanged(3)) {
			cv::Mat mMappingImg = GetOutputImage(3);
			mMappingImg.copyTo(mOutputImage(mvRects[3]));
		}
		if (isOutputTypeChanged(4)) {
			cv::Mat mMappingImg = GetOutputImage(4);
			mMappingImg.copyTo(mOutputImage(mvRects[4]));
		}
		if (isOutputTypeChanged(5)) {
			cv::Mat mMappingImg = GetOutputImage(5);
			mMappingImg.copyTo(mOutputImage(mvRects[5]));
		}
		if (nFrame % 100 == 0) {
			/*imshow("Output::Display", mOutputImage);
			cv::waitKey(1);*/
		}
		imshow("Output::Display", mOutputImage);
		cv::waitKey(1);
		////////Update Map Visualizer
	}
	std::cout << "MappingServer::Visualizer::End" << std::endl;
}
void UVR_SLAM::Visualizer::Run() {

	std::vector<UVR_SLAM::MapPoint*> allDummyPts;

	std::vector<cv::Scalar> colors;
	auto tempColors = UVR_SLAM::ObjectColors::mvObjectLabelColors;
	for (int i = 0; i < tempColors.size(); i++) {
		colors.push_back(cv::Scalar(tempColors[i].val[0], tempColors[i].val[1], tempColors[i].val[2]));
	}
	SetAxisMode();
	while (true) {
		
		if (bSaveMap) {
			auto frames = mpMap->GetGraphFrames();
			int nFrame = frames.size();
			auto frame = frames[nFrame - 1];
			std::stringstream sss;
			sss << mpSystem->GetDirPath(0) << "/map/map_"<< frame->mnKeyFrameID<<".txt";
			std::ofstream f;
			f.open(sss.str().c_str());
			auto mmpMap = mpMap->GetMap();
			for (auto iter = mmpMap.begin(); iter != mmpMap.end(); iter++) {
				auto pMPi = iter->first;
				int label = iter->second;
				if (!pMPi || pMPi->isDeleted())
					continue;
				cv::Mat Xw = pMPi->GetWorldPos();
				f << Xw.at<float>(0) << " " << Xw.at<float>(1) << " " << Xw.at<float>(2) <<" "<<label<<" "<<pMPi->GetNumObservations()<< std::endl;
				/*bool bPlane = pMPi->GetRecentLayoutFrameID() > 0;
				cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, -x3D.at<float>(mnAxis2) * mnVisScale);
				tpt += mVisMidPt;
				cv::Scalar color = cv::Scalar(0, 0, 0);
				if (label == 255) {
					color = cv::Scalar(125, 125, 0);
					if (bPlane)
						color = cv::Scalar(255, 255, 0);
				}
				else if (label == 150) {
					color = cv::Scalar(125, 0, 125);
					if (bPlane)
						color = cv::Scalar(255, 0, 255);
				}
				else if (label == 100) {
					color = cv::Scalar(0, 125, 125);
					if (bPlane)
						color = cv::Scalar(0, 255, 255);
				}
				cv::circle(tempVis, tpt, 2, color, -1);*/
			}
			f.close();
			////////save txt
			///*std::ofstream f;
			//std::stringstream sss;
			//sss << mStrPath.c_str() << "/plane.txt";
			//f.open(sss.str().c_str());
			//mvpMPs = mpTargetFrame->GetMapPoints();
			//mvpOPs = mpTargetFrame->GetObjectVector();
			//if(bLocalFloor)
			//	for (int j = 0; j < mvpMPs.size(); j++) {
			//		UVR_SLAM::MapPoint* pMP = mvpMPs[j];
			//		if (!pMP) {
			//			continue;
			//		}
			//		if (pMP->isDeleted()) {
			//			continue;
			//		}
			//		cv::Mat Xw = pMP->GetWorldPos();
			//	
			//		if (pMP->GetPlaneID() > 0) {
			//			if (pMP->GetPlaneID() == tempFloorID)
			//			{
			//				f << Xw.at<float>(0) << " " << Xw.at<float>(1) << " " << Xw.at<float>(2) << " 1" << std::endl;
			//			}
			//			else if (pMP->GetPlaneID() == tempWallID) {
			//				f << Xw.at<float>(0) << " " << Xw.at<float>(1) << " " << Xw.at<float>(2) << " 2" << std::endl;
			//			}
			//		}
			//		else
			//			f << Xw.at<float>(0) << " " << Xw.at<float>(1) << " " << Xw.at<float>(2) << " 0" << std::endl;
			//	}
			//f.close();*/
			////////save txt
			bSaveMap = false;
		}

		//if (true) {

		//update pt
		/*mVisMidPt += mPt;
		mPt = cv::Point2f(0, 0);
		mnVisScale = nScale;*/

		if (isDoingProcess()) {

			cv::Mat tempVis = mVisPoseGraph.clone();
			

			cv::Scalar color1 = cv::Scalar(0, 0, 255);
			cv::Scalar color2 = cv::Scalar(0, 255, 0);
			cv::Scalar color3 = cv::Scalar(0, 0, 0);

			///////////////////////////////////////////////////////////////////////////////
			////tracking results
			//local map
			if(bShowOnlyTrajectory){
				//////////////////////////////
				////전체 맵포인트 시각화
				////그리드
				//auto mvpMapGrid = mpMap->GetMapGrids();
				//for (auto iter = mvpMapGrid.begin(), iend = mvpMapGrid.end(); iter != iend; iter++) {
				//	auto pGrid = *iter;
				//	auto color = pGrid->mGridColor;
				//	cv::Scalar color2(0, 0, 0);
				//	auto vpMPs = pGrid->GetMapPoints();
				//	auto nGridID = pGrid->mnMapGridID;
				//	for (size_t i = 0, ii = vpMPs.size(); i < ii; i++) {
				//		auto pMPi = vpMPs[i];
				//		if (!pMPi || pMPi->isDeleted() || pMPi->GetMapGridID() != pGrid->mnMapGridID)
				//			continue;
				//		cv::Mat x3D = pMPi->GetWorldPos();
				//		cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
				//		tpt += mVisMidPt;
				//		cv::circle(tempVis, tpt, 2, color, -1);

				//		if (pGrid->mnTrackingID < pMatchInfo->mpRefFrame->mnFrameID)
				//		{
				//			cv::circle(tempVis, tpt, 2, color2, -1);
				//		}

				//		int diff = pGrid->mnTrackingID - pMPi->mnTrackingID;
				//		if(diff > 100 && pMPi->GetFVRatio() < 0.5f){
				//			//std::cout << "ratio : " << pMPi->GetFVRatio() << "::" << pMPi->mnTrackingID << std::endl;
				//			pMPi->Delete();
				//			//cv::circle(tempVis, tpt, 2, color2, -1);
				//		}
				//	}
				//}
				////그리드
				////맵
				auto mmpMap = mpMap->GetMap();
				for (auto iter = mmpMap.begin(); iter != mmpMap.end(); iter++) {
					auto pMPi = iter->first;
					if (!pMPi || pMPi->isDeleted())
						continue;
					//int label = iter->second;
					int label = pMPi->GetLabel();
					cv::Mat x3D = pMPi->GetWorldPos();
					bool bPlane = pMPi->GetPlaneID() > 0;
					cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
					tpt += mVisMidPt;
					
					cv::Scalar color = cv::Scalar(0, 0, 0);
					if (label == 255) {
						color = cv::Scalar(125, 125, 0);
						if(bPlane)
							color = cv::Scalar(255, 255, 0);
					}
					else if (label == 150) {
						color = cv::Scalar(125, 0, 125);
						if (bPlane)
							color = cv::Scalar(255, 0, 255);
					}
					else if (label == 100) {
						color = cv::Scalar(0, 125, 125);
						if (bPlane)
							color = cv::Scalar(0, 255, 255);
					}
					if(pMPi->GetNumObservations() < 7){
						color = cv::Scalar(0, 0, 0);
						continue;
					}
					cv::circle(tempVis, tpt, 2, UVR_SLAM::ObjectColors::mvObjectLabelColors[label], -1);
				}
				////맵
				////전체 맵포인트 시각화
				//////////////////////////////

				////////////////////////////////
				//////평면 시각화
				std::vector<cv::Mat> vpTempPlaneVisPTs;
				mpPlaneEstimator->GetTempPTs(vpTempPlaneVisPTs);
				cv::Scalar tempPlaneColor(255, 0, 255);
				for (size_t i = 0, iend = vpTempPlaneVisPTs.size(); i < iend; i++) {
					cv::Mat x3D = vpTempPlaneVisPTs[i];
					cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
					tpt += mVisMidPt;
					cv::circle(tempVis, tpt, 2, tempPlaneColor, -1);
				}
				//////평면 시각화
				////////////////////////////////
				

				////////////////////////////////////////////////////////////
				/////트래킹 결과 출력
				//if (pMatchInfo) {
				//	/*auto lastBAFrame = pMatchInfo->mpTargetFrame;*/
				//	{
				//		std::unique_lock<std::mutex> lock(mpSystem->mMutexUseCreateCP);
				//		mpSystem->cvUseCreateCP.wait(lock, [&] {return mpSystem->mbCreateCP; });
				//	}
				//	auto vpCPs = pMatchInfo->mvpMatchingCPs;
				//	cv::Scalar tracking_color = cv::Scalar(125, 125, 125);
				//	for(size_t i = 0, iend = vpCPs.size(); i < iend; i++){
				//		auto pCPi = vpCPs[i];
				//		auto pMPi = pCPi->GetMP();
				//		auto label = pMPi->GetLabel();
				//		if (!pMPi || pMPi->isDeleted())
				//			continue;
				//		cv::Mat x3D = pMPi->GetWorldPos();
				//		cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
				//		tpt += mVisMidPt;
				//		
				//		/*if (label == 255)
				//			color = cv::Scalar(255, 0, 0);
				//		else if (label == 150)
				//			color = cv::Scalar(0, 0, 255);*/
				//		cv::circle(tempVis, tpt, 2, tracking_color, -1);//UVR_SLAM::ObjectColors::mvObjectLabelColors[label]
				//	}
				//	
				//	////tracking results
				//}
				/////트래킹 결과 출력
				////////////////////////////////////////////////////////////
			}
			////trajectory
			/*auto sGraphKFs = mpMap->GetGraphFrames();
			for (auto iter = sGraphKFs.begin(), iend = sGraphKFs.end(); iter != iend; iter++) {
				auto pKFi = *iter;
				cv::Mat t1 = pKFi->GetCameraCenter();
				cv::Point2f pt1 = cv::Point2f(t1.at<float>(mnAxis1)* mnVisScale, t1.at<float>(mnAxis2)* mnVisScale);
				pt1 += mVisMidPt;
				cv::circle(tempVis, pt1, 2, cv::Scalar(0, 155, 248), -1);
			}
			auto lKFs = mpMap->GetWindowFramesVector();
			for (auto iter = lKFs.begin(); iter != lKFs.end(); iter++) {
				auto pKFi = *iter;
				cv::Mat t1 = pKFi->GetCameraCenter();
				cv::Point2f pt1 = cv::Point2f(t1.at<float>(mnAxis1)* mnVisScale, t1.at<float>(mnAxis2)* mnVisScale);
				pt1 += mVisMidPt;
				cv::circle(tempVis, pt1, 2, cv::Scalar(51, 0, 51), -1);
			}*/
			auto lKFs = mpMap->GetFrames();
			int nMaxID = 0;
			UVR_SLAM::Frame* lastKF = nullptr;
			for (auto iter = lKFs.begin(); iter != lKFs.end(); iter++) {
				auto pKFi = *iter;
				cv::Mat t1 = pKFi->GetCameraCenter();
				cv::Point2f pt1 = cv::Point2f(t1.at<float>(mnAxis1)* mnVisScale, t1.at<float>(mnAxis2)* mnVisScale);
				pt1 += mVisMidPt;
				cv::circle(tempVis, pt1, 2, cv::Scalar(0, 155, 248), -1);
				if (nMaxID < pKFi->mnKeyFrameID) {
					nMaxID = pKFi->mnKeyFrameID;
					lastKF = pKFi;
				}
			}
			if(lKFs.size()>0)
			{
				auto currKF = lastKF;// lKFs[lKFs.size() - 1];
				cv::Mat t1 = currKF->GetCameraCenter();
				cv::Point2f pt1 = cv::Point2f(t1.at<float>(mnAxis1)* mnVisScale, t1.at<float>(mnAxis2)* mnVisScale);
				pt1 += mVisMidPt;
				cv::circle(tempVis, pt1, 3, cv::Scalar(0, 0, 255), -1);
				
				cv::Mat directionZ = currKF->GetRotation().row(2);
				cv::Point2f dirPtZ = cv::Point2f(directionZ.at<float>(mnAxis1)* mnVisScale / 10.0, directionZ.at<float>(mnAxis2)* mnVisScale / 10.0) + pt1;
				cv::line(tempVis, pt1, dirPtZ, cv::Scalar(255, 0, 0), 2);
				
				cv::Mat directionY = currKF->GetRotation().row(1);
				cv::Point2f dirPtY = cv::Point2f(directionY.at<float>(mnAxis1)* mnVisScale / 10.0, directionY.at<float>(mnAxis2)* mnVisScale / 10.0) + pt1;
				cv::line(tempVis, pt1, dirPtY, cv::Scalar(0, 255, 0), 2);

				cv::Mat directionX = currKF->GetRotation().row(0);
				cv::Point2f dirPtX1 = pt1 + cv::Point2f(directionX.at<float>(mnAxis1)* mnVisScale / 10.0, directionX.at<float>(mnAxis2)* mnVisScale / 10.0);
				cv::Point2f dirPtX2 = pt1 - cv::Point2f(directionX.at<float>(mnAxis1)* mnVisScale / 10.0, directionX.at<float>(mnAxis2)* mnVisScale / 10.0);
				cv::line(tempVis, dirPtX1, dirPtX2, cv::Scalar(0, 0, 255), 2);
			}
			////trajectory	

			/////pose recovery test
			auto vReinit = mpMap->GetReinit();
			for (int i = 0; i < vReinit.size(); i++) {
				cv::Mat x3D = vReinit[i];
				cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
				tpt += mVisMidPt;
				cv::Scalar color = cv::Scalar(154,250,000);
				cv::circle(tempVis, tpt, 2, color, -1);
			}
			{
				cv::Scalar color = cv::Scalar(0, 255, 255);
				vReinit = mpMap->GetTempMPs();
				for (int i = 0; i < vReinit.size(); i++) {
					cv::Mat x3D = vReinit[i];
					cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
					tpt += mVisMidPt;
					cv::circle(tempVis, tpt, 4, color, -1);
				}
			}
			/////pose recovery test
			///////////////////////////////////////////////////////////////////////////////

			//save map
			//mpSystem->GetDirPath(mpMap->GetCurrFrame()->GetKeyFrameID());
			
			//fuse time text 
			//std::stringstream ss;
			////ss << "Fuse = " << mpFrameWindow->GetFuseTime()<<", PE = "<< mpFrameWindow->GetPETime();
			//cv::rectangle(tempVis, cv::Point2f(0, 0), cv::Point2f(tempVis.cols, 30), cv::Scalar::all(0), -1);
			//cv::putText(tempVis, ss.str(), cv::Point2f(0, 20), mnFontFace, mfFontScale, cv::Scalar::all(255));
			//fuse time text
			
			//tempVis.copyTo(mOutputImage(mvRects[2]));
			SetOutputImage(tempVis, 4);

			//time 
			cv::Mat imgTime = cv::Mat::zeros(500, 500, CV_8UC1);
			cv::putText(imgTime, mpSystem->GetTrackerString(), cv::Point2f(0, 20), mnFontFace, mfFontScale, cv::Scalar::all(255));
			cv::putText(imgTime, mpSystem->GetSegmentationString(), cv::Point2f(0, 50), mnFontFace, mfFontScale, cv::Scalar::all(255));
			cv::putText(imgTime, mpSystem->GetLocalMapperString(), cv::Point2f(0, 80), mnFontFace, mfFontScale, cv::Scalar::all(255));
			cv::putText(imgTime, mpSystem->GetMapOptimizerString(), cv::Point2f(0, 110), mnFontFace, mfFontScale, cv::Scalar::all(255));
			cv::putText(imgTime, mpSystem->GetPlaneString(), cv::Point2f(0, 140), mnFontFace, mfFontScale, cv::Scalar::all(255));
			cv::imshow("Output::Time", imgTime);

			SetBoolDoingProcess(false);
		}//if
		if (isOutputTypeChanged(0)) {
			cv::Mat mTrackImg = GetOutputImage(0);
			mTrackImg.copyTo(mOutputImage(mvRects[0]));
		}
		if (isOutputTypeChanged(1)) {
			cv::Mat mWinImg = GetOutputImage(1);
			mWinImg.copyTo(mOutputImage(mvRects[1]));
		}
		if (isOutputTypeChanged(2)) {
			cv::Mat mMapImg = GetOutputImage(2);
			mMapImg.copyTo(mOutputImage(mvRects[2]));
		}
		if (isOutputTypeChanged(3)) {
			cv::Mat mMappingImg = GetOutputImage(3);
			mMappingImg.copyTo(mOutputImage(mvRects[3]));
		}
		if (isOutputTypeChanged(4)) {
			cv::Mat mMappingImg = GetOutputImage(4);
			mMappingImg.copyTo(mOutputImage(mvRects[4]));
		}
		if (isOutputTypeChanged(5)) {
			cv::Mat mMappingImg = GetOutputImage(5);
			mMappingImg.copyTo(mOutputImage(mvRects[5]));
		}
		imshow("Output::Display", mOutputImage);
		cv::waitKey(1);
	}//while
}

//////////////////////////////////////