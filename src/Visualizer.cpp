#include <Visualizer.h>
#include <Frame.h>
#include <MapPoint.h>
#include <System.h>
#include <PlaneEstimator.h>
#include <Map.h>
#include <MapGrid.h>
#include <plane.h>

UVR_SLAM::Visualizer::Visualizer() {}
UVR_SLAM::Visualizer::Visualizer(int w, int h, int scale, Map* pMap) :mnWidth(w), mnHeight(h), mnVisScale(scale), mnFontFace(2), mfFontScale(0.6), mpMatchInfo(nullptr){
	mpMap = pMap;
}
UVR_SLAM::Visualizer::~Visualizer() {}

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
cv::Point2f sPt = cv::Point2f(0, 0);
cv::Point2f ePt = cv::Point2f(0, 0);
cv::Point2f mPt = cv::Point2f(0, 0);
cv::Mat M = cv::Mat::zeros(0, 0, CV_64FC1);
int mnMode = 2;
int mnMaxMode = 3;
int mnAxis1 = 0;
int mnAxis2 = 2;

bool bSaveMap = false;
bool bShowOnlyTrajectory = true;

int nScale;
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

void UVR_SLAM::Visualizer::CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		//std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
		sPt = cv::Point2f(x, y);

		////button interface
		if (sPt.x < 50 && sPt.y < 50) {
			bSaveMap = true;
		}else if (sPt.x < 50 && (sPt.y >= 50 && sPt.y < 100)) {
			bShowOnlyTrajectory = !bShowOnlyTrajectory;
		}
		////button interface
	}
	else if (event == cv::EVENT_LBUTTONUP)
	{
		//std::cout << "Left button of the mouse is released - position (" << x << ", " << y << ")" << std::endl;
		ePt = cv::Point2f(x, y);
		mPt = ePt - sPt;
		M = cv::Mat::zeros(2, 3, CV_64FC1);
		M.at<double>(0, 0) = 1.0;
		M.at<double>(1, 1) = 1.0;
		M.at<double>(0, 2) = mPt.x;
		M.at<double>(1, 2) = mPt.y;
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
			nScale += 20;
		}
		else {
			//scroll down
			nScale -= 20;
			if (nScale <= 0)
				nScale = 20;
		}
	}
}

void UVR_SLAM::Visualizer::Init() {
	
	//Visualization
	mVisPoseGraph = cv::Mat(mnHeight * 2, mnHeight * 2, CV_8UC3, cv::Scalar(255, 255, 255));
	rectangle(mVisPoseGraph, cv::Rect(0, 0, 50, 50), cv::Scalar(255, 255, 0), -1);
	rectangle(mVisPoseGraph, cv::Rect(0, 50, 50, 50), cv::Scalar(0, 255, 255), -1);
	mVisMidPt = cv::Point2f(mnHeight, mnHeight);
	mVisPrevPt = mVisMidPt;

	//set image
	int nImageWindowStartX = -1690;
	int nImageWindowEndX = 1920;
	int nImageWIndowStartY1 = 20;
	int nImageWIndowStartY2 = 50;

	/////////////////////////////
	////New 배치
	/*cv::namedWindow("Output::PE::PARAM");
	cv::moveWindow("Output::PE::PARAM", nImageWindowStartX, 0);*/
	cv::namedWindow("edge+optical");
	cv::moveWindow("edge+optical", nImageWindowStartX, 0);
	////New 배치
	/////////////////////////////
	

	///////////////
	/*cv::namedWindow("Output::Matching");
	cv::moveWindow("Output::Matching", nImageWindowStartX, nImageWIndowStartY1);*/
	cv::namedWindow("Output::PE::PARAM");
	cv::moveWindow("Output::PE::PARAM", nImageWindowStartX, 0);
	cv::namedWindow("Output::Segmentation");  
	cv::moveWindow("Output::Segmentation", nImageWindowStartX, nImageWIndowStartY1+2*mnHeight+30);
	cv::namedWindow("Output::LoopFrame");
	cv::moveWindow("Output::LoopFrame", nImageWindowStartX+mnWidth/2+15, nImageWIndowStartY1 + 2 * mnHeight + 35);
	cv::namedWindow("Output::Tracking");
	cv::moveWindow("Output::Tracking", nImageWindowStartX + mnWidth, nImageWIndowStartY1);
	cv::namedWindow("Output::Time");
	cv::moveWindow("Output::Time", nImageWindowStartX + mnWidth, 50 + mnHeight);
	/*cv::namedWindow("Output::PlaneEstimation");
	cv::moveWindow("Output::PlaneEstimation", nImageWindowStartX + mnWidth, 50 + mnHeight);*/
	//cv::namedWindow("Output::Segmentation");
	//cv::moveWindow("Output::Segmentation", nImageWindowStartX, nImageWIndowStartY1);
	//cv::namedWindow("Output::SegmentationMask");
	//cv::moveWindow("Output::SegmentationMask", nImageWindowStartX, 50 + mnHeight);
	
	cv::namedWindow("Output::Trajectory");
	cv::moveWindow("Output::Trajectory", nImageWindowStartX + mnWidth + mnWidth, 20);
	

	//cv::namedWindow("Output::Trajectory");
	//cv::moveWindow("Output::Trajectory", nImageWindowStartX + mnWidth + mnWidth, 20);

	//Output::Segmentation

	//Opencv Image Window
	int nAdditional1 = 355;
	/*namedWindow("Output::Trajectory");
	moveWindow("Output::Trajectory", nImageWindowStartX + nAdditional1 + mnWidth + mnWidth + mnWidth, 0);*/
	cv::namedWindow("Initialization::Frame::1");
	cv::moveWindow("Initialization::Frame::1", nImageWindowStartX + nAdditional1 + mnWidth + mnWidth + mnWidth + mnWidth, 0);
	cv::namedWindow("Initialization::Frame::2");
	cv::moveWindow("Initialization::Frame::2", nImageWindowStartX + nAdditional1 + mnWidth + mnWidth + mnWidth + mnWidth, 30 + mnHeight);
	cv::namedWindow("LocalMapping::CreateMPs");
	cv::moveWindow("LocalMapping::CreateMPs", nImageWindowStartX + nAdditional1 + mnWidth + mnWidth + mnWidth + mnWidth, 0);
	cv::namedWindow("Output::Matching::SemanticFrame");
	cv::moveWindow("Output::Matching::SemanticFrame", nImageWindowStartX + nAdditional1 + mnWidth + mnWidth + mnWidth, 30 + mnHeight);

	cv::namedWindow("Test::Matching::Frame");
	cv::moveWindow("Test::Matching::Frame", nImageWindowStartX + nAdditional1 + mnWidth + mnWidth + mnWidth*0.7, 30 + mnHeight);

	

	cv::setMouseCallback("Output::Trajectory", UVR_SLAM::Visualizer::CallBackFunc, NULL);
	nScale = mnVisScale;
}

void UVR_SLAM::Visualizer::SetSystem(UVR_SLAM::System* pSystem) {
	mpSystem = pSystem;
}

void UVR_SLAM::Visualizer::SetBoolDoingProcess(bool b) {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	mbDoingProcess = b;
}
bool UVR_SLAM::Visualizer::isDoingProcess() {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	return mbDoingProcess;
}

void UVR_SLAM::Visualizer::Run() {

	std::vector<UVR_SLAM::MapPoint*> allDummyPts;

	std::vector<cv::Scalar> colors;
	auto tempColors = UVR_SLAM::ObjectColors::mvObjectLabelColors;
	for (int i = 0; i < tempColors.size(); i++) {
		colors.push_back(cv::Scalar(tempColors[i].val[0], tempColors[i].val[1], tempColors[i].val[2]));
	}
	SetAxisMode();
	while (1) {
		
		if (bSaveMap) {
			auto frames = mpMap->GetFrames();
			int nFrame = frames.size();
			auto frame = frames[nFrame - 1];
			std::stringstream sss;
			sss << mpSystem->GetDirPath(0) << "/map/map_"<< frame->GetKeyFrameID()<<".txt";
			std::ofstream f;
			f.open(sss.str().c_str());
			auto mmpMap = mpMap->GetMap();
			for (auto iter = mmpMap.begin(); iter != mmpMap.end(); iter++) {
				auto pMPi = iter->first;
				int label = iter->second;
				if (!pMPi || pMPi->isDeleted())
					continue;
				cv::Mat Xw = pMPi->GetWorldPos();
				f << Xw.at<float>(0) << " " << Xw.at<float>(1) << " " << Xw.at<float>(2) <<" "<<label<<" "<<pMPi->GetNumConnectedFrames()<< std::endl;
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

		if (isDoingProcess()) {

			cv::Mat tempVis = mVisPoseGraph.clone();
			
			//update pt
			mVisMidPt += mPt;
			mPt = cv::Point2f(0, 0);
			mnVisScale = nScale;

			cv::Scalar color1 = cv::Scalar(0, 0, 255);
			cv::Scalar color2 = cv::Scalar(0, 255, 0);
			cv::Scalar color3 = cv::Scalar(0, 0, 0);

			///////////////////////////////////////////////////////////////////////////////
			////tracking results
			auto pMatchInfo = GetMatchInfo();
			////grid 출력
			//if (pMatchInfo) {
			//	auto lastBAFrame = pMatchInfo->mpTargetFrame;
			//	auto mvGrids = mpMap->GetMapGrids();
			//	int radius = mpMap->mfMapGridSize * mnVisScale;
			//	for (int i = 0; i < mvGrids.size(); i++) {
			//		auto pMGi = mvGrids[i];
			//		int nCount = pMGi->Count();
			//		if (nCount < 10)
			//			continue;
			//		cv::Mat x3D = pMGi->Xw.clone();
			//		cv::Point2f spt = cv::Point2f((x3D.at<float>(mnAxis1)) * mnVisScale, (-x3D.at<float>(mnAxis2)) * mnVisScale);
			//		cv::Point2f ept = cv::Point2f((x3D.at<float>(mnAxis1) + mpMap->mfMapGridSize) * mnVisScale, (-x3D.at<float>(mnAxis2) - mpMap->mfMapGridSize) * mnVisScale);
			//		spt += mVisMidPt;
			//		ept += mVisMidPt;
			//		if (lastBAFrame->isInFrustum(pMGi, 0.6f)) {
			//			cv::rectangle(tempVis, spt, ept, cv::Scalar(150, 150, 150),-1);
			//		}
			//		else {
			//			/*cv::Mat x3D = pMGi->Xw.clone();
			//			cv::Point2f spt = cv::Point2f((x3D.at<float>(mnAxis1) - mpMap->mfMapGridSize) * mnVisScale, (-x3D.at<float>(mnAxis2) + mpMap->mfMapGridSize) * mnVisScale);
			//			cv::Point2f ept = cv::Point2f((x3D.at<float>(mnAxis1) + mpMap->mfMapGridSize) * mnVisScale, (-x3D.at<float>(mnAxis2) - mpMap->mfMapGridSize) * mnVisScale);
			//			spt += mVisMidPt;
			//			ept += mVisMidPt;*/
			//			//::rectangle(tempVis, spt, ept, cv::Scalar(50, 50, 50), -1);
			//		}
			//	}
			//}
			////grid 출력

			//local map
			if(bShowOnlyTrajectory){
				//////////////////////////////
				//전체 맵포인트 시각화
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
					/*if (pMPi->GetFVRatio()<0.1)
						color = cv::Scalar(79, 79, 47);*/
					/*if(pMPi->GetConnedtedFrames().size() < 7){
						color = cv::Scalar(0, 0, 0);
						continue;
					}*/
					cv::circle(tempVis, tpt, 2, color, -1);
				}
				//전체 맵포인트 시각화
				//////////////////////////////

				////////////////////////////////////////////////////////////
				/////트래킹 결과 출력
				if (pMatchInfo) {
					auto lastBAFrame = pMatchInfo->mpTargetFrame;
					std::vector<MapPoint*> mvpMatchingMPs;
					std::vector<CandidatePoint*> mvpMatchingCPs;
					auto mvpMatchingPts = pMatchInfo->GetMatchingPts(mvpMatchingCPs, mvpMatchingMPs);
					for (int i = 0; i < mvpMatchingPts.size(); i++) {
						auto pMPi = mvpMatchingMPs[i];
						auto label = pMPi->GetLabel();
						if (!pMPi || pMPi->isDeleted())
							continue;
						cv::Mat x3D = pMPi->GetWorldPos();
						cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
						tpt += mVisMidPt;
						cv::Scalar color = cv::Scalar(125,125,125);
						if (label == 255)
							color = cv::Scalar(255, 0, 0);
						else if (label == 150)
							color = cv::Scalar(0, 0, 255);
						cv::circle(tempVis, tpt, 2, color, -1);
					}
					
					////tracking results
				}
				/////트래킹 결과 출력
				////////////////////////////////////////////////////////////
				}
			////trajectory
			auto lGraphKFs = mpMap->GetGraphFrames();
			auto siter = mpMap->GetGraphFramesStartIterator();
			auto eiter = mpMap->GetGraphFramesEndIterator();
			//for (auto iter = siter; iter != eiter; iter++) {
			for (std::list<Frame*>::const_iterator iter = lGraphKFs.begin(), iend = lGraphKFs.end(); iter != iend; iter++) {
				auto pKFi = *iter;
				cv::Mat t1 = pKFi->GetCameraCenter();
				cv::Point2f pt1 = cv::Point2f(t1.at<float>(mnAxis1)* mnVisScale, t1.at<float>(mnAxis2)* mnVisScale);
				pt1 += mVisMidPt;
				cv::circle(tempVis, pt1, 2, cv::Scalar(0, 155, 248), -1);
			}
			auto lKFs = mpMap->GetWindowFrames();
			auto siter2 = mpMap->GetWindowFramesStartIterator();
			auto eiter2 = mpMap->GetWindowFramesEndIterator();
			//for (auto iter = siter2; iter != eiter2; iter++) {
			for (std::list<Frame*>::const_iterator iter = lKFs.begin(), iend = lKFs.end(); iter != iend; iter++) {
				auto pKFi = *iter;
				cv::Mat t1 = pKFi->GetCameraCenter();
				cv::Point2f pt1 = cv::Point2f(t1.at<float>(mnAxis1)* mnVisScale, t1.at<float>(mnAxis2)* mnVisScale);
				pt1 += mVisMidPt;
				cv::circle(tempVis, pt1, 2, cv::Scalar(51, 0, 51), -1);
			} 
			{
				auto currKF = lKFs.back();
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

			///////////////////////////////////////////////////////////////////////////////

			//save map
			//mpSystem->GetDirPath(mpMap->GetCurrFrame()->GetKeyFrameID());
			
			//fuse time text 
			//std::stringstream ss;
			////ss << "Fuse = " << mpFrameWindow->GetFuseTime()<<", PE = "<< mpFrameWindow->GetPETime();
			//cv::rectangle(tempVis, cv::Point2f(0, 0), cv::Point2f(tempVis.cols, 30), cv::Scalar::all(0), -1);
			//cv::putText(tempVis, ss.str(), cv::Point2f(0, 20), mnFontFace, mfFontScale, cv::Scalar::all(255));
			//fuse time text

			
			cv::imshow("Output::Trajectory", tempVis);
			
			//time 
			cv::Mat imgTime = cv::Mat::zeros(500, 500, CV_8UC1);
			cv::putText(imgTime, mpSystem->GetTrackerString(), cv::Point2f(0, 20), mnFontFace, mfFontScale, cv::Scalar::all(255));
			cv::putText(imgTime, mpSystem->GetSegmentationString(), cv::Point2f(0, 50), mnFontFace, mfFontScale, cv::Scalar::all(255));
			cv::putText(imgTime, mpSystem->GetLocalMapperString(), cv::Point2f(0, 80), mnFontFace, mfFontScale, cv::Scalar::all(255));
			cv::putText(imgTime, mpSystem->GetMapOptimizerString(), cv::Point2f(0, 110), mnFontFace, mfFontScale, cv::Scalar::all(255));
			cv::putText(imgTime, mpSystem->GetPlaneString(), cv::Point2f(0, 140), mnFontFace, mfFontScale, cv::Scalar::all(255));
			cv::imshow("Output::Time", imgTime);

			cv::waitKey(1);
			SetBoolDoingProcess(false);
		}//if
	}//while
}

//////////////////////////////////////

void UVR_SLAM::Visualizer::SetMatchInfo(MatchInfo* pMatch){
	std::unique_lock<std::mutex> lock(mMutexMatchingInfo);
	mpMatchInfo = pMatch;
}
UVR_SLAM::MatchInfo* UVR_SLAM::Visualizer::GetMatchInfo(){
	std::unique_lock<std::mutex> lock(mMutexMatchingInfo);
	return mpMatchInfo;
}