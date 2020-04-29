#include <Visualizer.h>
#include <FrameWindow.h>
#include <Frame.h>
#include <MapPoint.h>
#include <System.h>
#include <PlaneEstimator.h>
#include <Map.h>
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
int mnMode = 0;
int mnMaxMode = 3;
int mnAxis1 = 0;
int mnAxis2 = 2;

int nScale;
int UVR_SLAM::Visualizer::GetScale(){
	std::unique_lock<std::mutex>lock(mMutexScale);
	return mnVisScale;
}
void UVR_SLAM::Visualizer::SetScale(int s){
	std::unique_lock<std::mutex>lock(mMutexScale);
	mnVisScale = s;
}

void UVR_SLAM::Visualizer::CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		//std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
		sPt = cv::Point2f(x, y);
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
	mVisualized2DMap = cv::Mat(mnHeight * 2, mnHeight * 2, CV_8UC3, cv::Scalar(255, 255, 255));
	mVisTrajectory = cv::Mat(mnHeight * 2, mnHeight * 2, CV_8UC3, cv::Scalar(255, 255, 255));
	mVisMapPoints = cv::Mat(mnHeight * 2, mnHeight * 2, CV_8UC3, cv::Scalar(255, 255, 255));
	mVisPoseGraph = cv::Mat(mnHeight * 2, mnHeight * 2, CV_8UC3, cv::Scalar(255, 255, 255));
	mVisMidPt = cv::Point2f(mnHeight, mnHeight);
	mVisPrevPt = mVisMidPt;

	//set image
	int nImageWindowStartX = -1690;
	int nImageWIndowStartY1 = 20;
	int nImageWIndowStartY2 = 50;

	///////////////
	cv::namedWindow("Output::Matching");
	cv::moveWindow("Output::Matching", nImageWindowStartX, nImageWIndowStartY1);
	cv::namedWindow("Output::Segmentation");
	cv::moveWindow("Output::Segmentation", nImageWindowStartX, nImageWIndowStartY1+2*mnHeight+30);
	cv::namedWindow("Output::LoopFrame");
	cv::moveWindow("Output::LoopFrame", nImageWindowStartX+mnWidth/2+15, nImageWIndowStartY1 + 2 * mnHeight + 35);
	cv::namedWindow("Output::Tracking");
	cv::moveWindow("Output::Tracking", nImageWindowStartX + mnWidth, nImageWIndowStartY1);
	cv::namedWindow("Output::PlaneEstimation");
	cv::moveWindow("Output::PlaneEstimation", nImageWindowStartX + mnWidth, 50 + mnHeight);
	//cv::namedWindow("Output::Segmentation");
	//cv::moveWindow("Output::Segmentation", nImageWindowStartX, nImageWIndowStartY1);
	//cv::namedWindow("Output::SegmentationMask");
	//cv::moveWindow("Output::SegmentationMask", nImageWindowStartX, 50 + mnHeight);
	
	cv::namedWindow("Output::Trajectory");
	cv::moveWindow("Output::Trajectory", nImageWindowStartX + mnWidth + mnWidth, 20);
	cv::namedWindow("Output::Time");
	cv::moveWindow("Output::Time", nImageWindowStartX + mnWidth + mnWidth + mVisPoseGraph.cols, 20);

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

void UVR_SLAM::Visualizer::SetFrameWindow(UVR_SLAM::FrameWindow* pFrameWindow) {
	mpFrameWindow = pFrameWindow;
}

void UVR_SLAM::Visualizer::SetTargetFrame(Frame* pFrame) {
	mpTargetFrame = pFrame;
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
	
	while (1) {

		if (isDoingProcess()) {

			if (mbFrameMatching) {
				//VisualizeFrameMatching();

				//VisualizeTracking();
			}

			cv::Mat tempVis = mVisPoseGraph.clone();
			
			//trajactory

			//update pt
			mVisMidPt += mPt;
			mPt = cv::Point2f(0, 0);
			mnVisScale = nScale;

			cv::Scalar color1 = cv::Scalar(0, 0, 255);
			cv::Scalar color2 = cv::Scalar(0, 255, 0);
			cv::Scalar color3 = cv::Scalar(0, 0, 0);

			//////////////////////////////////////////////////////////////////////////////
			/////////////전체 포인트 출력
			auto mvpGlobalFrames = mpMap->GetFrames();
			for (int i = 0; i < mvpGlobalFrames.size(); i++) {
				UVR_SLAM::Frame* pF = mvpGlobalFrames[i];
				auto mvpMPs = pF->GetMapPoints();
				for (int j = 0; j < mvpMPs.size(); j++) {
					UVR_SLAM::MapPoint* pMP = mvpMPs[j];
					if (!pMP)
						continue;
					if (pMP->isDeleted())
						continue;
					cv::Mat x3D = pMP->GetWorldPos();
					cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, -x3D.at<float>(mnAxis2) * mnVisScale);
					tpt += mVisMidPt;
					cv::circle(tempVis, tpt, 2,cv::Scalar(0,0,0), -1);
				}
			}
			/////////////전체 포인트 출력
			//////////////////////////////////////////////////////////////////////////////

			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//local map
			auto mvpWindowFrames = mpFrameWindow->GetLocalMapFrames();
			auto mvbLocalMapInliers = mpFrameWindow->GetLocalMapInliers();
			auto mvpLocalMPs = mpFrameWindow->GetLocalMap();
			mvbLocalMapInliers = std::vector<bool>(mvpLocalMPs.size(), false);

			/////////////line visualization
			//std::cout << "LINE TEST" << std::endl;
			RNG rng(12345);

			if (mpMap->isFloorPlaneInitialized() && mvpWindowFrames.size() > 0) {
				auto mvpWalls = mpMap->GetWallPlanes();
				UVR_SLAM::PlaneInformation* aplane = mpMap->mpFloorPlane;
				cv::Mat planeParam = aplane->GetParam();

				cv::Mat K = mvpWindowFrames[0]->mK.clone();
				cv::Mat invK = K.inv();

				for (int i = 0; i < mvpWalls.size(); i++) {
					auto mvpLines = mvpWalls[i]->GetLines();

					for (int j = 0; j < mvpLines.size(); j++) {

						auto mpFrame = mvpLines[j]->mpFrame;
						cv::Mat invP, invK, invT;
						mpFrame->mpPlaneInformation->GetInformation(invP, invT, invK);

						auto mat = mvpLines[j]->GetLinePts();
						for (int k = 0; k < mat.rows; k++) {
							cv::Point2f topt = cv::Point2f(mat.row(k).at<float>(mnAxis1) * mnVisScale, -mat.row(k).at<float>(mnAxis2) * mnVisScale);
							topt += mVisMidPt;
							//cv::circle(tempVis, topt, 3, ObjectColors::mvObjectLabelColors[mvpWalls[i]->mnPlaneID + 10], -1);
						}
						/*
						cv::Mat from = UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(mvpLines[j]->from, invP, invT, invK);
						cv::Mat to   = UVR_SLAM::PlaneInformation::CreatePlanarMapPoint(mvpLines[j]->to  , invP, invT, invK);

						cv::Point2f frompt = cv::Point2f(from.at<float>(0) * mnVisScale, -from.at<float>(2) * mnVisScale);
						frompt += mVisMidPt;
						cv::Point2f topt = cv::Point2f(to.at<float>(0) * mnVisScale, -to.at<float>(2) * mnVisScale);
						topt += mVisMidPt;

						cv::line(tempVis, frompt, topt, ObjectColors::mvObjectLabelColors[mvpWalls[i]->mnPlaneID-1],3);
						*/
					}
				}
			}
			/////////////line visualization

			
			//map points
			for (int i = 0; i < mvpLocalMPs.size(); i++) {
				UVR_SLAM::MapPoint* pMP = mvpLocalMPs[i];
				if (!pMP)
					continue;
				if (pMP->isDeleted())
					continue;
				cv::Mat x3D = pMP->GetWorldPos();
				cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, -x3D.at<float>(mnAxis2) * mnVisScale);

				/*if (pMP->GetNumConnectedFrames() == 1 && mvpWindowFrames[0]->GetKeyFrameID() == pMP->mnFirstKeyFrameID) {
					cv::circle(tempVis, tpt, 4, cv::Scalar(0, 255, 255), -1);
				}*/

				tpt += mVisMidPt;
				if (mvbLocalMapInliers[i]){
						cv::circle(tempVis, tpt, 2, ObjectColors::mvObjectLabelColors[pMP->GetObjectType()], -1);
				}
				else
					cv::circle(tempVis, tpt, 1, color2, 1);
				
			}

			//trajectory
			
			for (int i = 0; i < mvpWindowFrames.size(); i++) {
				//cv::Mat t2 = mvpWindowFrames[i + 1]->GetTranslation();
				cv::Mat t1 = mvpWindowFrames[i]->GetCameraCenter();
				cv::Point2f pt1 = cv::Point2f(t1.at<float>(mnAxis1)* mnVisScale, -t1.at<float>(mnAxis2)* mnVisScale);
				//cv::Point2f pt2 = cv::Point2f(t2.at<float>(0)* mnVisScale, t2.at<float>(2)* mnVisScale);
				pt1 += mVisMidPt;
				//pt2 += mVisMidPt;
				if(i != mvpWindowFrames.size()-1)
					cv::circle(tempVis, pt1, 3, cv::Scalar(255, 255, 0), -1);
				else{
					cv::circle(tempVis, pt1, 3, cv::Scalar(0, 0, 255), -1);
					cv::Mat directionZ = mvpWindowFrames[i]->GetRotation().row(2);
					cv::Point2f dirPtZ = cv::Point2f(directionZ.at<float>(mnAxis1)* mnVisScale/10.0, -directionZ.at<float>(mnAxis2)* mnVisScale / 10.0)+pt1;
					cv::line(tempVis, pt1, dirPtZ, cv::Scalar(0, 0, 255), 2);

					cv::Mat directionX = mvpWindowFrames[i]->GetRotation().row(0);
					cv::Point2f dirPtX1 = pt1 + cv::Point2f(directionX.at<float>(mnAxis1)* mnVisScale / 10.0, -directionX.at<float>(mnAxis2)* mnVisScale / 10.0);
					cv::Point2f dirPtX2 = pt1 - cv::Point2f(directionX.at<float>(mnAxis1)* mnVisScale / 10.0, -directionX.at<float>(mnAxis2)* mnVisScale / 10.0);
					cv::line(tempVis, dirPtX1, dirPtX2, cv::Scalar(0, 0, 255), 2);
				}
				//cv::line(tempVis, pt1, pt2, cv::Scalar(0, 0, 0), 2);
			}

			int nRecentLayoutFrameID = mpFrameWindow->GetLastLayoutFrameID();

			//tracking results
			auto pMatchInfo = GetMatchInfo();
			if(pMatchInfo)
				for (int i = 0; i < pMatchInfo->mvMatchingPts.size(); i++) {
					auto pMPi = pMatchInfo->mvpMatchingMPs[i];
					auto label = pMatchInfo->mvObjectLabels[i];
					if (!pMPi || pMPi->isDeleted())
						continue;
					cv::Mat x3D = pMPi->GetWorldPos();
					cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, -x3D.at<float>(mnAxis2) * mnVisScale);
					tpt += mVisMidPt;
					cv::Scalar color = cv::Scalar(0, 0, 0);
					if (label == 255)
						color = cv::Scalar(255, 0, 0);
					else if (label == 150)
						color = cv::Scalar(0, 0, 255);
					cv::circle(tempVis, tpt, 2, color, -1);
				}

			auto vpFrameMPs = GetMPs();
			for (int i = 0; i < vpFrameMPs.size(); i++) {
				UVR_SLAM::MapPoint* pMP = vpFrameMPs[i];
				if (!pMP)
					continue;
				if (pMP->isDeleted())
					continue;
				cv::Mat x3D = pMP->GetWorldPos();
				cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, -x3D.at<float>(mnAxis2) * mnVisScale);
				tpt += mVisMidPt;

				/*if (pMP->GetNumConnectedFrames() == 1 && mvpWindowFrames[0]->GetKeyFrameID() == pMP->mnFirstKeyFrameID) {
					cv::circle(tempVis, tpt, 4, cv::Scalar(0, 255, 255), -1);
				}*/

				if (pMP->GetMapPointType() == UVR_SLAM::PLANE_DENSE_MP){
					if(pMP->GetPlaneID() > 0){
						if(pMP->GetNumDensedFrames() > 2)
							cv::circle(tempVis, tpt, 2, cv::Scalar(255, 0, 255), -1);
						else {
							cv::circle(tempVis, tpt, 2, cv::Scalar(255, 255, 0), -1);
						}
					}
					else
						cv::circle(tempVis, tpt, 2, cv::Scalar(0, 255, 255), -1);
				}
				else
					cv::circle(tempVis, tpt, 2, ObjectColors::mvObjectLabelColors[pMP->GetObjectType()], -1);
				//cv::circle(tempVis, tpt, 2, ObjectColors::mvObjectLabelColors[pMP->GetObjectType()], -1);
				/*if (pMP->GetRecentLayoutFrameID() == nRecentLayoutFrameID) {
					cv::circle(tempVis, tpt, 4, ObjectColors::mvObjectLabelColors[pMP->GetObjectType()]/2, 2);
				}*/
			}

			//save map
			//mpSystem->GetDirPath(mpMap->GetCurrFrame()->GetKeyFrameID());
			

			//fuse time text 
			std::stringstream ss;
			//ss << "Fuse = " << mpFrameWindow->GetFuseTime()<<", PE = "<< mpFrameWindow->GetPETime();
			cv::rectangle(tempVis, cv::Point2f(0, 0), cv::Point2f(tempVis.cols, 30), cv::Scalar::all(0), -1);
			cv::putText(tempVis, ss.str(), cv::Point2f(0, 20), mnFontFace, mfFontScale, cv::Scalar::all(255));
			//fuse time text

			cv::imshow("Output::Trajectory", tempVis);
			

			//time 
			cv::Mat imgTime = cv::Mat::zeros(500, 500, CV_8UC1);
			cv::putText(imgTime, mpSystem->GetTrackerString(), cv::Point2f(0, 20), mnFontFace, mfFontScale, cv::Scalar::all(255));
			cv::putText(imgTime, mpSystem->GetSegmentationString(), cv::Point2f(0, 50), mnFontFace, mfFontScale, cv::Scalar::all(255));
			cv::putText(imgTime, mpSystem->GetPlaneString(), cv::Point2f(0, 80), mnFontFace, mfFontScale, cv::Scalar::all(255));
			cv::putText(imgTime, mpSystem->GetMapOptimizerString(), cv::Point2f(0, 140), mnFontFace, mfFontScale, cv::Scalar::all(255));
			float lm1, lm2;
			mpSystem->GetLocalMappingTime(lm1, lm2);
			std::stringstream ssTime;
			ssTime << "LocalMapping : " <<mpSystem->GetLocalMapperFrameID()<<"::"<< lm1 << " :: BA : " << lm2;
			cv::putText(imgTime, ssTime.str(), cv::Point2f(0, 110), mnFontFace, mfFontScale, cv::Scalar::all(255));

			cv::imshow("Output::Time", imgTime);

			cv::waitKey(1);
			SetBoolDoingProcess(false);
		}//if
	}//while
}

//////////////////////////////////////
void UVR_SLAM::Visualizer::SetFrameMatching(Frame* pF1, Frame* pF2, std::vector<cv::DMatch> vMatchInfos) {
	mpMatchingFrame1 = pF1;
	mpMatchingFrame2 = pF2;
	mvMatchInfos = std::vector<cv::DMatch>(vMatchInfos.begin(), vMatchInfos.end());
	mbFrameMatching = true;
}

void UVR_SLAM::Visualizer::VisualizeTracking() {
	//일단 테스트

	
	cv::Mat vis = mpMatchingFrame2->GetOriginalImage();
	//cvtColor(vis, vis, CV_RGBA2BGR);
	vis.convertTo(vis, CV_8UC3);



	for (int i = 0; i < mpMatchingFrame2->mvKeyPoints.size(); i++) {
		if (!mpMatchingFrame2->mvbMPInliers[i])
			continue;
		UVR_SLAM::MapPoint* pMP = mpMatchingFrame2->mvpMPs[i];
		cv::circle(vis, mpMatchingFrame2->mvKeyPoints[i].pt, 1, cv::Scalar(255, 0, 255), -1);
		if (pMP) {
			if (pMP->isDeleted()) {
				mpMatchingFrame2->mvbMPInliers[i] = false;
				continue;
			}
			cv::Point2f p2D;
			cv::Mat pCam;
			pMP->Projection(p2D, pCam, mpMatchingFrame2->GetRotation(), mpMatchingFrame2->GetTranslation(), mpMatchingFrame2->mK, mnWidth, mnHeight);
			UVR_SLAM::ObjectType type = pMP->GetObjectType();
			cv::line(vis, p2D, mpMatchingFrame2->mvKeyPoints[i].pt, cv::Scalar(255, 255, 0), 1);
			if (type != OBJECT_NONE)
				circle(vis, p2D, 3, UVR_SLAM::ObjectColors::mvObjectLabelColors[type], -1);
			if (pMP->GetPlaneID() > 0) {
				circle(vis, p2D, 4, cv::Scalar(255, 0, 255), -1);
			}
		}
	}

	
	auto mvpOPs = mpMatchingFrame2->GetObjectVector();
	cv::Mat vis2 = mpMatchingFrame2->GetOriginalImage();
	for (int i = 0; i < mpMatchingFrame2->mvKeyPoints.size(); i++) {
		UVR_SLAM::ObjectType type = mvpOPs[i];
		if (type != OBJECT_NONE)
			circle(vis2, mpMatchingFrame2->mvKeyPoints[i].pt, 2, ObjectColors::mvObjectLabelColors[type], -1);
	}
	cv::imshow("Output::Tracking", vis);
	cv::imshow("Output::Matching::SemanticFrame", vis2);
	cv::waitKey(1);

	std::stringstream ss;
	ss << "../../bin/segmentation/res/img/img_" << mpMatchingFrame2->GetFrameID() << ".jpg";
	cv::imwrite(ss.str(), vis);
	cv::imwrite("../../bin/segmentation/res/labeling.jpg", vis2);
}

void UVR_SLAM::Visualizer::VisualizeFrameMatching() {

	int nLastSIdx = mpFrameWindow->GetLastSemanticFrameIndex();
	int nLastKIdx = mpFrameWindow->size()-1;
	int nLastSFrameIdx = 0;
	int nLastKFrameIdx = 0;
	if(nLastSIdx >=0)
		nLastSFrameIdx = mpFrameWindow->GetFrame(nLastSIdx)->GetFrameID();
	if (nLastKIdx >= 0)
		nLastKFrameIdx = mpFrameWindow->GetFrame(nLastKIdx)->GetFrameID();

	cv::Mat img1 = mpMatchingFrame1->GetOriginalImage();
	cv::Mat img2 = mpMatchingFrame2->GetOriginalImage();
	cv::Point2f ptBottom = cv::Point2f(0, img1.rows);

	cv::Rect mergeRect1 = cv::Rect(0, 0, img1.cols, img1.rows);
	cv::Rect mergeRect2 = cv::Rect(0, img1.rows, img1.cols, img1.rows);
	cv::Mat debugging = cv::Mat::zeros(img1.rows * 2, img1.cols, img1.type());
	img1.copyTo(debugging(mergeRect1));
	img2.copyTo(debugging(mergeRect2));

	/*for (int i = 0; i < mpMatchingFrame1->mvKeyPoints.size(); i++) {
		cv::circle(debugging, mpMatchingFrame1->mvKeyPoints[i].pt, 1, cv::Scalar(255, 255, 0), -1);
	}
	for (int i = 0; i < mpMatchingFrame2->mvKeyPoints.size(); i++) {
		cv::circle(debugging, mpMatchingFrame2->mvKeyPoints[i].pt+ ptBottom, 1, cv::Scalar(255, 255, 0), -1);
	}*/
	for (int i = 0; i < mvMatchInfos.size(); i++) {
		if(mpMatchingFrame2->mvbMPInliers[mvMatchInfos[i].trainIdx])
			cv::line(debugging, mpMatchingFrame1->mvKeyPoints[mvMatchInfos[i].queryIdx].pt, mpMatchingFrame2->mvKeyPoints[mvMatchInfos[i].trainIdx].pt + ptBottom, cv::Scalar(255, 0, 255));
		else
			cv::line(debugging, mpMatchingFrame1->mvKeyPoints[mvMatchInfos[i].queryIdx].pt, mpMatchingFrame2->mvKeyPoints[mvMatchInfos[i].trainIdx].pt + ptBottom, cv::Scalar(255, 255, 0));
	}
	std::stringstream ss;
	ss <<"FID = "<< mpMatchingFrame2->GetFrameID()<<", KID = "<<nLastKFrameIdx<<", SID = "<<nLastSFrameIdx<<" ||"<< "Matching = " << mvMatchInfos.size()<<" || MPs = "<<mpMatchingFrame2->GetInliers();
	cv::rectangle(debugging, cv::Point2f(0, 0), cv::Point2f(img1.cols, 30), cv::Scalar::all(0), -1);
	cv::putText(debugging, ss.str(), cv::Point2f(0,20), mnFontFace, mfFontScale, cv::Scalar::all(255));

	cv::imshow("Test::Matching::Frame", debugging);
	mbFrameMatching = false;
}
void UVR_SLAM::Visualizer::SetMPs(std::vector<UVR_SLAM::MapPoint*> vpMPs){
	std::unique_lock<std::mutex> lock(mMutexFrameMPs);
	mvpFrameMPs = std::vector<UVR_SLAM::MapPoint*>(vpMPs.begin(), vpMPs.end());
}
std::vector<UVR_SLAM::MapPoint*> UVR_SLAM::Visualizer::GetMPs(){
	std::unique_lock<std::mutex> lock(mMutexFrameMPs);
	return std::vector<UVR_SLAM::MapPoint*>(mvpFrameMPs.begin(), mvpFrameMPs.end());
}
void UVR_SLAM::Visualizer::SetMatchInfo(MatchInfo* pMatch){
	std::unique_lock<std::mutex> lock(mMutexFrameMPs);
	mpMatchInfo = pMatch;
}
UVR_SLAM::MatchInfo* UVR_SLAM::Visualizer::GetMatchInfo(){
	std::unique_lock<std::mutex> lock(mMutexFrameMPs);
	return mpMatchInfo;
}