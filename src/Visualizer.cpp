#include <Visualizer.h>
#include <FrameWindow.h>
#include <Frame.h>

UVR_SLAM::Visualizer::Visualizer() {}
UVR_SLAM::Visualizer::Visualizer(int w, int h, int scale) :mnWidth(w), mnHeight(h), mnVisScale(scale), mnFontFace(2), mfFontScale(0.6){}
UVR_SLAM::Visualizer::~Visualizer() {}

////////////////////////////
cv::Point2f sPt = cv::Point2f(0, 0);
cv::Point2f ePt = cv::Point2f(0, 0);
cv::Point2f mPt = cv::Point2f(0, 0);
cv::Mat M = cv::Mat::zeros(0, 0, CV_64FC1);

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
		sPt = cv::Point2f(x, y);
	}
	else if (event == cv::EVENT_LBUTTONUP)
	{
		std::cout << "Left button of the mouse is released - position (" << x << ", " << y << ")" << std::endl;
		ePt = cv::Point2f(x, y);
		mPt = ePt - sPt;
		M = cv::Mat::zeros(2, 3, CV_64FC1);
		M.at<double>(0, 0) = 1.0;
		M.at<double>(1, 1) = 1.0;
		M.at<double>(0, 2) = mPt.x;
		M.at<double>(1, 2) = mPt.y;
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
	cv::namedWindow("Output::Segmentation");
	cv::moveWindow("Output::Segmentation", nImageWindowStartX, nImageWIndowStartY1);
	cv::namedWindow("Output::Tracking");
	cv::moveWindow("Output::Tracking", nImageWindowStartX + mnWidth, nImageWIndowStartY1);
	cv::namedWindow("Output::PlaneEstimation");
	cv::moveWindow("Output::PlaneEstimation", nImageWindowStartX + mnWidth, 50 + mnHeight);
	cv::namedWindow("Output::SegmentationMask");
	cv::moveWindow("Output::SegmentationMask", nImageWindowStartX, 50 + mnHeight);
	cv::namedWindow("Output::Trajectory");
	cv::moveWindow("Output::Trajectory", nImageWindowStartX + mnWidth + mnWidth, 20);

	//cv::namedWindow("Output::Trajectory");
	//cv::moveWindow("Output::Trajectory", nImageWindowStartX + mnWidth + mnWidth, 20);

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

	cv::setMouseCallback("Output::Trajectory", CallBackFunc, NULL);

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
	while (1) {
		if (isDoingProcess()) {

			if (mbFrameMatching) {
				VisualizeFrameMatching();
				//VisualizeTracking();
			}

			cv::Mat tempVis = mVisPoseGraph.clone();
			
			//trajactory

			//update pt
			mVisMidPt += mPt;
			mPt = cv::Point2f(0, 0);

			cv::Scalar color1 = cv::Scalar(0, 0, 255);
			cv::Scalar color2 = cv::Scalar(0, 255, 0);
			cv::Scalar color3 = cv::Scalar(0, 0, 0);

			/*if (mpFrameWindow->GetQueueSize() > 0) {

			if (M.rows > 0) {
			cv::warpAffine(mVisPoseGraph, mVisPoseGraph, M, mVisPoseGraph.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255,255,255));
			M = cv::Mat::zeros(0, 0, CV_64FC1);
			}

			UVR_SLAM::Frame* pF = mpFrameWindow->GetQueueLastFrame();
			for (int i = 0; i < pF->mvKeyPoints.size(); i++) {
			if (!pF->GetBoolInlier(i))
			continue;
			UVR_SLAM::MapPoint* pMP = pF->GetMapPoint(i);
			if (!pMP)
			continue;
			if (pMP->isDeleted())
			continue;
			cv::Mat x3D = pMP->GetWorldPos();
			cv::Point2f tpt2 = cv::Point2f(x3D.at<float>(0) * mnVisScale, -x3D.at<float>(2) * mnVisScale);
			tpt2 += mVisMidPt;
			cv::circle(mVisPoseGraph, tpt2, 2, color3, -1);

			}
			}*/

			//map points
			for (int i = 0; i < mpFrameWindow->GetLocalMapSize(); i++) {
				UVR_SLAM::MapPoint* pMP = mpFrameWindow->GetMapPoint(i);
				if (!pMP)
					continue;
				if (pMP->isDeleted())
					continue;
				cv::Mat x3D = pMP->GetWorldPos();
				cv::Point2f tpt = cv::Point2f(x3D.at<float>(0) * mnVisScale, -x3D.at<float>(2) * mnVisScale);
				tpt += mVisMidPt;
				if (mpFrameWindow->GetBoolInlier(i))
					cv::circle(tempVis, tpt, 2, ObjectColors::mvObjectLabelColors[pMP->GetObjectType()], -1);
				else
					cv::circle(tempVis, tpt, 1, color2, 1);
			}

			//trajectory
			auto mvpWindowFrames = mpFrameWindow->GetAllFrames();
			for (int i = 0; i < mvpWindowFrames.size() - 1; i++) {
				cv::Mat t2 = mvpWindowFrames[i + 1]->GetTranslation();
				cv::Mat t1 = mvpWindowFrames[i]->GetTranslation();
				cv::Point2f pt1 = cv::Point2f(t1.at<float>(0)* mnVisScale, t1.at<float>(2)* mnVisScale);
				cv::Point2f pt2 = cv::Point2f(t2.at<float>(0)* mnVisScale, t2.at<float>(2)* mnVisScale);
				pt1 += mVisMidPt;
				pt2 += mVisMidPt;
				cv::line(tempVis, pt1, pt2, cv::Scalar(0, 0, 0), 2);
			}
			
			cv::imshow("Output::Trajectory", tempVis);
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
		if (!mpMatchingFrame2->GetBoolInlier(i))
			continue;
		UVR_SLAM::MapPoint* pMP = mpMatchingFrame2->GetMapPoint(i);
		cv::circle(vis, mpMatchingFrame2->mvKeyPoints[i].pt, 1, cv::Scalar(255, 0, 255), -1);
		if (pMP) {
			if (pMP->isDeleted()) {
				mpMatchingFrame2->SetBoolInlier(false, i);
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

	cv::Mat vis2 = mpMatchingFrame2->GetOriginalImage();
	for (int i = 0; i < mpMatchingFrame2->mvKeyPoints.size(); i++) {
		UVR_SLAM::ObjectType type = mpMatchingFrame2->GetObjectType(i);
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
		if(mpMatchingFrame2->GetBoolInlier(mvMatchInfos[i].trainIdx))
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
