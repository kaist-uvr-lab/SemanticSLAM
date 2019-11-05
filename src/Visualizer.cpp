#include <Visualizer.h>
#include <FrameWindow.h>
#include <Frame.h>

UVR_SLAM::Visualizer::Visualizer() {}
UVR_SLAM::Visualizer::Visualizer(int w, int h, int scale) :mnWidth(w), mnHeight(h), mnVisScale(scale) {}
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

	
	cv::namedWindow("Output::Trajectory");
	cv::moveWindow("Output::Trajectory", -1690 + mnWidth + mnWidth, 20);
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
			for (int i = 0; i < mpFrameWindow->LocalMapSize; i++) {
				UVR_SLAM::MapPoint* pMP = mpFrameWindow->GetMapPoint(i);
				if (!pMP)
					continue;
				if (pMP->isDeleted())
					continue;
				cv::Mat x3D = pMP->GetWorldPos();
				cv::Point2f tpt = cv::Point2f(x3D.at<float>(0) * mnVisScale, -x3D.at<float>(2) * mnVisScale);
				tpt += mVisMidPt;
				if (mpFrameWindow->GetBoolInlier(i))
					cv::circle(tempVis, tpt, 2, color1, -1);
				else
					cv::circle(tempVis, tpt, 2, color2, -1);
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
				cv::line(tempVis, pt1, pt2, cv::Scalar(255, 0, 0), 2);
			}
			
			cv::imshow("Output::Trajectory", tempVis);
			cv::waitKey(1);

			SetBoolDoingProcess(false);
		}//if
	}//while
}

//////////////////////////////////////
