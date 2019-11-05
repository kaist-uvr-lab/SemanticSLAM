#include <System.h>
#include <FrameWindow.h>
#include <Initializer.h>
#include <IndoorLayoutEstimator.h>
#include <LocalMapper.h>
#include <PlaneEstimator.h>

UVR_SLAM::System::System(){}
UVR_SLAM::System::System(std::string strFilePath){
	LoadParameter(strFilePath);
	LoadVocabulary();
	Init();
}
UVR_SLAM::System::System(int nWidth, int nHeight, cv::Mat _K, cv::Mat _K2, cv::Mat _D, int _nFeatures, float _fScaleFactor, int _nLevels, int _fIniThFAST, int _fMinThFAST, std::string _strVOCPath):
	mnWidth(nWidth), mnHeight(nHeight), mK(_K), mKforPL(_K2), mD(_D),
	mnFeatures(_nFeatures), mfScaleFactor(_fScaleFactor), mnLevels(_nLevels), mfIniThFAST(_fIniThFAST), mfMinThFAST(_fMinThFAST), strVOCPath(_strVOCPath)
{
	LoadVocabulary();
	Init();
}
UVR_SLAM::System::~System() {}

void UVR_SLAM::System::LoadParameter(std::string strPath) {
	FileStorage fs(strPath, FileStorage::READ);
	fs["K"] >> mK;
	fs["D"] >> mD;

	fs["nFeatures"] >> mnFeatures;
	fs["fScaleFactor"] >> mfScaleFactor;
	fs["nLevels"] >> mnLevels;
	fs["fIniThFAST"] >> mfIniThFAST;
	fs["fMinThFAST"] >> mfMinThFAST;
	fs["WIDTH"] >> mnWidth;
	fs["HEIGHT"] >> mnHeight;
	fs["VocPath"] >> strVOCPath;
	fs["nVisScale"] >> mnVisScale;
	//fs["DirPath"] >> strDirPath; 데이터와 관련이 있는거여서 별도로 분리

	float fx = mK.at<float>(0, 0);
	float fy = mK.at<float>(1, 1);
	float cx = mK.at<float>(0, 2);
	float cy = mK.at<float>(1, 2);
	
	//Pluker Line Coordinate에 이용함.
	mKforPL = (cv::Mat_<float>(3, 3) << fx, 0, 0, 0, fy, 0, -fy*cx, -fx*cy, fx*fy);

	fs.release();
}

bool UVR_SLAM::System::LoadVocabulary() {
	fvoc = new fbow::Vocabulary();
	
	fvoc->readFromFile(strVOCPath);
	std::string desc_name = fvoc->getDescName();
	auto voc_words = fbow::VocabularyCreator::getVocabularyLeafNodes(*fvoc);
	
	if (voc_words.type() != fvoc->getDescType()) {
		std::cerr << "Invalid types for features according to the voc" << std::endl;
		return false;
	}
	std::cout << "voc desc name=" << desc_name << std::endl;
	std::cout << "number of words=" << voc_words.rows << std::endl;
	return true;
	
}

cv::Point2f sPt = cv::Point2f(0, 0);
cv::Point2f ePt = cv::Point2f(0, 0);
cv::Point2f mPt = cv::Point2f(0, 0);
cv::Mat M = cv::Mat::zeros(0,0, CV_64FC1);

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
		sPt = cv::Point2f(x, y);
	}
	else if (event == EVENT_LBUTTONUP)
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

void UVR_SLAM::System::Init() {

	//init
	mbInitialized = false;
	//mpInitFrame = nullptr;
	mpCurrFrame = nullptr;
	mpPrevFrame = nullptr;

	mpInitORBExtractor = new UVR_SLAM::ORBextractor(2 * mnFeatures, mfScaleFactor, mnLevels, mfIniThFAST, mfMinThFAST);
	mpPoseORBExtractor = new UVR_SLAM::ORBextractor(mnFeatures, mfScaleFactor, mnLevels, mfIniThFAST, mfMinThFAST);
	mpORBExtractor = mpInitORBExtractor;

	//FrameWindow
	mpFrameWindow = new UVR_SLAM::FrameWindow(15);
	mpFrameWindow->SetSystem(this);

	//optmizer
	mpOptimizer = new UVR_SLAM::Optimization();

	//matcher
	mpMatcher = new UVR_SLAM::Matcher(DescriptorMatcher::create("BruteForce-Hamming"), mnWidth, mnHeight);

	//initializer
	mpInitializer = new UVR_SLAM::Initializer(mK);
	mpInitializer->SetMatcher(mpMatcher);
	mpInitializer->SetFrameWindow(mpFrameWindow);

	//PlaneEstimator
	mpPlaneEstimator = new UVR_SLAM::PlaneEstimator(mK, mKforPL, mnWidth, mnHeight);
	mpPlaneEstimator->SetSystem(this);
	mpPlaneEstimator->SetFrameWindow(mpFrameWindow);
	mptPlaneEstimator = new std::thread(&UVR_SLAM::PlaneEstimator::Run, mpPlaneEstimator);

	//layout estimating thread
	mpLayoutEstimator = new UVR_SLAM::IndoorLayoutEstimator(mnWidth, mnHeight);
	mpLayoutEstimator->SetSystem(this);
	mpLayoutEstimator->SetFrameWindow(mpFrameWindow);
	mptLayoutEstimator = new std::thread(&UVR_SLAM::IndoorLayoutEstimator::Run, mpLayoutEstimator);

	//local mapping thread
	mpLocalMapper = new UVR_SLAM::LocalMapper(mnWidth, mnHeight);
	mpLocalMapper->SetFrameWindow(mpFrameWindow);
	mpLocalMapper->SetMatcher(mpMatcher);
	mpLocalMapper->SetPlaneEstimator(mpPlaneEstimator);
	mpLocalMapper->SetLayoutEstimator(mpLayoutEstimator);
	mptLocalMapper = new std::thread(&UVR_SLAM::LocalMapper::Run, mpLocalMapper);

	//loop closing thread

	

	//tracker thread
	mpTracker = new UVR_SLAM::Tracker(mnWidth, mnHeight, mK);
	//mptTracker = new std::thread(&UVR_SLAM::Tracker::Run, mpTracker);
	mpTracker->SetMatcher(mpMatcher);
	mpTracker->SetInitializer(mpInitializer);
	mpTracker->SetFrameWindow(mpFrameWindow);
	mpTracker->SetLocalMapper(mpLocalMapper);
	mpTracker->SetPlaneEstimator(mpPlaneEstimator);
	mpTracker->SetSystem(this);

	int nImageWindowStartX = -1690;
	int nImageWIndowStartY1 = 20;
	int nImageWIndowStartY2 = 50;
	namedWindow("Output::Segmentation");
	moveWindow("Output::Segmentation", nImageWindowStartX, nImageWIndowStartY1);
	namedWindow("Output::Tracking");
	moveWindow("Output::Tracking", nImageWindowStartX +mnWidth, nImageWIndowStartY1);
	namedWindow("Output::PlaneEstimation");
	moveWindow("Output::PlaneEstimation", nImageWindowStartX + mnWidth, 50 + mnHeight);
	namedWindow("Output::SegmentationMask");
	moveWindow("Output::SegmentationMask", nImageWindowStartX, 50 + mnHeight);
	namedWindow("Output::Trajectory");
	moveWindow("Output::Trajectory", nImageWindowStartX + mnWidth + mnWidth, 20);
	
	//Visualization
	mVisualized2DMap = cv::Mat(mnHeight * 2, mnHeight * 2, CV_8UC3, cv::Scalar(255, 255, 255));
	mVisTrajectory   = cv::Mat(mnHeight * 2, mnHeight * 2, CV_8UC3, cv::Scalar(255, 255, 255));
	mVisMapPoints    = cv::Mat(mnHeight * 2, mnHeight * 2, CV_8UC3, cv::Scalar(255, 255, 255));
	mVisPoseGraph	 = cv::Mat(mnHeight * 2, mnHeight * 2, CV_8UC3, cv::Scalar(255, 255, 255));
	mVisMidPt = cv::Point2f(mnHeight, mnHeight);
	mVisPrevPt = mVisMidPt;

	//Opencv Image Window
	int nAdditional1 = 355;
	/*namedWindow("Output::Trajectory");
	moveWindow("Output::Trajectory", nImageWindowStartX + nAdditional1 + mnWidth + mnWidth + mnWidth, 0);*/
	namedWindow("Initialization::Frame::1");
	moveWindow("Initialization::Frame::1", nImageWindowStartX + nAdditional1 + mnWidth+ mnWidth+ mnWidth+ mnWidth,0);
	namedWindow("Initialization::Frame::2");
	moveWindow("Initialization::Frame::2", nImageWindowStartX + nAdditional1 + mnWidth + mnWidth + mnWidth + mnWidth, 30+mnHeight);
	namedWindow("LocalMapping::CreateMPs");
	moveWindow("LocalMapping::CreateMPs", nImageWindowStartX + nAdditional1 + mnWidth+ mnWidth + mnWidth + mnWidth, 0);
	cv::setMouseCallback("Output::Trajectory", CallBackFunc, NULL);
}

void UVR_SLAM::System::SetCurrFrame(cv::Mat img) {
	mpPrevFrame = mpCurrFrame;
	mpCurrFrame = new UVR_SLAM::Frame(img, mnWidth, mnHeight);
	//std::cout << mpCurrFrame->mnFrameID << std::endl;
	mpCurrFrame->Init(mpORBExtractor, mK, mD);
	mpCurrFrame->SetBowVec(fvoc);

	//cv::Mat test = mpCurrFrame->GetOriginalImage();
	//for (int i = 0; i < mpCurrFrame->mvKeyPoints.size(); i++) {
	//	circle(test, mpCurrFrame->mvKeyPoints[i].pt, 3, cv::Scalar(255, 0, 255), -1);
	//}
	//imshow("set image test", test);
	////imwrite("./writed_img.jpg", mpCurrFrame->matOri);
	//cv::waitKey(10);
}


int Ntrial = 10;
void UVR_SLAM::System::SetLayoutFrame() {
	
	//window에 들어 있는 프레임
	if (mbInitialized && !mpLayoutEstimator->isDoingProcess()) {
		UVR_SLAM::Frame* pF = mpFrameWindow->back();
		if (!pF->CheckType(UVR_SLAM::FLAG_SEGMENTED_FRAME)) {
			pF->TurnOnFlag(UVR_SLAM::FLAG_SEGMENTED_FRAME);
			mpLayoutEstimator->SetBoolDoingProcess(true);
			mpLayoutEstimator->SetTargetFrame(pF);
		}
	}
}

void UVR_SLAM::System::Track() {
	//std::cout << "Track::Start\n";
	mpTracker->Tracking(mpPrevFrame , mpCurrFrame, mbInitialized);
	//std::cout << "Track::End\n";
}

void UVR_SLAM::System::Reset() {
	mbInitialized = false;
	mpInitializer->Init();
}


//이것들 전부다 private로 변경.
//trajecotoryMap, MPsMap;
void UVR_SLAM::System::VisualizeTranslation() {
	
	cv::Mat tempVis = mVisPoseGraph.clone();
	if (mbInitialized) {
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
	}
	cv::imshow("Output::Trajectory", tempVis);
	cv::waitKey(1);
}