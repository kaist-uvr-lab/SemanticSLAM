#include <System.h>
#include <FrameWindow.h>
#include <Initializer.h>
#include <IndoorLayoutEstimator.h>
#include <LocalMapper.h>
#include <PlaneEstimator.h>
#include <Visualizer.h>

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

	//Visualizer
	mpVisualizer = new Visualizer(mnWidth, mnHeight, mnVisScale);
	mpVisualizer->Init();
	mpVisualizer->SetFrameWindow(mpFrameWindow);
	mptVisualizer = new std::thread(&UVR_SLAM::Visualizer::Run, mpVisualizer);

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

	
	//set visualizer
	mpTracker->SetVisualizer(mpVisualizer);

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
