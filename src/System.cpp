#include <System.h>
#include <FrameWindow.h>
#include <Initializer.h>
#include <IndoorLayoutEstimator.h>
#include <LocalMapper.h>

UVR_SLAM::System::System() {}
UVR_SLAM::System::System(int nWidth, int nHeight, cv::Mat _K, cv::Mat _K2, cv::Mat _D, int _nFeatures, float _fScaleFactor, int _nLevels, int _fIniThFAST, int _fMinThFAST, std::string _strVOCPath):
	mnWidth(nWidth), mnHeight(nHeight), mK(_K), mKforPL(_K2), mD(_D),
	mnFeatures(_nFeatures), mfScaleFactor(_fScaleFactor), mnLevels(_nLevels), mfIniThFAST(_fIniThFAST), mfMinThFAST(_fMinThFAST), strVOCPath(_strVOCPath)
{
	LoadVocabulary();
	Init();
}
UVR_SLAM::System::~System() {}

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

	//initializer
	mpInitializer = new UVR_SLAM::Initializer(mK);
	mpInitializer->SetMatcher(mpMatcher);
	mpInitializer->SetFrameWindow(mpFrameWindow);

	//local mapping thread
	mpLocalMapper = new UVR_SLAM::LocalMapper(mnWidth, mnHeight);
	mpLocalMapper->SetFrameWindow(mpFrameWindow);
	mpLocalMapper->SetMatcher(mpMatcher);
	mptLocalMapper = new std::thread(&UVR_SLAM::LocalMapper::Run, mpLocalMapper);

	//tracker thread
	mpTracker = new UVR_SLAM::Tracker(mnWidth, mnHeight, mK);
	//mptTracker = new std::thread(&UVR_SLAM::Tracker::Run, mpTracker);
	mpTracker->SetMatcher(mpMatcher);
	mpTracker->SetInitializer(mpInitializer);
	mpTracker->SetFrameWindow(mpFrameWindow);
	mpTracker->SetLocalMapper(mpLocalMapper);
	mpTracker->SetSystem(this);

	//loop closing thread

	//layout estimating thread
	mpLayoutEstimator = new UVR_SLAM::IndoorLayoutEstimator(mnWidth, mnHeight);
	mpLayoutEstimator->SetSystem(this);
	mpLayoutEstimator->SetFrameWindow(mpFrameWindow);
	mptLayoutEstimator = new std::thread(&UVR_SLAM::IndoorLayoutEstimator::Run, mpLayoutEstimator);

	namedWindow("Output::Segmentation");
	moveWindow("Output::Segmentation", -1650, 20);
	namedWindow("Output::Tracking");
	moveWindow("Output::Tracking", -1650+mnWidth, 20);
	namedWindow("Output::Matching::SemanticFrame");
	moveWindow("Output::Matching::SemanticFrame", -1650 + mnWidth, 50 + mnHeight);
	namedWindow("Output::SegmentationMask");
	moveWindow("Output::SegmentationMask", -1650, 50 + mnHeight);
	/*namedWindow("Output::Trajectory");
	moveWindow("Output::Trajectory", -1650+ mnWidth + mnWidth, 20);*/
	

	int nAdditional1 = 355;
	namedWindow("Output::Trajectory");
	moveWindow("Output::Trajectory", -1650 + nAdditional1 + mnWidth + mnWidth + mnWidth, 0);
	namedWindow("Initialization::Frame::1");
	moveWindow("Initialization::Frame::1", -1650 + nAdditional1 + mnWidth+ mnWidth+ mnWidth+ mnWidth,0);
	namedWindow("Initialization::Frame::2");
	moveWindow("Initialization::Frame::2", -1650 + nAdditional1 + mnWidth + mnWidth + mnWidth + mnWidth, 30+mnHeight);
	namedWindow("LocalMapping::CreateMPs");
	moveWindow("LocalMapping::CreateMPs", -1650 + nAdditional1 + mnWidth+ mnWidth + mnWidth + mnWidth, 0);

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


cv::Mat mVisualized2DMap = cv::Mat::ones(720, 720, CV_8UC1)*255;
cv::Point2f midPt = cv::Point(360, 360);
cv::Point2f prevPt = midPt;
void UVR_SLAM::System::VisualizeTranslation() {
	if (mbInitialized) {
		int mnScale = 200;
		cv::Mat t = mpCurrFrame->GetTranslation();
		cv::Point2f pt = cv::Point2f(t.at<float>(0)* 200, t.at<float>(1)* 200);
		pt += midPt;
		cv::line(mVisualized2DMap, prevPt, pt, cv::Scalar(0, 0, 255), 1);
		//std::cout << "POSE=" << pt << ", " << prevPt <<"::"<<t<< std::endl;
		prevPt = pt;
		
	}
	cv::imshow("Output::Trajectory", mVisualized2DMap);
	cv::waitKey(1);
}