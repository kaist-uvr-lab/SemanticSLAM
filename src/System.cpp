#include <System.h>
#include <Map.h>
#include <FrameWindow.h>
#include <Initializer.h>
#include <SemanticSegmentator.h>
#include <LocalMapper.h>
#include <PlaneEstimator.h>
#include <Visualizer.h>
#include <MapOptimizer.h>
#include <direct.h>

int UVR_SLAM::System::nKeyFrameID = 0;

UVR_SLAM::System::System(){}
UVR_SLAM::System::System(std::string strFilePath):mstrFilePath(strFilePath), mbTrackingEnd(true), mbLocalMapUpdateEnd(true), mbSegmentationEnd(false), mbLocalMappingEnd(false), mbPlaneEstimationEnd(false), mbPlanarMPEnd(false){
	LoadParameter(strFilePath);
	LoadVocabulary();
	Init();
}
UVR_SLAM::System::System(int nWidth, int nHeight, cv::Mat _K, cv::Mat _K2, cv::Mat _D, int _nFeatures, float _fScaleFactor, int _nLevels, int _fIniThFAST, int _fMinThFAST, std::string _strVOCPath):
	mnWidth(nWidth), mnHeight(nHeight), mK(_K), mKforPL(_K2), mD(_D), mbTrackingEnd(true), mbLocalMapUpdateEnd(true), mbSegmentationEnd(false), mbLocalMappingEnd(false), mbPlaneEstimationEnd(false), mbPlanarMPEnd(false),
	mnFeatures(_nFeatures), mfScaleFactor(_fScaleFactor), mnLevels(_nLevels), mfIniThFAST(_fIniThFAST), mfMinThFAST(_fMinThFAST), strVOCPath(_strVOCPath)
{
	LoadVocabulary();
	Init();
}
UVR_SLAM::System::~System() {}

void UVR_SLAM::System::LoadParameter(std::string strPath) {
	FileStorage fs(strPath, FileStorage::READ);

	float fx = fs["Camera.fx"];
	float fy = fs["Camera.fy"];
	float cx = fs["Camera.cx"];
	float cy = fs["Camera.cy"];

	mK = cv::Mat::eye(3, 3, CV_32F);
	mK.at<float>(0, 0) = fx;
	mK.at<float>(1, 1) = fy;
	mK.at<float>(0, 2) = cx;
	mK.at<float>(1, 2) = cy;

	cv::Mat DistCoef(4, 1, CV_32F);
	DistCoef.at<float>(0) = fs["Camera.k1"];
	DistCoef.at<float>(1) = fs["Camera.k2"];
	DistCoef.at<float>(2) = fs["Camera.p1"];
	DistCoef.at<float>(3) = fs["Camera.p2"];
	const float k3 = fs["Camera.k3"];
	if (k3 != 0)
	{
		DistCoef.resize(5);
		DistCoef.at<float>(4) = k3;
	}
	DistCoef.copyTo(mD);

	mnFeatures = fs["ORBextractor.nFeatures"];
	mfScaleFactor = fs["ORBextractor.scaleFactor"];
	mnLevels = fs["ORBextractor.nLevels"];
	mfIniThFAST = fs["ORBextractor.iniThFAST"];
	mfMinThFAST = fs["ORBextractor.minThFAST"];

	mnWidth = fs["Image.width"];
	mnHeight = fs["Image.height"];

	std::cout << mK << mD << std::endl;

	fs["VocPath"] >> strVOCPath;
	fs["nVisScale"] >> mnVisScale;
	//fs["IP"] >> ip;
	//fs["port"] >> port;
	//std::cout << ip << "::" << port << std::endl;
	//fs["DirPath"] >> strDirPath; 데이터와 관련이 있는거여서 별도로 분리

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

	InitDirPath();

	//init
	mbInitialized = false;
	//mpInitFrame = nullptr;
	mpCurrFrame = nullptr;
	mpPrevFrame = nullptr;

	mpInitORBExtractor = new UVR_SLAM::ORBextractor(2 * mnFeatures, mfScaleFactor, mnLevels, mfIniThFAST, mfMinThFAST);
	mpPoseORBExtractor = new UVR_SLAM::ORBextractor(mnFeatures, mfScaleFactor, mnLevels, mfIniThFAST, mfMinThFAST);
	mpORBExtractor = mpInitORBExtractor;

	//Map
	mpMap = new UVR_SLAM::Map();

	//FrameWindow
	mpFrameWindow = new UVR_SLAM::FrameWindow(15);
	mpFrameWindow->SetSystem(this);

	//optmizer
	mpOptimizer = new UVR_SLAM::Optimization();

	//matcher
	mpMatcher = new UVR_SLAM::Matcher(DescriptorMatcher::create("BruteForce-Hamming"), mnWidth, mnHeight);

	//Visualizer
	mpVisualizer = new Visualizer(mnWidth, mnHeight, mnVisScale, mpMap);
	mpVisualizer->Init();
	mpVisualizer->SetFrameWindow(mpFrameWindow);
	mpVisualizer->SetSystem(this);
	mptVisualizer = new std::thread(&UVR_SLAM::Visualizer::Run, mpVisualizer);

	//initializer
	mpInitializer = new UVR_SLAM::Initializer(this, mpMap, mK);
	mpInitializer->SetMatcher(mpMatcher);
	mpInitializer->SetFrameWindow(mpFrameWindow);

	//PlaneEstimator
	mpPlaneEstimator = new UVR_SLAM::PlaneEstimator(mpMap,mstrFilePath, mK, mKforPL, mnWidth, mnHeight);
	mpPlaneEstimator->SetSystem(this);
	mpPlaneEstimator->SetFrameWindow(mpFrameWindow);
	mpPlaneEstimator->SetMatcher(mpMatcher);
	mptPlaneEstimator = new std::thread(&UVR_SLAM::PlaneEstimator::Run, mpPlaneEstimator);

	//layout estimating thread
	mpSegmentator = new UVR_SLAM::SemanticSegmentator(mstrFilePath);
	mpSegmentator->SetSystem(this);
	mpSegmentator->SetFrameWindow(mpFrameWindow);
	mpSegmentator->SetPlaneEstimator(mpPlaneEstimator);
	mptLayoutEstimator = new std::thread(&UVR_SLAM::SemanticSegmentator::Run, mpSegmentator);

	//local mapping thread
	mpLocalMapper = new UVR_SLAM::LocalMapper(mpMap, mnWidth, mnHeight);
	mpLocalMapper->SetFrameWindow(mpFrameWindow);
	mpLocalMapper->SetMatcher(mpMatcher);
	mpLocalMapper->SetPlaneEstimator(mpPlaneEstimator);
	mpLocalMapper->SetLayoutEstimator(mpSegmentator);
	mpLocalMapper->SetSystem(this);
	mptLocalMapper = new std::thread(&UVR_SLAM::LocalMapper::Run, mpLocalMapper);

	mpInitializer->SetLocalMapper(mpLocalMapper);
	mpInitializer->SetSegmentator(mpSegmentator);
	//loop closing thread

	//map optimizer
	mpMapOptimizer = new UVR_SLAM::MapOptimizer(mstrFilePath, mpMap);
	mpMapOptimizer->SetFrameWindow(mpFrameWindow);
	mpMapOptimizer->SetSystem(this);
	mptMapOptimizer = new std::thread(&UVR_SLAM::MapOptimizer::Run, mpMapOptimizer);

	//tracker thread
	mpTracker = new UVR_SLAM::Tracker(mpMap, mstrFilePath);
	//mptTracker = new std::thread(&UVR_SLAM::Tracker::Run, mpTracker);
	mpTracker->SetMatcher(mpMatcher);
	mpTracker->SetInitializer(mpInitializer);
	mpTracker->SetFrameWindow(mpFrameWindow);
	mpTracker->SetLocalMapper(mpLocalMapper);
	mpTracker->SetPlaneEstimator(mpPlaneEstimator);
	mpTracker->SetSystem(this);

	mpTracker->SetMapOptimizer(mpMapOptimizer);
	mpLocalMapper->SetMapOptimizer(mpMapOptimizer);
	
	//set visualizer
	mpTracker->SetSegmentator(mpSegmentator);
	mpTracker->SetVisualizer(mpVisualizer);
	mpPlaneEstimator->SetInitializer(mpInitializer);
	mpSegmentator->SetLocalMapper(mpLocalMapper);

	//Time
	mnSegID = mnLoalMapperID = mnPlaneID = mnMapOptimizerID = 0;
	mfMapOptimizerTime = 0.0;
	mfLocalMappingTime1 = mfLocalMappingTime2 = 0.0;
}

void UVR_SLAM::System::SetCurrFrame(cv::Mat img) {
	mpPrevFrame = mpCurrFrame;
	mpCurrFrame = new UVR_SLAM::Frame(img, mnWidth, mnHeight, mK);
	//std::cout << mpCurrFrame->mnFrameID << std::endl;
	mpCurrFrame->Init(mpORBExtractor, mK, mD);
	//mpCurrFrame->SetBowVec(fvoc);
	
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
	mpTracker->Tracking(mpPrevFrame , mpCurrFrame);
	//std::cout << "Track::End\n";
}

void UVR_SLAM::System::Reset() {
	mbInitialized = false;
	mpInitializer->Reset();
	mpFrameWindow->ClearLocalMapFrames();
	mpMap->ClearFrames();
	mpMap->ClearWalls();
	mlpNewMPs.clear();
	//mpLocalMapper->mlpNewMPs.clear();
	nKeyFrameID = 0;
	//frame reset

}

void UVR_SLAM::System::SetBoolInit(bool b) {
	std::unique_lock<std::mutex> lock(mMutexInitialized);
	mbInitialized = b;
}

bool UVR_SLAM::System::isInitialized() {
	std::unique_lock<std::mutex> lock(mMutexInitialized);
	return mbInitialized;
}

void UVR_SLAM::System::InitDirPath() {
	std::unique_lock<std::mutex> lock(mMutexDirPath);
	//확인용 폴더 생성하기 위한 것
	std::time_t curr_time;
	struct tm curr_tm;
	curr_time = time(NULL);
	localtime_s(&curr_tm, &curr_time);
	//curr_tm = std::localtime(&curr_time);

	std::stringstream ssDirPath;
	ssDirPath << "../../bin/SLAM/KeyframeDebugging/" << curr_tm.tm_year << "_" << curr_tm.tm_mon << "_" << curr_tm.tm_mday << "_" << curr_tm.tm_hour << "_" << curr_tm.tm_min << "_" << curr_tm.tm_sec;
	mStrBasePath = ssDirPath.str();
	_mkdir(mStrBasePath.c_str());
}

void UVR_SLAM::System::SetDirPath(int id) {

	std::stringstream ss;
	{
		std::unique_lock<std::mutex> lock(mMutexDirPath);
		ss << mStrBasePath.c_str() << "/" << id;
	}
	_mkdir(ss.str().c_str());
}
std::string UVR_SLAM::System::GetDirPath(int id){
	std::string strPath;
	std::stringstream ss;
	{
		std::unique_lock<std::mutex> lock(mMutexDirPath);
		ss << mStrBasePath.c_str() << "/" << id;
	}
	strPath = ss.str();
	return strPath;
}

//이것들 전부다 private로 변경.
//trajecotoryMap, MPsMap;

//void UVR_SLAM::System::SetLayoutTime(float t1) {
//	std::unique_lock<std::mutex> lock(mMutexLayoutTime);
//	mfLayoutTime = t1;
//}
//
//void UVR_SLAM::System::GetLayoutTime(float& t1) {
//	std::unique_lock<std::mutex> lock(mMutexLayoutTime);
//	t1 = mfLayoutTime;
//}

void UVR_SLAM::System::SetLocalMappingTime(float t1, float t2) {
	std::unique_lock<std::mutex> lock(mMutexLocalMappingTime);
	mfLocalMappingTime1 = t1;
	mfLocalMappingTime2 = t2;
}

void UVR_SLAM::System::GetLocalMappingTime(float& t1, float& t2){
	std::unique_lock<std::mutex> lock(mMutexLocalMappingTime);
	t1 = mfLocalMappingTime1;
	t2 = mfLocalMappingTime2;
}
void UVR_SLAM::System::SetSegFrameID(int n){
	std::unique_lock<std::mutex> lock(mMutexSegID);
	mnSegID = n;
}
int UVR_SLAM::System::GetSegFrameID(){
	std::unique_lock<std::mutex> lock(mMutexSegID);
	return mnSegID;
}
void UVR_SLAM::System::SetLocalMapperFrameID(int n)
{
	std::unique_lock<std::mutex> lock(mMutexLMID);
	mnLoalMapperID = n;
}
int UVR_SLAM::System::GetLocalMapperFrameID(){
	std::unique_lock<std::mutex> lock(mMutexLMID);
	return mnLoalMapperID;
}
void UVR_SLAM::System::SetPlaneFrameID(int n)
{
	std::unique_lock<std::mutex> lock(mMutexPlaneID);
	mnPlaneID = n;
}
int UVR_SLAM::System::GetPlaneFrameID() {
	std::unique_lock<std::mutex> lock(mMutexPlaneID);
	return mnPlaneID;
}



void UVR_SLAM::System::SetMapOptimizerTime(float t1) {
	std::unique_lock<std::mutex> lock(mMutexMapOptimizerTime);
	mfMapOptimizerTime = t1;
}

void UVR_SLAM::System::GetMapOptimizerTime(float& t1) {
	std::unique_lock<std::mutex> lock(mMutexMapOptimizerTime);
	t1 = mfMapOptimizerTime;
}
void UVR_SLAM::System::SetMapOptimizerID(int n)
{
	std::unique_lock<std::mutex> lock(mMutexMapOptimizerID);
	mnMapOptimizerID = n;
}
int UVR_SLAM::System::GetMapOptimizerID() {
	std::unique_lock<std::mutex> lock(mMutexMapOptimizerID);
	return mnMapOptimizerID;
}

void UVR_SLAM::System::SetPlaneString(std::string str){
	std::unique_lock<std::mutex> lock(mMutexPlaneString);
	mStrPlaneString = str;
}
std::string UVR_SLAM::System::GetPlaneString(){
	std::unique_lock<std::mutex> lock(mMutexPlaneString);
	return mStrPlaneString;
}
void UVR_SLAM::System::SetTrackerString(std::string str) {
	std::unique_lock<std::mutex> lock(mMutexTrackerString);
	mStrTrackerString = str;
}
std::string UVR_SLAM::System::GetTrackerString() {
	std::unique_lock<std::mutex> lock(mMutexTrackerString);
	return mStrTrackerString;
}
void UVR_SLAM::System::SetSegmentationString(std::string str) {
	std::unique_lock<std::mutex> lock(mMutexSegmentationString);
	mStrSegmentationString = str;
}
std::string UVR_SLAM::System::GetSegmentationString() {
	std::unique_lock<std::mutex> lock(mMutexSegmentationString);
	return mStrSegmentationString;
}