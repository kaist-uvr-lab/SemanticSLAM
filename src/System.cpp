#include <System.h>
#include <Map.h>
#include <FrameWindow.h>
#include <Initializer.h>
#include <SemanticSegmentator.h>
#include <LocalMapper.h>
#include <LoopCloser.h>
#include <PlaneEstimator.h>
#include <Visualizer.h>
#include <FrameVisualizer.h>
#include <MapOptimizer.h>
#include <direct.h>
#include <Converter.h>

int UVR_SLAM::System::nKeyFrameID = 0;

UVR_SLAM::System::System(){}
UVR_SLAM::System::System(std::string strFilePath):mstrFilePath(strFilePath), mbTrackingEnd(true), mbLocalMapUpdateEnd(true), mbSegmentationEnd(false), mbLocalMappingEnd(false), mbPlaneEstimationEnd(false), mbPlanarMPEnd(false), 
mbSegmentation(false), mbPlaneEstimation(false), mStrSegmentationString("Segmentation"), mStrPlaneString("PE"){
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

	mbSegmentation=fs["Segmentator.on"]=="on";
	mbPlaneEstimation = fs["PlaneEstimator.on"] == "on";

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

	mnPatchSize = fs["Dense.patch"];
	mnHalfWindowSize = fs["Dense.window"];
	
	mnMaxConnectedKFs = fs["KeyFrame.window"];
	mnMaxCandidateKFs = fs["KeyFrame.candidate"];

	mnDisplayX = fs["Display.x"];
	mnDisplayY = fs["Display.y"];

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

void UVR_SLAM::System::SaveTrajectory(std::string filename) {
	auto vpKFs = mpMap->GetTrajectoryFrames();
	/*auto vpKFs = mpMap->GetWindowFramesVector(3);
	auto vpGraphKFs = mpMap->GetGraphFrames();
	for (int i = 0; i < vpGraphKFs.size(); i++)
		vpKFs.push_back(vpGraphKFs[i]);*/

	std::string base = GetDirPath(0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ;
	std::stringstream ssdir, ssfile;
	/*std::s  tringstream ssDirPath;
	ssDirPath << "../../bin/SLAM/KeyframeDebugging/"*/
	//ssdir <<base<<"/trajectory";
	ssdir <<"./trajectory";
	_mkdir(ssdir.str().c_str());
	ssfile << ssdir.str() << "/"<<filename;
	std::ofstream f;
	f.open(ssfile.str().c_str());
	f << std::fixed;
	for (int i = 0; i < vpKFs.size(); i++) {
		auto pKF = vpKFs[i];
		
		cv::Mat R, t;     
		pKF->GetPose(R, t);
		R = R.t(); //inverse
		t = -R*t;  //camera center
		std::vector<float> q = Converter::toQuaternion(R);
		f << std::setprecision(6) << pKF->mdTimestamp << std::setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
			<< " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;
	}
	f.close();
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
	mpMap = new UVR_SLAM::Map(this, mnMaxConnectedKFs, mnMaxCandidateKFs);

	////FrameWindow
	//mpFrameWindow = new UVR_SLAM::FrameWindow(15);
	//mpFrameWindow->SetSystem(this);

	//optmizer
	mpOptimizer = new UVR_SLAM::Optimization();

	//Visualizer
	mpVisualizer = new Visualizer(this, mnWidth, mnHeight, mnVisScale);

	//FrameVisualizer
	mpFrameVisualizer = new FrameVisualizer(this, mnWidth, mnHeight, mK);

	//matcher
	mpMatcher = new UVR_SLAM::Matcher(this, DescriptorMatcher::create("BruteForce-Hamming"), mnWidth, mnHeight);

	//initializer
	mpInitializer = new UVR_SLAM::Initializer(this, mpMap, mK, mnWidth, mnHeight);

	//PlaneEstimator
	mpPlaneEstimator = new UVR_SLAM::PlaneEstimator(this,mstrFilePath, mK, mKforPL, mnWidth, mnHeight);

	//layout estimating thread
	mpSegmentator = new UVR_SLAM::SemanticSegmentator(this,mstrFilePath);

	//local mapping thread
	mpLocalMapper = new UVR_SLAM::LocalMapper(this, mstrFilePath, mnWidth, mnHeight);
	
	//loop closing thread
	mpLoopCloser = new UVR_SLAM::LoopCloser(this, mnWidth, mnHeight, mK);

	//map optimizer
	mpMapOptimizer = new UVR_SLAM::MapOptimizer(mstrFilePath, this);

	//tracker thread
	mpTracker = new UVR_SLAM::Tracker(this, mstrFilePath);
	
	/////초기화
	mpSegmentator->Init();
	mpPlaneEstimator->Init();
	mpLocalMapper->Init();
	mpLoopCloser->Init();
	mpMapOptimizer->Init();
	mpVisualizer->Init();
	mpFrameVisualizer->Init();
	mpInitializer->Init();
	mpTracker->Init();
	mpMatcher->Init();

	////쓰레드
	mptLocalMapper = new std::thread(&UVR_SLAM::LocalMapper::Run, mpLocalMapper);
	mptMapOptimizer = new std::thread(&UVR_SLAM::MapOptimizer::Run, mpMapOptimizer);
	mptLoopCloser = new std::thread(&UVR_SLAM::LoopCloser::Run, mpLoopCloser);
	if(mbSegmentation)
		mptLayoutEstimator = new std::thread(&UVR_SLAM::SemanticSegmentator::Run, mpSegmentator);
	if(mbPlaneEstimation)
		mptPlaneEstimator = new std::thread(&UVR_SLAM::PlaneEstimator::Run, mpPlaneEstimator);
	mptVisualizer = new std::thread(&UVR_SLAM::Visualizer::Run, mpVisualizer);
	mptFrameVisualizer = new std::thread(&UVR_SLAM::FrameVisualizer::Run, mpFrameVisualizer);

	//Time
	mnSegID = mnLoalMapperID = mnPlaneID = mnMapOptimizerID = 0;
	mfLocalMappingTime1 = mfLocalMappingTime2 = 0.0;
}

void UVR_SLAM::System::SetCurrFrame(cv::Mat img, double t) {
	mpPrevFrame = mpCurrFrame;
	mpCurrFrame = new UVR_SLAM::Frame(img, mnWidth, mnHeight, mK, t);
	//mpCurrFrame->DetectEdge();
	//std::cout << mpCurrFrame->mnFrameID << std::endl;
	//mpCurrFrame->Init(mpORBExtractor, mK, mD);
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
	mpTracker->Tracking(mpPrevFrame , mpCurrFrame);
	////제안서 작업시 이미지 특징점 저장용
	/*cv::Mat img = mpCurrFrame->GetOriginalImage();
	for (int i = 0; i < mpCurrFrame->mvKeyPoints.size(); i++) {
		cv::circle(img, mpCurrFrame->mvKeyPoints[i].pt, 3, cv::Scalar(255, 255, 0), -1);
	}
	imwrite("d:/res.png", img);*/
	////제안서 작업시 이미지 특징점 저장용
}

void UVR_SLAM::System::Reset() {
	mbInitialized = false;
	mpInitializer->Reset();
	mpFrameWindow->ClearLocalMapFrames();
	mpPlaneEstimator->Reset();
	mpLocalMapper->Reset();
	mpMap->ClearWalls();
	mlpNewMPs.clear();
	//mpLocalMapper->mlpNewMPs.clear();
	nKeyFrameID = 1;
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
		if (id == 0) {
			ss << mStrBasePath.c_str();
		}
		else
			ss << mStrBasePath.c_str() << "/" << id;
	}
	_mkdir(ss.str().c_str());
}
std::string UVR_SLAM::System::GetDirPath(int id){
	std::string strPath;
	std::stringstream ss;
	{
		std::unique_lock<std::mutex> lock(mMutexDirPath);
		if (id == 0) {
			ss << mStrBasePath.c_str();
		}else
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
void UVR_SLAM::System::SetLocalMapperString(std::string str) {
	std::unique_lock<std::mutex> lock(mMutexLocalMapperString);
	mStrLocalMapperString = str;
}
std::string UVR_SLAM::System::GetLocalMapperString() {
	std::unique_lock<std::mutex> lock(mMutexLocalMapperString);
	return mStrLocalMapperString;
}
void UVR_SLAM::System::SetSegmentationString(std::string str) {
	std::unique_lock<std::mutex> lock(mMutexSegmentationString);
	mStrSegmentationString = str;
}
std::string UVR_SLAM::System::GetSegmentationString() {
	std::unique_lock<std::mutex> lock(mMutexSegmentationString);
	return mStrSegmentationString;
}
void UVR_SLAM::System::SetMapOptimizerString(std::string str) {
	std::unique_lock<std::mutex> lock(mMutexMapOptimizer);
	mStrMapOptimizer = str;
}
std::string UVR_SLAM::System::GetMapOptimizerString() {
	std::unique_lock<std::mutex> lock(mMutexMapOptimizer);
	return mStrMapOptimizer;
}