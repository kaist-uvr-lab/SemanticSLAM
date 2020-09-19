#include <SemanticSegmentator.h>
#include <System.h>
#include <SegmentationData.h>
#include <FrameWindow.h>
#include <PlaneEstimator.h>
#include <Plane.h>
#include <LocalMapper.h>
#include <Map.h>

std::vector<cv::Vec3b> UVR_SLAM::ObjectColors::mvObjectLabelColors;

UVR_SLAM::SemanticSegmentator::SemanticSegmentator():mbDoingProcess(false){
	/*mVecLabelColors.push_back(COLOR_FLOOR);
	mVecLabelColors.push_back(COLOR_WALL);
	mVecLabelColors.push_back(COLOR_CEILING);
	mVecLabelColors.push_back(COLOR_NONE);*/
}
UVR_SLAM::SemanticSegmentator::SemanticSegmentator(std::string _ip, int _port, int w, int h): ip(_ip), port(_port),mbDoingProcess(false), mnWidth(w), mnHeight(h), mpPrevTargetFrame(nullptr){
	UVR_SLAM::ObjectColors::Init();
	/*mVecLabelColors.push_back(COLOR_FLOOR);
	mVecLabelColors.push_back(COLOR_WALL);
	mVecLabelColors.push_back(COLOR_CEILING);
	mVecLabelColors.push_back(COLOR_NONE);*/
}
UVR_SLAM::SemanticSegmentator::SemanticSegmentator(Map* pMap, const std::string & strSettingPath) :mbDoingProcess(false), mpPrevTargetFrame(nullptr) {
	UVR_SLAM::ObjectColors::Init();
	cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
	mbOn = static_cast<int>(fSettings["Segmentator.on"]) != 0;
	ip = fSettings["Segmentator.ip"];
	port = fSettings["Segmentator.port"];
	mnWidth = fSettings["Image.width"];
	mnHeight = fSettings["Image.height"];
	cx = fSettings["Camera.cx"];
	cy = fSettings["Camera.cy"];
	fSettings.release();
	/*mVecLabelColors.push_back(COLOR_FLOOR);
	mVecLabelColors.push_back(COLOR_WALL);
	mVecLabelColors.push_back(COLOR_CEILING);
	mVecLabelColors.push_back(COLOR_NONE);*/
	mpMap = pMap;
}

UVR_SLAM::SemanticSegmentator::~SemanticSegmentator(){}


void UVR_SLAM::SemanticSegmentator::InsertKeyFrame(UVR_SLAM::Frame *pKF)
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	//std::cout << "Segmentator::" << mKFQueue.size() << std::endl;
	mKFQueue.push(pKF);
}

bool UVR_SLAM::SemanticSegmentator::CheckNewKeyFrames()
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	return(!mKFQueue.empty());
}

void UVR_SLAM::SemanticSegmentator::ProcessNewKeyFrame()
{
	std::unique_lock<std::mutex> lock(mMutexNewKFs);
	mpPrevTargetFrame = mpTargetFrame;
	mpTargetFrame = mKFQueue.front();
	mpTargetFrame->TurnOnFlag(UVR_SLAM::FLAG_SEGMENTED_FRAME);
	
	//mpSystem->SetDirPath(mpTargetFrame->GetKeyFrameID()); //
	
	
	
	////이 내용도 pe에서 진행해도 될 거 같음.
	/*if (mpMap->isFloorPlaneInitialized()) {
		mpTargetFrame->mpPlaneInformation = new UVR_SLAM::PlaneProcessInformation(mpTargetFrame, mpMap->mpFloorPlane);
		mpTargetFrame->mpPlaneInformation->Calculate();
	}*/
	//mpSystem->SetSegFrameID(mpTargetFrame->GetKeyFrameID());	//setsegframeid, getsegframeid는 이용 안하는 듯
	mKFQueue.pop();
}

void UVR_SLAM::SemanticSegmentator::Run() {

	JSONConverter::Init();

	while (1) {
		std::string mStrDirPath;
		if (CheckNewKeyFrames()) {
			
			SetBoolDoingProcess(true);
			ProcessNewKeyFrame();
			//std::cout << "segmentation::start::" << std::endl;
			
			std::chrono::high_resolution_clock::time_point s_start = std::chrono::high_resolution_clock::now();
			cv::Mat colorimg, resized_color, segmented;
			cv::cvtColor(mpTargetFrame->GetOriginalImage(), colorimg, CV_RGBA2BGR);
			colorimg.convertTo(colorimg, CV_8UC3);
			cv::resize(colorimg, resized_color, cv::Size(mnWidth/2, mnHeight/2));
			//cv::resize(colorimg, resized_color, cv::Size(160, 90));

			///////////////////////////////////////////
			///////////세그멘테이션이 시작하는 것을 알림.
			//lock
			std::unique_lock<std::mutex> lock(mpSystem->mMutexUseSegmentation);
			mpSystem->mbSegmentationEnd = false;
			/////////////////////////////////////////
			//insert kyeframe to plane estimator

			//mpPlaneEstimator->InsertKeyFrame(mpTargetFrame);
			//if (mpSystem->isInitialized()) {
			//	//std::cout << "insert??????????" << std::endl;
			//	mpLocalMapper->InsertKeyFrame(mpTargetFrame);
			//}
			
			
			//request post
			//리사이즈 안하면 칼라이미지로
			int status = 0;
			JSONConverter::RequestPOST(ip, port, resized_color, segmented, mpTargetFrame->GetFrameID(), status);

			//////////////////////////////
			////제안서 세그멘테이션 테스트
			//cv::Mat tempImg = cv::imread("D:/abcabcabc.png");
			//cv::Mat tempSeg, tempSegColor;
			//JSONConverter::RequestPOST(ip, port, tempImg, tempSeg, mpTargetFrame->GetFrameID(), status);
			//tempSegColor = cv::Mat::zeros(tempSeg.size(), CV_8UC3);
			//for (int x = 0; x < tempSeg.cols; x++) {
			//	for (int y = 0; y < tempSeg.rows; y++) {
			//		int label = tempSeg.at<uchar>(y, x);
			//		tempSegColor.at<cv::Vec3b>(y, x) = UVR_SLAM::ObjectColors::mvObjectLabelColors[label];
			//	}
			//}
			//cv::resize(tempSegColor, tempSegColor, cv::Size(tempImg.cols, tempImg.rows));
			//cv::addWeighted(tempSegColor, 0.5, tempImg, 0.5, 0.0, tempImg);
			//imshow("aaaaaa", tempImg); waitKey(1);
			//imwrite("D:/aaaseg.jpg", tempSegColor);
			//////////////////////////////

			//cv::resize(segmented, segmented, colorimg.size());
			
			int nRatio = colorimg.rows / segmented.rows;
			//ratio 버전이 아닌 다르게
			ImageLabeling(segmented, mpTargetFrame->matLabeled);
			//ObjectLabeling(mpTargetFrame->matLabeled, nRatio);
			//mpMap->SetCurrFrame(mpTargetFrame);
			mpTargetFrame->matSegmented = segmented.clone();
			mpTargetFrame->SetBoolSegmented(true);
			////////////////////////////////////////////////////////
			///////////세그멘테이션이 끝난 것을 알림.
			//unlock & notify
			mpSystem->mbSegmentationEnd = true;
			lock.unlock();
			mpSystem->cvUseSegmentation.notify_all();
			//unlock & notify
			////////////////////////////////////////////////////////

			//시간체크
			std::chrono::high_resolution_clock::time_point s_end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_start).count();
			float tttt = duration / 1000.0;

			//////////////////////
			////lock
			//std::cout << "seg::s" << std::endl;
			////plane estimation에서 맵포인트를 생성할 때까지 락.
			/*{
				std::unique_lock<std::mutex> lock(mpSystem->mMutexUsePlaneEstimation);
				while (!mpSystem->mbPlaneEstimationEnd) {
					mpSystem->cvUsePlaneEstimation.wait(lock);
				}
			}*/
			//std::cout << "seg::e" << std::endl;
			////lock
			//시간체크
			std::chrono::high_resolution_clock::time_point p_end = std::chrono::high_resolution_clock::now();
			auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(p_end - s_end).count();
			float tttt2 = duration2 / 1000.0;
			//////////////////////

			//////////////////////////////////////////////
			//////디버깅 값 전달
			std::stringstream ssa;
			/*if (mpPrevTargetFrame)
				ssa << "Segmentation : " << mpTargetFrame->GetKeyFrameID() << "=" << mpPrevTargetFrame->GetFrameID() << ", " << mpTargetFrame->GetFrameID() << " : " << tttt << "||" << tttt2 << std::endl;
			else
				ssa << "Seg";*/
			ssa << "Segmentation : " << mpTargetFrame->GetKeyFrameID() <<" : " << tttt << "||" << tttt2 << std::endl;
			//ssa << "Segmentation : " << mpTargetFrame->GetKeyFrameID() << " : " << tttt << ", " << tttt5 << "||" << n1 << ", " << n2 << "::" << numOfLables;
			mpSystem->SetSegmentationString(ssa.str());
			//////디버깅 값 전달
			//////////////////////////////////////////////

			//////////////////////////////////////////////////
			////////디버깅을 위한 이미지 저장
			//mStrDirPath = mpSystem->GetDirPath(mpTargetFrame->GetKeyFrameID());
			//std::stringstream ss;
			//ss << mStrDirPath.c_str() << "/segmentation.jpg";
			//cv::imwrite(ss.str(), segmented);
			//ss.str("");
			//ss << mStrDirPath.c_str() << "/segmentation_color_"<<mpTargetFrame->GetFrameID()<<".jpg";
			//cv::imwrite(ss.str(), colorimg);
			//cv::waitKey(1);
			////////디버깅을 위한 이미지 저장
			//////////////////////////////////////////////////
			//std::cout << "segmentation::end::"<< std::endl;
			mpTargetFrame->mpMatchInfo->SetLabel();
			SetBoolDoingProcess(false);
		}
	}
}
void UVR_SLAM::SemanticSegmentator::SetSystem(System* pSystem) {
	mpSystem = pSystem;
}
void UVR_SLAM::SemanticSegmentator::SetFrameWindow(UVR_SLAM::FrameWindow* pWindow) {
	mpFrameWindow = pWindow;
}
void UVR_SLAM::SemanticSegmentator::SetPlaneEstimator(UVR_SLAM::PlaneEstimator* pEstimator) {
	mpPlaneEstimator = pEstimator;
}
void UVR_SLAM::SemanticSegmentator::SetTargetFrame(Frame* pFrame) {
	mpTargetFrame = pFrame;
}
void UVR_SLAM::SemanticSegmentator::SetLocalMapper(LocalMapper* pEstimator) {
	mpLocalMapper = pEstimator;
}
void UVR_SLAM::SemanticSegmentator::SetBoolDoingProcess(bool b) {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	mbDoingProcess = b;
}
bool UVR_SLAM::SemanticSegmentator::isDoingProcess() {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	return mbDoingProcess;
}
bool UVR_SLAM::SemanticSegmentator::isRun() {
	return mbOn;
}

/////////////////////////////////////////////////////////////////////////////////////////
//모든 특징점과 트래킹되고 있는 맵포인트를 레이블링함.
void UVR_SLAM::SemanticSegmentator::ImageLabeling(cv::Mat segmented, cv::Mat& labeld) {

	labeld = cv::Mat::zeros(segmented.size(), CV_8UC1);

	for (int i = 0; i < segmented.rows; i++) {
		for (int j = 0; j < segmented.cols; j++) {
			int val = segmented.at<uchar>(i, j);
			switch (val) {
			case 1://벽
			case 9://유리창
			case 11://캐비넷
			case 15://문
			
			case 36://옷장
					//case 43://기둥
			case 44://갚난
					//case 94://막대기
			case 23: //그림
			//case 101://포스터
				labeld.at<uchar>(i, j) = 255;
				break;
			case 4:  //floor
			case 29: //rug
			case 101://포스터
				labeld.at<uchar>(i, j) = 150;
				break;
			case 6:
				labeld.at<uchar>(i, j) = 100;
				break;
			case 13: //person, moving object
				labeld.at<uchar>(i, j) = 20;
				break;
			default:
				labeld.at<uchar>(i, j) = 50;
				break;
			}
		}
	}

}

//참조하고 있는 오브젝트 클래스 별로 마스크 이미지를 생성하고 픽셀 단위로 마스킹함.
void UVR_SLAM::SemanticSegmentator::SetSegmentationMask(cv::Mat segmented) {
	mVecLabelMasks.clear();
	for (int i = 0; i < ObjectColors::mvObjectLabelColors.size()-1; i++) {
		cv::Mat temp = cv::Mat::zeros(segmented.size(), CV_8UC1);
		mVecLabelMasks.push_back(temp.clone());
	}
	for (int y = 0; y < segmented.rows; y++) {
		for (int x = 0; x < segmented.cols; x++) {
			cv::Vec3b tempColor = segmented.at<Vec3b>(y, x);
			for (int i = 0; i < ObjectColors::mvObjectLabelColors.size()-1; i++) {
				if (tempColor == ObjectColors::mvObjectLabelColors[i]) {
					mVecLabelMasks[i].at<uchar>(y, x) = 255;
				}
			}
		}
	}
}




