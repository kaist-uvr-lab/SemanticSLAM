#include <SemanticSegmentator.h>
#include <System.h>
#include <SegmentationData.h>
#include <FrameWindow.h>
#include <PlaneEstimator.h>
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
	mpTargetFrame = mKFQueue.front();
	mpTargetFrame->TurnOnFlag(UVR_SLAM::FLAG_SEGMENTED_FRAME);
	mpSystem->SetSegFrameID(mpTargetFrame->GetKeyFrameID());
	mKFQueue.pop();
}

void UVR_SLAM::SemanticSegmentator::Run() {

	JSONConverter::Init();

	while (1) {
		std::string mStrDirPath;
		if (CheckNewKeyFrames()) {
			std::cout << "segmentation::start" << std::endl;
			SetBoolDoingProcess(true);
			ProcessNewKeyFrame();
			mStrDirPath = mpSystem->GetDirPath(mpTargetFrame->GetKeyFrameID());

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
			mpPlaneEstimator->InsertKeyFrame(mpTargetFrame);
			if (mpSystem->isInitialized()) {
				//std::cout << "insert??????????" << std::endl;
				mpLocalMapper->InsertKeyFrame(mpTargetFrame);
			}
			
			
			//request post
			//리사이즈 안하면 칼라이미지로
			int status = 0;
			JSONConverter::RequestPOST(ip, port, resized_color, segmented, mpTargetFrame->GetFrameID(), status);
			//cv::resize(segmented, segmented, colorimg.size());
			
			int nRatio = colorimg.rows / segmented.rows;
			//ratio 버전이 아닌 다르게
			ObjectLabeling(segmented, nRatio);
			
			mpMap->SetCurrFrame(mpTargetFrame);
			mpTargetFrame->matSegmented = segmented.clone();
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
			std::cout << "seg::s" << std::endl;
			////plane estimation에서 맵포인트를 생성할 때까지 락.
			{
				std::unique_lock<std::mutex> lock(mpSystem->mMutexUsePlaneEstimation);
				while (!mpSystem->mbPlaneEstimationEnd) {
					mpSystem->cvUsePlaneEstimation.wait(lock);
				}
			}
			std::cout << "seg::e" << std::endl;
			////lock
			//시간체크
			std::chrono::high_resolution_clock::time_point p_end = std::chrono::high_resolution_clock::now();
			auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(p_end - s_end).count();
			float tttt2 = duration2 / 1000.0;
			//////////////////////

			//////////////////////////////////////////////
			//////디버깅 값 전달
			std::stringstream ssa;
			ssa << "Segmentation : " << mpTargetFrame->GetKeyFrameID() << " : " << tttt << "||" << tttt2 << std::endl;
			//ssa << "Segmentation : " << mpTargetFrame->GetKeyFrameID() << " : " << tttt << ", " << tttt5 << "||" << n1 << ", " << n2 << "::" << numOfLables;
			mpSystem->SetSegmentationString(ssa.str());
			//////디버깅 값 전달
			//////////////////////////////////////////////

			//////////////////////////////////////////////////
			////////디버깅을 위한 이미지 저장
			//std::stringstream ss;
			//ss << mStrDirPath.c_str() << "/segmentation.jpg";
			//cv::imwrite(ss.str(), segmented);
			//ss.str("");
			//ss << mStrDirPath.c_str() << "/segmentation_color_"<<mpTargetFrame->GetFrameID()<<".jpg";
			//cv::imwrite(ss.str(), colorimg);
			//cv::waitKey(1);
			////////디버깅을 위한 이미지 저장
			//////////////////////////////////////////////////
			std::cout << "segmentation::end" << std::endl;
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
void UVR_SLAM::SemanticSegmentator::ObjectLabeling(cv::Mat masked, int ratio) {

	//레이블링을 매칭에 이용하기 위한 디스크립터 설정.
	/*bool bMatch = false;
	if (mpTargetFrame->mPlaneDescriptor.rows == 0)
		bMatch = true;*/
	//초기화 하지 말고 중복 체크하면서 추가하도록 해야 함.
	bool bMatch = true;
	/*mpTargetFrame->mPlaneDescriptor = cv::Mat::zeros(0, mpTargetFrame->matDescriptor.cols, mpTargetFrame->matDescriptor.type());
	mpTargetFrame->mPlaneIdxs.clear();
	mpTargetFrame->mWallDescriptor = cv::Mat::zeros(0, mpTargetFrame->matDescriptor.cols, mpTargetFrame->matDescriptor.type());
	mpTargetFrame->mWallIdxs.clear();*/

	//오브젝트 및 맵포인트 정보 획득
	std::vector<UVR_SLAM::ObjectType> vObjTypes(mpTargetFrame->mvKeyPoints.size(), UVR_SLAM::OBJECT_NONE);
	auto mvpMPs = mpTargetFrame->GetMapPoints();
	//다른 프레임들과의 매칭 결과를 반영해야 함.

	for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {

		//get object type
		cv::Point2f pt = mpTargetFrame->mvKeyPoints[i].pt;
		pt.x /= ratio;
		pt.y /= ratio;
		int val = masked.at<uchar>(pt);
		UVR_SLAM::ObjectType type = vObjTypes[i];
		
		////object type 인식
		//bool bMP = false;
		//UVR_SLAM::MapPoint* pMP = mvpMPs[i];
		//if (pMP) {
		//	if (!pMP->isDeleted()) {
		//		if (mpTargetFrame->isInFrustum(pMP, 0.5)) {
		//			bMP = true;
		//		}
		//	}
		//}

		////find object type in map
		//auto val2 = static_cast<ObjectType>(val);
		//auto iter = mpTargetFrame->mvMapObjects[i].find(val2);
		//if (iter != mpTargetFrame->mvMapObjects[i].end()) {
		//	iter->second++;
		//}
		//else {
		//	mpTargetFrame->mvMapObjects[i].insert(std::make_pair(val2, 1));
		//}
		//
		//int maxVal = 0;
		//
		//for (std::multimap<ObjectType, int, std::greater<int>>::iterator iter = mpTargetFrame->mvMapObjects[i].begin(); iter != mpTargetFrame->mvMapObjects[i].end(); iter++) {
		//	//std::cout << "begin ::" << iter->first << ", " << iter->second << std::endl;
		//	if (maxVal < iter->second) {
		//		val2 = iter->first;
		//		maxVal = iter->second;
		//	}
		//}

		switch (val) {
		case 1://벽
		case 9://유리창
		case 11://캐비넷
		case 15://문
		case 23: //그림
		case 36://옷장
		//case 43://기둥
		case 44://갚난
		//case 94://막대기
		case 101://포스터
			type = ObjectType::OBJECT_WALL;
			////맵포인트에 추가
			//if (bMP){
			//	mpTargetFrame->mspWallMPs.insert(pMP);
			//	mpFrameWindow->mspWallMPs.insert(pMP);
			//}
			//레이블별 인덱스 설정
			//사용 안함.
			/*if (mpTargetFrame->mLabelStatus.at<uchar>(i) == 0) {
				mpTargetFrame->mLabelStatus.at<uchar>(i) = val;
				mpTargetFrame->mWallDescriptor.push_back(mpTargetFrame->matDescriptor.row(i));
				mpTargetFrame->mWallIdxs.push_back(i);
			}*/
			break;
		case 4:
		case 29: //rug
			type = ObjectType::OBJECT_FLOOR;
			////맵포인트에 추가
			//if (bMP){
			//	mpTargetFrame->mspFloorMPs.insert(pMP);
			//	mpFrameWindow->mspFloorMPs.insert(pMP);
			//}
			//레이블별 인덱스 설정
			//사용 안함.
			/*if (mpTargetFrame->mLabelStatus.at<uchar>(i) == 0) {
				mpTargetFrame->mLabelStatus.at<uchar>(i) = val;
				mpTargetFrame->mPlaneDescriptor.push_back(mpTargetFrame->matDescriptor.row(i));
				mpTargetFrame->mPlaneIdxs.push_back(i);
			}*/
			break;
		case 6:
			type = ObjectType::OBJECT_CEILING;
			////맵포인트에 추가
			//if (bMP){
			//	mpTargetFrame->mspCeilMPs.insert(pMP);
			//	mpFrameWindow->mspCeilMPs.insert(pMP);
			//}
			break;
		case 13:
			type = ObjectType::OBJECT_PERSON;
		default:
			break;
		}
		vObjTypes[i] = type;
		/*if (bMP) {
			pMP->SetObjectType(type);
		}*/
	}
	mpTargetFrame->SetObjectVector(vObjTypes);
	mpTargetFrame->SetBoolSegmented(true);
}
void UVR_SLAM::SemanticSegmentator::ObjectLabeling() {

	std::vector<UVR_SLAM::ObjectType> vObjTypes(mpTargetFrame->mvKeyPoints.size(), UVR_SLAM::OBJECT_NONE);

	for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {
		cv::Point2f pt = mpTargetFrame->mvKeyPoints[i].pt;
		for (int j = 0; j < mVecLabelMasks.size(); j++) {
			if (mVecLabelMasks[j].at<uchar>(pt) == 255) {
				UVR_SLAM::ObjectType type = static_cast<ObjectType>(j);
				//mpTargetFrame->SetObjectType(type, i);
				vObjTypes[i] = type;
				if (mpTargetFrame->mvbMPInliers[i]) {
					UVR_SLAM::MapPoint* pMP = mpTargetFrame->mvpMPs[i];
					if (pMP) {
						if (pMP->isDeleted())
							continue;
						cv::Point2f p2D;
						cv::Mat pcam;
						if (pMP->Projection(p2D, pcam, mpTargetFrame->GetRotation(), mpTargetFrame->GetTranslation(), mpTargetFrame->mK, mnWidth, mnHeight)) {
							pMP->SetObjectType(type);
						}//projection
					}//pMP
				}//inlier
			}
		}
	}
	mpTargetFrame->SetObjectVector(vObjTypes);

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




