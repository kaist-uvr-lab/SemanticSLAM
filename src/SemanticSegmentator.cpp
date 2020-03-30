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
	mpTargetFrame->TurnOnFlag(UVR_SLAM::FLAG_KEY_FRAME);
	mpSystem->SetDirPath(mpTargetFrame->GetKeyFrameID());
	mpTargetFrame->SetBowVec(mpSystem->fvoc);
	if (mpMap->isFloorPlaneInitialized()) {
		mpTargetFrame->mpPlaneInformation = new UVR_SLAM::PlaneProcessInformation(mpTargetFrame, mpMap->mpFloorPlane);
	}
	mpSystem->SetSegFrameID(mpTargetFrame->GetKeyFrameID());
	mKFQueue.pop();
}

void UVR_SLAM::SemanticSegmentator::Run() {

	JSONConverter::Init();

	while (1) {
		std::string mStrDirPath;
		if (CheckNewKeyFrames()) {
			
			SetBoolDoingProcess(true);
			ProcessNewKeyFrame();
			std::cout << "segmentation::start::" << mpTargetFrame->GetFrameID() << ", " << mpTargetFrame->GetKeyFrameID() << std::endl;
			mStrDirPath = mpSystem->GetDirPath(mpTargetFrame->GetKeyFrameID());

			std::chrono::high_resolution_clock::time_point s_start = std::chrono::high_resolution_clock::now();
			cv::Mat colorimg, resized_color, segmented;
			cv::cvtColor(mpTargetFrame->GetOriginalImage(), colorimg, CV_RGBA2BGR);
			colorimg.convertTo(colorimg, CV_8UC3);
			cv::resize(colorimg, resized_color, cv::Size(mnWidth/2, mnHeight/2));
			//cv::resize(colorimg, resized_color, cv::Size(160, 90));

			///////////////////////////////////////////
			///////////���׸����̼��� �����ϴ� ���� �˸�.
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
			//�������� ���ϸ� Į���̹�����
			int status = 0;
			JSONConverter::RequestPOST(ip, port, resized_color, segmented, mpTargetFrame->GetFrameID(), status);
			//cv::resize(segmented, segmented, colorimg.size());
			
			int nRatio = colorimg.rows / segmented.rows;
			//ratio ������ �ƴ� �ٸ���
			ImageLabeling(segmented, mpTargetFrame->matLabeled);
			ObjectLabeling(mpTargetFrame->matLabeled, nRatio);
			
			mpMap->SetCurrFrame(mpTargetFrame);
			mpTargetFrame->matSegmented = segmented.clone();
			////////////////////////////////////////////////////////
			///////////���׸����̼��� ���� ���� �˸�.
			//unlock & notify
			mpSystem->mbSegmentationEnd = true;
			lock.unlock();
			mpSystem->cvUseSegmentation.notify_all();
			//unlock & notify
			////////////////////////////////////////////////////////

			//�ð�üũ
			std::chrono::high_resolution_clock::time_point s_end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_start).count();
			float tttt = duration / 1000.0;

			//////////////////////
			////lock
			//std::cout << "seg::s" << std::endl;
			////plane estimation���� ������Ʈ�� ������ ������ ��.
			{
				std::unique_lock<std::mutex> lock(mpSystem->mMutexUsePlaneEstimation);
				while (!mpSystem->mbPlaneEstimationEnd) {
					mpSystem->cvUsePlaneEstimation.wait(lock);
				}
			}
			//std::cout << "seg::e" << std::endl;
			////lock
			//�ð�üũ
			std::chrono::high_resolution_clock::time_point p_end = std::chrono::high_resolution_clock::now();
			auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(p_end - s_end).count();
			float tttt2 = duration2 / 1000.0;
			//////////////////////

			//////////////////////////////////////////////
			//////����� �� ����
			std::stringstream ssa;
			/*if (mpPrevTargetFrame)
				ssa << "Segmentation : " << mpTargetFrame->GetKeyFrameID() << "=" << mpPrevTargetFrame->GetFrameID() << ", " << mpTargetFrame->GetFrameID() << " : " << tttt << "||" << tttt2 << std::endl;
			else
				ssa << "Seg";*/
			ssa << "Segmentation : " << mpTargetFrame->GetKeyFrameID() <<" : " << tttt << "||" << tttt2 << std::endl;
			//ssa << "Segmentation : " << mpTargetFrame->GetKeyFrameID() << " : " << tttt << ", " << tttt5 << "||" << n1 << ", " << n2 << "::" << numOfLables;
			mpSystem->SetSegmentationString(ssa.str());
			//////����� �� ����
			//////////////////////////////////////////////

			//////////////////////////////////////////////////
			////////������� ���� �̹��� ����
			//std::stringstream ss;
			//ss << mStrDirPath.c_str() << "/segmentation.jpg";
			//cv::imwrite(ss.str(), segmented);
			//ss.str("");
			//ss << mStrDirPath.c_str() << "/segmentation_color_"<<mpTargetFrame->GetFrameID()<<".jpg";
			//cv::imwrite(ss.str(), colorimg);
			//cv::waitKey(1);
			////////������� ���� �̹��� ����
			//////////////////////////////////////////////////
			std::cout << "segmentation::end::" << mpTargetFrame->GetKeyFrameID() << ", " << mpTargetFrame->GetFrameID() << std::endl;
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
//��� Ư¡���� Ʈ��ŷ�ǰ� �ִ� ������Ʈ�� ���̺���.
void UVR_SLAM::SemanticSegmentator::ImageLabeling(cv::Mat segmented, cv::Mat& labeld) {

	labeld = cv::Mat::zeros(segmented.size(), CV_8UC1);

	for (int i = 0; i < segmented.rows; i++) {
		for (int j = 0; j < segmented.cols; j++) {
			int val = segmented.at<uchar>(i, j);
			switch (val) {
			case 1://��
			case 9://����â
			case 11://ĳ���
			case 15://��
			case 23: //�׸�
			case 36://����
					//case 43://���
			case 44://����
					//case 94://�����
			case 101://������
				labeld.at<uchar>(i, j) = 255;
				break;
			case 4:
			case 29: //rug
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

void UVR_SLAM::SemanticSegmentator::ObjectLabeling(cv::Mat masked, int ratio) {

	//���̺��� ��Ī�� �̿��ϱ� ���� ��ũ���� ����.
	/*bool bMatch = false;
	if (mpTargetFrame->mPlaneDescriptor.rows == 0)
		bMatch = true;*/
	//�ʱ�ȭ ���� ���� �ߺ� üũ�ϸ鼭 �߰��ϵ��� �ؾ� ��.
	bool bMatch = true;
	/*mpTargetFrame->mPlaneDescriptor = cv::Mat::zeros(0, mpTargetFrame->matDescriptor.cols, mpTargetFrame->matDescriptor.type());
	mpTargetFrame->mPlaneIdxs.clear();
	mpTargetFrame->mWallDescriptor = cv::Mat::zeros(0, mpTargetFrame->matDescriptor.cols, mpTargetFrame->matDescriptor.type());
	mpTargetFrame->mWallIdxs.clear();*/

	//������Ʈ �� ������Ʈ ���� ȹ��
	std::vector<UVR_SLAM::ObjectType> vObjTypes(mpTargetFrame->mvKeyPoints.size(), UVR_SLAM::OBJECT_NONE);
	auto mvpMPs = mpTargetFrame->GetMapPoints();
	//�ٸ� �����ӵ���� ��Ī ����� �ݿ��ؾ� ��.

	for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {

		//get object type
		cv::Point2f pt = mpTargetFrame->mvKeyPoints[i].pt;
		pt.x /= ratio;
		pt.y /= ratio;
		int val = masked.at<uchar>(pt);
		UVR_SLAM::ObjectType type = vObjTypes[i];
		
		switch (val) {
		case 255:
			type = ObjectType::OBJECT_WALL;
			break;
		case 150:
			type = ObjectType::OBJECT_FLOOR;
			break;
		case 100:
			type = ObjectType::OBJECT_CEILING;
			break;
		case 20:
			type = ObjectType::OBJECT_PERSON;
			break;
		case 50:
			break;
		}
		vObjTypes[i] = type;
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

//�����ϰ� �ִ� ������Ʈ Ŭ���� ���� ����ũ �̹����� �����ϰ� �ȼ� ������ ����ŷ��.
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




