#include <SemanticSegmentator.h>
#include <System.h>
#include <SegmentationData.h>
#include <FrameWindow.h>
#include <PlaneEstimator.h>

std::vector<cv::Vec3b> UVR_SLAM::ObjectColors::mvObjectLabelColors;

UVR_SLAM::SemanticSegmentator::SemanticSegmentator():mbDoingProcess(false){
	/*mVecLabelColors.push_back(COLOR_FLOOR);
	mVecLabelColors.push_back(COLOR_WALL);
	mVecLabelColors.push_back(COLOR_CEILING);
	mVecLabelColors.push_back(COLOR_NONE);*/
}
UVR_SLAM::SemanticSegmentator::SemanticSegmentator(std::string _ip, int _port, int w, int h): ip(_ip), port(_port),mbDoingProcess(false), mnWidth(w), mnHeight(h){
	UVR_SLAM::ObjectColors::Init();
	/*mVecLabelColors.push_back(COLOR_FLOOR);
	mVecLabelColors.push_back(COLOR_WALL);
	mVecLabelColors.push_back(COLOR_CEILING);
	mVecLabelColors.push_back(COLOR_NONE);*/
}
UVR_SLAM::SemanticSegmentator::SemanticSegmentator(const std::string & strSettingPath) :mbDoingProcess(false){
	UVR_SLAM::ObjectColors::Init();
	cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
	mbOn = static_cast<int>(fSettings["Segmentator.on"]) != 0;
	ip = fSettings["Segmentator.ip"];
	port = fSettings["Segmentator.port"];
	mnWidth = fSettings["Image.width"];
	mnHeight = fSettings["Image.height"];
	fSettings.release();
	/*mVecLabelColors.push_back(COLOR_FLOOR);
	mVecLabelColors.push_back(COLOR_WALL);
	mVecLabelColors.push_back(COLOR_CEILING);
	mVecLabelColors.push_back(COLOR_NONE);*/
}

UVR_SLAM::SemanticSegmentator::~SemanticSegmentator(){}


void UVR_SLAM::SemanticSegmentator::Run() {

	JSONConverter::Init();

	while (1) {

		if (isDoingProcess()) {
			std::cout << "SemanticSegmentator::RUN::Start" << std::endl;

			cv::Mat colorimg, segmented;
			cvtColor(mpTargetFrame->GetOriginalImage(), colorimg, CV_RGBA2BGR);
			colorimg.convertTo(colorimg, CV_8UC3);
			JSONConverter::RequestPOST(ip, port, colorimg, segmented, mpTargetFrame->GetFrameID());
			cv::resize(segmented, segmented, cv::Size(mnWidth, mnHeight));
			SetSegmentationMask(segmented);
			ObjectLabeling();

			//PlaneEstimator
			/*if (!mpPlaneEstimator->isDoingProcess()) {
				mpTargetFrame->TurnOnFlag(UVR_SLAM::FLAG_LAYOUT_FRAME);
				mpPlaneEstimator->SetBoolDoingProcess(true, 3);
				mpPlaneEstimator->SetTargetFrame(mpTargetFrame);
			}*/

			//여기는 시각화로 보낼 수 있으면 보내는게 좋을 듯.
			cv::addWeighted(segmented, 0.5, colorimg, 0.5, 0.0, colorimg);
			cv::Mat maskSegmentation = mVecLabelMasks[0].clone() + mVecLabelMasks[1].clone() / 255 * 50 + mVecLabelMasks[2].clone() / 255 * 150;

			cv::imshow("Output::SegmentationMask", maskSegmentation);
			cv::imshow("Output::Segmentation", colorimg);

			cv::imwrite("../../bin/segmentation/res/seg.jpg", colorimg);
			cv::imwrite("../../bin/segmentation/res/mask.jpg", maskSegmentation);

			/*cv::imshow("Output::LABEL::FLOOR"   , mVecLabelMasks[0]);
			cv::imshow("Output::LABEL::WALL"    , mVecLabelMasks[1]);
			cv::imshow("Output::LABEL::CEILING" , mVecLabelMasks[2]);*/
			//cv::imwrite("../../bin/segmentation/res.jpg", colorimg);
			//cv::imwrite("../../bin/segmentation/res_ceil.jpg", mVecLabelMasks[2]);
			//cv::imwrite("../../bin/segmentation/res_wall.jpg", mVecLabelMasks[1]);
			//cv::imwrite("../../bin/segmentation/res_floor.jpg", mVecLabelMasks[0]);


			//피쳐 레이블링 결과 확인
			/*cv::Mat test = mpTargetFrame->undistorted.clone();
			for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {
				UVR_SLAM::ObjectType type = mpTargetFrame->GetObjectType(i);
				if(type != OBJECT_NONE)
				circle(test, mpTargetFrame->mvKeyPoints[i].pt, 3, ObjectColors::mvObjectLabelColors[type], 1);
				UVR_SLAM::MapPoint* pMP = mpTargetFrame->GetMapPoint(i);
				if (pMP) {
				if (pMP->isDeleted())
				continue;
				UVR_SLAM::ObjectType type2 = pMP->GetObjectType();
				if (type2 != OBJECT_NONE)
				circle(test, mpTargetFrame->mvKeyPoints[i].pt, 1, ObjectColors::mvObjectLabelColors[type2], -1);
				}
			}*/
			//imshow("segmendted feature", test);

			/*std::stringstream ss;
			ss << "../../bin/segmentation/res_keypoints_" << mpTargetFrame->GetFrameID() << ".jpg";
			imwrite(ss.str(), test);*/
			//cv::imwrite("../../bin/segmentation/res/label_kp.jpg", test);
			cv::waitKey(10);

			SetBoolDoingProcess(false);
			std::cout << "SemanticSegmentator::RUN::End" << std::endl;
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
void UVR_SLAM::SemanticSegmentator::ObjectLabeling() {
	for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {
		cv::Point2f pt = mpTargetFrame->mvKeyPoints[i].pt;
		for (int j = 0; j < mVecLabelMasks.size(); j++) {
			if (mVecLabelMasks[j].at<uchar>(pt) == 255) {
				UVR_SLAM::ObjectType type = static_cast<ObjectType>(j);
				mpTargetFrame->SetObjectType(type, i);
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



