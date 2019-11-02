#include <IndoorLayoutEstimator.h>
#include <System.h>
#include <SegmentationData.h>
#include <FrameWindow.h>

std::vector<cv::Vec3b> UVR_SLAM::ObjectColors::mvObjectLabelColors;

UVR_SLAM::IndoorLayoutEstimator::IndoorLayoutEstimator():mbDoingProcess(false){
	/*mVecLabelColors.push_back(COLOR_FLOOR);
	mVecLabelColors.push_back(COLOR_WALL);
	mVecLabelColors.push_back(COLOR_CEILING);
	mVecLabelColors.push_back(COLOR_NONE);*/
}
UVR_SLAM::IndoorLayoutEstimator::IndoorLayoutEstimator(int w, int h): mbDoingProcess(false), mnWidth(w), mnHeight(h){
	UVR_SLAM::ObjectColors::Init();
	/*mVecLabelColors.push_back(COLOR_FLOOR);
	mVecLabelColors.push_back(COLOR_WALL);
	mVecLabelColors.push_back(COLOR_CEILING);
	mVecLabelColors.push_back(COLOR_NONE);*/
}
UVR_SLAM::IndoorLayoutEstimator::~IndoorLayoutEstimator(){}


void UVR_SLAM::IndoorLayoutEstimator::Run() {

	JSONConverter::Init();

	while (1) {

		if (isDoingProcess()) {
			std::cout << "IndoorLayoutEstimator::RUN::Start" << std::endl;

			//semantic frame index
			int nLastFrameIndex = mpFrameWindow->GetLastSemanticFrameIndex();
			mpFrameWindow->SetLastSemanticFrameIndex();
			int nCurrFrameIndex = mpFrameWindow->GetLastSemanticFrameIndex();

			UVR_SLAM::Frame* prevSemanticFrame;
			if (nLastFrameIndex >= 0) {
				prevSemanticFrame = mpFrameWindow->GetFrame(nLastFrameIndex);
				auto pType = prevSemanticFrame->GetType();
				std::cout << "type test = " << "::" << (int)pType << std::endl << std::endl << std::endl;
			}
			std::cout << "SemanticFrame::Last=" << nLastFrameIndex << "|| Curr=" << nCurrFrameIndex << std::endl;

			cv::Mat colorimg, segmented;
			cvtColor(mpTargetFrame->GetOriginalImage(), colorimg, CV_RGBA2BGR);
			colorimg.convertTo(colorimg, CV_8UC3);
			JSONConverter::RequestPOST(colorimg, segmented, mpTargetFrame->GetFrameID());
			cv::resize(segmented, segmented, cv::Size(mnWidth, mnHeight));
			SetSegmentationMask(segmented);
			ObjectLabeling();

			cv::addWeighted(segmented, 0.5, colorimg, 0.5, 0.0, colorimg);

			cv::Mat maskSegmentation = mVecLabelMasks[0].clone() + mVecLabelMasks[1].clone() / 255 * 50 + mVecLabelMasks[2].clone() / 255 * 150;
			cv::imshow("Output::SegmentationMask", maskSegmentation);
			cv::imshow("Output::Segmentation", colorimg);
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
			}
			imshow("segmendted feature", test);

			std::stringstream ss;
			ss << "../../bin/segmentation/res_keypoints_" << mpTargetFrame->GetFrameID() << ".jpg";
			imwrite(ss.str(), test);*/
			cv::waitKey(300);

			SetBoolDoingProcess(false);
			std::cout << "IndoorLayoutEstimator::RUN::End" << std::endl;
		}
	}
}
void UVR_SLAM::IndoorLayoutEstimator::SetSystem(System* pSystem) {
	mpSystem = pSystem;
}
void UVR_SLAM::IndoorLayoutEstimator::SetFrameWindow(UVR_SLAM::FrameWindow* pWindow) {
	mpFrameWindow = pWindow;
}
void UVR_SLAM::IndoorLayoutEstimator::SetTargetFrame(Frame* pFrame) {
	mpTargetFrame = pFrame;
}
void UVR_SLAM::IndoorLayoutEstimator::SetBoolDoingProcess(bool b) {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	mbDoingProcess = b;
}
bool UVR_SLAM::IndoorLayoutEstimator::isDoingProcess() {
	std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
	return mbDoingProcess;
}

/////////////////////////////////////////////////////////////////////////////////////////

void UVR_SLAM::IndoorLayoutEstimator::ObjectLabeling() {
	for (int i = 0; i < mpTargetFrame->mvKeyPoints.size(); i++) {
		cv::Point2f pt = mpTargetFrame->mvKeyPoints[i].pt;
		for (int j = 0; j < mVecLabelMasks.size(); j++) {
			if (mVecLabelMasks[j].at<uchar>(pt) == 255) {
				UVR_SLAM::ObjectType type = static_cast<ObjectType>(j);
				mpTargetFrame->SetObjectType(type, i);
				if (mpTargetFrame->GetBoolInlier(i)) {
					UVR_SLAM::MapPoint* pMP = mpTargetFrame->GetMapPoint(i);
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

void UVR_SLAM::IndoorLayoutEstimator::SetSegmentationMask(cv::Mat segmented) {
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



