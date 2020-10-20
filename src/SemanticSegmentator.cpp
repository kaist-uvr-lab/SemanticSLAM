#include <SemanticSegmentator.h>
#include <System.h>
#include <SegmentationData.h>
#include <FrameWindow.h>
#include <PlaneEstimator.h>
#include <Plane.h>
#include <LocalMapper.h>
#include <Visualizer.h>
#include <CandidatePoint.h>
#include <Map.h>

std::vector<cv::Vec3b> UVR_SLAM::ObjectColors::mvObjectLabelColors;

UVR_SLAM::SemanticSegmentator::SemanticSegmentator():mbDoingProcess(false){
	/*mVecLabelColors.push_back(COLOR_FLOOR);
	mVecLabelColors.push_back(COLOR_WALL);
	mVecLabelColors.push_back(COLOR_CEILING);
	mVecLabelColors.push_back(COLOR_NONE);*/
}
UVR_SLAM::SemanticSegmentator::SemanticSegmentator(std::string _ip, int _port, int w, int h): ip(_ip), port(_port),mbDoingProcess(false), mnWidth(w), mnHeight(h), mpPrevFrame(nullptr){
	UVR_SLAM::ObjectColors::Init();
	/*mVecLabelColors.push_back(COLOR_FLOOR);
	mVecLabelColors.push_back(COLOR_WALL);
	mVecLabelColors.push_back(COLOR_CEILING);
	mVecLabelColors.push_back(COLOR_NONE);*/
}
UVR_SLAM::SemanticSegmentator::SemanticSegmentator(System* pSys, const std::string & strSettingPath) :mbDoingProcess(false), mpPrevFrame(nullptr) {
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
	
	mpSystem = pSys;
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
	mpPrevFrame = mpTargetFrame;
	mpTargetFrame = mKFQueue.front();
	mKFQueue.pop();
}

void UVR_SLAM::SemanticSegmentator::Init() {
	mpMap = mpSystem->mpMap;
	mpLocalMapper = mpSystem->mpLocalMapper;
	mpPlaneEstimator = mpSystem->mpPlaneEstimator;
	mpVisualizer = mpSystem->mpVisualizer;
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
			/*cv::cvtColor(mpTargetFrame->GetOriginalImage(), colorimg, CV_RGBA2BGR);
			colorimg.convertTo(colorimg, CV_8UC3);*/
			cv::resize(mpTargetFrame->GetOriginalImage().clone(), resized_color, cv::Size(mnWidth/2, mnHeight/2));
			//cv::resize(colorimg, resized_color, cv::Size(160, 90));
			
			//request post
			//리사이즈 안하면 칼라이미지로
			int status = 0;
			JSONConverter::RequestPOST(ip, port, resized_color, segmented, mpTargetFrame->mnFrameID, status);

			int nRatio = colorimg.rows / segmented.rows;
			//ratio 버전이 아닌 다르게
			/////////미리 지정된 오브젝트만 레이블링 하는 코드
			//ImageLabeling(segmented, mpTargetFrame->matLabeled);
			mpTargetFrame->matLabeled = segmented.clone();
			mpTargetFrame->matSegmented = segmented.clone();
			
			////세그멘테이션 칼라 전달
			int nMaxLabel;
			int nMaxLabelCount = 0;
			cv::Mat seg_color = cv::Mat::zeros(segmented.size(), CV_8UC3);
			for (int y = 0; y < seg_color.rows; y++) {
				for (int x = 0; x < seg_color.cols; x++) {
					int label = segmented.at<uchar>(y, x);
					seg_color.at<cv::Vec3b>(y, x) = UVR_SLAM::ObjectColors::mvObjectLabelColors[label];
					
					if (!mpTargetFrame->mpMatchInfo->mmLabelMasks.count(label)) {
						mpTargetFrame->mpMatchInfo->mmLabelMasks[label] = cv::Mat::zeros(segmented.size(), CV_8UC1);
					}
					else {
						mpTargetFrame->mpMatchInfo->mmLabelAcc[label]++;
						if (mpTargetFrame->mpMatchInfo->mmLabelAcc[label] > nMaxLabelCount)
						{
							nMaxLabel = label;
							nMaxLabelCount = mpTargetFrame->mpMatchInfo->mmLabelAcc[label];
						}
						mpTargetFrame->mpMatchInfo->mmLabelMasks[label].at<uchar>(y, x) = 255;
					}
				}
			}
			
			///////////CCL TEST
			cv::addWeighted(seg_color, 0.5, resized_color, 0.5, 0.0, resized_color);
			for (auto iter = mpTargetFrame->mpMatchInfo->mmLabelMasks.begin(), eiter = mpTargetFrame->mpMatchInfo->mmLabelMasks.end(); iter != eiter; iter++) {
				Mat img_labels, stats, centroids;
				int label = iter->first;
				cv::Mat mask = iter->second;
				
				int numOfLables = connectedComponentsWithStats(mask, img_labels, stats, centroids, 8, CV_32S);
				if (numOfLables > 0) {
					int maxArea = 0;
					int maxIdx = 0;
					////라벨링 된 이미지에 각각 직사각형으로 둘러싸기 
					//for (int j = 0; j < numOfLables; j++) {
					//	int area = stats.at<int>(j, CC_STAT_AREA);
					//	
					//	//std::cout << area << std::endl;
					//	if (area > maxArea) {
					//		maxArea = area;
					//		maxIdx = j;
					//	}
					//}
					for (int j = 0; j < numOfLables; j++) {
						/*if (j == maxIdx)
						continue;*/
						int area = stats.at<int>(j, CC_STAT_AREA);
						if (area < 300)
							continue;
						int left = stats.at<int>(j, CC_STAT_LEFT);
						int top = stats.at<int>(j, CC_STAT_TOP);
						int width = stats.at<int>(j, CC_STAT_WIDTH);
						int height = stats.at<int>(j, CC_STAT_HEIGHT);
						cv::Rect rect(Point(left, top), Point(left + width, top + height));
						/*mpTargetFrame->mpMatchInfo->mmLabelRects.insert(std::make_pair(label, rect));
						mpTargetFrame->mpMatchInfo->mmLabelCPs.insert(std::make_pair(label, std::list<CandidatePoint*>()));*/
						auto info = std::make_pair(rect, std::list<CandidatePoint*>());
						mpTargetFrame->mpMatchInfo->mmLabelRectCPs.insert(std::make_pair(label, info));
						rectangle(resized_color, rect, Scalar(255, 255, 255), 2);
						//rectangle(ccl_res, Point(left, top), Point(left + width, top + height), Scalar(255, 255, 255), 2);
					}
				}
			}

			mpTargetFrame->SetBoolSegmented(true);
			mpTargetFrame->mpMatchInfo->SetLabel();

			for (auto iter = mpTargetFrame->mpMatchInfo->mmLabelRectCPs.begin(), eiter = mpTargetFrame->mpMatchInfo->mmLabelRectCPs.end(); iter != eiter; iter++) {
				auto label = iter->first;
				auto rect = iter->second.first;
				auto lpCPs = iter->second.second;
				for (auto liter = lpCPs.begin(), leiter = lpCPs.end(); liter != leiter; liter++) {
					auto pCPi = *liter;
					int idx = pCPi->GetPointIndexInFrame(mpTargetFrame->mpMatchInfo);
					if (idx > -1) {
						auto pt = mpTargetFrame->mpMatchInfo->mvMatchingPts[idx] / 2;
						cv::circle(resized_color, pt, 2, UVR_SLAM::ObjectColors::mvObjectLabelColors[label], -1);
					}
				}
				//std::cout << "Object Info = " << label << ", " << lpCPs.size() << std::endl;
			}

			mpVisualizer->SetOutputImage(resized_color, 4);
			////시간체크
			std::chrono::high_resolution_clock::time_point s_end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_start).count();
			float tttt = duration / 1000.0;
			////시간체크

			//if (mmLabelMasks.size() > 0) {
			//	Mat img_labels, stats, centroids;
			//	cv::Mat maskA = mmLabelMasks[nMaxLabel];
			//	cv::Mat ccl_res = seg_color.clone();
			//	int numOfLables = connectedComponentsWithStats(maskA, img_labels, stats, centroids, 8, CV_32S);
			//	if (numOfLables > 0) {
			//		int maxArea = 0;
			//		int maxIdx = 0;
			//		//라벨링 된 이미지에 각각 직사각형으로 둘러싸기 
			//		for (int j = 0; j < numOfLables; j++) {
			//			int area = stats.at<int>(j, CC_STAT_AREA);
			//			if (area > maxArea) {
			//				maxArea = area;
			//				maxIdx = j;
			//			}
			//		}
			//		for (int j = 0; j < numOfLables; j++) {
			//			/*if (j == maxIdx)
			//			continue;*/
			//			int area = stats.at<int>(j, CC_STAT_AREA);
			//			int left = stats.at<int>(j, CC_STAT_LEFT);
			//			int top = stats.at<int>(j, CC_STAT_TOP);
			//			int width = stats.at<int>(j, CC_STAT_WIDTH);
			//			int height = stats.at<int>(j, CC_STAT_HEIGHT);

			//			rectangle(ccl_res, Point(left, top), Point(left + width, top + height), Scalar(255, 255, 255), 2);
			//		}
			//		imshow("seg mask", maskA);
			//		imshow("aaabbb", ccl_res); cv::waitKey(1);
			//	}
			//}
			///////////CCL TEST
		
			//////////////////////////////////////////////
			//////디버깅 값 전달
			std::stringstream ssa;
			ssa << "Segmentation : " << mpTargetFrame->mnKeyFrameID <<" : " << tttt << std::endl;
			mpSystem->SetSegmentationString(ssa.str());
			//////디버깅 값 전달
			//////////////////////////////////////////////

			
			SetBoolDoingProcess(false);
		}
	}
}
void UVR_SLAM::SemanticSegmentator::SetInitialSegFrame(UVR_SLAM::Frame* pKF1) {
	mpPrevFrame = pKF1;
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
			case 101://포스터
				labeld.at<uchar>(i, j) = 255;
				break;
			case 4:  //floor
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

