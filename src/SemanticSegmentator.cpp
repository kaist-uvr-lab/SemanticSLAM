#include <SemanticSegmentator.h>
#include <System.h>
#include <SegmentationData.h>
#include <FrameWindow.h>
#include <PlaneEstimator.h>
#include <Plane.h>
#include <LocalMapper.h>
#include <Visualizer.h>
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
UVR_SLAM::SemanticSegmentator::SemanticSegmentator(System* pSys, const std::string & strSettingPath) :mbDoingProcess(false), mpPrevTargetFrame(nullptr) {
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
	mpPrevTargetFrame = mpTargetFrame;
	mpTargetFrame = mKFQueue.front();
	mpTargetFrame->TurnOnFlag(UVR_SLAM::FLAG_SEGMENTED_FRAME);
	
	//mpSystem->SetDirPath(mpTargetFrame->GetKeyFrameID()); //
	
	
	
	////�� ���뵵 pe���� �����ص� �� �� ����.
	/*if (mpMap->isFloorPlaneInitialized()) {
		mpTargetFrame->mpPlaneInformation = new UVR_SLAM::PlaneProcessInformation(mpTargetFrame, mpMap->mpFloorPlane);
		mpTargetFrame->mpPlaneInformation->Calculate();
	}*/
	//mpSystem->SetSegFrameID(mpTargetFrame->GetKeyFrameID());	//setsegframeid, getsegframeid�� �̿� ���ϴ� ��
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
			cv::cvtColor(mpTargetFrame->GetOriginalImage(), colorimg, CV_RGBA2BGR);
			colorimg.convertTo(colorimg, CV_8UC3);
			cv::resize(colorimg, resized_color, cv::Size(mnWidth/2, mnHeight/2));
			//cv::resize(colorimg, resized_color, cv::Size(160, 90));
			
			//request post
			//�������� ���ϸ� Į���̹�����
			int status = 0;
			JSONConverter::RequestPOST(ip, port, resized_color, segmented, mpTargetFrame->GetFrameID(), status);

			int nRatio = colorimg.rows / segmented.rows;
			//ratio ������ �ƴ� �ٸ���
			/////////�̸� ������ ������Ʈ�� ���̺� �ϴ� �ڵ�
			//ImageLabeling(segmented, mpTargetFrame->matLabeled);
			mpTargetFrame->matLabeled = segmented.clone();
			mpTargetFrame->matSegmented = segmented.clone();
			mpTargetFrame->SetBoolSegmented(true);
			mpTargetFrame->mpMatchInfo->SetLabel();

			////���׸����̼� Į�� ����
			mmLabelAcc.clear();
			mmLabelMasks.clear();
			int nMaxLabel;
			int nMaxLabelCount = 0;
			cv::Mat seg_color = cv::Mat::zeros(segmented.size(), CV_8UC3);
			for (int y = 0; y < seg_color.rows; y++) {
				for (int x = 0; x < seg_color.cols; x++) {
					int label = segmented.at<uchar>(y, x);
					seg_color.at<cv::Vec3b>(y, x) = UVR_SLAM::ObjectColors::mvObjectLabelColors[label];

					if (!mmLabelMasks.count(label)) {
						mmLabelMasks[label] = cv::Mat::zeros(segmented.size(), CV_8UC1);
					}
					else {
						mmLabelAcc[label]++;
						if (mmLabelAcc[label] > nMaxLabelCount)
						{
							nMaxLabel = label;
							nMaxLabelCount = mmLabelAcc[label];
						}
						mmLabelMasks[label].at<uchar>(y, x) = 255;
					}

				}
			}
			
			///////////CCL TEST
			cv::Mat ccl_res = seg_color.clone();
			for (auto iter = mmLabelMasks.begin(), eiter = mmLabelMasks.end(); iter != eiter; iter++) {
				Mat img_labels, stats, centroids;
				int label = iter->first;
				cv::Mat mask = iter->second;
				
				int numOfLables = connectedComponentsWithStats(mask, img_labels, stats, centroids, 8, CV_32S);
				if (numOfLables > 0) {
					int maxArea = 0;
					int maxIdx = 0;
					////�󺧸� �� �̹����� ���� ���簢������ �ѷ��α� 
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
						if (area < 200)
							continue;
						int left = stats.at<int>(j, CC_STAT_LEFT);
						int top = stats.at<int>(j, CC_STAT_TOP);
						int width = stats.at<int>(j, CC_STAT_WIDTH);
						int height = stats.at<int>(j, CC_STAT_HEIGHT);
						cv::Rect rect(Point(left, top), Point(left + width, top + height));
						mmLabelRects.insert(std::make_pair(label, rect));
						rectangle(ccl_res, rect, Scalar(255, 255, 255), 2);
						//rectangle(ccl_res, Point(left, top), Point(left + width, top + height), Scalar(255, 255, 255), 2);
					}
					/*imshow("seg mask", maskA);
					*/
				}
			}
			//imshow("aaabbb", ccl_res); cv::waitKey(1);

			mpVisualizer->SetOutputImage(ccl_res, 4);
			////�ð�üũ
			std::chrono::high_resolution_clock::time_point s_end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_start).count();
			float tttt = duration / 1000.0;
			////�ð�üũ

			//if (mmLabelMasks.size() > 0) {
			//	Mat img_labels, stats, centroids;
			//	cv::Mat maskA = mmLabelMasks[nMaxLabel];
			//	cv::Mat ccl_res = seg_color.clone();
			//	int numOfLables = connectedComponentsWithStats(maskA, img_labels, stats, centroids, 8, CV_32S);
			//	if (numOfLables > 0) {
			//		int maxArea = 0;
			//		int maxIdx = 0;
			//		//�󺧸� �� �̹����� ���� ���簢������ �ѷ��α� 
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
			//////����� �� ����
			std::stringstream ssa;
			ssa << "Segmentation : " << mpTargetFrame->GetKeyFrameID() <<" : " << tttt << std::endl;
			mpSystem->SetSegmentationString(ssa.str());
			//////����� �� ����
			//////////////////////////////////////////////

			
			SetBoolDoingProcess(false);
		}
	}
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
			
			case 36://����
					//case 43://���
			case 44://����
					//case 94://�����
			case 23: //�׸�
			case 101://������
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

