#include <Database.h>
#include <SegmentationData.h>
namespace UVR_SLAM {
	Database::Database(){
		mnObjColorSize = ObjectColors::mvObjectLabelColors.size();
	}
	Database::~Database(){}

	void Database::AddData(unsigned long long id, int data) {
		std::unique_lock<std::mutex> lockMP(mMutexLBP);
		if (mmObjLBP.count(id) == 0) {
			mmObjLBP[id] = cv::Mat::zeros(1, mnObjColorSize, CV_32SC1);
		}
		mmObjLBP[id].at<int>(data)++;
	}
	int  Database::GetData(unsigned long long id){
		cv::Mat objData;
		{
			std::unique_lock<std::mutex> lockMP(mMutexLBP);
			if (mmObjLBP.count(id) == 0)
				return -1;
			objData = mmObjLBP[id].clone();
		}
		double max_val = 0;
		cv::Point minLoc2, maxLoc2;
		minMaxLoc(objData, NULL, &max_val, NULL, &maxLoc2);
		return maxLoc2.x;
	}
}