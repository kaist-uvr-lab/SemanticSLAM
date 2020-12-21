#include <LocalBinaryPatternProcessor.h>

namespace UVR_SLAM {
	LocalBinaryPatternProcessor::LocalBinaryPatternProcessor() :mnRadius(2), mnNeighbor(4), mnNumID(10), mnDiscrete(10){
		mpLBP = new lbplibrary::SCSLBP(mnRadius, mnNeighbor);
		mnNumPatterns = mpLBP->numPatterns;
	}
	LocalBinaryPatternProcessor::LocalBinaryPatternProcessor(int r, int n, int m, int d) : mnRadius(r), mnNeighbor(n), mnNumID(m), mnDiscrete(d){
		mpLBP = new lbplibrary::SCSLBP(mnRadius, mnNeighbor);
		mnNumPatterns = mpLBP->numPatterns;
	}
	LocalBinaryPatternProcessor:: ~LocalBinaryPatternProcessor(){}
	unsigned long long LocalBinaryPatternProcessor::GetID(cv::Mat hist) {
		unsigned long long id = 0;
		unsigned long long base = 1;
		for (size_t i = 0, iend = hist.cols; i < iend; i++) {
			//std::cout << "curr base id::" << base << std::endl;
			/*unsigned long long tempID = base-1;*/
			int val = hist.at<int>(i);
			if (val >= mnNumID)
				val = mnNumID - 1;
			/*if(val > 0)
			id += (tempID + val);*/
			id += (base*val);
			base *= mnNumID;
			//std::cout << "next base id::" <<base<< std::endl;
		}
		//std::cout << id << "::" << hist << std::endl;
		return id;
	}
	cv::Mat LocalBinaryPatternProcessor::ConvertDescriptor(cv::Mat src){
		cv::Mat res;
		mpLBP->run(src, res);
		return res;
	}
	cv::Mat LocalBinaryPatternProcessor::ConvertHistogram(cv::Mat src, cv::Rect rect) {
		cv::Mat hist = lbplibrary::histogram(src(rect), mnNumPatterns);
		hist /= mnDiscrete;
		return hist;
	}
}