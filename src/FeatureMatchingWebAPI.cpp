#include <FeatureMatchingWebAPI.h>

int feaeture_count = 0;
std::stringstream ssWebData, ssDepthEstimateData;
std::vector<cv::Point2f> resPts;
std::vector<int> resMatches;
cv::Mat resDepth;


int depth_count = 0;

std::string ConvertNumberToString(int id) {
	std::stringstream ss;
	ss << "{\"id\":" << (int)id <<"}";
	return ss.str();
}
std::string ConvertImageToString(cv::Mat img, int id) {
	int r = img.rows;
	int c = img.cols;
	int total = r*c;

	std::stringstream ss;
	ss << "{\"img\":";
	ss << "\"";
	int params[3] = { 0 };
	params[0] = CV_IMWRITE_JPEG_QUALITY;
	params[1] = 100;
	std::vector<uchar> buf;
	bool code = cv::imencode(".jpg", img, buf, std::vector<int>(params, params + 2));
	uchar* result = reinterpret_cast<uchar*> (&buf[0]);

	std::string strimg = Base64Encoder::base64_encode(result, buf.size());
	ss << strimg;
	ss << "\"";
	ss << ",\"w\":" << (int)c << ",\"h\":" << (int)r << ",\"c\":" << (int)img.channels() << ",\"id\":" << (int)id << "}";
	return ss.str();
}

std::string ConvertNumberToString(int id1, int id2) {
	std::stringstream ss;
	ss << "{\"id1\":" << (int)id1 << ",\"id2\":" << (int)id2  << "}";
	return ss.str();
}


std::vector<cv::Point2f> ConvertStringToPoints(const char* data, int N) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "return JSON parsing error" << std::endl;
	}
	std::vector<cv::Point2f> res;
	//cv::Mat res = cv::Mat::zeros(0, 0, CV_8UC1);
	if (document.HasMember("res") && document["res"].IsArray()) {

		const rapidjson::Value& a = document["res"];
		int n = document["n"].GetInt();
		for (size_t i = 0; i < n; i++) {
			cv::Point2f pt(a[i][0].GetFloat(), a[i][1].GetFloat());
			res.push_back(pt);
		}
		//std::cout << "detect::" << n << std::endl;
		//int w = document["w"].GetInt();
		////int c = a[0][0].Size();

		//res = cv::Mat::zeros(h, w, CV_8UC1);
		//for (int y = 0; y < h; y++) {
		//	for (int x = 0; x < w; x++) {
		//		res.at<uchar>(y, x) = a[y][x].GetInt();
		//	}
		//}
	}
	/*else {
		std::cout << "???????????" << std::endl;
	}*/

	return res;
}

std::vector<int> ConvertStringToLabels(const char* data, int N) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "return JSON parsing error" << std::endl;
	}
	std::vector<int> res;
	//cv::Mat res = cv::Mat::zeros(0, 0, CV_8UC1);
	if (document.HasMember("res") && document["res"].IsArray()) {

		const rapidjson::Value& a = document["res"];
		int n = document["n"].GetInt();
		for (size_t i = 0; i < n; i++) {
			res.push_back(a[i].GetInt());
		}
	}
	return res;
}

std::vector<std::string> split(std::string input, char delimiter) {
	std::vector<std::string> answer;
	std::stringstream ss(input);
	std::string temp;

	while (getline(ss, temp, delimiter)) {
		answer.push_back(temp);
	}

	return answer;
}

cv::Mat ConvertStringToDepthImage(const char* data, int N) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "Depth Estimate JSON parsing error" << std::endl;
	}
	cv::Mat res;
	if (document["b"].GetBool()) {
		if (document.HasMember("res") && document["res"].IsArray()) {

			const rapidjson::Value& a = document["res"];
			int w = document["w"].GetInt();
			int h = document["h"].GetInt();

			res = cv::Mat::zeros(h, w, CV_32FC1);
			for (int y = 0; y < h; y++) {
				for (int x = 0; x < w; x++) {
					res.at<float>(y, x) = 1.0/a[y][x].GetFloat();
				}
			}
			//auto temp = document["res"].GetString();
			/*auto temp_strs = split(std::string(temp), ' ');
			res = cv::Mat::zeros(h, w, CV_32FC1);

			if (temp_strs.size() != (w*h))
				std::cout << "????????????????????????????????????????????????????????" << std::endl;

			for (size_t i = 0, iend = temp_strs.size(); i < iend; i++) {
				int x = i % w;
				int y = i / w;
				float val = std::stof(temp_strs[i]);
				res.at<float>(y, x) = 1.0/val;
			}*/
		
		}
		else {
			std::cout << "depth estimate!!" << std::endl;
		}
	}
	else {
		res = cv::Mat::zeros(0, 0, CV_8UC1);
	}
	return res;
}

void FeatueOnBegin(const happyhttp::Response* r, void* userdata)
{
	ssWebData.str("");
	feaeture_count = 0;
}

void FeatueOnData(const happyhttp::Response* r, void* userdata, const unsigned char* data, int n)
{
	ssWebData.write((const char*)data, n);
	feaeture_count += n;
}

void FeatureDetectOnComplete(const happyhttp::Response* r, void* userdata)
{
	resPts = ConvertStringToPoints(ssWebData.str().c_str(), feaeture_count);
}
void FeatureMatchOnComplete(const happyhttp::Response* r, void* userdata)
{
	resMatches = ConvertStringToLabels(ssWebData.str().c_str(), feaeture_count);
}

void FeatureResetOnComplete(const happyhttp::Response* r, void* userdata)
{
}

void DepthOnBegin(const happyhttp::Response* r, void* userdata)
{
	ssDepthEstimateData.str("");
	depth_count = 0;
}

void DepthOnData(const happyhttp::Response* r, void* userdata, const unsigned char* data, int n)
{
	ssDepthEstimateData.write((const char*)data, n);
	depth_count += n;
}
void DepthEstimateOnComplete(const happyhttp::Response* r, void* userdata)
{
	resDepth = ConvertStringToDepthImage(ssDepthEstimateData.str().c_str(), depth_count);
}

bool FeatureMatchingWebAPI::Reset(std::string ip, int port) {
	std::string strJSON = "";

	happyhttp::Connection* mpConnection = new happyhttp::Connection(ip.c_str(), port);

	mpConnection->setcallbacks(FeatueOnBegin, FeatueOnData, FeatureResetOnComplete, 0);
	mpConnection->request("POST",
		"/api/reset",
		Base64Encoder::headers,
		(const unsigned char*)strJSON.c_str(),
		strlen(strJSON.c_str())
	);

	while (mpConnection->outstanding())
		mpConnection->pump();

	return false;
}

bool FeatureMatchingWebAPI::SendImage(std::string ip, int port, cv::Mat src, int id) {
	std::string strJSON = ConvertImageToString(src, id);

	happyhttp::Connection* mpConnection = new happyhttp::Connection(ip.c_str(), port);

	mpConnection->setcallbacks(FeatueOnBegin, FeatueOnData, FeatureResetOnComplete, 0);
	mpConnection->request("POST",
		"/api/receiveimage",
		Base64Encoder::headers,
		(const unsigned char*)strJSON.c_str(),
		strlen(strJSON.c_str())
	);

	while (mpConnection->outstanding())
		mpConnection->pump();

	return true;
}

bool FeatureMatchingWebAPI::RequestDetect(std::string ip, int port, int id, std::vector<cv::Point2f>& vPTs) {
	std::string strJSON = ConvertNumberToString(id);

	happyhttp::Connection* mpConnection = new happyhttp::Connection(ip.c_str(), port);

	mpConnection->setcallbacks(FeatueOnBegin, FeatueOnData, FeatureDetectOnComplete, 0);
	mpConnection->request("POST",
		"/api/detect",
		Base64Encoder::headers,
		(const unsigned char*)strJSON.c_str(),
		strlen(strJSON.c_str())
	);

	while (mpConnection->outstanding())
		mpConnection->pump();

	vPTs = resPts;
	//dst = res.clone();

	return false;
}

bool FeatureMatchingWebAPI::RequestMatch(std::string ip, int port, int id1, int id2, std::vector<int>& vMatches) {
	std::string strJSON = ConvertNumberToString(id1, id2);

	happyhttp::Connection* mpConnection = new happyhttp::Connection(ip.c_str(), port);

	mpConnection->setcallbacks(FeatueOnBegin, FeatueOnData, FeatureMatchOnComplete, 0);
	mpConnection->request("POST",
		"/api/match",
		Base64Encoder::headers,
		(const unsigned char*)strJSON.c_str(),
		strlen(strJSON.c_str())
	);

	while (mpConnection->outstanding())
		mpConnection->pump();

	vMatches = resMatches;
	//dst = res.clone();

	return false;
}

bool FeatureMatchingWebAPI::RequestDepthEstimate(std::string ip, int port, int id, cv::Mat& dst) {
	std::string strJSON = ConvertNumberToString(id);

	happyhttp::Connection* mpConnection = new happyhttp::Connection(ip.c_str(), port);

	mpConnection->setcallbacks(DepthOnBegin, DepthOnData, DepthEstimateOnComplete, 0);
	mpConnection->request("POST",
		"/api/depthestimate",
		Base64Encoder::headers,
		(const unsigned char*)strJSON.c_str(),
		strlen(strJSON.c_str())
	);

	while (mpConnection->outstanding())
		mpConnection->pump();

	dst = resDepth.clone();
	if (dst.empty())
		return false;
	else
		return true;
}

bool FeatureMatchingWebAPI::RequestDetect(std::string ip, int port, cv::Mat src, int id, std::vector<cv::Point2f>& vPTs){
	std::string strJSON = ConvertImageToString(src, id);

	happyhttp::Connection* mpConnection = new happyhttp::Connection(ip.c_str(), port);
	
	mpConnection->setcallbacks(FeatueOnBegin, FeatueOnData, FeatureDetectOnComplete, 0);
	mpConnection->request("POST",
		"/api/detect",
		Base64Encoder::headers,
		(const unsigned char*)strJSON.c_str(),
		strlen(strJSON.c_str())
	);

	while (mpConnection->outstanding())
		mpConnection->pump();

	vPTs = resPts;
	//dst = res.clone();

	return false;
}

bool FeatureMatchingWebAPI::RequestDepthEstimate(std::string ip, int port, cv::Mat src, int id, cv::Mat& dst) {
	cv::resize(src, src, src.size() / 2);
	std::string strJSON = ConvertImageToString(src, id);

	happyhttp::Connection* mpConnection = new happyhttp::Connection(ip.c_str(), port);

	mpConnection->setcallbacks(DepthOnBegin, DepthOnData, DepthEstimateOnComplete, 0);
	mpConnection->request("POST",
		"/api/depthestimate",
		Base64Encoder::headers,
		(const unsigned char*)strJSON.c_str(),
		strlen(strJSON.c_str())
	);

	while (mpConnection->outstanding())
		mpConnection->pump();

	dst = resDepth.clone();

	return false;
}