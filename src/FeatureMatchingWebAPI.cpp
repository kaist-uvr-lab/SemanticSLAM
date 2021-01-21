#include <FeatureMatchingWebAPI.h>

int feaeture_count = 0;
std::stringstream ssWebData;
std::vector<cv::Point2f> resPts;
std::vector<int> resMatches;
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
			res.push_back(a[i].GetFloat());
		}
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

