#include <WebAPIDataConverter.h>
#include <Base64Encoder.h>
#include <rapidjson\document.h>

///////ÀÎÇ² ¾Æ¿ôÇ² º¯È¯ ÇÔ¼öµé
std::string WebAPIDataConverter::ConvertImageToString(cv::Mat img, int id) {
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

std::string WebAPIDataConverter::ConvertNumberToString(int id) {
	std::stringstream ss;
	ss << "{\"id\":" << (int)id << "}";
	return ss.str();
}

std::string WebAPIDataConverter::ConvertNumberToString(int id1, int id2) {
	std::stringstream ss;
	ss << "{\"id1\":" << (int)id1 << ",\"id2\":" << (int)id2 << "}";
	return ss.str();
}

void WebAPIDataConverter::ConvertStringToPoints(const char* data, std::vector<cv::Point2f>& vPTs, cv::Mat& desc) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "return JSON parsing error" << std::endl;
	}
	//cv::Mat res = cv::Mat::zeros(0, 0, CV_8UC1);
	if (document.HasMember("res") && document["res"].IsArray()) {

		const rapidjson::Value& a = document["res"];
		int n = document["n"].GetInt();
		for (size_t i = 0; i < n; i++) {
			cv::Point2f pt(a[i][0].GetFloat(), a[i][1].GetFloat());
			vPTs.push_back(pt);
		}

		desc = cv::Mat::zeros(n, 256, CV_32FC1);
		const rapidjson::Value& b = document["desc"];
		for (size_t j = 0; j < 256; j++) {
			for (size_t i = 0; i < n; i++) {
				desc.at<float>(i, j) = b[j][i].GetFloat();
			}
		}

	}
}

void WebAPIDataConverter::ConvertStringToLabels(const char* data, std::vector<int>& vMatches) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "return JSON parsing error" << std::endl;
	}
	if (document.HasMember("res") && document["res"].IsArray()) {

		const rapidjson::Value& a = document["res"];
		int n = document["n"].GetInt();
		for (size_t i = 0; i < n; i++) {
			vMatches.push_back(a[i].GetInt());
		}
	}
}

void WebAPIDataConverter::ConvertStringToDepthImage(const char* data, cv::Mat& res) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "Depth Estimate JSON parsing error" << std::endl;
	}
	if (document["b"].GetBool()) {
		if (document.HasMember("res") && document["res"].IsArray()) {

			const rapidjson::Value& a = document["res"];
			int w = document["w"].GetInt();
			int h = document["h"].GetInt();

			res = cv::Mat::zeros(h, w, CV_32FC1);
			for (int y = 0; y < h; y++) {
				for (int x = 0; x < w; x++) {
					res.at<float>(y, x) = 1.0 / a[y][x].GetFloat();
				}
			}

		}
		else {
			std::cout << "depth estimate!!" << std::endl;
		}
	}
	else {
		res = cv::Mat::zeros(0, 0, CV_8UC1);
	}
}