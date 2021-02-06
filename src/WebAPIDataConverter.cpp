#include <WebAPIDataConverter.h>
#include <Base64Encoder.h>
#include <rapidjson\document.h>

///////인풋 아웃풋 변환 함수들
std::string WebAPIDataConverter::ConvertImageToString(cv::Mat img, int id) {
	int r = img.rows;
	int c = img.cols;
	int total = r*c;

	std::stringstream ss;
	ss << "{\"img\":";
	ss << "\"";
	std::vector<int> params;
	params.push_back(CV_IMWRITE_JPEG_QUALITY);
	params.push_back(95);
	/*int params[3] = { 0 };
	params[0] = CV_IMWRITE_JPEG_QUALITY;
	params[1] = 95;*/
	std::vector<uchar> buf;
	bool code = cv::imencode(".jpg", img, buf, params);
	uchar* result = reinterpret_cast<uchar*> (&buf[0]);
	
	std::string strimg = Base64Encoder::base64_encode(result, buf.size());

	/*uchar temp[1000000];
	temp[999999] = 0;
	tb64avx2enc(result, buf.size(), temp);
	std::string strimg = static_cast<std::string>(reinterpret_cast<const char *>(temp));*/
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

void WebAPIDataConverter::ConvertStringToDesc(const char* data, int n, cv::Mat& desc) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "ConvertStringToDesc::JSON parsing error" << std::endl;
	}
	if (document.HasMember("desc") && document["desc"].IsString()) {
		
		////디스크립터 정보 확인 필요.
		//256 x n 으로 받은 후 트랜스포즈하던가
		//보내는 쪽을 n x 256으로 변경하던가
		desc = cv::Mat::zeros(256, n, CV_32FC1);
		auto resstr = Base64Encoder::base64_decode(std::string(document["desc"].GetString()));
		std::memcpy(desc.data, resstr.c_str(), n * 256 * sizeof(float));
		/*auto a = std::string(document["desc"].GetString()).c_str();
		int len = strlen(a);
		uchar c[500000]; c[499999] = 0;
		tb64avx2dec((const uchar*)a, len, c);
		std::memcpy(desc.data, c, n * 256 * sizeof(float));*/
		desc = desc.t();
	}
}

//void WebAPIDataConverter::ConvertStringToPoints(const char* data, std::vector<cv::Point2f>& vPTs, cv::Mat& desc) {
//	rapidjson::Document document;
//	if (document.Parse(data).HasParseError()) {
//		std::cout << "return JSON parsing error" << std::endl;
//	}
//	if (document.HasMember("pts") && document["pts"].IsString()) {
//
//		int n = document["n"].GetInt();
//		desc = cv::Mat::zeros(n, 256, CV_32FC1);
//		auto resstrkpts = Base64Encoder::base64_decode(std::string(document["pts"].GetString()));// , n2);
//		float* tempFloat1 = (float*)malloc(n * 2 * sizeof(float));
//		std::memcpy(tempFloat1, resstrkpts.c_str(), n * 2 * sizeof(float));
//		for (int i = 0; i < n; i++) {
//			vPTs.push_back(std::move(cv::Point2f(tempFloat1[2*i], tempFloat1[2 * i+1])));
//		}
//		std::free(tempFloat1);
//
//		auto resstr = Base64Encoder::base64_decode(std::string(document["desc"].GetString()));
//		float* tempFloat2 = (float*)malloc(n *256* sizeof(float));
//		std::memcpy(tempFloat2, resstr.c_str(), n * 256* sizeof(float));
//		for (int i = 0, iend = n * 256; i < iend; i++) {
//			int x = i % n;
//			int y = i / n;
//			desc.at<float>(x, y) = tempFloat2[i];
//		}
//		std::free(tempFloat2);
//	}
//}


void WebAPIDataConverter::ConvertStringToNumber(const char* data, int &n) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "ConvertStringToNumber::JSON parsing error" << std::endl;
	}
	if (document.HasMember("num") && document["num"].IsInt()) {
		n = document["num"].GetInt();
	}
}

void WebAPIDataConverter::ConvertStringToPoints(const char* data, int n, std::vector<cv::Point2f>& vPTs) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "ConvertStringToPoints::JSON parsing error" << std::endl;
	}
	if (document.HasMember("pts") && document["pts"].IsString()) {
		auto resstrkpts = Base64Encoder::base64_decode(std::string(document["pts"].GetString()));// , n2);
		float* tempFloat1 = (float*)malloc(n * 2 * sizeof(float));
		std::memcpy(tempFloat1, resstrkpts.c_str(), n * 2 * sizeof(float));
		for (int i = 0; i < n; i++) {
			vPTs.push_back(std::move(cv::Point2f(tempFloat1[2 * i], tempFloat1[2 * i + 1])));
		}
		std::free(tempFloat1);
		/*vPTs = std::vector<cv::Point2f>(n);
		std::memcpy(&vPTs[0], resstrkpts.c_str(), n * 2 * sizeof(float));*/
	}
}

void WebAPIDataConverter::ConvertStringToMatches(const char* data, int n, std::vector<int>& vMatches) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "JSON parsing error::ConvertStringToMatches" << std::endl;
	}
	if (document.HasMember("matches") && document["matches"].IsString()) {
		auto resstrmatches = Base64Encoder::base64_decode(std::string(document["matches"].GetString()));// , n2);
		//int* temp = (int*)malloc(n * sizeof(int));
		vMatches = std::vector<int>(n);
		std::memcpy(&vMatches[0], resstrmatches.c_str(), n * sizeof(int));
		/*for (size_t i = 0; i < n; i++) {
			vMatches[i] = temp[i];
		}*/
	}
}

void WebAPIDataConverter::ConvertStringToDepthImage(const char* data, cv::Mat& res) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "Depth Estimate JSON parsing error" << std::endl;
	}
	if (document["b"].GetBool()) {
		if (document.HasMember("res") && document["res"].IsString()) {

			
			int w = document["w"].GetInt();
			int h = document["h"].GetInt();
			res = cv::Mat::zeros(h, w, CV_32FC1);

			auto resstr = Base64Encoder::base64_decode(std::string(document["res"].GetString()));
			float* tempFloat2 = (float*)malloc(w* h * sizeof(float));
			std::memcpy(tempFloat2, resstr.c_str(), w * h * sizeof(float));
			for (int i = 0, iend = w * h; i < iend; i++) {
				int x = i % w;
				int y = i / w;
				res.at<float>(y, x) = tempFloat2[i];
			}
			std::free(tempFloat2);

			/*const rapidjson::Value& a = document["res"];
			for (int y = 0; y < h; y++) {
				for (int x = 0; x < w; x++) {
					res.at<float>(y, x) = 1.0 / a[y][x].GetFloat();
				}
			}*/

		}
		else {
			std::cout << "depth estimate!!" << std::endl;
		}
	}
	else {
		res = cv::Mat::zeros(0, 0, CV_8UC1);
	}
}