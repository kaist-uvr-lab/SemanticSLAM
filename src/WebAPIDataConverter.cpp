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

void WebAPIDataConverter::ConvertBytesToDesc(const char* data, int n, cv::Mat& desc) {
	desc = cv::Mat::zeros(n, 256, CV_32FC1);
	std::memcpy(desc.data, data, n * 256 * sizeof(float));
	//desc = desc.t();
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
		std::memcpy(desc.data, resstr.data(), n * 256 * sizeof(float));
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

void WebAPIDataConverter::ConvertStringToImage(const char* data, cv::Mat& img) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "ConvertStringToImage::JSON parsing error" << std::endl;
	}
	if (document.HasMember("img") && document["img"].IsString()) {
		/*int w = document["w"].GetInt();
		int h = document["h"].GetInt();*/
		auto resstr = Base64Encoder::base64_decode(std::string(document["img"].GetString()));// , n2);
		auto temp = std::vector<uchar>(resstr.length());
		std::memcpy(&temp[0], resstr.c_str(), temp.size() * sizeof(uchar));
		img = cv::imdecode(temp, cv::IMREAD_COLOR);
	}
}
void WebAPIDataConverter::ConvertStringToNumber(const char* data, int &n) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "ConvertStringToNumber::JSON parsing error" << std::endl;
	}
	if (document.HasMember("n") && document["n"].IsInt()) {
		n = document["n"].GetInt();
	}
}
void WebAPIDataConverter::ConvertStringToNumber(const char* data, int &id1, int& id2) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "ConvertStringToNumber::JSON parsing error" << std::endl;
	}
	if (document.HasMember("id1") && document["id1"].IsInt()) {
		id1 = document["id1"].GetInt();
		id2 = document["id2"].GetInt();
	}
}

void WebAPIDataConverter::ConvertStringToPoints(const char* data, std::vector<cv::Point2f>& vPTs) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "ConvertStringToPoints::JSON parsing error" << std::endl;
	}
	//if (document.HasMember("pts") && document["pts"].IsString()) {
	//	auto resstrkpts = Base64Encoder::base64_decode(std::string(document["pts"].GetString()));//length / (4*2) = n임
	//	int n = document["n"].GetInt();
	//	float* tempFloat1 = (float*)malloc(n * 2 * sizeof(float));
	//	std::memcpy(tempFloat1, resstrkpts.data(), n * 2 * sizeof(float));
	//	for (int i = 0; i < n; i++) {
	//		/*if (tempFloat1[2 * i] < 0.0 || tempFloat1[2 * i] >= 640.0) {
	//			std::cout << "Error::Get Points X " << tempFloat1[2 * i] << std::endl;
	//		}
	//		if (tempFloat1[2 * i+1] < 0.0 || tempFloat1[2 * i+1] >= 480) {
	//			std::cout << "Error::Get Points Y " << tempFloat1[2 * i+1] << std::endl;
	//		}*/
	//		vPTs.push_back(std::move(cv::Point2f(tempFloat1[2 * i], tempFloat1[2 * i + 1])));
	//	}
	//	std::free(tempFloat1);
	//	/*vPTs = std::vector<cv::Point2f>(n);
	//	std::memcpy(&vPTs[0], resstrkpts.c_str(), n * 2 * sizeof(float));*/
	//}
	if (document.HasMember("pts")&& document["pts"].IsArray()){
		const rapidjson::Value& a = document["pts"];
		int n = document["n"].GetInt();
		for (int i = 0; i < n; i++) {
			vPTs.push_back(std::move(cv::Point2f(a[i][0].GetFloat(), a[i][1].GetFloat())));
		}
	}
	
}

void WebAPIDataConverter::ConvertStringToMatches(const char* data, int n, cv::Mat& mMatches) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "JSON parsing error::ConvertStringToMatches" << std::endl;
	}
	//if (document.HasMember("matches") && document["matches"].IsString()) {
	//	auto resstrmatches = Base64Encoder::base64_decode(std::string(document["matches"].GetString()));// , n2);
	//	
	//	if (resstrmatches.size() != 4 * n)
	//		std::cout << "Error::ConvertStringToMatches::Invaild Received message size::" << resstrmatches.size() << " ," << 4 * n << std::endl;

	//	mMatches = cv::Mat::zeros(1, n, CV_32SC1);
	//	std::memcpy(mMatches.data, resstrmatches.c_str(), n * sizeof(int));
	//}
	if (document.HasMember("matches")) {
		mMatches = cv::Mat::zeros(1, n, CV_32SC1);
		const rapidjson::Value& a = document["matches"];
		for (int i = 0; i < n; i++) {
			mMatches.at<int>(i) = a[i].GetInt();
		}
	}
}
void WebAPIDataConverter::ConvertStringToMatches(const char* data, int n, std::vector<int>& vMatches) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "JSON parsing error::ConvertStringToMatches" << std::endl;
	}
	//if (document.HasMember("matches") && document["matches"].IsString()) {
	//	auto resstrmatches = Base64Encoder::base64_decode(std::string(document["matches"].GetString()));// , n2);
	//	vMatches = std::vector<int>(n);
	//	std::memcpy(&vMatches[0], resstrmatches.c_str(), n * sizeof(int));
	//}
	if (document.HasMember("matches")) {
		const rapidjson::Value& a = document["matches"];
		for (int i = 0; i < n; i++) {
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////단말과 통신 부분
void WebAPIDataConverter::ConvertMapName(const char* data, std::string& map) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "JSON parsing error::ConvertMapName" << std::endl;
	}
	map = document["map"].GetString();
}
void WebAPIDataConverter::ConvertDisconnectToServer(const char* data, std::string& _u) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "JSON parsing error::ConvertDisconnectToServer" << std::endl;
	}
	_u = document["u"].GetString();
}
void WebAPIDataConverter::ConvertConnectToServer(const char* data, std::string& _u, float& _fx, float& _fy, float& _cx, float& _cy, int& _w, int & _h, bool& _b, std::string& _n){
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "JSON parsing error::ConvertConnectToServer" << std::endl;
	}
	_fx = document["fx"].GetFloat();
	_fy = document["fy"].GetFloat();
	_cx = document["cx"].GetFloat();
	_cy = document["cy"].GetFloat();
	_w = document["w"].GetInt();
	_h = document["h"].GetInt();
	_b = document["b"].GetBool();
	_n = document["n"].GetString();
	_u = document["u"].GetString();
}
void WebAPIDataConverter::ConvertDeviceFrameIDToServer(const char* data, std::string& user, int& id) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "JSON parsing error::ConvertDeviceFrameIDToServer" << std::endl;
	}
	user = document["user"].GetString();
	id = document["id"].GetInt();
}
void WebAPIDataConverter::ConvertDeviceToServer(const char* data, int& id, bool& init) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "JSON parsing error::ConvertDeviceToServer" << std::endl;
	}
	if (document.HasMember("id2") && document["id2"].IsInt()) {
		id = document["id2"].GetInt();
		init = document["init"].GetBool();
	}
}

std::string WebAPIDataConverter::ConvertInitializationToJsonString(int id, bool bInit, cv::Mat R, cv::Mat t, cv::Mat keypoints, cv::Mat mappoints){
	std::stringstream ss;

	std::string strpose = "";
	std::string strkey = "";
	std::string strmap = "";
	if (bInit) {
		cv::Mat P;
		cv::vconcat(R, t.t(), P);
		/*auto data2 = t.data;
		float *fdata1 = (float*)malloc(sizeof(float) * 9);
		float *fdata2 = (float*)malloc(sizeof(float) * 3);
		memcpy(fdata1, data1, sizeof(float) * 9);
		memcpy(fdata2, data2, sizeof(float) * 3);
		////base64 코딩 후에 전송하기
		//확인 완료
		std::cout << R.at<float>(0, 0) << ", " << R.at<float>(0, 1) << " " << R.at<float>(0, 2) << "::" << fdata1[0] << ", " << fdata1[1] << ", " << fdata1[2] << std::endl;
		std::cout << R.at<float>(1, 0) << ", " << R.at<float>(1, 1) << " " << R.at<float>(1, 2) << "::" << fdata1[3] << ", " << fdata1[4] << ", " << fdata1[5] << std::endl;
		std::cout << R.at<float>(2, 0) << ", " << R.at<float>(2, 1) << " " << R.at<float>(2, 2) << "::" << fdata1[6] << ", " << fdata1[7] << ", " << fdata1[8] << std::endl;*/
		strpose = Base64Encoder::base64_encode(P.data, sizeof(float)*12);
		strkey = Base64Encoder::base64_encode(keypoints.data, sizeof(float) * keypoints.rows);
		strmap = Base64Encoder::base64_encode(mappoints.data, sizeof(float) * mappoints.rows);
	}

	ss << "{\"id1\":" << (int)id << ",\"init\":" << bInit << ",\"pose\":\"" << strpose <<"\""<< ",\"keypoints\":\"" << strkey << "\"" << ",\"mappoints\":\"" << strmap << "\"" << "}";
	return ss.str();
}

std::string WebAPIDataConverter::ConvertMapDataToJson(cv::Mat mpIDs, cv::Mat x3Ds, cv::Mat kfids, cv::Mat poses, cv::Mat idxs) {
	std::string strids = Base64Encoder::base64_encode(mpIDs.data, sizeof(int) * mpIDs.rows);
	std::string strx3ds = Base64Encoder::base64_encode(x3Ds.data, sizeof(float) * x3Ds.rows);
	std::string strkfids = Base64Encoder::base64_encode(kfids.data, sizeof(int) * kfids.rows);
	std::string strposes = Base64Encoder::base64_encode(poses.data, sizeof(float) * poses.rows*3);
	std::string stridxs = Base64Encoder::base64_encode(idxs.data, sizeof(float) * idxs.rows);
	std::stringstream ss;
	ss << "{\"ids\":\"" << strids << "\",\"x3ds\":\"" << strx3ds <<"\",\"kfids\":\""<<strkfids<< "\",\"poses\":\"" << strposes << "\",\"idxs\":\"" << stridxs << "\"}";
	return ss.str();
}

////단말과 통신 부분
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
