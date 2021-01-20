#include <JSONConverter.h>
#include <winsock2.h>
#include <chrono>


//static const std::string base64_chars =
//"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
//"abcdefghijklmnopqrstuvwxyz"
//"0123456789+/";

//static inline bool is_base64(unsigned char c)
//{
//	return (isalnum(c) || (c == '+') || (c == '/'));
//}

//std::string base64_encode(uchar const* bytes_to_encode, unsigned int in_len)
//{
//	std::string ret;
//
//	int i = 0;
//	int j = 0;
//	unsigned char char_array_3[3];
//	unsigned char char_array_4[4];
//
//	while (in_len--)
//	{
//		char_array_3[i++] = *(bytes_to_encode++);
//		if (i == 3)
//		{
//			char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
//			char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
//			char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
//			char_array_4[3] = char_array_3[2] & 0x3f;
//
//			for (i = 0; (i <4); i++)
//			{
//				ret += base64_chars[char_array_4[i]];
//			}
//			i = 0;
//		}
//	}
//
//	if (i)
//	{
//		for (j = i; j < 3; j++)
//		{
//			char_array_3[j] = '\0';
//		}
//
//		char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
//		char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
//		char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
//		char_array_4[3] = char_array_3[2] & 0x3f;
//
//		for (j = 0; (j < i + 1); j++)
//		{
//			ret += base64_chars[char_array_4[j]];
//		}
//
//		while ((i++ < 3))
//		{
//			ret += '=';
//		}
//	}
//
//	return ret;
//
//}


std::string JSONConverter::ConvertImageToJSONStr(int nFrameID, cv::Mat img) {
	/*
	auto reqJsonData = R"(
	{
	"UserSeq": 1,
	"UserID": "jacking75",
	"UserPW": "123qwe"
	}
	)";
	const char json[] = " { \"hello\" : \"world\", \"t\" : true , \"f\" : false, \"n\": null, \"i\":123, \"pi\": 3.1416, \"a\":[1, 2, 3, 4] } ";
	*/
	std::string json;

	int r = img.rows;
	int c = img.cols;
	int total = r*c;
	std::vector<cv::Mat> channels;
	cv::split(img, channels);

	//cv::base64_encode

	/*std::vector<uchar> data1(channels[0].ptr(), channels[0].ptr() + total);
	std::vector<uchar> data2(channels[1].ptr(), channels[1].ptr() + total);
	std::vector<uchar> data3(channels[2].ptr(), channels[2].ptr() + total);
	std::string s1(data1.begin(), data1.end());
	std::string s2(data2.begin(), data2.end());
	std::string s3(data3.begin(), data3.end());*/

	std::stringstream ss;
	
	ss << "{\"image\":";
	/////////////////
	ss << "\"";
	//ss << s1.c_str() << s2.c_str() << s3.c_str();
	int params[3] = { 0 };
	params[0] = CV_IMWRITE_JPEG_QUALITY;
	params[1] = 100;

	std::vector<uchar> buf;
	bool code = cv::imencode(".jpg", img, buf, std::vector<int>(params, params + 2));
	uchar* result = reinterpret_cast<uchar*> (&buf[0]);

	std::string strimg = Base64Encoder::base64_encode(result, buf.size());
	ss << strimg;
	ss << "\"";
	
	/////////////////
	//ss<<"[";
	//for (int y = 0; y < data1.size(); y++) {
	//	ss << data1[y] << ",";
	//}
	////std::cout <<data1.size()<<"::"<< ss.str() << std::endl;
	//for (int y = 0; y < data2.size(); y++) {
	//	ss << data2[y] << ",";
	//}
	//for (int y = 0; y < data3.size()-1; y++) {
	//	ss << data3[y] << ",";
	//}
	////std::string temp;
	////std::stringstream tempss;
	////std::copy(data1.begin(), data1.begin()+total, std::ostream_iterator<int>(tempss, ","));
	//////std::cout << tempss.str() << std::endl;
	////ss << data3[total - 1];
	/////////////////
	//ss << s1.c_str() << s2.c_str() << s3.c_str();
	/*for (int cha = 0; cha < 3; cha++) {
		for (int y = 0; y < r; y++) {
			for (int x = 0; x < c; x++) {
				cv::Vec3b colorVec = img.at<cv::Vec3b>(y, x);
				ss << (int)colorVec.val[cha];
				if (cha != 2 || x != c - 1 || y != r - 1)
					ss << ",";
			}
		}
	}*/
	//ss << "]";
	/////////////////
	ss<<",\"w\":" << (int)c << ",\"h\":" << (int)r << "}";
	return ss.str();
}

cv::Mat JSONConverter::ConvertStringToImage(const char* data, int N) {
	rapidjson::Document document;
	
	if (document.Parse(data).HasParseError()) {
		std::cout << "JSON parsing error" << std::endl;
	}

	cv::Mat res = cv::Mat::zeros(0, 0, CV_8UC3);
	if (document.HasMember("seg_img") && document["seg_img"].IsArray()) {

		const rapidjson::Value& a = document["seg_img"];
		int h = a.Size();
		int w = a[0].Size();
		int c = a[0][0].Size();

		res = cv::Mat::zeros(h, w, CV_8UC3);
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				cv::Vec3b val;
				val[0] = a[y][x][0].GetInt();
				val[1] = a[y][x][1].GetInt();
				val[2] = a[y][x][2].GetInt();
				res.at<cv::Vec3b>(y, x) = val;
			}
		}
		

		//std::cout << a.Size()<<", "<<a[0].Size()<<", "<<a[0][0].Size()<< std::endl;
		//std::cout << document["image"].GetArray();
		//document["image"].G
	}
	
	return res;
}

cv::Mat JSONConverter::ConvertStringToLabel(const char* data, int N) {
	rapidjson::Document document;
	if (document.Parse(data).HasParseError()) {
		std::cout << "return JSON parsing error" << std::endl;
	}

	cv::Mat res = cv::Mat::zeros(0, 0, CV_8UC1);
	if (document.HasMember("seg_label") && document["seg_label"].IsArray()) {

		const rapidjson::Value& a = document["seg_label"];
		int h = document["h"].GetInt();
		int w = document["w"].GetInt();
		//int c = a[0][0].Size();

		res = cv::Mat::zeros(h, w, CV_8UC1);
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				res.at<uchar>(y, x) = a[y][x].GetInt();
			}
		}

	}

	return res;
}


//const char* JSONConverter::headers[] = {
//	"Connection", "close",
//	"Content-type", "application/json",
//	"Accept", "text/plain",
//	0
//};

int count = 0;
std::stringstream ss;
cv::Mat res;

void OnBegin(const happyhttp::Response* r, void* userdata)
{
	ss.str("");
	count = 0;
}

void OnData(const happyhttp::Response* r, void* userdata, const unsigned char* data, int n)
{
	ss.write((const char*)data, n);
	count += n;
}

void OnComplete(const happyhttp::Response* r, void* userdata)
{
	res = JSONConverter::ConvertStringToLabel(ss.str().c_str(), count);
}


//void JSONConverter::Init() {
//	//WINSOCK for RESTAPI
//	WSAData wsaData;
//	int code = WSAStartup(MAKEWORD(1, 1), &wsaData);
//}

bool JSONConverter::RequestPOST(std::string ip, int port, cv::Mat img, cv::Mat& dst,int mnFrameID, int& stat) {
	
	std::string strJSON = ConvertImageToJSONStr(mnFrameID, img);
	
	//rapidjson::Document document;
	//if (document.Parse(strJSON.c_str()).HasParseError()) {
	//	std::cout << "intpu JSON parsing error" << std::endl;
	//}
	////if (document.HasMember("w") && document.HasMember("h") && document["w"].IsInt()) {
	////	std::cout << "a;lsdjf;alskdjfl;asdjkf" << std::endl;
	////}
	//if (document.HasMember("image") && document["image"].IsString()) {
	//	std::cout << "success string " << std::endl;
	//	//std::cout << document["image"].GetArray();
	//	//document["image"].G
	//}
	//else if (document.HasMember("image") && document["image"].IsArray()) {
	//	std::cout << "success array " << std::endl;
	//}

	stat = 1;
	happyhttp::Connection* mpConnection = new happyhttp::Connection(ip.c_str(), port);

	mpConnection->setcallbacks(OnBegin, OnData, OnComplete, 0);
	mpConnection->request("POST",
		"/api/predict",
		Base64Encoder::headers,
		(const unsigned char*)strJSON.c_str(),
		strlen(strJSON.c_str())
	);

	while (mpConnection->outstanding())
		mpConnection->pump();
	
	dst = res.clone();
	
	return false;
}