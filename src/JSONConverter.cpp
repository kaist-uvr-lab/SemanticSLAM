#include <JSONConverter.h>
#include <winsock2.h>


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

const char* JSONConverter::headers[] = {
	"Connection", "close",
	"Content-type", "application/json",
	"Accept", "text/plain",
	0
};

int count = 0;
std::stringstream ss;
cv::Mat res;
void OnBegin(const happyhttp::Response* r, void* userdata)
{
	ss.str("");
	//printf("BEGIN (%d %s)\n", r->getstatus(), r->getreason());
	count = 0;
}

void OnData(const happyhttp::Response* r, void* userdata, const unsigned char* data, int n)
{
	ss.write((const char*)data, n);
	count += n;
	//fwrite(data, 1, n, (FILE*)ss);
}

void OnComplete(const happyhttp::Response* r, void* userdata)
{
	res = JSONConverter::ConvertStringToImage(ss.str().c_str(), count);
	//printf("COMPLETE (%d bytes)\n", count);
}


void JSONConverter::Init() {
	//WINSOCK for RESTAPI
	WSAData wsaData;
	int code = WSAStartup(MAKEWORD(1, 1), &wsaData);
	
}

bool JSONConverter::RequestPOST(std::string ip, int port, cv::Mat img, cv::Mat& dst,int mnFrameID) {
	
	std::string strJSON = ConvertImageToJSONStr(mnFrameID, img);
	
	//rapidjson::Document document;
	

	//if (document.Parse(strJSON.c_str()).HasParseError()) {
	//	std::cout << "JSON parsing error" << std::endl;
	//}

	//if (document.HasMember("image") && document["image"].IsArray()) {
	//	std::cout << "success" << std::endl;
	//	//std::cout << document["image"].GetArray();
	//	//document["image"].G
	//}

	
	happyhttp::Connection* mpConnection = new happyhttp::Connection(ip.c_str(), port);

	mpConnection->setcallbacks(OnBegin, OnData, OnComplete, 0);
	
	mpConnection->request("POST",
		"/api/predict",
		headers,
		(const unsigned char*)strJSON.c_str(),
		strlen(strJSON.c_str())
	);
	
	while (mpConnection->outstanding())
		mpConnection->pump();
	
	dst = res.clone();
	
	return false;
}