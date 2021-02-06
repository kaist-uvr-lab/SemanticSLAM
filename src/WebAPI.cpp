#include <WebAPI.h>
#include <winsock2.h>

WebAPI::WebAPI(std::string a, int b):ip(a), port(b), datastream(""){}
WebAPI::~WebAPI() {}

const char* WebAPI::headers[] = {
	"Connection", "close",
	"Content-type", "application/json",
	"Accept", "text/plain",
	0
};

void WebAPI::Init() {
	//WINSOCK for RESTAPI
	WSAData wsaData;
	int code = WSAStartup(MAKEWORD(1, 1), &wsaData);
}

void OnBegin(const happyhttp::Response* r, void* userdata)
{
	static_cast<std::stringstream*>(userdata)->str("");
}

void OnData(const happyhttp::Response* r, void* userdata, const unsigned char* data, int n)
{
	static_cast<std::stringstream*>(userdata)->write((const char*)data, n);
}

void OnComplete(const happyhttp::Response* r, void* userdata)
{
}

std::string WebAPI::Send(std::string method, std::string input) {
	happyhttp::Connection* mpConnection = new happyhttp::Connection(ip.c_str(), port);

	mpConnection->setcallbacks(OnBegin, OnData, OnComplete, (void*)&datastream);
	mpConnection->request("POST",
		method.c_str(),
		headers,
		(const unsigned char*)input.c_str(),
		strlen(input.c_str())
	);

	while (mpConnection->outstanding()){
		mpConnection->pump();
	}

	return datastream.str();
}


