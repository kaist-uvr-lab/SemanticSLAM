#include <Base64Encoder.h>
#include <winsock2.h>

const std::string Base64Encoder::base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

const char* Base64Encoder::headers[] = {
	"Connection", "close",
	//"Content-type", "application/json",
	"Content-type", "application/json",
	"Accept", "text/plain",
	0
};

void Base64Encoder::Init() {
	//WINSOCK for RESTAPI
	WSAData wsaData;
	int code = WSAStartup(MAKEWORD(1, 1), &wsaData);
}