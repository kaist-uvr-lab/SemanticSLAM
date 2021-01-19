#include <Base64Encoder.h>

const std::string Base64Encoder::base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

const char* Base64Encoder::headers[] = {
	"Connection", "close",
	//"Content-type", "application/json",
	"Content-type", "application/json",
	"Accept", "text/plain",
	0
};