#include <Base64Encoder.h>
#include <winsock2.h>
#include <iostream>

const std::string Base64Encoder::base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";



void Base64Encoder::Init() {
	//WINSOCK for RESTAPI
	WSAData wsaData;
	int code = WSAStartup(MAKEWORD(1, 1), &wsaData);
}

std::string Base64Encoder::base64_decode(std::string const& encoded_string) {
	int in_len = encoded_string.size() - 2;
	int i = 0;
	int j = 0;
	int in_ = 2;
	unsigned char char_array_4[4], char_array_3[3];
	std::string ret = "";

	while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
		char_array_4[i++] = encoded_string[in_]; in_++;
		if (i == 4) {
			for (i = 0; i <4; i++)
				char_array_4[i] = base64_chars.find(char_array_4[i]);

			char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
			char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
			char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

			for (i = 0; (i < 3); i++)
				ret += char_array_3[i];
			i = 0;
		}
	}
	if (i) {
		for (j = i; j <4; j++)
			char_array_4[j] = 0;

		for (j = 0; j <4; j++)
			char_array_4[j] = base64_chars.find(char_array_4[j]);

		char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
		char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
		char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

		for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
	}
	
	return ret;
}


std::vector<unsigned char> Base64Encoder::base64_decode(char const* encoded_string, int in_len) {
	int i = 0;
	int j = 0;
	
	unsigned char char_array_4[4], char_array_3[3];
	std::vector<unsigned char> res;
	for (size_t in = 2; in < in_len; in++) {
		char a = encoded_string[in];
		
		if (a == '=' || !is_base64(a))
			break;
		char_array_4[i++] = a;
		if (i == 4) {
			for (i = 0; i <4; i++)
				char_array_4[i] = base64_chars.find(char_array_4[i]);

			char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
			char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
			char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

			for (i = 0; (i < 3); i++){
				res.push_back(char_array_3[i]);
				
				//ret += char_array_3[i];
			}
			i = 0;
		}
	}
	
	if (i) {
		for (j = i; j <4; j++)
			char_array_4[j] = 0;

		for (j = 0; j <4; j++)
			char_array_4[j] = base64_chars.find(char_array_4[j]);

		char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
		char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
		char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

		for (j = 0; (j < i - 1); j++)
			res.push_back(char_array_3[j]);
	}
	/*for (size_t a = 0, aend = res.size(); a < aend; a+=4) {
		unsigned char temp = res[a + 3];
		res[a + 3] = res[a];
		res[a] = temp;
		temp = res[a + 2];
		res[a + 2] = res[a+1];
		res[a+1] = temp;
	}*/
	return res;
}

//일단 float 변환용
void Base64Encoder::base64_decode(char const* encoded_string, int in_len, char* res) {
	int i = 0;
	int j = 0;

	int idx = 0;
	unsigned char char_array_4[4], char_array_3[3];
	for (size_t in = 2; in < in_len; in++) {
		char a = encoded_string[in];

		if (a == '=' || !is_base64(a))
			break;
		char_array_4[i++] = a;
		if (i == 4) {
			for (i = 0; i <4; i++)
				char_array_4[i] = base64_chars.find(char_array_4[i]);

			char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
			char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
			char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
			if (idx >= 307200)
				std::cout << "asl;djfa;lsdjf;ladksj;kladsjkladsf" << std::endl;
			std::cout << idx <<", "<<(char)res[idx]<< std::endl;
			for (i = 0; (i < 3); i++) {
				(res[idx++]) = char_array_3[i];
				//res.push_back(char_array_3[i]);
				//ret += char_array_3[i];
			}
			i = 0;
		}
	}
	std::cout << "test1::" << idx << std::endl;
	if (i) {
		for (j = i; j <4; j++)
			char_array_4[j] = 0;

		for (j = 0; j <4; j++)
			char_array_4[j] = base64_chars.find(char_array_4[j]);

		char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
		char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
		char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

		for (j = 0; (j < i - 1); j++) {
			(res[idx++]) = char_array_3[j];
			//res.push_back(char_array_3[j]);
		}
			
	}
	std::cout << "test2::" << idx << std::endl;
}