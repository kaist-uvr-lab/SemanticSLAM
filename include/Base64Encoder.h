#ifndef BASE64_ENCODER_H
#define BASE64_ENCODER_H
#pragma once
/*
ÄÚµå
https://www.ics.uci.edu/~magda/Courses/ics167/chatroom_demo/server/base64.cpp
*/
#include <string>
#include <vector>
#include <iostream>
//#include <cstdio>
//#include <cstring>

class Base64Encoder {
public:
	
	static void Init();
	const static std::string base64_chars;
	
	static inline bool is_base64(unsigned char c) {
		return (isalnum(c) || (c == '+') || (c == '/'));
	}

	static std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len);
	static std::string base64_decode(std::string const& encoded_string);
	////////////////////////////////
	static const char MimeBase64[];
	static const int DecodeMimeBase64[256];

	static int base64_decode(char *text, unsigned char *dst, int numBytes);

	//static int base64_encode(char *text, int numBytes, char **encodedText);
	static std::string base64_encode2(unsigned char *text, int numBytes);
	static std::string base64_decode2(std::string const& encoded_string);

};

#endif