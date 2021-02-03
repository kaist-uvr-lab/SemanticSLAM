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

	static std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len)
	{
		std::string ret;

		int i = 0;
		int j = 0;
		unsigned char char_array_3[3];
		unsigned char char_array_4[4];

		while (in_len--)
		{
			char_array_3[i++] = *(bytes_to_encode++);
			if (i == 3)
			{
				char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
				char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
				char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
				char_array_4[3] = char_array_3[2] & 0x3f;

				for (i = 0; (i <4); i++)
				{
					ret += base64_chars[char_array_4[i]];
				}
				i = 0;
			}
		}

		if (i)
		{
			for (j = i; j < 3; j++)
			{
				char_array_3[j] = '\0';
			}

			char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
			char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
			char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
			char_array_4[3] = char_array_3[2] & 0x3f;

			for (j = 0; (j < i + 1); j++)
			{
				ret += base64_chars[char_array_4[j]];
			}

			while ((i++ < 3))
			{
				ret += '=';
			}
		}

		return ret;

	}
	//static std::string base64_decode(char const* encoded_string, int in_len);
	static std::vector<unsigned char> base64_decode(char const* encoded_string, int in_len);
	static void base64_decode(char const* encoded_string, int in_len, char* res);
	static std::string base64_decode(std::string const& encoded_string);

};

#endif