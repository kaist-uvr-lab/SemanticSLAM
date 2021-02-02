#ifndef WEBAPI_H
#define WEBAPI_H
#pragma once

//#include <rapidjson\document.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

//#include <Base64Encoder.h>
#include <WebAPIDataConverter.h>
#include "happyhttp.h"
#include <string>

//typedef void(*ResponseBegin_CB)(const happyhttp::Response* r, void* userdata);

class WebAPI {
public:
	WebAPI(std::string a, int b);
	virtual ~WebAPI();

	const static char* headers[];
	static void Init();

	//void (*funcbegin)(const happyhttp::Response* r, void* userdata);

	std::string Send(std::string method, std::string input);

protected:
	std::string ip;
	int port;
	std::stringstream datastream;
	int count;

private:
	/*void OnBegin(const happyhttp::Response* r, void* userdata);
	void OnData(const happyhttp::Response* r, void* userdata, const unsigned char* data, int n);
	void OnComplete(const happyhttp::Response* r, void* userdata);*/
};
#endif