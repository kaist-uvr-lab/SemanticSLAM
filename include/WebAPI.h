#ifndef WEBAPI_H
#define WEBAPI_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <WebAPIDataConverter.h>
#include "happyhttp.h"
#include <string>

class WebAPI {
public:
	WebAPI(std::string a, int b);
	virtual ~WebAPI();

	const static char* headers[];
	static void Init();
	std::string Send(std::string method, std::string input);

protected:
	std::string ip;
	int port;
	std::stringstream datastream;
};
#endif