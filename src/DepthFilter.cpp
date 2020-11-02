#include <DepthFilter.h>
#include <Frame.h>

namespace UVR_SLAM {
	Seed::Seed(float depth_mean, float depth_min):a(10), b(10), mu(1.0/depth_mean), z_range(1.0/depth_min), sigma2(z_range*z_range/36){}
	DepthFilter::DepthFilter(){
	
	}
	DepthFilter::~DepthFilter() {

	}
	void DepthFilter::Initialize(Frame* pF) {
		auto pMatch = pF->mpMatchInfo;
		auto vpCPs = pMatch->mvpMatchingCPs;
	}
}