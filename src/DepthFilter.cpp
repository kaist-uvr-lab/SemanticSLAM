#include <DepthFilter.h>
#include <System.h>
#include <Map.h>
#include <Frame.h>
#include <CandidatePoint.h>
#include <MapPoint.h>
#include <Optimization.h>
#include <Visualizer.h>

namespace UVR_SLAM {
	float Seed::px_err_angle;
	Seed::Seed(cv::Mat _ray, float depth_mean, float depth_min):count(0), a(10), b(10), mu(1.0/depth_mean), z_range(1.0/depth_min), sigma2(z_range*z_range/36), ray(_ray){
	}
	
	float Seed::ComputeDepth(cv::Point2f& est, cv::Point2f src, cv::Mat R, cv::Mat t, cv::Mat K) {
		float sig_sqrt = sqrt(sigma2);
		float z_inv_min = mu + sig_sqrt;
		float z_inv_max = max(mu - sig_sqrt, 0.00000001f);
		float z_min = 1. / z_inv_min;
		float z_max = 1. / z_inv_max;

		cv::Mat Xcmin = R*ray*z_min + t;
		cv::Mat Xcmax = R*ray*z_max + t;
		
		float fx = K.at<float>(0, 0);
		float fy = K.at<float>(1, 1);
		float cx = K.at<float>(0, 2);
		float cy = K.at<float>(1, 2);

		cv::Point2f XprojMin(Xcmin.at<float>(0) / Xcmin.at<float>(2), Xcmin.at<float>(1) / Xcmin.at<float>(2));
		cv::Point2f XprojMax(Xcmax.at<float>(0) / Xcmax.at<float>(2), Xcmax.at<float>(1) / Xcmax.at<float>(2));
		cv::Point2f epi_dir = XprojMin - XprojMax;
		cv::Point2f XimgMin(XprojMin.x*fx + cx, XprojMin.y*fy + cy);
		cv::Point2f XimgMax(XprojMax.x*fx + cx, XprojMax.y*fy + cy);
		float epi_length = cv::norm(XimgMin - XimgMax) / 2.0;
		size_t n_steps = epi_length / 0.7; // one step per pixel
		cv::Point2f step(epi_dir.x / n_steps, epi_dir.y / n_steps);
		cv::Point2f uv = XprojMax - step;
		cv::Point2f uv_best;
		float minDist = FLT_MAX;
		
		for (size_t i = 0; i < n_steps; ++i, uv += step)
		{
			cv::Point2f pt(uv.x*fx + cx, uv.y*fy + cy);
			auto diffPt = pt - src;
			float dist = diffPt.dot(diffPt);
			if (dist < minDist) {
				minDist = dist;
				uv_best = pt;
			}
		}//for
		est = uv_best;
		return minDist;
	}

	float Seed::ComputeTau(cv::Mat t, float z) {
		cv::Mat a = ray*z - t;
		float t_norm = cv::norm(t);
		float a_norm = cv::norm(a);
		float alpha = acos(ray.dot(t) / t_norm);
		float beta = acos(a.dot(-t) / (t_norm*a_norm));
		float beta_plus = beta + px_err_angle;
		float gamma_plus = CV_PI - alpha - beta_plus; // triangle angles sum to PI
		float z_plus = t_norm*sin(beta_plus) / sin(gamma_plus); // law of sines
		return (z_plus - z); // tau
	}
	
	void Seed::UpdateDepth(float invz, float tau2) {
		float norm_scale = sqrt(sigma2 + tau2);
		if (std::isnan(norm_scale))
			return;
		float nd_mu = mu;
		float nd_sd = norm_scale;

		float pdf_val;
		{
			if (std::isnan(invz))
				pdf_val = 0.f;
			else {
				float exponent = invz - nd_mu;
				exponent *= -exponent;
				exponent /= (2 * nd_sd*nd_sd);
				float res = exp(exponent);
				res /= nd_sd*sqrt(2 * CV_PI);
				/*if (std::isinf(res)) {
				std::cout << "inf::" << res << std::endl;
				res = 0.0;
				}*/
				pdf_val = res;
			}
		}

		float s2 = 1. / (1. / sigma2 + 1. / tau2);
		float m = s2*(mu / sigma2 + invz / tau2);
		float C1 = a / (a + b) * pdf_val;
		float C2 = b / (a + b) * 1. / z_range;
		float normalization_constant = C1 + C2;
		C1 /= normalization_constant;
		C2 /= normalization_constant;
		float f = C1*(a + 1.) / (a + b + 1.) + C2*a / (a + b + 1.);
		float e = C1*(a + 1.)*(a + 2.) / ((a + b + 1.)*(a + b + 2.))
			+ C2*a*(a + 1.0f) / ((a + b + 1.0f)*(a + b + 2.0f));

		// update parameters
		float mu_new = C1*m + C2*mu;
		sigma2 = C1*(s2 + m*m) + C2*(sigma2 + mu*mu) - mu_new*mu_new;
		mu = mu_new;
		a = (e - f) / (f - e / f);
		b = a*(1.0f - f) / f;
		count++;
	}

	DepthFilter::DepthFilter(){
	}
	DepthFilter::DepthFilter(System* pSys):mpSystem(pSys) {
	}
	DepthFilter::~DepthFilter() {
	}
	void DepthFilter::Init() {
		K = mpSystem->mK;
		invK = mpSystem->mInvK;
		////시드 생성 관련 변수
		float fx = mpSystem->mK.at<float>(0, 0);
		float noise = 1.0;
		Seed::px_err_angle = atan(noise / (2.0*fx))*2.0;
		////시드 생성 관련 변수

	}
	void DepthFilter::UpdateSeed(Seed* pSeed, float invz, float tau2) {
		float norm_scale = sqrt(pSeed->sigma2 + tau2);
		if (std::isnan(norm_scale))
			return;
		float nd_mu = pSeed->mu;
		float nd_sd = norm_scale;

		float pdf_val;
		{
			if (std::isnan(invz))
				pdf_val = 0.f;
			else {
				float exponent = invz - nd_mu;
				exponent *= -exponent;
				exponent /= (2 * nd_sd*nd_sd);
				float res = exp(exponent);
				res /= nd_sd*sqrt(2 * CV_PI);
				/*if (std::isinf(res)) {
					std::cout << "inf::" << res << std::endl;
					res = 0.0;
				}*/
				pdf_val = res;
			}
		}
		
		float s2 = 1. / (1. / pSeed->sigma2 + 1. / tau2);
		float m = s2*(pSeed->mu / pSeed->sigma2 + invz / tau2);
		float C1 = pSeed->a / (pSeed->a + pSeed->b) * pdf_val;
		float C2 = pSeed->b / (pSeed->a + pSeed->b) * 1. / pSeed->z_range;
		float normalization_constant = C1 + C2;
		C1 /= normalization_constant;
		C2 /= normalization_constant;
		float f = C1*(pSeed->a + 1.) / (pSeed->a + pSeed->b + 1.) + C2*pSeed->a / (pSeed->a + pSeed->b + 1.);
		float e = C1*(pSeed->a + 1.)*(pSeed->a + 2.) / ((pSeed->a + pSeed->b + 1.)*(pSeed->a + pSeed->b + 2.))
			+ C2*pSeed->a*(pSeed->a + 1.0f) / ((pSeed->a + pSeed->b + 1.0f)*(pSeed->a + pSeed->b + 2.0f));

		// update parameters
		float mu_new = C1*m + C2*pSeed->mu;
		pSeed->sigma2 = C1*(s2 + m*m) + C2*(pSeed->sigma2 + pSeed->mu*pSeed->mu) - mu_new*mu_new;
		pSeed->mu = mu_new;
		pSeed->a = (e - f) / (f - e / f);
		pSeed->b = pSeed->a*(1.0f - f) / f;
	}
	void DepthFilter::Update(Frame* pF, Frame* pPrev) {

		std::cout << "DepthFilter::Update::start" << std::endl;
		float thresh = 100.; //200.0
		auto pMatch = pF->mpMatchInfo;
		auto vpCPs = pMatch->mvpMatchingCPs;
		auto vPTs = pMatch->mvMatchingPts;

		cv::Mat Rcurr, Tcurr, Pcurr;
		pF->GetPose(Rcurr, Tcurr);
		cv::hconcat(Rcurr, Tcurr, Pcurr);
		cv::Mat Rinvcurr = Rcurr.t();
		cv::Mat Tinvcurr = -Rinvcurr*Tcurr;
		//mpSystem->mpMap->ClearReinit();

		cv::Mat testImg = pF->GetOriginalImage().clone();
		cv::Mat testImg2 = pPrev->GetOriginalImage().clone();

		int nFail = 0;
		int nCandidate = 0;
		for (size_t i = 0, iend = vpCPs.size(); i < iend; i++) {
			auto pCPi = vpCPs[i];
			auto pMPi = pCPi->GetMP();
			
			/*if (pMPi && !pMPi->isDeleted())
				continue;*/

			auto pSeed = pCPi->mpSeed;
			if (!pSeed) {
				continue;
			}
			nCandidate++;
			auto pt = vPTs[i];
			cv::Mat X3D;
			float z;
			bool bNewMP = pCPi->CreateMapPoint(X3D, z, K, invK, Pcurr, Rcurr, Tcurr, pt);
			/*if (!bNewMP){
				nFail++;
				continue;
			}*/
			float invz = 1. / z;
			float z_inv_min = pSeed->mu + sqrt(pSeed->sigma2);
			float z_inv_max = max(pSeed->mu - sqrt(pSeed->sigma2), 0.00000001f);
			
			float z_min = 1. / z_inv_min;
			float z_max = 1. / z_inv_max;

			//if (invz > z_inv_min || invz < z_inv_max){
			//	//std::cout << z << " " << 1. / z_inv_min << ", " << 1. / z_inv_max <<"::"<<invz<<", "<<z_inv_min<<", "<<z_inv_max<< std::endl;
			//	continue;
			//}

			pSeed->count++;
			cv::Mat Rref, Tref;
			pCPi->mpRefKF->GetPose(Rref, Tref);
			cv::Mat Trefcurr = Rref*Tinvcurr + Tref;
			cv::Mat Rrefcurr = Rref*Rinvcurr;
			cv::Mat Rinv = Rrefcurr.t();
			cv::Mat Tinv = -Rinv*Trefcurr;
			cv::Mat tempRay1 = Rinv*(pSeed->ray*1./ pSeed->mu)+ Tinv;
			cv::Mat tempRay2 = Rinv*(pSeed->ray*z_min) + Tinv;
			cv::Mat tempRay3 = Rinv*(pSeed->ray*z_max) + Tinv;

			cv::Mat proj1 = K*tempRay1; //mu
			cv::Mat proj2 = K*tempRay2; //min
			cv::Mat proj3 = K*tempRay3; //max

			cv::Point2f projPt1(proj1.at<float>(0) / proj1.at<float>(2), proj1.at<float>(1) / proj1.at<float>(2));
			cv::Point2f projPt2(proj2.at<float>(0) / proj2.at<float>(2), proj2.at<float>(1) / proj2.at<float>(2));
			cv::Point2f projPt3(proj3.at<float>(0) / proj3.at<float>(2), proj3.at<float>(1) / proj3.at<float>(2));

			/*if (tempRay1.at<float>(2) < 0.0)
			{
				continue;
			}
			if (!pF->isInImage(projPt1.x, projPt1.y)) {
				continue;
			}*/
			int idx = pCPi->GetPointIndexInFrame(pPrev->mpMatchInfo);
			if(idx > -1 && !std::isnan(z_min) && !std::isnan(z_max) && z_min > 0.){
				cv::line(testImg, projPt2, projPt3, cv::Scalar(255, 255, 0), 3);
				cv::line(testImg, projPt1, pt, cv::Scalar(0, 255, 0), 1);
				cv::circle(testImg, projPt1, 3, cv::Scalar(0, 255, 255), -1);
				cv::circle(testImg, pt, 2, cv::Scalar(255, 0, 255), -1);

				cv::Point2f prevPt = pPrev->mpMatchInfo->mvMatchingPts[idx];
				cv::circle(testImg2, prevPt, 2, cv::Scalar(255, 0, 255), -1);
			}
			float tau = pSeed->ComputeTau(Trefcurr, z);
			float tau_inverse = 0.5 * (1.0 / max(0.0000001, z - tau) - 1.0 / (z + tau));
			UpdateSeed(pSeed, invz, tau_inverse*tau_inverse);

			float val = sqrt(pSeed->sigma2);
			bool bConverged = val < pSeed->z_range/ thresh;

			//std::cout << "depth test::" << pSeed->count << "::" << val << "=" << pSeed->z_range / thresh << std::endl; //1./z << ", " << z_inv_min << ", " << z_inv_max << std::endl;

			/*if(bConverged){
				std::cout << "depth filter : " << pSeed->count <<"::"<< val <<"="<< pSeed->z_range/ thresh << std::endl;
				cv::Mat Rrefinv = Rref.t();
				cv::Mat Trefinv = -Rrefinv*Tref;
				cv::Mat Xw = Rrefinv*(pSeed->ray*(1. / pSeed->mu)) + Trefinv;
				int label = pCPi->GetLabel();
				mpSystem->mpMap->AddReinit(X3D);
				auto pMP = new UVR_SLAM::MapPoint(mpSystem->mpMap, pF, pCPi, X3D, cv::Mat(), label, pCPi->octave);
				pMP->SetOptimization(true);
				mpSystem->mlpNewMPs.push_back(pMP);

				auto mmpFrames = pCPi->GetFrames();
				for (auto iter = mmpFrames.begin(); iter != mmpFrames.end(); iter++) {
					auto pMatch = iter->first;
					auto pKF = pMatch->mpRefFrame;
					int idx = iter->second;
					pMP->ConnectFrame(pMatch, idx);
				}

			}*/
			//update 함수 && convergence test && triangulation
		}//for
		
		std::cout << "DepthFilter::Update::end::" << nFail << ", " << nCandidate << std::endl;

		/*cv::Mat resized;
		cv::resize(testImg, resized, cv::Size(testImg.cols / 2, testImg.rows / 2));
		mpSystem->mpVisualizer->SetOutputImage(resized, 1);
		cv::moveWindow("Output::DepthFilter", mpSystem->mnDisplayX, mpSystem->mnDisplayY);
		cv::Mat debugImg = cv::Mat::zeros(testImg.rows * 2, testImg.cols, testImg.type());
		cv::Rect mergeRect1 = cv::Rect(0, 0, testImg.cols, testImg.rows);
		cv::Rect mergeRect2 = cv::Rect(0, testImg.rows, testImg.cols, testImg.rows);
		testImg.copyTo(debugImg(mergeRect1));
		testImg2.copyTo(debugImg(mergeRect2));
		imshow("Output::DepthFilter", debugImg);
		cv::waitKey(1);*/
	}
}