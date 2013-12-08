#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

#include "tracking.hpp"

using namespace std;
using namespace cv;

tracking::tracking(){
	//kalman filter setup

	kalman = *cv::KalmanFilter( 6, 3, 0 );//state(x,y,z,deltaX,deltaY,deltaZ)
	process_noise = cv::Mat( 6, 1, CV_32FC1 );
	measurement = cv::Mat( 3, 1, CV_32FC1 );//measurement(x,y,z)
	rng = cv::RNG(-1);


};

tracking::~tracking(){
	delete kalman;
	delete process_noise;
	delete measurement;
};

void tracking::initial_tracker(){

	kalman->transitionMatrix = *(cv::Mat_<float>(6,6) <<
			1,0,0,1,0,0,
			0,1,0,0,1,0,
			0,0,1,0,0,1,
			0,0,0,1,0,0,
			0,0,0,0,1,0,
			0,0,0,0,0,1);
	cv::setIdentity(kalman->measurementMatrix,cv::Scalar(1) );
	cv::setIdentity(kalman->processNoiseCov,cv::Scalar(1e-5));
	cv::setIdentity(kalman->measurementNoiseCov,cv::Scalar(1e-1));
	cv::setIdentity(kalman->errorCovPost,cv::Scalar(1));
	//initialize post state of kalman filter at zero
	cv::randn(kalman->statePost, cv::Scalar(0), cv::Scalar(0.1));
}

cv::Point3f tracking::predict_tracker(){
	const cv::Mat* prediction=kalman->predict();
	//cv::Point3f predict_pt=cv::Point3f(prediction->data.fl);
	return predict_pt;
}

void tracking::update_measurement_tracker(Point3f realPosition){
	measurement->data.fl[0]=(float)realPosition.x;
	measurement->data.fl[1]=(float)realPosition.y;
	measurement->data.fl[2]=(float)realPosition.z;
}

void tracking::update_tracker(){
	kalman->correct(measurement);
}

cv::Point3f tracking::track(cv::Point3f realPosition){
//here still have same problem please try on your computer if your computer recognize it
	cv::Point3f predict=tracking::predict_tracker();
	update_measurement_tracker(realPosition);
	update_tracker();
	return predict;

};
