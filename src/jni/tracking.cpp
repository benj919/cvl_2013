#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

#include "tracking.hpp"

using namespace std;
using namespace cv;

tracking::tracking(){
	//kalman filter setup

	kalman = cv::KalmanFilter( 6, 3, 0 );   //state(x,y,z,deltaX,deltaY,deltaZ)
	process_noise = cv::Mat( 6, 1, CV_32FC1 );
	measurement = cv::Mat( 3, 1, CV_32FC1 );//measurement(x,y,z)
	rng = cv::RNG(-1);


};

tracking::~tracking(){
	//delete kalman;
	//delete process_noise;
	//delete measurement;
};

void tracking::initial_tracker(){

	kalman.transitionMatrix = (cv::Mat_<float>(6,6) <<
			1,0,0,1,0,0,
			0,1,0,0,1,0,
			0,0,1,0,0,1,
			0,0,0,1,0,0,
			0,0,0,0,1,0,
			0,0,0,0,0,1);
	cv::setIdentity(kalman.measurementMatrix,cv::Scalar(1) );
	cv::setIdentity(kalman.processNoiseCov,cv::Scalar(1e-5));
	cv::setIdentity(kalman.measurementNoiseCov,cv::Scalar(1e-1));
	cv::setIdentity(kalman.errorCovPost,cv::Scalar(1));
	//initialize post state of kalman filter at zero
	cv::randn(kalman.statePost, cv::Scalar(0), cv::Scalar(0.1));
}

void tracking::predict_tracker(){
	const cv::Mat prediction=kalman.predict();
	//cv::Point3f predict_pt=cv::Point3f(prediction->data.fl);

}

void tracking::update_measurement_tracker(Point3f realPosition){
	measurement = (cv::Mat_<float>(3, 1) <<
			(float)realPosition.x,(float)realPosition.y,(float)realPosition.z);
}

cv::Point3f tracking::update_tracker(){
	const cv::Mat prediction=kalman.correct(measurement);
	Point3f predict_pt;
	predict_pt.x = prediction.at<float>(0);
	predict_pt.y = prediction.at<float>(1);
	predict_pt.z = prediction.at<float>(2);
	return predict_pt;
}

cv::Point3f tracking::track(cv::Point3f realPosition){

	predict_tracker();
	update_measurement_tracker(realPosition);
	cv::Point3f predict=update_tracker();
	return predict;

};
