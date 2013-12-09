#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#pragma once


class tracking{
	// features
public:

	cv::KalmanFilter kalman;
	cv::Mat process_noise;
	cv::Mat measurement;
	cv::RNG rng;
	cv::Point3f predict_pt;

	tracking();
	~tracking();

    void initial_tracker();
    cv::Point3f predict_tracker();
    void update_measurement_tracker(cv::Point3f realPosition);
    void update_tracker();
    cv::Point3f track(cv::Point3f realPosition);

};
