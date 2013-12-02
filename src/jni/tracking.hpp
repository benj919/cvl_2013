#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>
#include <cmath>

#include <iostream>

#pragma once

class tracking{
	// features
public:

	CvKalman* kalman;
	CvMat* process_noise;
	CvMat* measurement;
	CvRNG rng;
	float A[4][4];
	CvPoint predict_pt;

	tracking();
	~tracking();

    void initial_tracker();
    void pridict_tracker();
    void update_measurement_tracker();
    void update_tracker();
	void track();

};
