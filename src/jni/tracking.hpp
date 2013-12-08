#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>
#include <cmath>
#include <opencv2/legacy/legacy.hpp>
//#include <opencv2/core/matx.hpp>
#include <iostream>

#pragma once


class tracking{
	// features
public:

	CvKalman* kalman;
	CvMat* process_noise;
	CvMat* measurement;
	CvRNG rng;
	//problem that it never recognize Matx66f, I am sure it is in documentation, but not work, try to include matx.hpp but not successful
	Matx66f A;
	Point3f predict_pt;

	tracking();
	~tracking();

    void initial_tracker();
    Point3f pridict_tracker();
    void update_measurement_tracker(Point3f realPosition);
    void update_tracker();
    Point3f track(Point3f realPosition);

};
