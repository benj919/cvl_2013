#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

#pragma once

class detection{
	// features
public:
    cv::OrbFeatureDetector  detector;
    cv::BFMatcher matcher;
    std::vector<cv::KeyPoint> initial_keypoints;
    std::vector<cv::Mat> initial_descriptors;
    std::vector<cv::Mat> raw_descriptors;
    cv::Point2i previous_location;
    bool initialized;

	// init & destructor
	detection();
	~detection();


	// functionality
	std::vector<cv::KeyPoint> detect_keypoints(cv::Mat& img);

	// setup feature tracking
	void extract_and_add_raw_features(cv::Mat& img);
	void setup_initial_features();

	//feature tracking
	// allow multiple trackings?
	cv::Mat detect(cv::Mat& img);

	void show_target_rectangle(cv::Mat& img, cv::Point2i top_left, cv::Point2i bottom_right);
	// for aquiring the raw features: the desired object to track should be in the center of the frame

	void show_features(cv::Mat& img, std::vector<cv::KeyPoint>& points);
	// displays some features in the image

	void overlay_status_info(cv::Mat& img);
	// overlay image with status info (number of key frames, number of features etc...)

	void set_feature(int idx);

	std::vector<cv::KeyPoint> non_max_suppression(std::vector<cv::KeyPoint> keypoints, int max_dist);
	// non-maximum suppression for feature detectors which lack this;
	// should probably be used for initial feature setup only
};
