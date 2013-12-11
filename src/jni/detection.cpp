#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <vector>

#include "detection.hpp"

#define f 690.3
#define px 320.0
#define py 240.0

detection::detection():
	initial_keypoints(0),
	initial_descriptors(0),
	raw_descriptors(0),
	initialized(false),
	detector(250),
	matcher(cv::NORM_HAMMING, true){

	K = *(cv::Mat_<float>(3,3) << f,0.0,px, 0.0,f,py, 0.0,0.0,1.0);
	K_inv = K.inv();
	//set up wire house
	set_up_house();
};

detection::~detection() {
};


void detection::extract_and_add_raw_features(cv::Mat& img){
	// img should be a ROI-Matrix representing the center of the frame targeted for object recognition
	// add descriptors to the initial raw set.
	// call setup_initial_features to calculate the initial set of features from the raw collection
	std::vector<cv::KeyPoint> keypoints;
	std::vector<cv::KeyPoint> nonmaxed_keypoints;
	cv::Mat descriptors;
	cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
	cv::Mat roi(mask, cv::Rect(220,140,500,340));
	roi = cv::Scalar(255);

	detector.detect(img, keypoints, mask);

	nonmaxed_keypoints = non_max_suppression(keypoints, 5);

	detector.compute(img, nonmaxed_keypoints, descriptors);

	initial_keypoints = nonmaxed_keypoints;
	raw_descriptors.push_back(descriptors);

	// for debugging
	setup_initial_features();
};

void detection::setup_initial_features(){
	// calculate the set of initial features (descriptors only) from the raw feature sets
	// Useless right now, we just take the first frame.
	std::vector<std::vector<cv::DMatch> > matches;
	initialized = true;
	if(raw_descriptors.size() == 0){
		//not enough frames for matching
	}
	else if(raw_descriptors.size() >= 1){
		initial_descriptors.push_back(raw_descriptors[0]);
	}
	else {
		//TODO not used right now
		initial_descriptors.reserve(2);
		// match and filter
		matcher.knnMatch( raw_descriptors[0], raw_descriptors[1], matches, 2 );
		for( unsigned int i = 0; i < matches.size(); i++ )
			{
				cv::DMatch m1 = matches[i][0];
				cv::DMatch m2 = matches[i][1];
				if(m1.distance < 0.6* m2.distance){
					//cv::Mat& v1, v2;
					// TODO single feature extraction from descriptor matrix :/
					//v1 = initial_descriptors[0];
					//v1.push_back(m1.trainIdx);
					//v2 = initial_descriptors[1];
					//v2.push_back(m1.queryIdx);
				}
			}
	}
};


bool detection::detect(cv::Mat& img){
	// try to find and track the initial features in the given image

	std::vector<cv::KeyPoint> keypoints_new;
	cv::Mat result;
	cv::Mat descriptors_new;
	std::vector<cv::DMatch> matches;

	detector.detect(img, keypoints_new);
	detector.compute(img, keypoints_new, descriptors_new);

	if(keypoints_new.size() == 0 or !initialized){
		// no features detected or not yet initialized
		return false;
	}

	matcher.match(initial_descriptors[0], descriptors_new, matches);

	if(matches.size() < 10){
		return false;
	}

	std::vector<cv::Point2f> initial_pts;
	std::vector<cv::Point2f> new_pts;

	for(int i = 0; i < matches.size(); i++ ){
		cv::DMatch& d = matches[i];
		cv::KeyPoint& init_pt = initial_keypoints[d.queryIdx];
		cv::KeyPoint& new_pt = keypoints_new[d.trainIdx];
		initial_pts.push_back(init_pt.pt);
		new_pts.push_back(new_pt.pt);
	}

	homography = cv::findHomography(initial_pts, new_pts, CV_RANSAC);

//	cv::RotatedRect box = cv::minAreaRect(cv::Mat(initial_pts));
//	cv::Point2f vertices[4], dst[4];
//	box.points(vertices);
//	cv::perspectiveTransform(vertices, dst, homography);
//	for (int i = 0; i < 4; ++i){
//	cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 1, CV_AA);
//	}

	return true;
};

void detection::warp_rectangle(cv::Mat& img){
	std::vector<cv::Point2f> src(4);
	std::vector<cv::Point2f> dst(4);

	src[0] = cv::Point(480,320);
	src[1] = cv::Point(480,160);
	src[2] = cv::Point(240,160);
	src[3] = cv::Point(240,320);

	cv::perspectiveTransform(src, dst, homography);

	cv::Scalar color(125,125,0);
	cv::line(img, dst[0], dst[1], color);
	cv::line(img, dst[1], dst[2], color);
	cv::line(img, dst[2], dst[3], color);
	cv::line(img, dst[3], dst[0], color);

	cv::Mat rvec, tvec, dist_coeffs;
	rvec.zeros(3,1,cv::DataType<float>::type);
	tvec.zeros(3,1,cv::DataType<float>::type);
	dist_coeffs.zeros(4,1,cv::DataType<float>::type);

	cv::Mat K = (cv::Mat_<double>(3,3) << 	1.0, 0.0, 0.0,
											0.0, 1.0, 0.0,
											0.0, 0.0, 1.0);
	//std::vector<cv::Point3f> base =

	//cv::solvePnP()
};

void detection::show_target_rectangle(cv::Mat& img, cv::Point2i top_left, cv::Point2i bottom_right){
	//add targeting rectangle for initial acquisition
	cv::rectangle(img, top_left, bottom_right, cv::Scalar(100,100,100,100), 2);
};

void detection::show_features(cv::Mat& img, std::vector<cv::KeyPoint>& points){
	// draw the points onto the image img
	for(int i = 0; i < points.size(); i++){
		cv::KeyPoint& tmp = points[i];
		cv::circle(img, cv::Point(tmp.pt.x, tmp.pt.y), 5, cv::Scalar(0,125,0,100));
	}
};

void detection::overlay_status_info(cv::Mat& img){
	std::stringstream str_stream;
	if(not raw_descriptors.empty()){
		cv::Mat tmp = raw_descriptors[0];
		str_stream << "# raw frames: " << raw_descriptors.size() << " # raw descr: " << tmp.size().height; //<< s.height;
	} else {
		str_stream << "no features captured";
	}
	cv::putText(img, str_stream.str() , cv::Point2i(50,460), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0)); //"# init ft: %i", initial_descriptors.size()
};

void detection::set_feature(int idx){
	//create new feature detector, extractor, and recalculated initial features if they are already there
	initial_keypoints.clear();
	initial_descriptors.clear();

	switch(idx){
	// TODO change to available descriptors/extractors
	case 0:
		// ORB
		break;
	case 1:
		//SURF
		break;
	case 2:
		//SIFT
		break;
	case 3:
		//STAR
		break;
	case 4:
		//MSER
		break;
	default:
		//whatever
		break;
	}

}

std::vector<cv::KeyPoint> detection::non_max_suppression(std::vector<cv::KeyPoint> keypoints, int max_dist){
	// sort the keypoints according to x values, then search for neighbours within max_dist pixels
	// finally only keep the point with max response.
	std::vector<cv::KeyPoint> result(0);
	//std::sort(keypoints.begin(), keypoints.end(), compare_keypoints());
	for(std::vector<cv::KeyPoint>::iterator cur_it = keypoints.begin(); cur_it != keypoints.end(); ++cur_it){
		bool keep = true;
		for(std::vector<cv::KeyPoint>::iterator cmp_it = keypoints.begin(); cmp_it != keypoints.end(); ++cmp_it){
			// brute force bad, yeah yeah
			if(cur_it == cmp_it ||
					std::abs( (*cur_it).pt.x - (*cmp_it).pt.x ) > max_dist ||
					std::abs( (*cur_it).pt.y - (*cmp_it).pt.y ) > max_dist ||
					(*cur_it).response > (*cmp_it).response ) {
				continue;
			}
			else{
				keep = false;
			}
		}
		if(keep){
			result.push_back(*cur_it);
		}
	}
	return result;
};

// could be used to get non_max_suppression more efficient. as its only used for initialization it doesn't matter to much.
bool compare_keypoints(const cv::KeyPoint& a, const cv::KeyPoint& b){
	return a.pt.x > b.pt.x;
};

void detection::set_up_house(){
	std::vector<int>& fu = house_edges[0];

	house_vertices.push_back(cv::Point3f(0.0,0.0,0.0));
	house_vertices.push_back(cv::Point3f(0.0,1.0,0.0));
	house_vertices.push_back(cv::Point3f(1.0,1.0,0.0));
	house_vertices.push_back(cv::Point3f(1.0,0.0,0.0));
	house_vertices.push_back(cv::Point3f(0.0,0.0,1.0));
	house_vertices.push_back(cv::Point3f(0.0,1.0,1.0));
	house_vertices.push_back(cv::Point3f(1.0,1.0,1.0));
	house_vertices.push_back(cv::Point3f(1.0,0.0,1.0));
	house_vertices.push_back(cv::Point3f(0.0,0.5,2.0));
	house_vertices.push_back(cv::Point3f(1.0,0.5,2.0));
	house_edges.resize(10);
	fu = house_edges[0]; fu.push_back(1); fu.push_back(4);
	fu = house_edges[1]; fu.push_back(2); fu.push_back(5);
	fu = house_edges[2]; fu.push_back(3); fu.push_back(6);
	fu = house_edges[3]; fu.push_back(0); fu.push_back(7);
	fu = house_edges[4]; fu.push_back(5); fu.push_back(8);
	fu = house_edges[5]; fu.push_back(6); fu.push_back(8);
	fu = house_edges[6]; fu.push_back(7); fu.push_back(9);
	fu = house_edges[7]; fu.push_back(4); fu.push_back(9);
	fu = house_edges[8]; fu.push_back(9);

};
