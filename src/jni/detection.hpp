#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

class detection{
	// features
public:
    cv::OrbFeatureDetector  detector;
    cv::BFMatcher matcher;
    std::vector<cv::KeyPoint> initial_keypoints;
    std::vector<cv::Mat> initial_descriptors;
    std::vector<cv::Mat> raw_descriptors;

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
	std::vector<cv::KeyPoint> track(cv::Mat& img);

	void add_target_rectangle(cv::Mat& img, cv::Point2i top_left, cv::Point2i bottom_right);
	// for aquiring the raw features: the desired object to track should be in the center of the frame
};
