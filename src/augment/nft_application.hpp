#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

#include "tracking.hpp"
#include "detection.hpp"

class nft_application{
	// features
public:
	// initial object capture
	bool object_aquisition;
	bool capt_frame;
	bool show_status_info;
	bool show_target_rectangle;
	bool track;
	int capture_idx;
	tracking* tracker;
	detection* detector;


	nft_application();
	~nft_application();
	//changed by hongyi
	void process_frame(cv::Mat& rgba, cv::Mat& gray);
	// process image

	void set_feature(int feature);
	// set feature type

	void capture_frame();
	// capture single frame

	void show_info(bool status);
	// disable/enable info overlay

	void object_acquisition(bool status);
	// enter/leave target mode

};
