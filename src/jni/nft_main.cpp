#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

#include<android/log.h>

#include "detection.hpp"

using namespace std;
int selected_feature = 0;

// initial object capture
bool object_aquisition = false;
bool capture_frame = false;
bool show_status_info = false;
bool tracking = false;
int capture_idx = 0;
detection* detector;

extern "C" {

// Function definitions for the compiler to recognize them
JNIEXPORT void JNICALL Java_org_nft_nftActivity_ObjectAquisition(JNIEnv*, jobject, jboolean aquisition);

JNIEXPORT void JNICALL Java_org_nft_nftActivity_TogleStatusInfo(JNIEnv*, jobject);

JNIEXPORT void JNICALL Java_org_nft_nftActivity_InitializeDetector(JNIEnv*, jobject);

JNIEXPORT void JNICALL Java_org_nft_nftActivity_SetFeature(JNIEnv*, jobject, jint feature_idx);

JNIEXPORT void JNICALL Java_org_nft_nftActivity_ProcessFrame(JNIEnv*, jobject, jlong matAddrGray, jlong matAddrRgba);

JNIEXPORT void JNICALL Java_org_nft_nftActivity_CaptureFrame(JNIEnv*, jobject, jint capture_idx);

// Function implementations
JNIEXPORT void JNICALL Java_org_nft_nftActivity_ProcessFrame(JNIEnv*, jobject, jlong matAddrGray, jlong matAddrRgba)
{
    cv::Mat& mGr  = *(cv::Mat*)matAddrGray;
    cv::Mat& mRgb = *(cv::Mat*)matAddrRgba;
    std::vector<cv::KeyPoint> new_keypoints;

    if (capture_frame){
    	detector->extract_and_add_raw_features(mGr);
//    	if(detector->raw_descriptors.size() == 2){
//    		detector->setup_initial_features();
//    	}
    	capture_frame = false;
    	return;
    }

    new_keypoints = detector->track(mGr);

	detector->show_features(mRgb, new_keypoints);

	if(show_status_info){
		detector->overlay_status_info(mRgb);
	}

	detector->add_target_rectangle(mRgb, cv::Point2i(480,320), cv::Point2i(240,160));
}
JNIEXPORT void JNICALL Java_org_nft_nftActivity_InitializeDetector(JNIEnv*, jobject){
	// setup detection object/"framework"
	detector = new detection();
}

JNIEXPORT void JNICALL Java_org_nft_nftActivity_SetFeature(JNIEnv*, jobject, jint feature_idx){
	selected_feature = feature_idx;
}

JNIEXPORT void JNICALL Java_org_nft_nftActivity_CaptureFrame(JNIEnv*, jobject, jint capture_idx){
	capture_frame = true;
}
JNIEXPORT void JNICALL Java_org_nft_nftActivity_TogleStatusInfo(JNIEnv*, jobject){
	show_status_info = !show_status_info;
}


JNIEXPORT void JNICALL Java_org_nft_nftActivity_ObjectAquisition(JNIEnv*, jobject, jboolean aquisition){
	object_aquisition = aquisition;
}

}
