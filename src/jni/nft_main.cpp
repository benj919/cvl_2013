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
bool capture_frame = false;
int capture_idx = 0;
vector<vector<cv::KeyPoint> > initial_keypoints;
vector<cv::Mat> initial_descriptors;
detection* detector;

extern "C" {

// Function definitions for the compiler to recognize them
JNIEXPORT void JNICALL Java_org_nft_nftActivity_InitializeDetector(JNIEnv*, jobject);

JNIEXPORT void JNICALL Java_org_nft_nftActivity_SetFeature(JNIEnv*, jobject, jint feature_idx);

JNIEXPORT void JNICALL Java_org_nft_nftActivity_FindFeatures(JNIEnv*, jobject, jlong matAddrGray, jlong matAddrRgba);

JNIEXPORT void JNICALL Java_org_nft_nftActivity_CaptureFrame(JNIEnv*, jobject, jint capture_idx);

// Function implementations
JNIEXPORT void JNICALL Java_org_nft_nftActivity_FindFeatures(JNIEnv*, jobject, jlong matAddrGray, jlong matAddrRgba)
{
    cv::Mat& mGr  = *(cv::Mat*)matAddrGray;
    cv::Mat& mRgb = *(cv::Mat*)matAddrRgba;
    std::vector<cv::KeyPoint> new_keypoints;

    if (capture_frame){
    	detector->extract_and_add_raw_features(mGr);
    	if(detector->raw_descriptors.size() == 2){
    		detector->setup_initial_features();
    	}
    	capture_frame = false;
    	return;
    }

    new_keypoints = detector->track(mGr);

    for( unsigned int i = 0; i < new_keypoints.size(); i++ )
    {
    	cv::KeyPoint p = new_keypoints[i];
    	cv::circle(mRgb, cv::Point(p.pt.x, p.pt.y), 10, cv::Scalar(0,0,255,100));
    }
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

}
