#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

using namespace std;
int selected_feature = 0;

// initial object capture
bool capture_frame = false;
int capture_idx = 0;
vector<vector<cv::KeyPoint> > initial_keypoints;
vector<cv::Mat> initial_descriptors;

extern "C" {
// Function definitions for the compiler to recognize them
JNIEXPORT void JNICALL Java_org_nft_nftActivity_SetFeature(JNIEnv*, jobject, jint feature_idx);

JNIEXPORT void JNICALL Java_org_nft_nftActivity_FindFeatures(JNIEnv*, jobject, jlong matAddrGray, jlong matAddrRgba);

JNIEXPORT void JNICALL Java_org_nft_nftActivity_CaptureFrame(JNIEnv*, jobject, jint capture_idx);

// Function implementations
JNIEXPORT void JNICALL Java_org_nft_nftActivity_FindFeatures(JNIEnv*, jobject, jlong matAddrGray, jlong matAddrRgba)
{
    cv::Mat& mGr  = *(cv::Mat*)matAddrGray;
    cv::Mat& mRgb = *(cv::Mat*)matAddrRgba;
    vector<cv::KeyPoint> v_new;
    cv::Mat descriptors_new;
    std::vector<cv::DMatch> matches;

    //FastFeatureDetector detector(50);
    cv::OrbFeatureDetector  detector(250);
    cv::OrbDescriptorExtractor extractor;
    cv::BFMatcher bf_matcher(cv::NORM_HAMMING, true);

    detector.detect(mGr, v_new);
    detector.compute(mRgb, v_new, descriptors_new);

    if (capture_frame){
    	initial_keypoints.push_back(v_new);
    	initial_descriptors.push_back(descriptors_new);
    	capture_frame = false;
    	return;
    }

    if(initial_descriptors.empty()){
    	return;
    }

    bf_matcher.match( initial_descriptors[0], descriptors_new, matches );

    for( unsigned int i = 0; i < matches.size(); i++ )
    {
    	cv::DMatch d = matches[i];
        const cv::KeyPoint& kp_new = v_new[d.trainIdx];
        const cv::KeyPoint& kp_old = initial_keypoints[0][d.queryIdx];
        if(abs(kp_new.pt.x - kp_old.pt.x) < 20 and abs(kp_new.pt.y - kp_old.pt.y) < 20 and d.distance < 0.5){
        	cv::circle(mRgb, cv::Point(kp_new.pt.x, kp_new.pt.y), 10, cv::Scalar(0,0,255,100));
        	cv::line(mRgb, cv::Point(kp_old.pt.x, kp_old.pt.y), cv::Point(kp_new.pt.x, kp_new.pt.y),cv::Scalar(0,255,0,255));
        }
    }
}

JNIEXPORT void JNICALL Java_org_nft_nftActivity_SetFeature(JNIEnv*, jobject, jint feature_idx){
	selected_feature = feature_idx;
}

JNIEXPORT void JNICALL Java_org_nft_nftActivity_CaptureFrame(JNIEnv*, jobject, jint capture_idx){
	capture_frame = true;
}

}
