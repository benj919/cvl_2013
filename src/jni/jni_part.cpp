#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

using namespace std;

cv::Mat descriptors_old;
cv::Mat descriptors_new;
vector<cv::KeyPoint> v_old;
int selected_feature = 0;

// initial object capture
bool capture_frame = false;
int capture_idx = 0;
vector<vector<cv::KeyPoint> > initial_keypoints;
vector<vector<cv::Mat> > initial_descriptors;

extern "C" {
JNIEXPORT void JNICALL Java_org_natural_feature_tracking_nftActivity_SetFeature(JNIEnv*, jobject, jint feature_idx);

JNIEXPORT void JNICALL Java_org_natural_feature_tracking_nftActivity_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba);

JNIEXPORT void JNICALL Java_org_natural_feature_tracking_nftActivity_CaptureFrame(JNIEnv*, jobject, jint capture_idx);

JNIEXPORT void JNICALL Java_org_natural_feature_tracking_nftActivity_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
{
    cv::Mat& mGr  = *(cv::Mat*)addrGray;
    cv::Mat& mRgb = *(cv::Mat*)addrRgba;
    vector<cv::KeyPoint> v_new;
    std::vector<cv::DMatch> filteredMatches12, matches12, matches21;

    //FastFeatureDetector detector(50);
    cv::OrbFeatureDetector  detector(50);
    cv::OrbDescriptorExtractor extractor;
    cv::BFMatcher bf_matcher(cv::NORM_HAMMING, true);

    detector.detect(mGr, v_new);
    detector.compute(mRgb, v_new, descriptors_new);

    if (capture_frame){
    	initial_keypoints.push_back(v_new);
    	initial_descriptors.push_back(descriptors_new);
    	capture_frame = false;
    }

    if(descriptors_old.empty()){
    	descriptors_old = descriptors_new;
    	v_old = v_new;
    }

    bf_matcher.match( descriptors_old, descriptors_new, matches12 );
    bf_matcher.match( descriptors_new, descriptors_old, matches21 );

    for( size_t i = 0; i < matches12.size(); i++ )
    {
        cv::DMatch forward = matches12[i];
        cv::DMatch backward = matches21[forward.trainIdx];
        if( backward.trainIdx == forward.queryIdx
        		and forward.distance < 0.5)
            filteredMatches12.push_back( forward );
    }

    for( unsigned int i = 0; i < filteredMatches12.size(); i++ )
    {
    	cv::DMatch d = filteredMatches12[i];
        const cv::KeyPoint& kp_new = v_new[d.trainIdx];
        const cv::KeyPoint& kp_old = v_old[d.queryIdx];
        if(abs(kp_new.pt.x - kp_old.pt.x) < 20 and abs(kp_new.pt.y - kp_old.pt.y) < 20){
        	cv::circle(mRgb, cv::Point(kp_new.pt.x, kp_new.pt.y), 10, cv::Scalar(0,0,255,100));
        	cv::line(mRgb, cv::Point(kp_old.pt.x, kp_old.pt.y), cv::Point(kp_new.pt.x, kp_new.pt.y),cv::Scalar(0,255,0,255));
        }
    }


    v_old = v_new;
    descriptors_old = descriptors_new;
}

JNIEXPORT void JNICALL Java_org_natural_feature_tracking_nftActivity_SetFeature(JNIEnv*, jobject, jint feature_idx){
	selected_feature = feature_idx;
}

JNIEXPORT void JNICALL Java_org_natural_feature_tracking_nftActivity_CaptureFrame(JNIEnv*, jobject, jint capture_idx){
	capture_frame = true;
}

}
