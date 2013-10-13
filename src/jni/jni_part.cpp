#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

using namespace std;
using namespace cv;

cv::Mat descriptors_old;
cv::Mat descriptors_new;
vector<KeyPoint> v_old;

extern "C" {
JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial2_Tutorial2Activity_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba);

JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial2_Tutorial2Activity_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
{
    Mat& mGr  = *(Mat*)addrGray;
    Mat& mRgb = *(Mat*)addrRgba;
    vector<KeyPoint> v_new;
    std::vector<cv::DMatch> matches;

    //FastFeatureDetector detector(50);
    OrbFeatureDetector  detector(50);
    OrbDescriptorExtractor extractor;
    cv::BFMatcher bf_matcher(cv::NORM_HAMMING, true);

    detector.detect(mGr, v_new);
    extractor.compute(mRgb, v_new, descriptors_new);

    if(descriptors_old.empty()){
    	descriptors_old = descriptors_new;
    	v_old = v_new;
    }

    bf_matcher.match(descriptors_old, descriptors_new, matches);

    for( unsigned int i = 0; i < matches.size(); i++ )
    {
    	DMatch d = matches[i];
        const KeyPoint& kp_new = v_new[d.trainIdx];
        const KeyPoint& kp_old = v_old[d.queryIdx];
        circle(mRgb, Point(kp_new.pt.x, kp_new.pt.y), 10, Scalar(0,0,255,100));
        line(mRgb, Point(kp_old.pt.x, kp_old.pt.y), Point(kp_new.pt.x, kp_new.pt.y),Scalar(0,255,0,255));
    }

    v_old = v_new;
    descriptors_old = descriptors_new;
}
}
