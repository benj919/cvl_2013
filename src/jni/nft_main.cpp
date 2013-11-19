#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

#include<android/log.h>

#include "nft_application.hpp"

using namespace std;

nft_application* application; // = new nft_application();

extern "C" {

// Function definitions for the compiler to recognize them
JNIEXPORT void JNICALL Java_org_nft_nftActivity_ObjectAquisition(JNIEnv*, jobject, jboolean aquisition);

JNIEXPORT void JNICALL Java_org_nft_nftActivity_ShowStatusInfo(JNIEnv*, jobject, jboolean status);

JNIEXPORT void JNICALL Java_org_nft_nftActivity_ShowTargetRectangle(JNIEnv*, jobject, jboolean rectangle);

JNIEXPORT void JNICALL Java_org_nft_nftActivity_InitializeDetector(JNIEnv*, jobject);

JNIEXPORT void JNICALL Java_org_nft_nftActivity_SetFeature(JNIEnv*, jobject, jint feature_idx);

JNIEXPORT void JNICALL Java_org_nft_nftActivity_ProcessFrame(JNIEnv*, jobject, jlong matAddrGray, jlong matAddrRgba);

JNIEXPORT void JNICALL Java_org_nft_nftActivity_CaptureFrame(JNIEnv*, jobject, jint capture_idx);

// Function implementations
JNIEXPORT void JNICALL Java_org_nft_nftActivity_ProcessFrame(JNIEnv*, jobject, jlong matAddrGray, jlong matAddrRgba)
{
    cv::Mat& mGr  = *(cv::Mat*)matAddrGray;
    cv::Mat& mRgb = *(cv::Mat*)matAddrRgba;

    application->process_frame(mRgb, mGr);
}
JNIEXPORT void JNICALL Java_org_nft_nftActivity_InitializeDetector(JNIEnv*, jobject){
	// setup detection object/"framework"
	application = new nft_application();
}

JNIEXPORT void JNICALL Java_org_nft_nftActivity_SetFeature(JNIEnv*, jobject, jint feature_idx){
	application->set_feature(feature_idx);
}

JNIEXPORT void JNICALL Java_org_nft_nftActivity_CaptureFrame(JNIEnv*, jobject, jint capture_idx){
	application->capture_frame();
}
JNIEXPORT void JNICALL Java_org_nft_nftActivity_ShowStatusInfo(JNIEnv*, jobject, jboolean status){
	application->show_info(status);
}


JNIEXPORT void JNICALL Java_org_nft_nftActivity_ObjectAquisition(JNIEnv*, jobject, jboolean acquisition){
	application->object_acquisition(acquisition);
}

}
