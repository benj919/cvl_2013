#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

#include "tracking.hpp"

using namespace std;
using namespace cv;

tracking::tracking(){
	//kalman filter setup

	kalman = cvCreateKalman( 4, 2, 0 );//state(x,y,detaX,detaY)
	process_noise = cvCreateMat( 4, 1, CV_32FC1 );
	measurement = cvCreateMat( 2, 1, CV_32FC1 );//measurement(x,y)
	rng = cvRNG(-1);


};

tracking::~tracking(){
	delete kalman;
	delete process_noise;
	delete measurement;
};

void tracking::initial_tracker(){
    //question1 about WHY this A always complain!
	A[4][4]={//transition matrix
		{1,0,1,0},
		{0,1,0,1},
		{0,0,1,0},
		{0,0,0,1}
	};

	memcpy( kalman->transition_matrix->data.fl,A,sizeof(A));
	cvSetIdentity(kalman->measurement_matrix,cvRealScalar(1) );
	cvSetIdentity(kalman->process_noise_cov,cvRealScalar(1e-5));
	cvSetIdentity(kalman->measurement_noise_cov,cvRealScalar(1e-1));
	cvSetIdentity(kalman->error_cov_post,cvRealScalar(1));
	//initialize post state of kalman filter at random
	//coordinate1 about image size
	Size imageSize;
	imageSize = Size((int) mGr.get(CV_CAP_PROP_FRAME_WIDTH), (int) mGr.get(CV_CAP_PROP_FRAME_HEIGHT));
	cvRandArr(&rng,kalman->state_post,CV_RAND_UNI,cvRealScalar(0),cvRealScalar(imageSize.height>imageSize.width?imageSize.width:imageSize.height));
}

void tracking::pridict_tracker(){
	const CvMat* prediction=cvKalmanPredict(kalman,0);
	predict_pt=cvPoint((int)prediction->data.fl[0],(int)prediction->data.fl[1]);
}

void tracking::update_measurement_tracker(){
	//coordinate2 about measurement value
	measurement->data.fl[0]=(float)newposition.x;
	measurement->data.fl[1]=(float)newposition.y;
}

void tracking::update_tracker(){
	cvKalmanCorrect( kalman, measurement );
}

void tracking::track(){

};
