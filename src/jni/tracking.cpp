#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

#include "tracking.hpp"

using namespace std;
using namespace cv;

tracking::tracking(){
	//kalman filter setup

	kalman = cvCreateKalman( 6, 3, 0 );//state(x,y,z,detaX,detaY,detaZ)
	process_noise = cvCreateMat( 6, 1, CV_32FC1 );
	measurement = cvCreateMat( 3, 1, CV_32FC1 );//measurement(x,y,z)
	rng = cvRNG(-1);


};

tracking::~tracking(){
	delete kalman;
	delete process_noise;
	delete measurement;
};

void tracking::initial_tracker(){
	A={//transition matrix
			1,0,0,0.5,0  ,0,
			0,1,0,0  ,0.5,0,
			0,0,1,0  ,0  ,0.5,
			0,0,0,1  ,0  ,0,
			0,0,0,0  ,1  ,0,
			0,0,0,0  ,0  ,1
	};

	memcpy( kalman->transition_matrix->data.fl,A,sizeof(A));
	cvSetIdentity(kalman->measurement_matrix,cvRealScalar(1) );
	cvSetIdentity(kalman->process_noise_cov,cvRealScalar(1e-5));
	cvSetIdentity(kalman->measurement_noise_cov,cvRealScalar(1e-1));
	cvSetIdentity(kalman->error_cov_post,cvRealScalar(1));
	//initialize post state of kalman filter at zero
	cvRandArr(&rng,kalman->state_post,CV_RAND_UNI,cvRealScalar(0),cvRealScalar(0));
}

Point3f tracking::pridict_tracker(){
	const CvMat* prediction=cvKalmanPredict(kalman,0);
	predict_pt=cvPoint((int)prediction->data.fl[0],(int)prediction->data.fl[1],(int)prediction->data.fl[2]);
	return predict_pt;
}

void tracking::update_measurement_tracker(Point3f realPosition){

	measurement->data.fl[0]=(float)realPosition.x;
	measurement->data.fl[1]=(float)realPosition.y;
	measurement->data.fl[2]=(float)realPosition.z;
}

void tracking::update_tracker(){
	cvKalmanCorrect( kalman, measurement );
}

Point3f tracking::track(Point3f realPosition){
//here still have same problem please try on your computer if your computer recognize it
	CvPoint predict=tracking::pridict_tracker();
	update_measurement_tracker(Point3f realPosition);
	update_tracker();
	return predict;

};
