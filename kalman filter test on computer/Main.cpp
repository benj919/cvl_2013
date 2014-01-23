#include "tracking.cpp"
#include "tracking.hpp"
#include <vector>
#include <iostream>
using namespace std;

const int winHeight=600;
const int winWidth=800;
bool farenough=false;

Point3f mousePosition=Point3f(winWidth>>1,winHeight>>1,0);

//mouse event callback

void mouseEvent(int event, int x, int y, int flags, void *param )
{
	if (event==CV_EVENT_MOUSEMOVE) {
		mousePosition=Point3f((float)x,(float)y,0);
	}
}

int main (void)
{
	//1.kalman filter setup
	tracking track;

	track.initial_tracker();
	CvFont font;
	cvInitFont(&font,CV_FONT_HERSHEY_SCRIPT_COMPLEX,1,1);

	cvNamedWindow("kalman");
	cvSetMouseCallback("kalman",mouseEvent);
	IplImage* img=cvCreateImage(cvSize(winWidth,winHeight),8,3);
	while (1){
		//2.kalman prediction
		track.predict_tracker();

    	//3.update measurement
		track.update_measurement_tracker(mousePosition);

		//4.update
		Point3f predict_pt=track.update_tracker();		

		//draw 
		cvSet(img,cvScalar(255,255,255,0));

		CvPoint measurementer;
		measurementer.x=(int)mousePosition.x;
		measurementer.y=(int)mousePosition.y;
		cvCircle(img,measurementer,5,CV_RGB(0,255,0),3);//current position with red
		char buf[256];
		sprintf_s(buf,256,"annoying kalman position:(%3d,%3d)",predict_pt.x,predict_pt.y);
		cvPutText(img,buf,cvPoint(10,30),&font,CV_RGB(0,0,0));
		sprintf_s(buf,256,"your position :(%3d,%3d)",mousePosition.x,mousePosition.y);
		cvPutText(img,buf,cvPoint(10,60),&font,CV_RGB(0,0,0));
		sprintf_s(buf,256,"can you get rid of kalman?");
		cvPutText(img,buf,cvPoint(10,570),&font,CV_RGB(0,0,0));
		if (((predict_pt.x-mousePosition.x)*(predict_pt.x-mousePosition.x))+((predict_pt.y-mousePosition.y)*(predict_pt.y-mousePosition.y))>16000){
			farenough=true;
			CvPoint predicter;
			predicter.x=(int)predict_pt.x;
			predicter.y=(int)predict_pt.y;
			cvCircle(img,predicter,5,CV_RGB(255,0,0),3);//current position with red
		}
		else{
			farenough=false;
			CvPoint predicter;
			predicter.x=(int)predict_pt.x;
			predicter.y=(int)predict_pt.y;
			cvCircle(img,predicter,5,CV_RGB(0,255,0),3);//current position with green
			sprintf_s(buf,256,"kalman: haha! slow boy!");
		    cvPutText(img,buf,cvPoint(10,90),&font,CV_RGB(0,0,0));
		}
		if (farenough){
		sprintf_s(buf,256,"kalman: Wait!!!");
		cvPutText(img,buf,cvPoint(10,90),&font,CV_RGB(0,0,0));
	}


		cvShowImage("kalman", img);
		int key=cvWaitKey(3);
		if (key==27){//esc   
			break;   
		}
	}      

	cvReleaseImage(&img);
	return 0;
}

