package org.nft;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SubMenu;
import android.view.WindowManager;

public class nftActivity extends Activity implements CvCameraViewListener2 {
    private static final String    TAG = "OCVnft::Activity";

    private static final int       VIEW_MODE_FT_SELECTION     = 0;
    private static final int       VIEW_MODE_CAPTURE     = 1;
    private static final int       VIEW_MODE_INFO    = 2;
    private static final int       VIEW_MODE_FEATURES = 5;
    
    private int 				   selected_feature;
    private static final int	   FEATURE_ORB = 0;
    private static final int	   FEATURE_SURF = 1;
    private static final int	   FEATURE_SIFT = 2;
    private static final int	   FEATURE_STAR = 3;
    private static final int	   FEATURE_MSER = 4;

    private int                    mViewMode;
    private Mat                    mRgba;
    private Mat                    mIntermediateMat;
    private Mat                    mGray;

    private MenuItem               mItemFeatureSelection;
    private MenuItem[] 			   mFeatureMenuItems;
    private SubMenu 			   mFeatureMenu;
    
    private MenuItem               mItemPreviewCapture;
    private MenuItem               mItemPreviewCanny;
    private MenuItem               mItemPreviewFeatures;

    private CameraBridgeViewBase   mOpenCvCameraView;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("mixed_sample");
                    InitializeDetector();
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public nftActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.nft_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.nft_activity_surface_view);
        mOpenCvCameraView.setMaxFrameSize(720, 480);
        mOpenCvCameraView.setCvCameraViewListener(this);
        
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        //mItemFeatureSelection = menu.add("Features");
        mFeatureMenu = menu.addSubMenu("Features");
        mFeatureMenuItems = new MenuItem[5];
        mFeatureMenuItems[0] = mFeatureMenu.add(2, FEATURE_ORB, Menu.NONE, "ORB");
        mFeatureMenuItems[1] = mFeatureMenu.add(2, FEATURE_SURF, Menu.NONE, "SURF");
        mFeatureMenuItems[2] = mFeatureMenu.add(2, FEATURE_SIFT, Menu.NONE, "SIFT");
        mFeatureMenuItems[3] = mFeatureMenu.add(2, FEATURE_STAR, Menu.NONE, "STAR");
        mFeatureMenuItems[4] = mFeatureMenu.add(2, FEATURE_MSER, Menu.NONE, "MSER");
        
        mItemPreviewCapture = menu.add("Capture");
        mItemPreviewCanny = menu.add("Info");
        mItemPreviewFeatures = menu.add("Find features");
        return true;
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_6, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
    }

    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        mIntermediateMat.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        final int viewMode = mViewMode;
        switch (viewMode) {
        case VIEW_MODE_FT_SELECTION:
            // input frame has RBGA format
            mRgba = inputFrame.rgba();
            break;
        case VIEW_MODE_INFO:
            // input frame has gray scale format
            mRgba = inputFrame.rgba();
            mViewMode = VIEW_MODE_FEATURES;
            TogleStatusInfo();
            break;
        case VIEW_MODE_CAPTURE:
            // input frame has gray scale format
            CaptureFrame(0);
            mRgba = inputFrame.rgba();
            mViewMode = VIEW_MODE_FEATURES;
            break;
        case VIEW_MODE_FEATURES:
            // input frame has RGBA format
            mRgba = inputFrame.rgba();
            mGray = inputFrame.gray();
            ProcessFrame(mGray.getNativeObjAddr(), mRgba.getNativeObjAddr());
            break;
        }

        return mRgba;
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);

        if (item.getGroupId() == 1){
            selected_feature = item.getItemId();
            SetFeature(selected_feature);
            mViewMode = VIEW_MODE_FT_SELECTION;
        } else if (item == mItemPreviewCapture) {
            mViewMode = VIEW_MODE_FT_SELECTION; //VIEW_MODE_FEATURES;
        } else if (item == mItemPreviewCanny) {
            mViewMode = VIEW_MODE_INFO;
        } else if (item == mItemPreviewFeatures) {
            mViewMode = VIEW_MODE_FEATURES;
        }

        return true;
    }
    public native void ObjectAquisition(boolean aquisition);
    public native void TogleStatusInfo();
    public native void InitializeDetector();
    public native void SetFeature(int feature_idx);
    public native void ProcessFrame(long matAddrGray, long matAddrRgba);
    public native void CaptureFrame(int capture_idx);
}
