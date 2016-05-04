#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include "dirent.h"

using namespace std;
using namespace cv;


/*
 The cascade classifiers that come with opencv are kept in the
 following folder: bulid/etc/haarscascades
 Set OPENCV_ROOT to the location of opencv in your system
 */
//string OPENCV_ROOT = "C:/opencv/";
//string cascades = OPENCV_ROOT + "build/etc/haarcascades/";
//string FACES_CASCADE_NAME = cascades + "haarcascade_frontalface_alt.xml";
string FACES_CASCADE_NAME = "/Users/sathishgp/Downloads/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt.xml";

/*  The mouth cascade is assumed to be in the local folder */
string MOUTH_CASCADE_NAME = "Mouth.xml";
string FINGER_CASCADE_NAME = "cascade.xml";

void drawEllipse(Mat frame, const Rect rect, int r, int g, int b) {
    int width2 = rect.width/2;
    int height2 = rect.height/2;
    Point center(rect.x + width2, rect.y + height2);
    ellipse(frame, center, Size(width2, height2), 0, 0, 360,
            Scalar(r, g, b), 2, 8, 0 );
}


vector<Rect> detectSilence(Mat frame, Point location, Mat ROI, CascadeClassifier cascade)
{
    // frame,location are used only for drawing the detected mouths
    vector<Rect> mouths;
    cascade.detectMultiScale(ROI, mouths, 1.2, 4, 0|CV_HAAR_SCALE_IMAGE, Size(24, 24));
    
    int nmouths = (int)mouths.size();
    for( int i = 0; i < nmouths ; i++ ) {
        Rect mouth_i = mouths[i];
        drawEllipse(frame, mouth_i + location, 255, 255, 0);
    }
    
    return(mouths);
}


int detectMouth(Mat frame, Point location, vector<Rect> mouths, CascadeClassifier cascade)
{
    // frame,location are used only for drawing the detected mouths
    vector<Rect> fingers;
    int total =0;
    for(int j=0; j < (int)mouths.size(); j++){
    Mat frame_gray;
    Rect mouth = mouths[j];
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    Mat ROI = frame_gray(mouth);
    cascade.detectMultiScale(ROI, fingers, 1.2, 4, 0|CV_HAAR_SCALE_IMAGE, Size(40, 40));
    
    int nfingers = (int)fingers.size();
    total+= nfingers;
    for( int i = 0; i < nfingers ; i++ ) {
        Rect finger_i = fingers[i];
        drawEllipse(frame, finger_i + location, 100, 100, 100);
    }
    }
    
    return(total == 0);
}



// you need to rewrite this function
int detect(Mat frame,
           CascadeClassifier cascade_face, CascadeClassifier cascade_mouth, CascadeClassifier cascade_hand) {
    Mat frame_gray;
    vector<Rect> faces;
    
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    
    //  equalizeHist(frame_gray, frame_gray); // input, outuput
    //  medianBlur(frame_gray, frame_gray, 5); // input, output, neighborhood_size
    //  blur(frame_gray, frame_gray, Size(5,5), Point(-1,-1));
    /*  input,output,neighborood_size,center_location (neg means - true center) */
    
    
    cascade_face.detectMultiScale(frame_gray, faces,
                                  1.2, 2, 0|CV_HAAR_SCALE_IMAGE, Size(40, 40));
    
    /* frame_gray - the input image
     faces - the output detections.
     1.1 - scale factor for increasing/decreasing image or pattern resolution
     3 - minNeighbors.
     larger (4) would be more selective in determining detection
     smaller (2,1) less selective in determining detection
     0 - return all detections.
     0|CV_HAAR_SCALE_IMAGE - flags. This flag means scale image to match pattern
     Size(30, 30)) - size in pixels of smallest allowed detection
     */
    
    int detected = 0;
    int hand =0;
    int num = 0;
    int nfaces = (int)faces.size();
    for( int i = 0; i < nfaces ; i++ ) {
        Rect face = faces[i];
        drawEllipse(frame, face, 255, 0, 255);
        int x1 = face.x;
        int y1 = face.y + face.height/2;
        
        
        Rect lower_face =  Rect(x1, y1, face.width, face.height/2);
       
        
        drawEllipse(frame, lower_face, 100, 0, 255);
        Mat lower_faceROI = frame_gray(lower_face);
        vector<Rect> mouths = detectSilence(frame, Point(x1, y1), lower_faceROI, cascade_mouth);
        detected = (int)mouths.size();
        if(detected ==0) {
            drawEllipse(frame, face, 0, 255, 0);
            num++;
            
        }//else{
       //     hand += detectMouth(frame, Point(0, 0), mouths, cascade_hand);
      //  }
   // }
       /* if(hand!=0){
        if((detected<=nfaces && hand <= detected))
            num += hand;
        
    }*/
    }
        return(num);
}

int runonFolder(const CascadeClassifier cascade1,
                const CascadeClassifier cascade2, const CascadeClassifier cascade3,
                string folder) {
    if(folder.at(folder.length()-1) != '/') folder += '/';
    DIR *dir = opendir(folder.c_str());
    if(dir == NULL) {
        cerr << "Can't open folder " << folder << endl;
        exit(1);
    }
    bool finish = false;
    string windowName;
    struct dirent *entry;
    int detections = 0;
    while (!finish && (entry = readdir(dir)) != NULL) {
        char *name = entry->d_name;
        string dname = folder + name;
        Mat img = imread(dname.c_str(), CV_LOAD_IMAGE_UNCHANGED);
        if(!img.empty()) {
            int d = detect(img, cascade1, cascade2,cascade3);
            cerr << d << " detections" << endl;
            detections += d;
            if(!windowName.empty()) destroyWindow(windowName);
            windowName = name;
            namedWindow(windowName.c_str(),CV_WINDOW_AUTOSIZE);
            imshow(windowName.c_str(), img);
            int key = waitKey(0); // Wait for a keystroke
            switch(key) {
                case 27 : // <Esc>
                    finish = true; break;
                default :
                    break;
            }
        } // if image is available
    }
    closedir(dir);
    return(detections);
}

void runonVideo(const CascadeClassifier cascade1,
                const CascadeClassifier cascade2,const CascadeClassifier cascade3) {
    VideoCapture videocapture(0);
    if(!videocapture.isOpened()) {
        cerr <<  "Can't open default video camera" << endl ;
        exit(1);
    }
    string windowName = "Live Video";
    namedWindow("video", CV_WINDOW_AUTOSIZE);
    Mat frame;
    bool finish = false;
    while(!finish) {
        if(!videocapture.read(frame)) {
            cout <<  "Can't capture frame" << endl ;
            break;
        }
        detect(frame, cascade1, cascade2,cascade3);
        imshow("video", frame);
        if(waitKey(30) >= 0) finish = true;
    }
}

int main(int argc, char** argv) {
    if(argc != 1 && argc != 2) {
        cerr << argv[0] << ": "
        << "got " << argc-1
        << " arguments. Expecting 0 or 1 : [image-folder]"
        << endl;
        return(-1);
    }
    
    string foldername = (argc == 1) ? "" : argv[1];
    CascadeClassifier faces_cascade, mouth_cascade,hand_cascade;
    
    if(
       !faces_cascade.load(FACES_CASCADE_NAME)
       || !mouth_cascade.load(MOUTH_CASCADE_NAME) || !hand_cascade.load(FINGER_CASCADE_NAME)) {
        cerr << FACES_CASCADE_NAME << " or " << MOUTH_CASCADE_NAME
        << " are not in a proper cascade format" << endl;
        return(-1);
    }
    
    int detections = 0;
    if(argc == 2) {
        detections = runonFolder(faces_cascade, mouth_cascade, hand_cascade,foldername);
        cout << "Total of " << detections << " detections" << endl;
    }
    else runonVideo(faces_cascade, mouth_cascade,hand_cascade);
    
    return(0);
}
