#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;
CascadeClassifier eye_cascade;

void detectAndDisplay(Mat frame);

int main ()
{
    if (!face_cascade.load("/home/m/code/cpp/face-detect/src/cascades/haarcascade_frontalface_alt.xml")) {
        cout << "Error loading face cascade!" << endl;
        return -1;
    }

    if (!eye_cascade.load("/home/m/code/cpp/face-detect/src/cascades/haarcascade_eye_tree_eyeglasses.xml")) {
        cout << "Error loading eye cascade!" << endl;
        return -1;
    }

    VideoCapture capture;
    Mat frame, resized, image;

    namedWindow("Capture", WINDOW_AUTOSIZE);

    capture.open(4);
    capture.set(CAP_PROP_FPS, 30);

    bool negative = false;

    if (capture.isOpened())
    {
        while (1) {
            capture >> frame;
            if (frame.empty()) break;

            resize(frame, resized, Size(1280, 720));
            detectAndDisplay(resized);
            
            imshow("Capture", resized);

            char c = (char)waitKey(10);

            if (c == 27 || c == 'q' || c == 'Q') break;
        }
    }
    else
    {
        cout << "Could not open webcam." << endl;
        return 1;
    }

    return 0;
}

void detectAndDisplay(Mat frame) {
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

    vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);

    for(size_t i = 0; i < faces.size(); i++) {
        Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
        ellipse(frame, center, Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(255, 255, 0), 4);

        Mat faceROI = frame_gray( faces[i] );

        vector<Rect> eyes;
        eye_cascade.detectMultiScale(faceROI, eyes);

        for(size_t j = 0; j < eyes.size(); j++) {
            Point eye_center(faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
            circle(frame, eye_center, radius, Scalar(255, 255, 0), 4);
        }
    }
}