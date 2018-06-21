/**
 * \brief C++ implementation of eval_all.m script of ICNet
 * \author Ankit Ramchandani
 * \date  
*/

#include "Refiner.h"
#include<fstream>
#include<stdlib.h>

using namespace cv;
	
Refiner::Refiner(string _data_root, string _data_file, string _segmented_root, string _refined_root) {
	#ifdef DEBUG
		debug_ = true;
	#else
		debug_ = false;
	#endif

	if(debug_)
		cout << "Entering Refiner::Refiner()" << endl;

	data_root_ = _data_root;
	data_file_ = _data_file;
	segmented_root_ = _segmented_root;
	refined_root_ = _refined_root;
	stat_file_stream_ = std::ofstream(refined_root_ + "/stats.csv");
	namedWindow( "show", WINDOW_NORMAL );
	if(debug_)
		cout << "Exiting Refiner::Refiner()" << endl;
}

void Refiner::Run() {

	if(debug_)
		cout << "Entering Refiner::Run()" << endl;

	std::ifstream fin(data_file_.c_str());
	string line;

	while(std::getline(fin, line)) {
		
		img_base_name_ = string(basename(const_cast<char*>(line.c_str())));
		string fullImgName = data_root_ + "/" + line, fullSegName = segmented_root_ + "/segmented_" + line; 
		Mat img = imread(fullImgName);
		cvtColor(img, img, COLOR_BGR2HLS);
		vector<Mat> channels;
		split(img, channels);
		img = channels[1];

		Mat segmented_img = imread(fullSegName, IMREAD_GRAYSCALE);
		Mat thresholded_img;
		if(img.empty()) { 
			cout << "could not open or find " << fullImgName << endl;
			exit(1);
		}
		if(segmented_img.empty()) {
			cout << "could not open or find " << fullSegName << endl;
			exit(1);
		}

		if(debug_)
			cout << "Successfully opened " << fullImgName << " and " << fullSegName << endl;

		bitwise_and(img, segmented_img, img);
		ThresholdImage(img,  thresholded_img);
	
	}
	
	if(debug_)
		cout << "Exiting Refiner::Run()" << endl;
}

void Refiner::ThresholdImage(const Mat& _extracted_img, Mat& _thresholded_img) {
	//Fails when input = output
	if(debug_)
		cout << "Entering Refiner::ThresholdImage()" << endl;

	string saved_name = refined_root_ + "/thresholded_" + img_base_name_;

	Mat blurred;
	GaussianBlur(_extracted_img, blurred, Size(3,3), 1);

	Mat lane_scores;
	FindLaneScores(blurred, lane_scores);
	
	imshow("show", lane_scores * 255);
	waitKey(0);
	
	multiply(blurred, lane_scores, _thresholded_img);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat dilated;
	morphologyEx(_thresholded_img, dilated, MORPH_DILATE, kernel);
	subtract(dilated, _thresholded_img, _thresholded_img);
	
	imshow("show", _thresholded_img);
	waitKey(0);
	
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(_thresholded_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	for(long long int i = 0; i < contours.size(); i++) {
		if(contourArea(contours[i]) > 15)
			drawContours(_thresholded_img, contours, i, Scalar(255), -1, 4, hierarchy);
	}	
	
//	kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	threshold(_thresholded_img, _thresholded_img, 254, 255, THRESH_BINARY); //removing changes by findContours
//	morphologyEx(_thresholded_img, _thresholded_img, MORPH_OPEN, kernel, Point(-1, -1), 1);

	imshow("show", _thresholded_img);
	waitKey(0);

	imwrite(saved_name,  _thresholded_img);

	if(debug_) {
		cout << "Saving " << saved_name << " in " << refined_root_ << endl;
		cout << "Exiting Refiner::ThresholdedImage()" << endl;
	}
}

void Refiner::FindLaneScores(const Mat& _extracted_img, Mat& _lane_scores, int thresh, int m) {
	//Based off of the GOLD Paper
    _lane_scores = Mat::zeros(_extracted_img.size(), CV_8U);
	const unsigned char *p_ex;
	unsigned char* p_ls;
    for(int r = 0; r < _extracted_img.rows; r++) {
        p_ex = _extracted_img.ptr<unsigned char>(r);
		p_ls = _lane_scores.ptr<unsigned char>(r);
        for (int c = m; c < _extracted_img.cols - m; c++) {
           	if(p_ex[c - m] > 0 && p_ex[c + m] > 0 &&
			   p_ex[c] - p_ex[c - m] > thresh &&
			   p_ex[c] - p_ex[c + m] > thresh) {
				
				p_ls[c] = 1;
						
			} 
        }
    }
}
