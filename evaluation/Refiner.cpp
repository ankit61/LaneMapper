/**
 * \brief C++ implementation of eval_all.m script of ICNet
 * \author Ankit Ramchandani
 * \date  
*/

#include "Refiner.h"
#include<fstream>
#include<stdlib.h>

using namespace cv;
	
Refiner::Refiner(string _data_root, string _data_file, string _seg_root, string _refined_root) {
	#ifdef DEBUG
		debug_ = true;
	#else
		debug_ = false;
	#endif

	if(debug_)
		cout << "Entering Refiner::Refiner()" << endl;

	data_root_ = _data_root;
	data_file_ = _data_file;
	seg_root_ = _seg_root;
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
		string fullImgName = data_root_ + "/" + line, fullSegName = seg_root_ + "/segmented_" + line; 
		string overlayed_name = refined_root_ + "/overlayed_" + img_base_name_;
		Mat original = imread(fullImgName);
		Mat img, overlayed;
		cvtColor(original, img, COLOR_RGB2GRAY);

		Mat seg_img = imread(fullSegName, IMREAD_GRAYSCALE);
		Mat thresholded_img;
		if(img.empty()) { 
			cout << "could not open or find " << fullImgName << endl;
			exit(1);
		}
		if(seg_img.empty()) {
			cout << "could not open or find " << fullSegName << endl;
			exit(1);
		}

		if(debug_)
			cout << "Successfully opened " << fullImgName << " and " << fullSegName << endl;
		
//		morphologyEx(seg_img, seg_img, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(7,7)));
//		morphologyEx(seg_img, seg_img, MORPH_DILATE, getStructuringElement(MORPH_RECT, Size(3,3)));
		bitwise_and(img, seg_img, img);
		ThresholdImage(img, thresholded_img);
		bitwise_and(img, thresholded_img, thresholded_img);

		Mat regions;
		vector<Mat> channels;
		channels.push_back(Mat::zeros(thresholded_img.size(), CV_8U));
		channels.push_back(Mat::zeros(thresholded_img.size(), CV_8U));
		channels.push_back(thresholded_img);
		merge(channels, regions);
		addWeighted(regions, 0.5, original, 0.5, 0, overlayed);

		imwrite(overlayed_name, overlayed);
		
		if(debug_)
			cout << "Saving " << overlayed_name << endl;
	
	}
	
	if(debug_)
		cout << "Exiting Refiner::Run()" << endl;
}

void Refiner::ThresholdImage(const Mat& _extracted_img,  Mat& _thresholded_img) {
	if(debug_)
		cout << "Entering Refiner::ThresholdImage()" << endl;
	
	Mat samples;
	const unsigned char* p_img;

	for(int r = 0; r < _extracted_img.rows; r++) {
		p_img = _extracted_img.ptr<unsigned char>(r);
		for(int c = 0; c < _extracted_img.cols; c++) {
			if(p_img[c])
				samples.push_back(p_img[c]); //keep only road pixels
		}
	}

	Ptr<ml::EM> em_model = ml::EM::create();
	em_model->setClustersNumber(3);

	em_model->trainEM(samples);  //default mode involves 100 iterations

	Mat means = em_model->getMeans();
	vector<Mat> covs;
	em_model->getCovs(covs);

	Point max_pt;
	double max_mean, max_std;
	minMaxLoc(means, nullptr, &max_mean, nullptr, &max_pt);
	max_std = std::sqrt(covs[max_pt.y].at<double>(0));
	
	double thresh = max_mean - 3 * max_std;

	if(debug_) {
		cout << "Maximum mean: " << max_mean << endl;
		cout << "Corresponding standard deviation: " << max_std << endl;
		cout << "Threshold set to " << thresh << endl;
	}

	threshold(_extracted_img, _thresholded_img, thresh, 255, THRESH_TOZERO);

	if(debug_) 
		cout << "Exiting Refiner::ThresholdedImage()" << endl;
}
