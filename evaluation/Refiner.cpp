/**
 * \brief C++ implementation of eval_all.m script of ICNet
 * \author Ankit Ramchandani
 * \date  
*/

#include "Refiner.h"
#include<fstream>
#include<stdlib.h>

namespace LD {

	using namespace cv;
		
	Refiner::Refiner(string _xmlFile) : Solver(_xmlFile) {

		if(m_debug)
			cout << "Entering Refiner::Refiner()" << endl;

		ParseXML();
		m_statFileStream = std::ofstream(m_refinedRoot + "/stats.csv");
		namedWindow( "show", WINDOW_NORMAL );
		if(m_debug)
			cout << "Exiting Refiner::Refiner()" << endl;
	}

	void Refiner::ParseXML() {
		m_xml = m_xml.child("Refiner");
		m_dataRoot = m_xml.child("SolverInstance").attribute("dataRoot").as_string();
		m_dataFile = m_xml.child("SolverInstance").attribute("dataFile").as_string();
		m_segRoot = m_xml.child("SolverInstance").attribute("segRoot").as_string();
		m_refinedRoot = m_xml.child("SolverInstance").attribute("refinedRoot").as_string();

		if(m_dataRoot.empty() || m_dataFile.empty() || m_segRoot.empty() || m_refinedRoot.empty())
			throw runtime_error("One of the following attributes are missing in SolverInstance node of Segmenter: dataRoot, dataFile, segRoot, refinedRoot");
	}

	void Refiner::Run() {

		if(m_debug)
			cout << "Entering Refiner::Run()" << endl;

		std::ifstream fin(m_dataFile.c_str());
		string line;

		while(std::getline(fin, line)) {
			
			m_imgBaseName = string(basename(const_cast<char*>(line.c_str())));
			string fullImgName = m_dataRoot + "/" + line, fullSegName = m_segRoot + "/segmented_" + line; 
			string _overlayedName = m_refinedRoot + "/overlayed_" + m_imgBaseName;
			Mat original = imread(fullImgName);
			Mat img, overlayed;
			cvtColor(original, img, COLOR_RGB2GRAY);

			Mat segImg = imread(fullSegName, IMREAD_GRAYSCALE);
			Mat thresholdedImg;
			if(img.empty()) { 
				cout << "could not open or find " << fullImgName << endl;
				exit(1);
			}
			if(segImg.empty()) {
				cout << "could not open or find " << fullSegName << endl;
				exit(1);
			}

			if(m_debug)
				cout << "Successfully opened " << fullImgName << " and " << fullSegName << endl;
			
	//		morphologyEx(segImg, segImg, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(7,7)));
	//		morphologyEx(segImg, segImg, MORPH_DILATE, getStructuringElement(MORPH_RECT, Size(3,3)));
			bitwise_and(img, segImg, img);
			ThresholdImage(img, thresholdedImg);
			
			bitwise_and(img, thresholdedImg, thresholdedImg);
			Mat regions;
			vector<Mat> channels;
			channels.push_back(Mat::zeros(thresholdedImg.size(), CV_8U));
			channels.push_back(Mat::zeros(thresholdedImg.size(), CV_8U));
			channels.push_back(thresholdedImg);
			merge(channels, regions);
			addWeighted(regions, 0.5, original, 0.5, 0, overlayed);

			imwrite(_overlayedName, overlayed);
			
			if(m_debug)
				cout << "Saving " << _overlayedName << endl;
		
		}
		
		if(m_debug)
			cout << "Exiting Refiner::Run()" << endl;
	}

	void Refiner::ThresholdImage(const Mat& _extractedImg,  Mat& _thresholdedImg) {
		if(m_debug)
			cout << "Entering Refiner::ThresholdImage()" << endl;
		
		threshold(_extractedImg, _thresholdedImg, 170, 255, THRESH_TOZERO);

		if(m_debug) 
			cout << "Exiting Refiner::ThresholdedImage()" << endl;
	}

}
