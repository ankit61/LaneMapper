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
		
		if(m_debug)
			cout << "Exiting Refiner::Refiner()" << endl;
	}

	void Refiner::ParseXML() {
		m_xml = m_xml.child("Refiner");
		pugi::xml_node solverInstance = m_xml.child("SolverInstance");

		m_dataRoot			= solverInstance.attribute("dataRoot").as_string();
		m_dataFile			= solverInstance.attribute("dataFile").as_string();
		m_segRoot      		= solverInstance.attribute("segRoot").as_string();
		m_refinedRoot  		= solverInstance.attribute("refinedRoot").as_string();
		m_vizImgPrefix 		= solverInstance.attribute("vizImgPrefix").as_string();
		m_saveVizImg   		= solverInstance.attribute("saveVizImg").as_bool(true);
		m_refinedImgPrefix 	= solverInstance.attribute("refinedImgPrefix").as_string();

		if(m_dataRoot.empty() || m_dataFile.empty() || m_segRoot.empty() || m_refinedRoot.empty() || (m_saveVizImg && m_vizImgPrefix.empty()) || m_refinedImgPrefix.empty())
			throw runtime_error("One of the following attributes are missing in SolverInstance node of Segmenter: dataRoot, dataFile, segRoot, refinedRoot, vizImgPrefix, saveVizImg, refinedImgPrefix");
	}

	void Refiner::operator()(const Mat& _original, const Mat& _segImg, Mat& _refinedImg) {
		if(m_debug)
			cout << "Entering Refiner::()" << endl;
		
		Mat preprocessed;
		Preprocess(_original, _segImg, preprocessed);
		Refine(preprocessed, _refinedImg);

		if(m_debug)
			cout << "Exiting Refiner::()" << endl;
	}

	void Refiner::Run() {

		if(m_debug)
			cout << "Entering Refiner::Run()" << endl;

		std::ifstream fin(m_dataFile.c_str());
		string line;

		while(std::getline(fin, line)) {
			
			m_imgBaseName = string(basename(const_cast<char*>(line.c_str())));
			string fullImgName = m_dataRoot + "/" + line, fullSegName = m_segRoot + "/segmented_" + line; 
			Mat original = imread(fullImgName);

			Mat segImg = imread(fullSegName, IMREAD_GRAYSCALE);
			
			if(original.empty()) { 
				cout << "could not open or find " << fullImgName << endl;
				exit(1);
			}
			if(segImg.empty()) {
				cout << "could not open or find " << fullSegName << endl;
				exit(1);
			}

			if(m_debug)
				cout << "Successfully opened " << fullImgName << " and " << fullSegName << endl;
			
			Mat refinedImg;
			this->operator()(original, segImg, refinedImg);
			
			imwrite(m_refinedRoot + "/" + m_refinedImgPrefix + m_imgBaseName, refinedImg);
			
			if(m_saveVizImg) {  //save overlayed image
				Mat regions, overlayed;
				string overlayedName = m_refinedRoot + "/" + m_vizImgPrefix + m_imgBaseName;
				vector<Mat> channels;
				channels.push_back(refinedImg);
				channels.push_back(Mat::zeros(refinedImg.size(), CV_8U));
				channels.push_back(Mat::zeros(refinedImg.size(), CV_8U));
				merge(channels, regions);
				addWeighted(regions, 0.5, original, 0.5, 0, overlayed);

				imwrite(overlayedName, overlayed);
			}	
		}
		
		if(m_debug)
			cout << "Exiting Refiner::Run()" << endl;
	}

}
