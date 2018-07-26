#include"LaneDetector.h"
#include<fstream>
#include<libgen.h>

namespace LD {
	
	LaneDetector::LaneDetector(string _xmlFile) : Solver(_xmlFile), m_segmenter(_xmlFile), m_refiner(_xmlFile), m_resultIntersector(_xmlFile), m_line3DTLinkage(_xmlFile)  { ParseXML(); assert(!m_resultIntersector.isMode2D()); }

	void LaneDetector::ParseXML() {
		if(m_debug)
			cout << "Entering LaneDetector::ParseXML()" << endl;
			
		m_xml = m_xml.child("LaneDetector");
		
		m_imgRoot    = m_xml.attribute("imgRoot").as_string();
		m_imgFile 	 = m_xml.attribute("imgFile").as_string();
		m_veloRoot	 = m_xml.attribute("veloRoot").as_string();

		if(m_imgRoot.empty() || m_imgRoot.empty() || m_veloRoot.empty())
			throw runtime_error("at least one of the following attributes is missing: imgRoot, imgFile, veloRoot");	

		if(m_debug)
			cout << "Entering LaneDetector::ParseXML()" << endl;
	}
	
	void LaneDetector::operator()(const cv::Mat& _inputImg, cv::Mat& _veloPoints, vector<Eigen::ArrayXf>& _models) {
		if(m_debug)
			cout << "Entering LaneDetector::()" << endl;

		cv::Mat segImg, refinedImg;
		Eigen::ArrayXXf intersectedPts;
		Eigen::ArrayXf clusters;
		m_segmenter(_inputImg, segImg);	
		m_refiner(_inputImg, segImg, refinedImg);
		m_resultIntersector(_veloPoints, segImg, refinedImg, intersectedPts);
		m_line3DTLinkage(intersectedPts, clusters, _models);

		if(m_debug)
			cout << "Exiting LaneDetector::()" << endl;
	}

	void LaneDetector::Run() {
		if(m_debug)
			cout << "Entering LaneDetector::Run()" << endl;

		std::ifstream fin(m_imgFile.c_str());
		string line;
	
		while(std::getline(fin, line)) {
			m_imgBaseName = basename(const_cast<char*>(line.c_str()));
			cv::Mat inputImg, veloPoints;
			vector<Eigen::ArrayXf> models;
			inputImg = cv::imread(m_imgRoot + "/" + line);
			m_resultIntersector.ReadVeloData(m_veloRoot + "/" + line.substr(0, line.size() - 3) + "bin", veloPoints);
			this->operator()(inputImg, veloPoints, models);
			m_line3DTLinkage.PrintModelsToFile(models, m_imgBaseName);
		}

		if(m_debug)
			cout << "Exiting LaneDetector::Run()" << endl;
	}
}
