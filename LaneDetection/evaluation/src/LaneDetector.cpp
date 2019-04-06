#include"LaneDetector.h"
#include<fstream>
#include<libgen.h>

namespace LD {
	
	LaneDetector::LaneDetector(string _xmlFile) : Solver(_xmlFile), m_segmenter(_xmlFile), m_refiner(_xmlFile), m_resultIntersector(_xmlFile), m_bSplineTLinkage(_xmlFile), m_laneQualityChecker(_xmlFile), m_visualizer(_xmlFile), m_mapGenerator(_xmlFile)  { 
		ParseXML(); 
		assert(!m_resultIntersector.isMode2D()); 
	}

	void LaneDetector::ParseXML() {
		if(m_debug)
			cout << "Entering LaneDetector::ParseXML()" << endl;
			
		m_xml = m_xml.child("LaneDetector");
		
		m_imgRoot    	  		= m_xml.attribute("imgRoot").as_string();
		m_imgFile 	 	  		= m_xml.attribute("imgFile").as_string();
		m_veloRoot	 	  		= m_xml.attribute("veloRoot").as_string();
		m_saveVizImg	  		= m_xml.attribute("saveVizImg").as_bool(true);
		m_vizImgPrefix	  		= m_xml.attribute("vizImgPrefix").as_string();

		if(m_imgRoot.empty() || m_imgRoot.empty() || m_veloRoot.empty() || (m_saveVizImg && m_vizImgPrefix.empty()))
			throw runtime_error("at least one of the following attributes is missing: imgRoot, imgFile, veloRoot, ratiosFile, vizImgPrefix, saveVizImg");	

		if(m_debug)
			cout << "Exiting LaneDetector::ParseXML()" << endl;
	}
	
	void LaneDetector::operator()(const cv::Mat& _inputImg, cv::Mat& _veloPoints, vector<Eigen::ArrayXf>& _models, vector<Eigen::ArrayXXf>& _worldCtrlPts, vector<MapGenerator::LaneType>& _laneTypes) {
		if(m_debug)
			cout << "Entering LaneDetector::()" << endl;

		cv::Mat segImg, refinedImg;
		Eigen::ArrayXXf intersectedPts;
		Eigen::ArrayXf clusters;
		Mat reflectivity;
		Eigen::MatrixXf veloImg;
		m_segmenter(_inputImg, segImg);

		if(m_debug)
			m_segmenter.SaveOverlaidImg(_inputImg, segImg, m_imgBaseName);

		m_refiner(_inputImg, segImg, refinedImg);

		if(m_debug) {
			m_refiner.SaveOverlaidImg(_inputImg, refinedImg, m_imgBaseName);
			Mat vizImg = _inputImg;
			m_resultIntersector(_veloPoints, segImg, refinedImg, intersectedPts, reflectivity, veloImg, vizImg);
			m_resultIntersector.PrintToFile(intersectedPts);
			m_resultIntersector.SaveVizImg(vizImg, m_imgBaseName);
		}
		else
			m_resultIntersector(_veloPoints, segImg, refinedImg, intersectedPts, reflectivity, veloImg);

		m_bSplineTLinkage(intersectedPts, clusters, _models);
		m_mapGenerator.GetWorldCtrlPts(m_imgBaseName, _models, _worldCtrlPts);

		_laneTypes.clear();
		for(int i = 0; i < _models.size(); i++) {
			_laneTypes.push_back((m_bSplineTLinkage.IsModelOnRight(_models[i]) ? MapGenerator::LaneType::RIGHT : MapGenerator::LaneType::LEFT));
			_worldCtrlPts[i].transposeInPlace();
		}
		
		if(m_saveVizImg) {
			for(int i = 0; i < _models.size(); i++) {
				ArrayXXf coordinates;
				m_bSplineTLinkage.VisualizeModel(_models[i], coordinates);
				Mat vizImg = _inputImg;
				m_visualizer(vizImg, coordinates);
				imwrite(m_vizImgPrefix + m_imgBaseName, _inputImg);	
			}
		}	

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

			try {
				vector<Eigen::ArrayXXf> worldCtrlPts;
				vector<MapGenerator::LaneType> laneTypes;
				this->operator()(inputImg, veloPoints, models, worldCtrlPts, laneTypes);
				for(int i = 0; i < worldCtrlPts.size(); i++)
					m_mapGenerator.PrintWorldCtrlPts(worldCtrlPts[i], laneTypes[i]);
			}
			catch(Sampler::MinSamplesNotFound) {
				std::cerr << m_imgBaseName << endl;
				std::cerr << "---------------------WARNING: Could not find enough samples-----------" << endl;	
			}
			catch(alglib::ap_error& e) {
				std::cerr << m_imgBaseName << endl;
				std::cerr << "---------------------WARNING: ALGLIB Error: " << e.msg << "-----------" << endl;
			}
			catch(...) {
				std::cerr << m_imgBaseName << endl;
				std::cerr << "----------------------------Warning: Unknown Error------------------------------" << endl;
			}
		}

		if(m_debug)
			cout << "Exiting LaneDetector::Run()" << endl;
	}
}