#include"SurfaceDataMaker.h"

namespace LD {
	void SurfaceDataMaker::ParseXML() {
		m_xml = m_xml.child("SurfaceDataMaker");
		
		m_segRoot			 = m_xml.attribute("segRoot").as_string();
		m_segImgPrefix 		 = m_xml.attribute("segImgPrefix").as_string();
		m_outputFile 		 = m_xml.attribute("outputFile").as_string();
		m_saveVizImg 		 = m_xml.attribute("saveVizImg").as_bool(true);
		m_vizImgPrefix		 = m_xml.attribute("vizImgPrefix").as_string();
		m_minPoints			 = m_xml.attribute("minPoints").as_int();

		if(m_segRoot.empty() || m_segImgPrefix.empty() || m_outputFile.empty() || (m_saveVizImg && m_vizImgPrefix.empty()) || !m_minPoints)
			throw runtime_error("at least one of the following attributes are missing in SurfaceDataMaker node: segRoot, saveVizImg, outputFile, segImgPrefix, vizImgPrefix, minPoints");
	}
	
	
	void SurfaceDataMaker::ProcessProjectedLidarPts(Eigen::MatrixXf& _veloImg) {
		
		if(m_debug)
			cout << "Entering ProcessProjectedLidarPts()" << endl;
		
		string segImgName = m_segRoot + "/" + m_segImgPrefix + m_imgBaseName;
		Mat segImg = imread(segImgName, IMREAD_GRAYSCALE);
		
		if(segImg.empty())
			throw std::runtime_error("Can't open " + segImgName);
		else if(m_debug) 
			cout << "Successfully read segmented image: " << segImgName << endl;


		int rows = 0, cols = 3;

		//count rows

		for(int i = 0; i < _veloImg.rows(); i++) {
			int x = _veloImg(i, 0), y = _veloImg(i, 1);
			int reflect = m_reflectivity.at<unsigned char>(i, 0);
			if(isValid(y, x, segImg.rows, segImg.cols) && segImg.at<unsigned char>(y, x)) {
				rows++;
				if(m_saveVizImg)
					circle(m_inputImg, Point(x, y), 5, Scalar(reflect, 0, 0));		
			}
		}
		
		if(rows > m_minPoints) {

			m_fout << m_imgBaseName << endl;
			m_fout << rows << "\t" << cols << endl;

			for(int i = 0; i < _veloImg.rows(); i++) {
				int x = _veloImg(i, 0), y = _veloImg(i, 1);
				if(isValid(y, x, segImg.rows, segImg.cols) && segImg.at<unsigned char>(y, x))
					m_fout << m_veloPoints.at<float>(i, 0) << "\t" << m_veloPoints.at<float>(i, 1) << "\t" << m_veloPoints.at<float>(i, 2) << endl;
			}
			m_fout.flush();

			if(m_saveVizImg)
				imwrite(m_outputRoot + "/" + m_vizImgPrefix + m_imgBaseName, m_inputImg);
		}

		if(m_debug)
			cout << "Exiting ProcessProjectedLidarPts()" << endl;
	}
}
