#include"ResultIntersector.h"

namespace LD {
	
	void ResultIntersector::ParseXML() {
		m_xml 			= m_xml.child("ResultIntersector");
		m_segRoot       = m_xml.attribute("segRoot").as_string();
		m_refinedRoot   = m_xml.attribute("refinedRoot").as_string();
		m_segImgPrefix	= m_xml.attribute("segImgPrefix").as_string();
		m_refImgPrefix	= m_xml.attribute("refImgPrefix").as_string();
		m_outputFile	= m_xml.attribute("outputFile").as_string();
		m_saveVizImg	= m_xml.attribute("saveVizImg").as_bool();
		m_vizImgPrefix	= m_xml.attribute("vizImgPrefix").as_string();

		if(m_segRoot.empty() || m_refinedRoot.empty() || m_segImgPrefix.empty() || m_refImgPrefix.empty() || m_outputFile.empty() || (m_saveVizImg && m_vizImgPrefix.empty()))
			throw runtime_error("at least one of the following attributes are missing in ResultIntersector node: segRoot, refinedRoot, outputFile, segImgPrefix, refImgPrefix, vizImgPrefix");
	}

	void ResultIntersector::ProcessProjectedLidarPts(Eigen::MatrixXf& _veloImg) {

		if(m_debug)
			cout << "Exiting ResultIntersector::ProcessProjectedLidarPts()" << endl;

		string segImgName = m_segRoot + "/" + m_segImgPrefix + m_imgBaseName;
		string refImgName = m_refinedRoot + "/" + m_refImgPrefix + m_imgBaseName;

		Mat segImg = imread(segImgName, IMREAD_GRAYSCALE);
		Mat refinedImg = imread(refImgName, IMREAD_GRAYSCALE);

		if(segImg.empty())
			throw std::runtime_error("Can't open " + segImgName);
		else 
			cout << "Successfully read segmented image: " << segImgName << endl;

		if(refinedImg.empty())
			throw std::runtime_error("Can't open " + refImgName);
		else 
			cout << "Successfully read refined image: " << refImgName << endl;

		double thresh = OtsuThresholdRoad(_veloImg, segImg);
		if(m_debug)
			cout << "Threshold set to " << thresh << endl;
		m_fout << m_imgBaseName << endl;
		IntersectIn3D(_veloImg, refinedImg, thresh, m_inputImg);

		if(m_saveVizImg)
			imwrite(m_outputRoot + "/" + m_vizImgPrefix + m_imgBaseName, m_inputImg);

		if(m_debug)
			cout << "Exiting ResultIntersector::ProcessProjectedLidarPts()" << endl;

	}

	void ResultIntersector::IntersectIn3D(const Eigen::MatrixXf _veloImg, const Mat& _refinedImg, const double& _thresh, const Mat& _vizImg) {
		if(m_debug)
			cout << "Entering ResultIntersector::IntersectIn3D()" << endl;

		for(int i = 0; i < _veloImg.rows(); i++) {
			int x = _veloImg(i, 0), y = _veloImg(i, 1);
			int reflect = m_reflectivity.at<unsigned char>(i, 0);
			if(isValid(y, x, _refinedImg.rows, _refinedImg.cols) && _refinedImg.at<unsigned char>(y, x) && reflect > _thresh) { 
				m_fout << m_veloPoints.at<float>(i, 0) << "\t" << m_veloPoints.at<float>(i, 1) << "\t" << m_veloPoints.at<float>(i, 2) << endl;
				if(m_saveVizImg)
					circle(_vizImg, Point(x, y), 5, Scalar(reflect, 0, 0));
			}
		}
		m_fout.flush();

		if(m_debug)
			cout << "Exiting ResultIntersector::IntersectIn3D()" << endl;
	}

	double ResultIntersector::OtsuThresholdRoad(const Eigen::MatrixXf _veloImg, const Mat& _segImg) {
		if(m_debug)
			cout << "Entering ResultIntersector::OtsuThresholdRoad()" << endl;

		//Find Otsu thresholding for points that are on road and have positive reflectivity
		m_reflectivity = 255 * m_reflectivity;
		m_reflectivity.convertTo(m_reflectivity, CV_8UC1);
		vector<unsigned char> onRoadRef;
		onRoadRef.reserve(m_reflectivity.rows);

		for(long long int i = 0; i < m_reflectivity.rows; i++) {
			int x = _veloImg(i, 0), y = _veloImg(i, 1);
			int reflect = m_reflectivity.at<unsigned char>(i, 0);
			if(isValid(y, x, _segImg.rows, _segImg.cols) && _segImg.at<unsigned char>(y, x)) 
				onRoadRef.push_back(m_reflectivity.at<unsigned char>(i, 0));
		}

		double thresh = threshold(onRoadRef, onRoadRef, 0, 255, THRESH_TOZERO | THRESH_OTSU);

		if(m_debug)
			cout << "Exiting ResultIntersector::OtsuThresholdRoad()" << endl;

		return thresh;
	}

}
