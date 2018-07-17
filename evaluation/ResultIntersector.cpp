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

	void ResultIntersector::ProcessProjectedLidarPts(const Eigen::MatrixXf& _veloImg, const Mat& _veloPoints, const Mat& _reflectivity, Mat& _inputImg) {

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

		double thresh = OtsuThresholdRoad(_veloImg, segImg, _reflectivity);
		if(m_debug)
			cout << "Threshold set to " << thresh << endl;
		m_fout << m_imgBaseName << endl;
		IntersectIn3D(_veloImg, _veloPoints, _reflectivity, refinedImg, thresh, _inputImg);

		if(m_saveVizImg)
			imwrite(m_outputRoot + "/" + m_vizImgPrefix + m_imgBaseName, _inputImg);

		if(m_debug)
			cout << "Exiting ResultIntersector::ProcessProjectedLidarPts()" << endl;

	}

	void ResultIntersector::IntersectIn3D(const Eigen::MatrixXf _veloImg, const Mat& _veloPoints, const Mat& _reflectivity, const Mat& _refinedImg, const double& _thresh, const Mat& _vizImg) {
		if(m_debug)
			cout << "Entering ResultIntersector::IntersectIn3D()" << endl;

		ulli rows = 0, cols = 3;

		//find number of rows

		for(int i = 0; i < _veloImg.rows(); i++) {
			int x = _veloImg(i, 0), y = _veloImg(i, 1);
			float reflect = _reflectivity.at<float>(i, 0);
			if(isValid(y, x, _refinedImg.rows, _refinedImg.cols) && _refinedImg.at<unsigned char>(y, x) && reflect > _thresh) { 
				rows++;
				if(m_saveVizImg)
					circle(_vizImg, Point(x, y), 5, Scalar(int(255 * reflect), 0, 0));
			}
		}

		m_fout << rows << "\t" << cols << endl;

		//print actual coordinates now
		
		for(int i = 0; i < _veloImg.rows(); i++) {
			int x = _veloImg(i, 0), y = _veloImg(i, 1);
			float reflect = _reflectivity.at<float>(i, 0);
			if(isValid(y, x, _refinedImg.rows, _refinedImg.cols) && _refinedImg.at<unsigned char>(y, x) && reflect > _thresh) { 
				m_fout << _veloPoints.at<float>(i, 0) << "\t" << _veloPoints.at<float>(i, 1) << "\t" << _veloPoints.at<float>(i, 2) << endl;
			}
		}

		m_fout.flush();

		if(m_debug)
			cout << "Exiting ResultIntersector::IntersectIn3D()" << endl;
	}

	double ResultIntersector::OtsuThresholdRoad(const Eigen::MatrixXf _veloImg, const Mat& _segImg, const Mat& _reflectivity) {
		if(m_debug)
			cout << "Entering ResultIntersector::OtsuThresholdRoad()" << endl;

		//Find Otsu thresholding for points that are on road and have positive reflectivity
		Mat scaledReflectivity = 255 * _reflectivity;
		scaledReflectivity.convertTo(scaledReflectivity, CV_8UC1);
		vector<unsigned char> onRoadRef;
		onRoadRef.reserve(scaledReflectivity.rows);

		for(long long int i = 0; i < scaledReflectivity.rows; i++) {
			int x = _veloImg(i, 0), y = _veloImg(i, 1);
			int reflect = scaledReflectivity.at<unsigned char>(i, 0);
			if(isValid(y, x, _segImg.rows, _segImg.cols) && _segImg.at<unsigned char>(y, x)) 
				onRoadRef.push_back(scaledReflectivity.at<unsigned char>(i, 0));
		}

		double thresh = threshold(onRoadRef, onRoadRef, 0, 255, THRESH_TOZERO | THRESH_OTSU) / 255;

		if(m_debug)
			cout << "Exiting ResultIntersector::OtsuThresholdRoad()" << endl;

		return thresh;
	}

}
