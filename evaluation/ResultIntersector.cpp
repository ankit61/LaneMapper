#include"ResultIntersector.h"

namespace LD {
	
	void ResultIntersector::ParseXML() {
		m_xml = m_xml.child("ResultIntersector");

		m_segRoot       = m_xml.attribute("segRoot").as_string();
		m_refinedRoot   = m_xml.attribute("refinedRoot").as_string();
		m_segImgPrefix	= m_xml.attribute("segImgPrefix").as_string();
		m_refImgPrefix	= m_xml.attribute("refImgPrefix").as_string();
		m_outputFile	= m_xml.attribute("outputFile").as_string();
		m_saveVizImg	= m_xml.attribute("saveVizImg").as_bool();
		m_vizImgPrefix	= m_xml.attribute("vizImgPrefix").as_string();
		m_maxWidth		= m_xml.attribute("maxWidth").as_int();
		m_maxLength		= m_xml.attribute("maxLength").as_int();
		m_printOnly2D	= m_xml.attribute("printOnly2D").as_bool(false);

		if(m_segRoot.empty() || m_refinedRoot.empty() || m_segImgPrefix.empty() || m_refImgPrefix.empty() || m_outputFile.empty() || (m_saveVizImg && m_vizImgPrefix.empty()) || !m_maxWidth || !m_maxLength)
			throw runtime_error("at least one of the following attributes are missing in ResultIntersector node: segRoot, refinedRoot, outputFile, segImgPrefix, refImgPrefix, vizImgPrefix, maxWidth, maxHeight");
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
		else if(m_debug) 
			cout << "Successfully read segmented image: " << segImgName << endl;

		if(refinedImg.empty())
			throw std::runtime_error("Can't open " + refImgName);
		else if(m_debug)
			cout << "Successfully read refined image: " << refImgName << endl;

		double thresh = OtsuThresholdRoad(_veloImg, segImg, _reflectivity);
		if(m_debug)
			cout << "Threshold set to " << thresh << endl;
		Eigen::ArrayXXf intersectedPts;
		IntersectIn3D(_veloImg, _veloPoints, _reflectivity, refinedImg, thresh, intersectedPts, _inputImg);
		PrintToFile(intersectedPts);

		if(m_saveVizImg)
			imwrite(m_outputRoot + "/" + m_vizImgPrefix + m_imgBaseName, _inputImg);

		if(m_debug)
			cout << "Exiting ResultIntersector::ProcessProjectedLidarPts()" << endl;

	}

	void ResultIntersector::operator()(Mat& _veloPoints, const Mat& _segImg, const Mat& _refinedImg, Eigen::ArrayXXf& _intersectedPts, Mat& _reflectivity, Eigen::MatrixXf& _veloImgPts) {
		if(m_debug)
			cout << "Entering ResultIntersector::()" << endl;
		
		Project(_veloPoints, _veloImgPts, _reflectivity);
		double thresh = OtsuThresholdRoad(_veloImgPts, _segImg, _reflectivity);
		IntersectIn3D(_veloImgPts, _veloPoints, _reflectivity, _refinedImg, thresh, _intersectedPts);

		if(m_debug)
			cout << "Entering ResultIntersector::()" << endl;
	}

	void ResultIntersector::IntersectIn3D(const Eigen::MatrixXf _veloImg, const Mat& _veloPoints, const Mat& _reflectivity, const Mat& _refinedImg, const double& _thresh, Eigen::ArrayXXf& _intersectedPts) {
		Mat test;
		IntersectIn3D(_veloImg, _veloPoints, _reflectivity, _refinedImg, _thresh, _intersectedPts, test);
	}

	void ResultIntersector::IntersectIn3D(const Eigen::MatrixXf _veloImg, const Mat& _veloPoints, const Mat& _reflectivity, const Mat& _refinedImg, const double& _thresh, Eigen::ArrayXXf& _intersectedPts, Mat& _vizImg) {
		if(m_debug)
			cout << "Entering ResultIntersector::IntersectIn3D()" << endl;

		vector<vector<float> > intersectedPtsVec;

		for(ulli i = 0; i < _veloImg.rows(); i++) {
			int xImg = _veloImg(i, 0), yImg = _veloImg(i, 1);
			float xLidar = _veloPoints.at<float>(i, 0), yLidar = _veloPoints.at<float>(i, 1);
			float reflect = _reflectivity.at<float>(i, 0);
			if(isValid(yImg, xImg, _refinedImg.rows, _refinedImg.cols) && std::abs(yLidar) <= m_maxWidth && std::abs(xLidar) <= m_maxLength && _refinedImg.at<unsigned char>(yImg, xImg) && reflect > _thresh) { 
				if(m_printOnly2D)
					intersectedPtsVec.push_back({xLidar, yLidar});
				else
					intersectedPtsVec.push_back({xLidar, yLidar, _veloPoints.at<float>(i, 2)});

				if(m_saveVizImg && !_vizImg.empty())
					circle(_vizImg, Point(xImg, yImg), 5, Scalar(int(255 * reflect)), 0, 0);
			}
		}

		_intersectedPts.resize(intersectedPtsVec.size() ? intersectedPtsVec[0].size() : 0, intersectedPtsVec.size());

		for(ulli r = 0; r < _intersectedPts.rows(); r++)
			for(ulli c = 0; c < _intersectedPts.cols(); c++)
				_intersectedPts(r, c) = intersectedPtsVec[c][r];

		if(m_debug)
			cout << "Exiting ResultIntersector::IntersectIn3D()" << endl;
	}

	void ResultIntersector::PrintToFile(Eigen::ArrayXXf& _intersectedPts) {
		if(m_debug)
			cout << "Entering ResultIntersector::PrintToFile()" << endl;
		
		m_fout << m_imgBaseName << endl;
		m_fout << _intersectedPts.cols() << "\t" <<  _intersectedPts.rows() << endl;

		for(ulli c = 0; c < _intersectedPts.cols(); c++) {
			for(ulli r = 0; r < _intersectedPts.rows(); r++)
				m_fout << _intersectedPts(r, c) << "\t";

			m_fout << endl;
		}

		if(m_debug)
			cout << "Exiting ResultIntersector::PrintToFile()" << endl;
	}

	double ResultIntersector::OtsuThresholdRoad(const Eigen::MatrixXf _veloImg, const Mat& _segImg, const Mat& _reflectivity) {
		if(m_debug)
			cout << "Entering ResultIntersector::OtsuThresholdRoad()" << endl;

		//Find Otsu thresholding for points that are on road and have positive reflectivity
		Mat scaledReflectivity = 255 * _reflectivity;
		scaledReflectivity.convertTo(scaledReflectivity, CV_8UC1);
		vector<unsigned char> onRoadRef;
		onRoadRef.reserve(scaledReflectivity.rows);

		for(ulli i = 0; i < scaledReflectivity.rows; i++) {
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
