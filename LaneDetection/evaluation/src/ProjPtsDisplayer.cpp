#include"ProjPtsDisplayer.h"
#include"Utilities.h"

namespace LD {
	
	void ProjPtsDisplayer::ProcessProjectedLidarPts(const Eigen::MatrixXf& _veloImg, const cv::Mat& _veloPoints, const cv::Mat& _reflectivity, cv::Mat& _inputImg) {
		if(m_debug)
			cout << "Entering ProjPtsDisplayer::ProcessProjectedLidarPts" << endl;

		for(ulli i = 0; i < _veloImg.rows(); i++) {
			int xImg = _veloImg(i, 0), yImg = _veloImg(i, 1);
			float reflect = _reflectivity.at<float>(i, 0);
			if(isValid(yImg, xImg, _inputImg.rows, _inputImg.cols))
				cv::circle(_inputImg, cv::Point(xImg, yImg), 5, cv::Scalar(int(255 * reflect)), 0, 0);

		}
		cv::imshow("debug", _inputImg);
		cv::waitKey(0);

		if(m_debug)
			cout << "Exiting ProjPtsDisplayer::ProcessProjectedLidarPts" << endl;
		

	}

}
