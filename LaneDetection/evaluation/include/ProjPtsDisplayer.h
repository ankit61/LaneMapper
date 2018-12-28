#ifndef PROJ_PTS_DISPLAYER
#define PROJ_PTS_DISPLAYER

#include"Solver.h"
#include"VeloProjector.h"
#include<Eigen/Dense>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

namespace LD {
	class ProjPtsDisplayer : public VeloProjector {
		public:
			ProjPtsDisplayer(string _xmlFile) : VeloProjector(_xmlFile) { ParseXML(); }

			virtual void ParseXML() override {}

			virtual void ProcessProjectedLidarPts(const Eigen::MatrixXf& _veloImg, const cv::Mat& _veloPoints, const cv::Mat& _reflectivity, cv::Mat& _inputImg) override;

	};
}

#endif