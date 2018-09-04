#ifndef MODEL_VISUALIZER_H_
#define MODEL_VISUALIZER_H_

#include"VeloProjector.h"

namespace LD {

	class PointsVisualizer : public VeloProjector {

		public:

		PointsVisualizer(const string& _xmlFile) : VeloProjector(_xmlFile) {}
		
		virtual void operator()(Mat& _inputImg, Eigen::ArrayXXf& _coordinates);
		
		virtual void ProcessProjectedLidarPts(const Eigen::MatrixXf& _veloImg, const Mat& _veloPoints, const Mat& _reflectivity, Mat& _inputImg) {}
	};

}

#endif
