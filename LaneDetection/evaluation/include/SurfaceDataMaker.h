#ifndef SURFACE_DATA_MAKER_H_
#define SURFACE_DATA_MAKER_H_

#include"VeloProjector.h"
#include"Utilities.h"

namespace LD {
	class SurfaceDataMaker : public VeloProjector {
		protected:
			string m_segRoot;
			string m_segImgPrefix;
			string m_outputFilePrefix;
			string m_vizImgPrefix;
			bool m_saveVizImg;
			bool m_printProjectedPts;
			int m_minPoints;

			virtual void ParseXML() override;

		public:
			virtual void ProcessProjectedLidarPts(const Eigen::MatrixXf& _veloImg, const Mat& _veloPoints, const Mat& _reflectivity, Mat& _inputImg) override;

			SurfaceDataMaker(string _xmlFile) : VeloProjector(_xmlFile) { ParseXML(); }
	};
}

#endif
