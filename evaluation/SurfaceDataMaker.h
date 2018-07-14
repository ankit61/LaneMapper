#ifndef SURFACE_DATA_MAKER_H_
#define SURFACE_DATA_MAKER_H_

#include"VeloProjector.h"
#include"Utilities.h"

namespace LD {
	class SurfaceDataMaker : public VeloProjector {
		protected:
			string m_segRoot;
			string m_segImgPrefix;
			string m_outputFile;
			string m_vizImgPrefix;
			bool m_saveVizImg;
			int m_minPoints;
			std::ofstream m_fout;

			virtual void ParseXML() override;

		public:
			virtual void ProcessProjectedLidarPts(Eigen::MatrixXf& _veloImg) override;

			SurfaceDataMaker(string _xmlFile) : VeloProjector(_xmlFile) { ParseXML(); m_fout.open(m_outputRoot + "/" + m_outputFile); }
	};
}

#endif
