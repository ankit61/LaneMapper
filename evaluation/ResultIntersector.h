#ifndef RESULT_INTERSECTOR_H_
#define RESULT_INTERSECTOR_H_

#include"VeloProjector.h"
#include"Utilities.h"

namespace LD {
	class ResultIntersector : public VeloProjector {
		public:
			ResultIntersector(string _xmlFile) : VeloProjector(_xmlFile) { ParseXML(); m_fout.open(m_outputRoot + "/" +m_outputFile); } 	

		protected:

			string m_segRoot;
			string m_refinedRoot;
			string m_refImgPrefix;
			string m_segImgPrefix;
			string m_vizImgPrefix;
			string m_outputFile;
			std::ofstream m_fout;
			bool m_saveVizImg;
			
			virtual void ProcessProjectedLidarPts(Eigen::MatrixXf& _veloImg) override;
			
			void IntersectIn3D(const Eigen::MatrixXf _veloImg, const Mat& _refinedImg, const double& _thresh, const Mat& _vizImg);
			
			double OtsuThresholdRoad(const Eigen::MatrixXf _veloImg, const Mat& _segImg);

			virtual void ParseXML() override;
	};
}

#endif
