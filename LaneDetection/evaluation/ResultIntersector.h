#ifndef RESULT_INTERSECTOR_H_
#define RESULT_INTERSECTOR_H_

#include"VeloProjector.h"
#include"Utilities.h"

//TODO: Remove repetition of code when counting number of rows in IntersectIn3D

namespace LD {
	class ResultIntersector : public VeloProjector {
		public:
			ResultIntersector(string _xmlFile) : VeloProjector(_xmlFile) { ParseXML(); m_fout.open(m_outputRoot + "/" +m_outputFile, std::ios_base::app); } 	

			void operator()(Mat& _veloPoints, const Mat& _segImg, const Mat& _refinedImg, Eigen::ArrayXXf& _intersectedPts, Mat& _reflectivity, Eigen::MatrixXf& _veloImgPoints);

			void operator()(Mat& _veloPoints, const Mat& _segImg, const Mat& _refinedImg, Eigen::ArrayXXf& _intersectedPts, Mat& _reflectivity, Eigen::MatrixXf& _veloImgPts, Mat& _vizImg);

			bool isMode2D() { return m_printOnly2D; }
		protected:

			string m_segRoot;
			string m_refinedRoot;
			string m_refImgPrefix;
			string m_segImgPrefix;
			string m_vizImgPrefix;
			string m_outputFile;
			std::ofstream m_fout;
			bool m_saveVizImg;
			bool m_printOnly2D;
			int m_maxWidth, m_maxLength;
			
			virtual void ProcessProjectedLidarPts(const Eigen::MatrixXf& _veloImg, const Mat& _veloPoints, const Mat& _reflectivity, Mat& _inputImg) override;
			
			void IntersectIn3D(const Eigen::MatrixXf _veloImg, const Mat& _veloPoints, const Mat& _reflectivity, const Mat& _refinedImg, const double& _thresh, Eigen::ArrayXXf& _intersectedImg, Mat& _vizImg);
			void IntersectIn3D(const Eigen::MatrixXf _veloImg, const Mat& _veloPoints, const Mat& _reflectivity, const Mat& _refinedImg, const double& _thresh, Eigen::ArrayXXf& _intersectedPts);

			double OtsuThresholdRoad(const Eigen::MatrixXf _veloImg, const Mat& _segImg, const Mat& _reflectivity);

			virtual void ParseXML() override;
			
		public:
			void PrintToFile(Eigen::ArrayXXf& _intersectedPts);

			void SaveVizImg(const Mat& _vizImg, string _imgBaseName);

	};
}

#endif
