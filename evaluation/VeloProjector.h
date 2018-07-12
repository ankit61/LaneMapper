#include<string>
#include<libgen.h>
#include<vector>
#include<iostream>
#include<unordered_set>
#include<Eigen/Dense>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/eigen.hpp>
#include<stdio.h>
#include<exception>
#include<stdlib.h>
#include "CalibDataLoader.h"
#include<fstream>

#include"Solver.h"

namespace LD {

	using namespace cv;
	using std::string;
	using std::vector;
	using std::cout;
	using std::endl;

	class VeloProjector : public Solver {
		private:
			
			string m_dataRoot;
			string m_calibRoot;
			string m_segRoot;
			string m_refinedRoot;
			string m_veloRoot;
			string m_outputRoot;
			string m_dataFile;
			double m_minX;
			int m_retentionFrequency;
			int m_camNum;
			Mat m_veloPoints;
			Eigen::MatrixXf m_PRect, m_Tr, m_RRect;
			CalibDataLoader m_calibDataLoader;
			Mat m_reflectivity;
			std::ofstream m_ptsFile3D;

			void ReadVeloData(string _binFile);

			void IntersectIn3D(const Eigen::MatrixXf _veloImg, const Mat& _segImg, double _thresh, Mat _img = Mat());
			
			double OtsuThresholdRoad(const Eigen::MatrixXf _veloImg, const Mat& _segImg);

			void Project(const Eigen::MatrixXf& _PVeloToImg, Eigen::MatrixXf& _veloImg);
			
			void ComputeProjMat(Eigen::MatrixXf& _PVeloToImg);
			
		public:

			VeloProjector(string _xmlFile);

			void ParseXML() override;
			
			virtual void Run() override;
			
	};

}
