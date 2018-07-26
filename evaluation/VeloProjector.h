#ifndef VELO_PROJECTOR_H_
#define VELO_PROJECTOR_H_

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
#include"CalibDataLoader.h"
#include<fstream>

#include"Solver.h"

namespace LD {

	using namespace cv;

	class VeloProjector : public Solver {
		protected:
			
			string m_dataRoot;
			string m_calibRoot;
			string m_veloRoot;
			string m_outputRoot;
			string m_dataFile;
			string m_imgBaseName;
			double m_minX;
			int m_retentionFrequency;
			int m_camNum;
			Mat m_veloPoints;
			Eigen::MatrixXf m_PRect, m_Tr, m_RRect, m_projectionMat;
			CalibDataLoader m_calibDataLoader;

			virtual void Project(const Eigen::MatrixXf& _PVeloToImg, Mat& _veloPoints, Eigen::MatrixXf& _veloImg, Mat& _reflectivity) final;
			
			void ComputeProjMat(Eigen::MatrixXf& _PVeloToImg);

			virtual void ProcessProjectedLidarPts(const Eigen::MatrixXf& _veloImg, const Mat& _veloPoints, const Mat& _reflectivity, Mat& _inputImg) = 0;
			
		public:

			VeloProjector(string _xmlFile);

			virtual void ParseXML() override;
		
			virtual void Run() override;
			
			virtual void ReadVeloData(string _binFile, Mat& _veloPoints);
			
	};

}

#endif
