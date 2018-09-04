#ifndef LANE_DETECTOR_H_
#define LANE_DETECTOR_H_

#include"Solver.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<Eigen/Dense>

#include"Segmenter.h"
#include"LaneExtractor.h"
#include"KPercentExtractor.h"
#include"ResultIntersector.h"
#include"TLinkage/BSplineTLinkage.h"
#include"LaneQualityChecker.h"
#include"PointsVisualizer.h"

namespace LD {
	class LaneDetector : public Solver {
		public:
			void operator()(const cv::Mat& _inputImg, cv::Mat& _veloPoints, vector<Eigen::ArrayXf>& _models, float& _brightnessRatio, float& _reflectivityRatio);

			virtual void Run() override;

			LaneDetector(string _xmlFile);
		
		protected:
			Segmenter m_segmenter;
			LaneExtractor m_refiner;
			ResultIntersector m_resultIntersector;
			BSplineTLinkage m_bSplineTLinkage;
			LaneQualityChecker m_laneQualityChecker;
			PointsVisualizer m_visualizer;

			string m_imgFile, m_imgRoot, m_veloRoot;
			string m_imgBaseName;
			string m_ratiosFile;
			bool m_saveVizImg;
			string m_vizImgPrefix;

			virtual void ParseXML() override;
	};
}

#endif
