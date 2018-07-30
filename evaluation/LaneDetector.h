#ifndef LANE_DETECTOR_H_
#define LANE_DETECTOR_H_

#include"Solver.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<Eigen/Dense>

#include"Segmenter.h"
#include"KPercentExtractor.h"
#include"ResultIntersector.h"
#include"TLinkage/Line3DTLinkage.h"

namespace LD {
	class LaneDetector : public Solver {
		public:
			void operator()(const cv::Mat& _inputImg, cv::Mat& _veloPoints, vector<Eigen::ArrayXf>& _models);

			virtual void Run() override;

			LaneDetector(string _xmlFile);
		
		protected:
			Segmenter m_segmenter;
			KPercentExtractor m_refiner;
			ResultIntersector m_resultIntersector;
			Line3DTLinkage m_line3DTLinkage;

			string m_imgFile, m_imgRoot, m_veloRoot;
			string m_imgBaseName;

			virtual void ParseXML() override;
	};
}

#endif
