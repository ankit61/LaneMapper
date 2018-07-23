#ifndef LANES_EXTRACTOR_H_
#define LANES_EXTRACTOR_H_

#include"Refiner.h"

namespace LD {

	/*
		TODO: calculate height and width in Lidar coordinates
			  consider only a section of road by constraining x and y Lidar coordinates
	
	*/


	class LaneExtractor : public Refiner {
		
		protected:

			virtual void ParseXML() override; 

			void FindLaneScores(const Mat& _road, const Mat& _objects, Mat& _lane_scores, int thresh = 40);

			void DrawBottomBorder(Mat& _img, int thickness = 2, unsigned char val = 1);

			std::pair<int, int> FindMaxDiffs(const Mat& _road, const Mat& _objects, const int& _row, const int& _c, const int& _l, const int& _r, const int& _max_its = 5);


			int FindRightBorder(const Mat& _border, const int& _r, const int& _current, const int& _prevBorder);
	
			void KeepOnlyPeaks(Mat& _in, const int& _peakNeighbors);
		
			FloodFill m_filler;
			int m_peakNeighbors, m_minDiff, m_minArea, m_maxArea, m_minWidth, m_maxWidth, m_minLength;
			
		public:
			virtual void Preprocess(const Mat& _original, const Mat& _segImg, Mat& _preprocessed) override;

			virtual void Refine(const Mat& _extractedImg, Mat& _refinedImg) override;

			LaneExtractor(string _xmlFile) : Refiner(_xmlFile), m_filler(_xmlFile)  { ParseXML(); }

	};

}

#endif
