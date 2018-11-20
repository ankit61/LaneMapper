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

			void DrawBorders(Mat& _img, unsigned char thickness = 2, unsigned char val = 255);

			std::pair<int, int> FindMaxDiffs(const Mat& _road, const Mat& _objects, const int& _row, const int& _c, const int& _l, const int& _r, const int& _max_its = 5);

			int FindRightBorder(const Mat& _border, const int& _r, const int& _current, const int& _prevBorder);
	
			void KeepOnlyPeaks(Mat& _in, const int& _peakNeighbors);

			void StretchHistogramHorizontally(Mat& _img);
	
			void DrawHorizontalLines(Mat& _img, unsigned char _val = 1);
			
			void FindSubtractedImg(const Mat& _original, Mat& _subtracted);
		
			FloodFill m_filler;
			int m_peakNeighbors;
			int m_minDiff;
			int m_minArea, m_maxArea;
			int m_minWidth, m_maxWidth, m_minLength;
			int m_stretchingFragments;
			int m_numHorizontalLines;
			bool m_showStepByStep;
			
		public:

			virtual void Refine(const Mat& _original, const Mat& _segImg, Mat& _refinedImg) override;

			LaneExtractor(string _xmlFile) : Refiner(_xmlFile), m_filler(_xmlFile)  { ParseXML(); }

	};

}

#endif
