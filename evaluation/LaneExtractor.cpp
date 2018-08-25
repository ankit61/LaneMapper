#include"LaneExtractor.h"

namespace LD {
	
	void LaneExtractor::ParseXML() { 
		if(m_debug)
			cout << "Entering LaneExtractor::ParseXML()" << endl;
		
		m_xml = m_xml.child("LaneExtractor");

		m_minDiff 				= m_xml.attribute("minDiff").as_int();
		m_peakNeighbors 		= m_xml.attribute("peakNeighbors").as_int();
		m_minArea				= m_xml.attribute("minArea").as_int(-1); 
		m_maxArea				= m_xml.attribute("maxArea").as_int();
		m_minLength				= m_xml.attribute("minLength").as_int(-1);
		m_minWidth				= m_xml.attribute("minWidth").as_int(-1);
		m_maxWidth				= m_xml.attribute("maxWidth").as_int();
		m_stretchingFragments	= m_xml.attribute("stretchingFragments").as_int();
		m_numHorizontalLines	= m_xml.attribute("numHorizontalLines").as_int();
		m_showStepByStep		= m_xml.attribute("showStepByStep").as_bool();

		if((m_minDiff <= 0) || (m_peakNeighbors <= 0) || (m_minWidth < 0) || (m_maxWidth <= 0) || (m_minLength < 0) || (m_minArea < 0) || (m_maxArea <= 0) || (m_stretchingFragments <= 0) || (m_numHorizontalLines < 0))
			throw runtime_error("at least one of the following parameters missing/invalid in xml: minDiff, peakNeighbors, minArea, maxArea, minWidth, maxWidth, minLength, stretchingFragments, numHorizontalLines");

		if(m_debug)
			cout << "Exiting LaneExtractor::ParseXML()" << endl;
	
	}

	void LaneExtractor::FindSubtractedImg(const Mat& _original, Mat& _subtracted) {
		if(m_debug)
			cout << "Entering LaneExtractor::FindSubtractedImg()" << endl;
		
		Mat kernel = getStructuringElement(MORPH_RECT, Size(4, 4));
		Mat dilated;
		morphologyEx(_original, dilated, MORPH_DILATE, kernel);
		subtract(dilated, _original, _subtracted);
		threshold(_subtracted, _subtracted, 20, 255, THRESH_TOZERO);  //FIXME: Better way of thresholding

		if(m_debug)
			cout << "Entering LaneExtractor::FindSubtractedImg()" << endl;
	}

	void LaneExtractor::StretchHistogramHorizontally(Mat& _img) {
		if(m_debug)
			cout << "Entering LaneExtractor::StretchingFragments()" << endl;
	
		int incrementBy = _img.cols / m_stretchingFragments;
		for(int r = 0; r < _img.rows; r++) {
			double minVal, maxVal;
			for(int c = 0; c < _img.cols; c += incrementBy) {
				minMaxLoc(_img.row(r).colRange(c, std::min(_img.cols, c + incrementBy)), &minVal, &maxVal);
				_img.row(r).colRange(c, std::min(_img.cols, c + incrementBy)) = (_img.row(r).colRange(c, std::min(_img.cols, c + incrementBy)) - minVal) * (255 / (maxVal - minVal));
			}
		}

		if(m_debug)
			cout << "Exiting LaneExtractor::StretchingFragments()" << endl;
	}
	
	void LaneExtractor::Refine(const Mat& _original, const Mat& _segImg, Mat& _refinedImg) {
		if(m_debug)
			cout << "Entering LaneExtractor::Refine()" << endl;

		Mat subtracted, extractedImg, grayOriginal;
		
		//fill small holes in segmented image
		Mat filledSeg = _segImg.clone(); //FIXME: deep copy
		m_filler(filledSeg, 0, 500);
		
		cvtColor(_original, grayOriginal, COLOR_RGB2GRAY);
		bitwise_and(grayOriginal, filledSeg, extractedImg);

		//StretchHistogramHorizontally(grayOriginal);
		FindSubtractedImg(extractedImg, subtracted);

		_refinedImg = subtracted.clone(); //FIXME: temp
		
		_refinedImg -= 1;  //making max intensity = 254

		DrawHorizontalLines(_refinedImg);
		
		if(m_showStepByStep) {
			imshow("show", _refinedImg);
			waitKey(0);
		}
		m_filler(_refinedImg, m_minArea, m_maxArea); 

		threshold(_refinedImg, _refinedImg, 254, 255, THRESH_BINARY); //keep only contours

		if(m_showStepByStep) {
			imshow("show", _refinedImg);
			waitKey(0);
		}

		Mat laneScores;
		FindLaneScores(grayOriginal, _refinedImg, laneScores);

		morphologyEx(_refinedImg, _refinedImg, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(4, 4)));

		if(m_showStepByStep) {
			imshow("show", laneScores);
			waitKey(0);
		}

		laneScores -= 1;

		m_filler(laneScores, m_minWidth, m_maxWidth, false, 0, FloodFill::OTHER_COLORS, FloodFill::WIDTH);
		threshold(laneScores, laneScores, 254, 255, THRESH_BINARY); //keep only contours
		
		if(m_showStepByStep) {
			imshow("show", laneScores);
			waitKey(0);
		}

		laneScores -= 1;

		m_filler(laneScores, m_minLength, laneScores.cols + 1, false, 0, FloodFill::OTHER_COLORS, FloodFill::LENGTH);
		threshold(laneScores, _refinedImg, 254, 255, THRESH_BINARY); //keep only contours
		
		if(m_showStepByStep) {
			imshow("show", _refinedImg);
			waitKey(0);
		}

		//post process: fill points that are inside lanes
		morphologyEx(_refinedImg, _refinedImg, MORPH_DILATE, getStructuringElement(MORPH_RECT, Size(3, 3)));
		m_filler(_refinedImg, 0, 500);

		if(m_showStepByStep) {
			imshow("show", _refinedImg);
			waitKey(0);
		}

		if(m_debug)
			cout << "Exiting LaneExtractor::Refine()" << endl;
	}

	void LaneExtractor::FindLaneScores(const Mat& _road, const Mat& _objects, Mat& _laneScores, int _thresh) {
		//Inspired by the GOLD Paper
		//FIXME: terrible implementation, especially the way of finding right border

		if(m_debug)
			cout << "Entering LaneExtractor::FindLaneScores()" << endl;

		Mat_<float> loHi(1, 3), HiLo(1,3);
		Mat loHiRes, hiLoRes;

		loHi << -1,  1, 0;
		HiLo <<  1, -1, 0;

		filter2D(_road, loHiRes, CV_32F, loHi, Point(-1, -1), 0, BORDER_REPLICATE);
		filter2D(_road, hiLoRes, CV_32F, HiLo, Point(-1, -1), 0, BORDER_REPLICATE);

		hiLoRes.convertTo(hiLoRes, CV_8U);
		loHiRes.convertTo(loHiRes, CV_8U);

		threshold(loHiRes, loHiRes, m_minDiff, 255, THRESH_TOZERO);  //FIXME: constant threshold
		threshold(hiLoRes, hiLoRes, m_minDiff, 255, THRESH_TOZERO);  //FIXME: constant threshold

		KeepOnlyPeaks(loHiRes, m_peakNeighbors);
		KeepOnlyPeaks(hiLoRes, m_peakNeighbors);
		
		const unsigned char *pObj;
		unsigned char *pLH, *pHL;
		unsigned char *pLanes;

		_laneScores = Mat::zeros(_road.size(), CV_8U);

		for(int r = 0; r < _road.rows; r++) {
			pObj   = _objects.ptr<unsigned char>(r);
			pLH    = loHiRes.ptr<unsigned char>(r);
			pHL    = hiLoRes.ptr<unsigned char>(r);
			pLanes = _laneScores.ptr<unsigned char>(r);
			int rightHL = 0, rightLH = 0, leftLH = 0, leftHL = 0;

			for (int c = 0; c < _road.cols; c++) {

				if(pLH[c] > 0) 
					leftLH  = c - 1;

				if(pHL[c] > 0)
					leftHL = c;
				
				if(pObj[c] > 0 && leftLH > leftHL) {
					
					rightHL = FindRightBorder(hiLoRes, r, c, rightHL);
					rightLH = std::max(-1, FindRightBorder(loHiRes, r, c, rightLH) - 1);
					
					if(rightHL != -1 && (rightLH > rightHL || rightLH == -1))
						pLanes[c] = 255;
				}
			}
		}

		if(m_debug)
			cout << "Exiting LaneExtractor::FindLaneScores()" << endl;
	}

	void LaneExtractor::KeepOnlyPeaks(Mat& _in, const int& _peakNeighbors) {
		if(m_debug)
			cout << "Entering LaneExtractor::KeepOnlyPeaks()" << endl;

		//FIXME: Terrible implementation: O(nk). can be improved to at least O(k*log(n))
		unsigned char *pIn;

		for(int r = 0; r < _in.rows; r++) {
			pIn = _in.ptr<unsigned char>(r);
			unsigned char* maxIt = std::max_element(pIn, pIn + 2 * _peakNeighbors + 1);
			for(int c = _peakNeighbors; c < _in.cols - _peakNeighbors; c++) {
				if(maxIt < &pIn[c - _peakNeighbors])
					maxIt = std::max_element(pIn + c - _peakNeighbors, pIn + c + _peakNeighbors + 1);
				else if(pIn[c + _peakNeighbors] > *maxIt)
					maxIt = &pIn[c + _peakNeighbors];
				pIn[c] = (pIn[c] == *maxIt) ? pIn[c] : 0;
			}
		}

		if(m_debug)
			cout << "Exiting LaneExtractor::KeepOnlyPeaks()" << endl;
	}

	void LaneExtractor::DrawHorizontalLines(Mat& _img, unsigned char _val) {
		if(m_debug)
			cout << "Entering LaneExtractor::DrawHorizontalLines()" << endl;

		int incrementBy = _img.rows / (m_numHorizontalLines + 1);
		for(int r = incrementBy; r < _img.rows; r += incrementBy)
			_img.row(r) = _val;	
			
		if(m_debug)
			cout << "Entering LaneExtractor::DrawHorizontalLines()" << endl;
	}

	void LaneExtractor::DrawBorders(Mat& _img, unsigned char _thickness, unsigned char _val) {
		
		if(m_debug)
			cout << "Entering LaneExtractor::DrawBorders()" << endl;
		
		//draw bottom border
		for(int r = _img.rows - _thickness; r < _img.rows; r++)
			_img.row(r) = _val;

		//draw right border	
		for(int c = _img.cols - _thickness; c < _img.cols; c++)
			_img.col(c) = _val;

		//draw left border
		for(int c = 0; c <= _thickness; c++)
			_img.col(c) = _val;

		if(m_debug)
			cout << "Entering LaneExtractor::DrawBorders()" << endl;
	}

	std::pair<int, int> LaneExtractor::FindMaxDiffs(const Mat& _road, const Mat& _edges, const int& _row, const int& _c, const int& _l, const int& _r, const int& _maxIts) {
		
		if(m_debug)
			cout << "Entering Refiner::FindMaxDiffs()" << endl;

		std::pair<int, int> maxDiffs = std::make_pair(0, 0);
		int i = 0, j = 0;
		int dx = 1, maxDx = std::min(_maxIts, std::max(_l, _road.cols - _r - 1));
		int nextL, nextR;
		const unsigned char* pRoad = _road.ptr<unsigned char>(_row);
		const unsigned char* pEdge = _edges.ptr<unsigned char>(_row);

		while((i < _maxIts || j < _maxIts) && dx <= maxDx) {

			nextL = _l - dx, nextR = _r + dx;
			
			if(i < _maxIts && nextL >= 0) {
				if(pEdge[nextL] == 0 && pRoad > 0) {
					maxDiffs.first += pRoad[_c] - pRoad[nextL];
					i++;
				}
				else if(pEdge[nextL] > 0)
					i = _maxIts; //entered another obj
			}

			if(j < _maxIts && nextR < _road.cols) {
				if(pEdge[nextR] == 0 && pRoad[nextR] > 0) {
					maxDiffs.second += pRoad[_c] - pRoad[nextR];
					j++;
				}
				else if(pEdge[nextR] > 0)
					j = _maxIts; //entered another obj
			}

			dx++;
		}

		if(i)
			maxDiffs.first /= i;

		if(j)
			maxDiffs.second /= j;
		
		if(m_debug)
			cout << "Exiting Refiner::FindMaxDiffs()" << endl;

		return maxDiffs;
	}

	int LaneExtractor::FindRightBorder(const Mat& _border, const int& _r, const int& _current, const int& _prevBorder) {
		if(_current < _prevBorder)
			return _prevBorder;

		const unsigned char* p = _border.ptr<unsigned char>(_r);

		for(int i = _current; i < _border.cols; i++)
			if(p[i] > 0)
				return i;

		return -1;
	}
}
