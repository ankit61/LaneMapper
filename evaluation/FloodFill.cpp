#include "FloodFill.h"
#include<queue>

namespace LD {

	void FloodFill::operator()(cv::Mat& _in, const ulli& _lo, const ulli& _hi, const unsigned char _color, MatchingCriteria _matchBy, bool _includeBorders, Constraint _constraint) {

		if(m_debug)
			cout << "Entering FloodFill::()" << endl;

		unsigned char* p;
		vector<vector<bool> > isVisited(_in.rows, vector<bool>(_in.cols, false));

		for(int r = 0; r < _in.rows; r++) {
			p = _in.ptr<unsigned char>(r);
			for (int c = 0; c < _in.cols; c++) {
				if(!isVisited[r][c] &&
				  ((_matchBy == THIS_COLOR  &&  p[c] == _color) ||
				  (_matchBy == OTHER_COLORS && p[c] != _color)))
					Fill(_in, isVisited, cv::Point(c, r), _lo, _hi, _color, _matchBy, _includeBorders, _constraint);
			}
		}
		if(m_debug)
			cout << "Exiting FloodFill::()" << endl;

	}
			
	unsigned long long int FloodFill::Fill(cv::Mat& _in, vector<vector<bool> >& _isVisited, const cv::Point& _start, const ulli& _lo, const ulli& _hi, const unsigned char _color, MatchingCriteria _matchBy, bool _includeBorders, Constraint _constraint) {
		
		//borders are considered to be of any color except _color; _in must be an edge image

		if(m_debug)
			cout << "Entering FloodFill::Fill()" << endl;
		
		std::queue<cv::Point> q;
		q.push(_start);
		cv::Mat clone = _in.clone(); //FIXME: Better way?

		cv::Point cur, next, borderStart;
		ulli area = 0, length = 0, width = 0, borderArea = 0, borderLength = 0, borderWidth = 0;
		_isVisited[_start.y][_start.x] = true;
		ulli minY = _start.y, maxY = _start.y, minX = _start.x, maxX = _start.x;
		bool borderRetrieved = false;

		while(!q.empty()) {

			cur = q.front();
			area++;
			clone.at<unsigned char>(cur) = 255;
			q.pop();
			
			minY = std::min((ulli)cur.y, minY);
			maxY = std::max((ulli)cur.y, maxY);
			minX = std::min((ulli)cur.x, minX);
			maxX = std::max((ulli)cur.x, maxX);
			
			for(int dx = -1; dx <= 1; dx++) {
				for(int dy = -1; dy <= 1; dy++) {

					next = cv::Point(cur.x + dx, cur.y + dy);
					
					if(InBounds(_in, next) && !_isVisited[next.y][next.x]) {
						if((_matchBy == THIS_COLOR && _in.at<unsigned char>(next) == _color) || 
						   (_matchBy == OTHER_COLORS && _in.at<unsigned char>(next) != _color)) {
							q.push(next);
							_isVisited[next.y][next.x] = true;
						}
						else {
							borderRetrieved = true;
							_isVisited[next.y][next.x] = true;
							area++;
							clone.at<unsigned char>(next) = 255;
							borderStart = next;
						}
					}
				}
			}
		}

		length = maxY - minY;
		width  = maxX - minX;

		switch(_constraint) {
			case AREA:
				if(_includeBorders)
					borderArea = Fill(clone, _isVisited, borderStart, std::max(_lo - area, (ulli)0), std::max(_hi - area, (ulli)0), _color, OTHER_COLORS, false, _constraint);
		
				if((area + borderArea >= _lo && area + borderArea <= _hi) ||
				   (area >= _lo && area <= _hi))
						clone.copyTo(_in);
				if(m_debug)
					cout << "Exiting FloodFill::Fill()" << endl;
				return area;

			case LENGTH:
				if(_includeBorders)
					borderLength = Fill(clone, _isVisited, next, _lo, _hi, _color, OTHER_COLORS, false, _constraint);
				else
					borderLength = length;
				
				if((borderLength >= _lo && borderLength <= _hi) && 
				   (length >= _lo && length <= _hi))
						clone.copyTo(_in);
				if(m_debug)
					cout << "Exiting FloodFill::Fill()" << endl;
				return length;

			case WIDTH:
				if(_includeBorders)
					borderWidth = Fill(clone, _isVisited, next, _lo, _hi, _color, OTHER_COLORS, false, _constraint);
				else
					borderWidth = width;
		
				if((borderWidth >= _lo && borderWidth <= _hi) && 
				   (width >= _lo && width <= _hi))
						clone.copyTo(_in);
				if(m_debug)
					cout << "Exiting FloodFill::Fill()" << endl;
				return width;
		}

		throw std::runtime_error("Invalid constraint passed to FloodFill::Fill()");

	}

	bool FloodFill::InBounds(const cv::Mat& _img, const cv::Point& _pt) {
		return _pt.x >= 0 && _pt.x < _img.cols && _pt.y >= 0 && _pt.y < _img.rows;
	}
}
