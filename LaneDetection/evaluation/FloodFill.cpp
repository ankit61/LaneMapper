#include "FloodFill.h"
#include<queue>

namespace LD {

	void FloodFill::operator()(cv::Mat& _in, const ulli& _lo, const ulli& _hi, bool _includeBorders, const unsigned char _color, MatchingCriteria _matchBy, Constraint _constraint) {

		if(m_debug)
			cout << "Entering FloodFill::()" << endl;

		unsigned char* p;
		vector<vector<long long int> > status(_in.rows, vector<long long int>(_in.cols, UNVISITED));

		for(int r = 0; r < _in.rows; r++) {
			p = _in.ptr<unsigned char>(r);
			for (int c = 0; c < _in.cols; c++) {
				if(status[r][c] == UNVISITED  &&
				  ((_matchBy == THIS_COLOR  &&  p[c] == _color) ||
				  (_matchBy == OTHER_COLORS && p[c] != _color)))
					Fill(_in, status, cv::Point(c, r), _lo, _hi, VISITED, _color, _matchBy, _includeBorders, _constraint);
			}
		}
		if(m_debug)
			cout << "Exiting FloodFill::()" << endl;

	}

	unsigned long long int FloodFill::Fill(cv::Mat& _in, vector<vector<long long int> >& _status, const cv::Point& _start, const ulli& _lo, const ulli& _hi, Status _curStatus, const unsigned char _color, MatchingCriteria _matchBy, bool _includeBorders, Constraint _constraint) {
		
		//borders are considered to be of any color except _color; _in must be an edge image

		if(_curStatus == BORDER)
			m_borderNum++;

		std::queue<cv::Point> q;
		q.push(_start);
		cv::Mat clone = _in.clone(); //FIXME: Better way?

		cv::Point cur, next, borderStart;
		ulli area = 0, length = 0, width = 0, borderArea = 0, borderLength = 0, borderWidth = 0;
		_status[_start.y][_start.x] = _curStatus;
		int minY = _start.y, maxY = _start.y, minX = _start.x, maxX = _start.x;
		bool borderRetrieved = false;

		while(!q.empty()) {

			cur = q.front();
			area++;
			clone.at<unsigned char>(cur) = 255;
			q.pop();
			
			minY = std::min(cur.y, minY);
			maxY = std::max(cur.y, maxY);
			minX = std::min(cur.x, minX);
			maxX = std::max(cur.x, maxX);
			
			for(int dx = -1; dx <= 1; dx++) {
				for(int dy = -1; dy <= 1; dy++) {

					next = cv::Point(cur.x + dx, cur.y + dy);
					
					if(InBounds(_in, next) && _status[next.y][next.x] != VISITED && _status[next.y][next.x] != m_borderNum) {
						if((_matchBy == THIS_COLOR && _in.at<unsigned char>(next) == _color) || 
						   (_matchBy == OTHER_COLORS && _in.at<unsigned char>(next) != _color)) {
							q.push(next);
							_status[next.y][next.x] = _curStatus == BORDER ? m_borderNum : _curStatus;
						}
						else if(!borderRetrieved) {
							borderRetrieved = true;
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
				if(_includeBorders && borderRetrieved)
					borderArea = Fill(clone, _status, borderStart, (_lo > area ? _lo - area : 0) , (_hi > area ? _hi - area : 0), BORDER, _color, OTHER_COLORS, false, _constraint);
		
				if((area + borderArea >= _lo && area + borderArea <= _hi) ||
				   (area >= _lo && area <= _hi))
						clone.copyTo(_in);
				return area;

			case LENGTH:
				if(_includeBorders && borderRetrieved)
					borderLength = Fill(clone, _status, next, _lo, _hi, BORDER, _color, OTHER_COLORS, false, _constraint);
				
				borderLength = std::max(borderLength, length);
				
				if(borderLength >= _lo && borderLength <= _hi)
						clone.copyTo(_in);
				return length;

			case WIDTH:
				if(_includeBorders && borderRetrieved)
					borderWidth = Fill(clone, _status, next, _lo, _hi, BORDER, _color, OTHER_COLORS, false, _constraint);
				
				borderWidth = std::max(borderWidth, width);
		
				if(borderWidth >= _lo && borderWidth <= _hi)
						clone.copyTo(_in);
				return width;
		}

		throw std::runtime_error("Invalid constraint passed to FloodFill::Fill()");

	}

	bool FloodFill::InBounds(const cv::Mat& _img, const cv::Point& _pt) {
		return _pt.x >= 0 && _pt.x < _img.cols && _pt.y >= 0 && _pt.y < _img.rows;
	}
}
