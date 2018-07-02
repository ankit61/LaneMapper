#include "FloodFill.h"


FloodFill::FloodFill() {
	#ifdef DEBUG
		debug_ = true;
	#else
		debug_ = false;
	#endif
}

void FloodFill::operator()(Mat& _in, const long long int& _lo, const long long int& _hi, const unsigned char _color, MatchingCriteria _match_by, bool _include_borders, Constraint _constraint) {

	if(debug_)
		cout << "Entering FloodFill::()" << endl;

	unsigned char* p;
	is_visited_ = vector<vector<bool> >(_in.rows, vector<bool>(_in.cols, false));
    for(int r = 0; r < _in.rows; r++) {
		p = _in.ptr<unsigned char>(r);
        for (int c = 0; c < _in.cols; c++) {
			if(!is_visited_[r][c] &&
			  ((_match_by == THIS_COLOR  &&  p[c] == _color) ||
			  (_match_by == OTHER_COLORS && p[c] != _color)))
				Fill(_in, Point(c, r), _lo, _hi, _color, _match_by, _include_borders, _constraint);
		}
    }
	if(debug_)
		cout << "Exiting FloodFill::()" << endl;

}
		
long long int FloodFill::Fill(Mat& _in, const Point& _start, const long long int& _lo, const long long int& _hi, const unsigned char _color, MatchingCriteria _match_by, bool _include_borders, Constraint _constraint) {
	
	//borders are considered to be of any color except _color; _in must be an edge image

	if(debug_)
		cout << "Entering FloodFill::Fill()" << endl;
	
	std::queue<Point> q;
	q.push(_start);
	Mat clone = _in.clone(); //FIXME: Better way?

	Point cur, next, border_start;
	long long int area = 0, height = 0, width = 0, border_area = 0, border_height = 0, border_width = 0;
	is_visited_[_start.y][_start.x] = true;
	long long int min_y = _start.y, max_y = _start.y, min_x = _start.x, max_x = _start.x;
	bool border_retrieved = false;

	while(!q.empty()) {

		cur = q.front();
		area++;
		clone.at<unsigned char>(cur) = 255;
		q.pop();
		
		min_y = std::min((long long int)cur.y, min_y);
		max_y = std::max((long long int)cur.y, max_y);
		min_x = std::min((long long int)cur.x, min_x);
		max_x = std::max((long long int)cur.x, max_x);
		
		for(int dx = -1; dx <= 1; dx++) {
			for(int dy = -1; dy <= 1; dy++) {

				next = Point(cur.x + dx, cur.y + dy);
				
				if(InBounds(_in, next) && !is_visited_[next.y][next.x]) {
					if((_match_by == THIS_COLOR && _in.at<unsigned char>(next) == _color) || 
				       (_match_by == OTHER_COLORS && _in.at<unsigned char>(next) != _color)) {
						q.push(next);
						is_visited_[next.y][next.x] = true;
					}
					else {
						border_retrieved = true;
						is_visited_[next.y][next.x] = true;
						area++;
						clone.at<unsigned char>(next) = 255;
						border_start = next;
					}
				}
			}
		}
	}

	height = max_y - min_y;
	width  = max_x - min_x;

	switch(_constraint) {
		case AREA:
			if(_include_borders)
				border_area = Fill(clone, border_start, std::max(_lo - area, (long long int)0), std::max(_hi - area, (long long int)0), _color, OTHER_COLORS, false, _constraint);
	
			if((area + border_area >= _lo && area + border_area <= _hi) ||
			   (area >= _lo && area <= _hi))
					clone.copyTo(_in);
			if(debug_)
				cout << "Exiting FloodFill::Fill()" << endl;
			return area;

		case HEIGHT:
			if(_include_borders)
				border_height = Fill(clone, next, _lo, _hi, _color, OTHER_COLORS, false, _constraint);
			
			if((border_height >= _lo && border_height <= _hi) && 
			   (height >= _lo && height <= _hi))
					clone.copyTo(_in);
			if(debug_)
				cout << "Exiting FloodFill::Fill()" << endl;
			return height;

		case WIDTH:
			if(_include_borders)
				border_width = Fill(clone, next, _lo, _hi, _color, OTHER_COLORS, false, _constraint);
	
			if((border_width >= _lo && border_width <= _hi) && 
			   (width >= _lo && width <= _hi))
			   		clone.copyTo(_in);
			if(debug_)
				cout << "Exiting FloodFill::Fill()" << endl;
			return width;
	}

	throw std::runtime_error("Invalid constraint passed to FloodFill::Fill()");

}

bool FloodFill::InBounds(const Mat& _img, const Point& _pt) {
	return _pt.x >= 0 && _pt.x < _img.cols && _pt.y >= 0 && _pt.y < _img.rows;
}
