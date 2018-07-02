#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<vector>
#include<limits>
#include<queue>
#include<algorithm>
#include<exception>

using namespace cv;
using std::vector;
using std::cout;
using std::endl;

class FloodFill {
	public:
		
		enum MatchingCriteria {
			THIS_COLOR,
			OTHER_COLORS
		};

		enum Constraint {
			AREA,
			HEIGHT,
			WIDTH
		};

		FloodFill();

		void operator()(Mat& _in, const long long int& _lo, const long long int& _hi, const unsigned char color = 0, MatchingCriteria _match_by = THIS_COLOR, bool _include_borders = false, Constraint _constraint = AREA);
		
	private:

		bool debug_;
		vector<vector<bool> > is_visited_;
		bool isImageSet;
	
		long long int Fill(Mat& _in, const Point& _start, const long long int& _lo, const long long int& _hi, const unsigned char _color = 0, MatchingCriteria _match_by = THIS_COLOR, bool _include_borders = true, Constraint _constraint = AREA);

		bool InBounds(const Mat& _img, const Point& _pt);
};
