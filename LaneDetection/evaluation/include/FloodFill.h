#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include"BaseLD.h"

namespace LD {

	class FloodFill : public BaseLD {
		public:
			
			enum MatchingCriteria {
				THIS_COLOR,
				OTHER_COLORS
			};

			enum Constraint {
				AREA,
				LENGTH,
				WIDTH
			};

			enum Status {
				UNVISITED = INT_MIN, 
				BORDER, 
				VISITED
			};

			FloodFill(string _xmlFile) : BaseLD(_xmlFile), m_borderNum(0) {}

			void operator()(cv::Mat& _in, const ulli& _lo, const ulli& _hi, bool _includeBorders = false, const unsigned char color = 0, MatchingCriteria _match_by = THIS_COLOR, Constraint _constraint = AREA);
			
		protected:

			unsigned long long int Fill(cv::Mat& _in, vector<vector<long long int> >& _status, const cv::Point& _start, const ulli& _lo, const ulli& _hi, Status _curStatus = VISITED, const unsigned char color = 0, MatchingCriteria _match_by = THIS_COLOR, bool _includeBorders = false, Constraint _constraint = AREA);

			bool InBounds(const cv::Mat& _img, const cv::Point& _pt);

			virtual void ParseXML() {}

			long long int m_borderNum;
	};

}
