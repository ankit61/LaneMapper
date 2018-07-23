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

			FloodFill(string _xmlFile) : BaseLD(_xmlFile) {}

			void operator()(cv::Mat& _in, const ulli& _lo, const ulli& _hi, const unsigned char color = 0, MatchingCriteria _match_by = THIS_COLOR, bool _includeBorders = false, Constraint _constraint = AREA);
			
		protected:

			unsigned long long int Fill(cv::Mat& _in, vector<vector<bool> >& _isVisited, const cv::Point& _start, const ulli& _lo, const ulli& _hi, const unsigned char color = 0, MatchingCriteria _match_by = THIS_COLOR, bool _includeBorders = false, Constraint _constraint = AREA);

			bool InBounds(const cv::Mat& _img, const cv::Point& _pt);

			virtual void ParseXML() {}
	};

}
