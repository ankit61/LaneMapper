#ifndef LINE_3D_T_LINKAGE_H_
#define LINE_3D_T_LINKAGE_H_

#include"TLinkage.h"

namespace LD {

	class Line3DTLinkage : public TLinkage {

		public:

			virtual ArrayXf GenerateHypothesis(const vector<ArrayXf>& _samples); 

			virtual double Distance(ArrayXf _dataPoint, ArrayXf _model) override;

			virtual void FitModel(const ArrayXXf& _clusters, ArrayXf& _model) override;

			Line3DTLinkage(string _xmlFile) : TLinkage(2, 6, _xmlFile) {}
	};

}
#endif
