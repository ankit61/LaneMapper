#ifndef SURFACE_T_LINKAGE_H_
#define SURFACE_T_LINKAGE_H_

#include"TLinkage.h"

namespace LD {

	class SurfaceTLinkage : public TLinkage {

		public:

			virtual ArrayXf GenerateHypothesis(const vector<ArrayXf>& _samples) override;

			virtual double Distance(ArrayXf _dataPoint, ArrayXf _model) override;

			virtual void FitModel(const ArrayXXf& _clusters, ArrayXf& _model) override;

			SurfaceTLinkage(string _xmlFile) : TLinkage(10, 10, _xmlFile) {} //FIXME: There should be a way to do this by 9 samples only
	};

}
#endif
