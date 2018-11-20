#ifndef PLANE_T_LINKAGE_H_
#define PLANE_T_LINKAGE_H_

#include"TLinkage.h"

namespace LD {

	class PlaneTLinkage : public TLinkage {

		public:

			virtual ArrayXf GenerateHypothesis(const vector<ArrayXf>& _samples) override;

			virtual double Distance(ArrayXf _dataPoint, ArrayXf _model) override;

			virtual void FitModel(const ArrayXXf& _cluster, ArrayXf& _models) override;

			PlaneTLinkage(string _xmlFile) : TLinkage(3, 4, _xmlFile) {}
	};

}

#endif
