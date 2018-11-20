#ifndef CIRCLE_T_LINKAGE_H_
#define CIRCLE_T_LINKAGE_H_

#include"TLinkage.h"
namespace LD {
	
	class Circle2DTLinkage : public TLinkage {

		public:

		virtual ArrayXf GenerateHypothesis(const vector<ArrayXf>& _samples) override;

		virtual double Distance(ArrayXf _dataPoint, ArrayXf _model) override;

		virtual void FitModel(const ArrayXXf& _cluster, ArrayXf& _model) override;

		Circle2DTLinkage(string _xmlFile) : TLinkage(3, 3, _xmlFile) {}
	};
}
#endif
