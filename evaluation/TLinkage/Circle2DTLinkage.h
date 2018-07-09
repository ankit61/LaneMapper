#ifndef CIRCLE_T_LINKAGE_H_
#define CIRCLE_T_LINKAGE_H_

#include"TLinkage.h"

class Circle2DTLinkage : public TLinkage {

	public:

	virtual void GenerateHypothesis(const ArrayXXf& _data, 
		const ArrayXXf& _sampleIndices, ArrayXXf& _hypotheses) override;

	virtual double Distance(ArrayXf _dataPoint, ArrayXf _model) override;

	virtual void FitModel(const ArrayXXf& _cluster, ArrayXf& _model) override;

	Circle2DTLinkage() : TLinkage(3, 3) {}
};

#endif
