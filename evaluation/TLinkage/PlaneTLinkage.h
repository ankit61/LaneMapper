#ifndef PLANE_T_LINKAGE_H_
#define PLANE_T_LINKAGE_H_

#include"TLinkage.h"

class PlaneTLinkage : public TLinkage {

	public:

	virtual void GenerateHypothesis(const ArrayXXf& _data, 
		const ArrayXXf& _sampleIndices, ArrayXXf& _hypotheses) override;

	virtual double Distance(ArrayXf _dataPoint, ArrayXf _model) override;

	virtual void FitModel(const ArrayXXf& _cluster, ArrayXf& _models) override;

	PlaneTLinkage() : TLinkage(3, 4) {}
};

#endif
