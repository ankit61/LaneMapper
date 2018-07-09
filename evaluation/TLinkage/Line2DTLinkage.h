#ifndef LINE_2D_T_LINKAGE_H_
#define LINE_2D_T_LINKAGE_H_

#include"TLinkage.h"

class Line2DTLinkage : public TLinkage {

	public:

	virtual void GenerateHypothesis(const ArrayXXf& _data, 
		const ArrayXXf& _sampleIndices, ArrayXXf& _hypotheses) override;

	virtual double Distance(ArrayXf _dataPoint, ArrayXf _model) override;

	virtual void FitModel(const ArrayXXf& _clusters, ArrayXf& _model) override;

	Line2DTLinkage() : TLinkage(2, 3) {}
};

#endif
