#ifndef ROAD_T_LINKAGE_H_
#define ROAD_T_LINKAGE_H_

#include"TLinkage.h"

namespace LD {

	class RoadTLinkage : public TLinkage {
		public:
			virtual ArrayXf GenerateHypothesis(const vector<ArrayXf>& _samples) override;

			virtual double Distance(ArrayXf _dataPoint, ArrayXf _model) override;

			virtual void FitModel(const ArrayXXf& _clusters, ArrayXf& _model) override;

			RoadTLinkage(string _xmlFile) : TLinkage(5, 5, _xmlFile) {}

	};

}

#endif
