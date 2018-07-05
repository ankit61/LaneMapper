#ifndef OUTLIER_REJECTOR_H_
#define OUTLIER_REJECTOR_H_

#include<Eigen/Dense>
#include "BaseLD.h"

using namespace Eigen;

class OutlierRejector : public BaseLD {
	public:
		virtual void operator()(const ArrayXf& _clusters, ArrayXf& _out) = 0;

		virtual void ParseXML() {}
};

#endif
