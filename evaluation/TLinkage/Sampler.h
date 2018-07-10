#ifndef SAMPLER_H_
#define SAMPLER_H_

#include<Eigen/Dense>
#include "BaseLD.h"

using namespace Eigen;

class Sampler : public BaseLD {
	public:
		virtual void operator()(const ArrayXXf& _data, ArrayXXf& _sampleIndices, 
			const ulli& _numSamples, const ulli& _minSamples) = 0;

		virtual void ParseXML() override {}
};

#endif
