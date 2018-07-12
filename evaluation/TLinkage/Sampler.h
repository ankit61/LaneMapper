#ifndef SAMPLER_H_
#define SAMPLER_H_

#include<Eigen/Dense>
#include "../BaseLD.h"

namespace LD {
	using namespace Eigen;

	class Sampler : public BaseLD {
		public:
			virtual void operator()(const ArrayXXf& _data, ArrayXXf& _sampleIndices, 
				const ulli& _numSamples, const ulli& _minSamples) = 0;

			virtual void ParseXML() override {
				m_xml = m_xml.child("Solvers").child("TLinkage").child("Samplers");
			}

			Sampler(string _xmlFile) : BaseLD(_xmlFile) { ParseXML(); }
	};
}
#endif
