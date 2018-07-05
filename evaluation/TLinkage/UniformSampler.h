#ifndef UNIFORM_SAMPLER_H_
#define UNIFORM_SAMPLER_H_

#include "Sampler.h"
#include<unordered_set>

class UniformSampler : public Sampler {
	public:
		virtual void operator()(const ArrayXXf& _data, ArrayXXf& _sampleIndices, 
			const ulli& _numSamples, const ulli& _minSamples) {

			if(m_debug)
				cout << "Entering UniformSampler::operator() " << endl;

			_sampleIndices = ArrayXXf(_minSamples, _numSamples);

			for(ulli i = 0; i < _numSamples; i++) {
				_sampleIndices(0, i) = i % _data.cols();
				std::unordered_set<ulli> uniques;
				uniques.insert(_sampleIndices(0,i));
				while(uniques.size() != _minSamples) {
					_sampleIndices(uniques.size(), i) = rand() % _data.cols();
					uniques.insert(_sampleIndices(uniques.size(), i));
				}
			}

			if(m_debug)
				cout << "Exiting UniformSampler::operator() " << endl;
		}
};

#endif
