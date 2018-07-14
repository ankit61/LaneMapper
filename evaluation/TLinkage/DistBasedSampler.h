#ifndef DIST_BASED_SAMPLER_H_
#define DIST_BASED_SAMPLER_H_

#include "Sampler.h"
#include<unordered_set>
#include<algorithm>


namespace LD {

	class DistBasedSampler : public Sampler {
		public:
			virtual void operator()(const ArrayXXf& _data, ArrayXXf& _sampleIndices, 
					const ulli& _numSamples, const ulli& _minSamples) override {
				if(m_debug)
					cout << "Entering DistBasedSampler::operator() " << endl;

				ArrayXXf cdfs;
				_sampleIndices.resize(_minSamples, _numSamples);

				FindCDF(_data, cdfs);

				for(ulli i = 0; i < _numSamples; i++) {
					_sampleIndices(0, i) = i % _data.cols();
					std::unordered_set<ulli> uniques;
					uniques.insert(_sampleIndices(0, i));
					while(uniques.size() != _minSamples) {
						auto it = std::upper_bound(cdfs.col(_sampleIndices(0, i)).data(), cdfs.col(_sampleIndices(0, i)).data() + cdfs.rows(), (double) rand() / RAND_MAX);

						_sampleIndices(uniques.size(), i) = std::distance(cdfs.col(_sampleIndices(0, i)).data(), it);
						if(_sampleIndices(uniques.size(), i) != _data.cols())
							uniques.insert(_sampleIndices(uniques.size(), i));
					}
				}

				if(m_debug)
					cout << "Exiting DistBasedSampler::operator() " << endl;

			}

			virtual void ParseXML() override {
				m_xml = m_xml.child("DistBased");
				m_sigma = m_xml.attribute("sigma").as_float();
				if(!m_sigma)
					throw runtime_error("sigma attribute not there in DistBased Sampler in TLinkage");
			}

			DistBasedSampler(string _xmlFile) : Sampler(_xmlFile) { ParseXML(); }

		protected:
			virtual void FindCDF(const ArrayXXf& _data, ArrayXXf& _cdfs) {
				if(m_debug)
					cout << "Entering DistBasedSampler::FindCDF() " << endl;

				double dist;
				_cdfs.resize(_data.cols(), _data.cols());
				for(ulli r = 0; r < _data.cols(); r++) {
					for(ulli c = 0; c < _data.cols(); c++) {
						dist = Distance(_data.col(r), _data.col(c));
						_cdfs(r, c) = (dist == 0) ? 0 : std::exp(-std::pow(dist, 2) / m_sigma);
					}
				}

				double colSum;
				for(ulli c = 0; c < _cdfs.cols(); c++) {	
					colSum = _cdfs.col(c).sum();
					_cdfs(0, c) /= colSum;
					for(ulli r = 1; r < _cdfs.rows(); r++)
						_cdfs(r, c) = _cdfs(r, c) / colSum + _cdfs(r - 1,c);
				}

				if(m_debug)
					cout << "Exiting DistBasedSampler::FindCDF() " << endl;	
			}

			double Distance(const ArrayXf& pt1, const ArrayXf& pt2) {
				return (pt1 - pt2).matrix().norm();	
			}
			double m_sigma;
	};
}
#endif
