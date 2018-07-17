#ifndef K_LARGEST_OR_H_
#define K_LARGEST_OR_H_

#include"OutlierRejector.h"
#include<unordered_map>

//TODO: Make the function that calculates clusters also calculate their respective sizes

namespace LD {
	using namespace Eigen;

	class KLargestOR : public OutlierRejector {
		public:
			virtual void operator()(const ArrayXf& _clusters, ArrayXf& _out) {
				
				std::unordered_map<ulli, ulli> clusterID2Index;
				vector<std::pair<ulli, ulli> > count;

				for(ulli i = 0; i < _clusters.size(); i++) {
					if(clusterID2Index.find(_clusters(i)) == clusterID2Index.end()) {
						clusterID2Index[_clusters(i)] = count.size();
						count.push_back(std::make_pair(_clusters(i), 1));
					}
					else
						count[clusterID2Index[_clusters(i)]].second++;
				}

				m_k = std::min(m_k, int(count.size()));

				std::partial_sort(count.begin(), count.begin() + m_k, count.end(), 
				[](std::pair<ulli, ulli>& a, std::pair<ulli, ulli>& b) {
					return a.second > b.second;	
				});

				std::unordered_set<ulli> clustersToKeep;

				for(ulli i = 0; i < m_k; i++)
					clustersToKeep.insert(count[i].first);
			
				_out = _clusters;
				for(ulli i = 0; i < _clusters.size(); i++)
					if(clustersToKeep.find(_clusters(i)) == clustersToKeep.end())
						_out(i) = -1;
			}

			virtual void ParseXML() override { 
				m_xml = m_xml.child("KLargest");
				m_k = m_xml.attribute("k").as_int();
				if(!m_k)
					throw runtime_error("at least one of the following attributes is missing in KLargest node: k");
			}

			KLargestOR(string _xmlFile) : OutlierRejector(_xmlFile) { ParseXML(); }

		protected:
			int m_k;
	};
}
#endif
