#include"KLargestOR.h"
#include<unordered_map>
#include<unordered_set>

namespace LD {

	void KLargestOR::operator()(const ArrayXf& _clusters, ArrayXf& _out) {
		if(m_debug)
			cout << "Entering KLargestOR::operator()" << endl;
		
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
		
		int k = std::min(m_k, int(count.size()));

		std::partial_sort(count.begin(), count.begin() + k, count.end(), 
				[](std::pair<ulli, ulli>& a, std::pair<ulli, ulli>& b) {
				return a.second > b.second;	
				});

		std::unordered_set<ulli> clustersToKeep;

		for(ulli i = 0; i < k; i++)
			clustersToKeep.insert(count[i].first);

		_out = _clusters;
		for(ulli i = 0; i < _clusters.size(); i++)
			if(clustersToKeep.find(_clusters(i)) == clustersToKeep.end())
				_out(i) = -1;
		
		if(m_debug)
			cout << "Exiting KLargestOR::operator()" << endl;
	}

	void KLargestOR::ParseXML() {
		m_xml = m_xml.child("KLargest");
		m_k = m_xml.attribute("k").as_int();
		if(!m_k)
			throw runtime_error("at least one of the following attributes is missing in KLargest node: k");
	}
}
