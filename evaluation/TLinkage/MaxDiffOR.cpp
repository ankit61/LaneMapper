#include"MaxDiffOR.h"

namespace LD {
	void MaxDiffOR::operator()(const ArrayXf& _clusters, ArrayXf& _out) {
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

		count.push_back(std::make_pair(count.size(), m_minSamples)); //add dummy

		sort(count.begin(), count.end(), [](std::pair<ulli, ulli>& a, std::pair<ulli, ulli>& b) {
				return a.second < b.second;	
				});

		ulli maxChange = 0, maxChangeIndex = 0;

		for(ulli i = 0; i < count.size() - 1; i++) {
			ulli diff = count[i + 1].second - count[i].second;
			if(diff > maxChange && i + 1 != count.size() - 1) {
				maxChange = diff;
				maxChangeIndex = i + 1;
			}
		}

		std::unordered_set<ulli> clustersToDiscard;
		for(ulli i = 0; i < maxChangeIndex; i++)
			clustersToDiscard.insert(count[i].first);

		_out = _clusters;
		for(ulli i = 0; i < _clusters.size(); i++)
			if(clustersToDiscard.find(_out(i)) != clustersToDiscard.end())
				_out(i) = -1;
	}
}
