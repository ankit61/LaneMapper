#include "TLinkage.h"
#include"UnionFind.h"
#include<cstdlib>
#include<limits>

//TODO: Make a Run function

void TLinkage::SetSampler(SamplingMethod _method) {
	switch(_method) {
		case PREFER_NEAR:
			m_sampler = std::make_unique<DistBasedSampler>();
			break;
		case UNIFORM:
			m_sampler = std::make_unique<UniformSampler>();
			break;
	}
}

void TLinkage::SetOutlierRejector(OutlierRejectionMethod _method) {
}

void TLinkage::Sample(const ArrayXXf& _data, ArrayXXf& _sampleIndices, 
	const unsigned long long int& _numSamples) {
	
	if(m_debug)
		cout << "Entering TLinkage::Sample()" << endl;
	
	if(!m_sampler)
		throw runtime_error("Sampler not set");

	(*m_sampler)(_data, _sampleIndices, _numSamples, m_minSamples);	
	
	if(m_debug)
		cout << "Exiting TLinkage::Sample()" << endl;
}

void TLinkage::FindResiduals(const ArrayXXf& _data, const ArrayXXf& _hypotheses, 
	ArrayXXf& _residuals) {
	
	if(m_debug)
		cout << "Entering TLinkage::FindResiduals()" << endl;

	_residuals = ArrayXXf(_hypotheses.cols(), _data.cols());  //column major memory considered

	for(ulli i = 0; i < _residuals.rows(); i++)
		for(ulli j = 0; j < _residuals.cols(); j++)
			_residuals(i, j) = Distance(_data.col(j), _hypotheses.col(i));

	if(m_debug)
		cout << "Exiting TLinkage::FindResiduals()" << endl;	
}

void TLinkage::FindPreferences(const ArrayXXf& _residuals, 
	ArrayXXf& _preferences, VotingScheme _method, const double& _eps) {
	
	if(m_debug)
		cout << "Entering TLinkage::FindPreferences()" << endl;

	_preferences = ArrayXXf::Zero(_residuals.rows(), _residuals.cols());
	
	switch(_method) {
		case EXP: {
			double tau = _eps / 5;
			for(ulli r = 0; r < _residuals.rows(); r++)
				for(ulli c = 0; c < _residuals.cols(); c++)
					if(_residuals(r, c) < _eps)
						_preferences(r, c) = std::exp(-_residuals(r, c) / tau);
			break;
		}
		case GAUSS: {
			double sigma = _eps / 4;  
			for(ulli r = 0; r < _residuals.rows(); r++)
				for(ulli c = 0; c < _residuals.cols(); c++)
					if(_residuals(r, c) < _eps)
						_preferences(r, c) = std::exp(-std::pow(_residuals(r, c), 2) / std::pow(sigma, 2));
			break;
		}
		case HARD: {
			for(ulli r = 0; r < _residuals.rows(); r++)
				for(ulli c = 0; c < _residuals.cols(); c++)
				   if(_residuals(r, c) < _eps)
					   _preferences(r, c) = 1;
			break;
		}
		case TUKEY: {
			for(ulli r = 0; r < _residuals.rows(); r++)
				 for(ulli c = 0; c < _residuals.cols(); c++)
					 if(_residuals(r, c) < _eps)
						 _preferences(r, c) = std::pow(1 - std::pow(_residuals(r, c) / _eps , 2), 3);
			break;
		}
		default:
			throw runtime_error("Voting scheme not available");
	}

	if(m_debug)
		cout << "Exiting TLinkage::FindPreferences()" << endl;

}

void TLinkage::Cluster(ArrayXXf& _preferences, ArrayXf& _clusters) {
	if(m_debug)
		cout << "Entering TLinkage::Cluster()" << endl;
	
	ArrayXXf distances;
	CalculateTanimotoDist(_preferences, distances);

	_clusters = ArrayXf(_preferences.cols());

	UnionFind uf(_preferences.cols());

	//find min dist
	ulli pt1, pt2;
	double minDist = distances.minCoeff(&pt1, &pt2);
	while(minDist < 1) {
	
		//remove other point
		if(pt2 > 0)
			distances.col(pt2).head(pt2) = std::numeric_limits<float>::max();
		distances.row(pt2).tail(distances.cols() - pt2) = std::numeric_limits<float>::max();
		
		//modify the preferences of the retained point so it expresses preferences of the union
		_preferences.col(pt1) =  _preferences.col(pt1).min(_preferences.col(pt2));

		//update the distances matrix
		for(ulli i = 0; i < pt1; i++)
			if(distances(i, pt1) != std::numeric_limits<float>::max())
				distances(i, pt1) = Tanimoto(_preferences.col(pt1), _preferences.col(i));

		for(ulli i = pt1 + 1; i < distances.cols(); i++)
			if(distances(pt1, i) != std::numeric_limits<float>::max())
				distances(pt1, i) = Tanimoto(_preferences.col(pt1), _preferences.col(i));
		uf.unionSet(pt1, pt2);
		minDist = distances.minCoeff(&pt1, &pt2);
	}

	for(ulli i = 0; i < _clusters.size(); i++)  //copy union find vector to Eigen::Array
		_clusters(i) = uf.findSet(i);

	if(m_debug)
		cout << "Exiting TLinkage::Cluster()" << endl;
}

void TLinkage::CalculateTanimotoDist(const ArrayXXf& _preferences, ArrayXXf& _distances) {
	if(m_debug)
		cout << "Entering TLinkage::CalculateTanimotoDist()" << endl;
	_distances = ArrayXXf(_preferences.cols(), _preferences.cols());
	//create symmetric matrix without storing two copies
	for(ulli r = 0; r < _distances.rows(); r++) {
		for(ulli c = 0; c < _distances.cols(); c++) {
			if(c <= r)
				_distances(r, c) = std::numeric_limits<float>::max(); //so it's not min ever
			else 
				_distances(r, c) = Tanimoto(_preferences.col(r), _preferences.col(c));
		}
	}
	
	if(m_debug)
		cout << "Exiting TLinkage::CalculateTanimotoDist()" << endl;
}

float TLinkage::Tanimoto(const ArrayXf& _a, const ArrayXf& _b) {
	float dotProd = _a.matrix().dot(_b.matrix());
	return 1 - dotProd / (_a.matrix().squaredNorm() + _b.matrix().squaredNorm() - dotProd);
}

void TLinkage::RejectOutliers(const ArrayXf& _clusters, ArrayXf& _out) {
	
	if(m_debug)
		cout << "Entering TLinkage::RejectOutliers()" << endl;
	
	if(!m_outlierRejector)
		throw runtime_error("OutlierRejector not set");
	
	(*m_outlierRejector)(_clusters, _out);
	
	if(m_debug)
		cout << "Exiting TLinkage::RejectOutliers()" << endl;
}

void TLinkage::FitModels(const ArrayXXf& _data, const ArrayXf& _clusters, vector<ArrayXf>& _models, const int& _noiseIndex) {
	if(m_debug)
		cout << "Entering TLinkage::FitModels()" << endl;
	
	vector<vector<float> > clusteredPoints;
	std::unordered_map<long long int, ulli> clusterID2Index;
	
	for(ulli i = 0; i < _clusters.size(); i++) {
		if(_clusters(i) != _noiseIndex) {
			if(clusterID2Index.find(_clusters(i)) == clusterID2Index.end()) {
				clusterID2Index[_clusters(i)] = clusteredPoints.size();
				clusteredPoints.push_back(vector<float>());
			}
			for(ulli j = 0; j < _data.rows(); j++)
				clusteredPoints[clusterID2Index[_clusters(i)]].push_back(_data(j, i));
		}
	}

	if(m_debug)
		cout << "Clubbed points of the same cluster" << endl;

	vector<ArrayXXf> clusters(clusteredPoints.size());
	_models.resize(clusteredPoints.size());

	for(ulli i = 0; i < clusters.size(); i++) {
	
		clusters[i] = Map<Matrix<float, Dynamic, Dynamic, RowMajor>, Unaligned, OuterStride<> >(clusteredPoints[i].data(),
						clusteredPoints[i].size() / _data.rows(), _data.rows(), 
						OuterStride<>(_data.rows())).array(); //this is a deep copy
		FitModel(clusters[i], _models[i]);
	}

	if(m_debug)
		cout << "Exiting TLinkage::FitModels()" << endl;
}
