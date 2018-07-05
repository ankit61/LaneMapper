#include"Line2DTLinkage.h"

void Line2DTLinkage::GenerateHypothesis(const ArrayXXf& _data, 
	const ArrayXXf& _sampleIndices, ArrayXXf& _hypotheses) {
	if(m_debug)
		cout << "Entering Line2DTLinkage::GenerateHypothesis()" << endl;
	
	_hypotheses = ArrayXXf(3, _sampleIndices.cols()); 
	
	Array2f pt1, pt2;

	for(ulli i = 0; i < _sampleIndices.cols(); i++) {
		pt1 = _data.col(_sampleIndices(0, i));
		pt2 = _data.col(_sampleIndices(1, i));
		
		_hypotheses(0, i) = pt1(1) - pt2(1); //y1 - pt2(1)
		_hypotheses(1, i) = pt2(0) - pt1(0); //pt2(0) - x1
		_hypotheses(2, i) = (pt1(0) - pt2(0)) * pt1(1) + (pt2(1) - pt1(1)) * pt1(0);
	}

	if(m_debug)
		cout << "Exiting Line2DTLinkage::GenerateHypothesis()" << endl;
}

double Line2DTLinkage::Distance(ArrayXf _dataPoint, ArrayXf _model) {
	return std::abs(_dataPoint(0) * _model(0) + _dataPoint(1) * _model(1) + _model(2)) / sqrt(std::pow(_model(0), 2) + std::pow(_model(1), 2)); //ax+by+c / sqrt(a^2 + b^2)
}

void Line2DTLinkage::FitModel(const ArrayXXf& _data, const ArrayXf& _clusters, ArrayXXf& _models, const int& _noiseIndex) {
	//FIXME: This wouldn't fit lines appropriately if the least square line is kind of vertical.  We can add this functionality too at the cost of over double the computation
	if(m_debug)
		cout << "Entering Line2DTLinkage::FitModel()" << endl;
	
	vector<vector<float> > clusteredPoints;
	std::unordered_map<long long int, ulli> clusterID2Index;
	
	for(ulli i = 0; i < _clusters.size(); i++) {
		if(_clusters(i) != _noiseIndex) {
			if(clusterID2Index.find(_clusters(i)) == clusterID2Index.end()) {
				clusterID2Index[_clusters(i)] = clusteredPoints.size();
				clusteredPoints.push_back(vector<float>());
			}
			clusteredPoints[clusterID2Index[_clusters(i)]].push_back(_data(0, i));
			clusteredPoints[clusterID2Index[_clusters(i)]].push_back(_data(1, i));
		}
	}

	if(m_debug)
		cout << "Clubbed points of the same cluster" << endl;

	vector<Matrix<float, Dynamic, Dynamic, RowMajor> > clustered(clusteredPoints.size());
	for(ulli i = 0; i < clustered.size(); i++) {
		clustered[i] = Matrix<float, Dynamic, Dynamic, RowMajor>(clusteredPoints[i].size() / 2, 3);
		clustered[i].col(0) = Matrix<float, Dynamic, Dynamic, RowMajor>::Ones(clusteredPoints[i].size() / 2, 1);
		clustered[i].rightCols(2) = Map<Matrix<float, Dynamic, Dynamic, RowMajor>, Unaligned, OuterStride<2> >(clusteredPoints[i].data(), clusteredPoints[i].size() / 2, 2); //this is a deep copy
	}

	if(m_debug)
		cout << "Finding least square error line" << endl;
	
	//Find Least Squares Lines
	
	_models = Array<float, Dynamic, Dynamic>(2, clustered.size());

	for(ulli i = 0; i < clustered.size(); i++) {
		_models.col(i) = clustered[i].leftCols(2).bdcSvd(ComputeThinU | ComputeThinV).solve(clustered[i].col(2));
	}

	if(m_debug)
		cout << "Exiting Line2DTLinkage::FitModel()" << endl;
}
