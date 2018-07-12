#include"Line2DTLinkage.h"

namespace LD {

	ArrayXf Line2DTLinkage::GenerateHypothesis(const vector<ArrayXf>& _samples) {
		if(m_debug)
			cout << "Entering Line2DTLinkage::GenerateHypothesis()" << endl;

		ArrayXf hypothesis(m_modelParams); 

		hypothesis(0) = _samples[0](1) - _samples[1](1); //y1 - _samples[1](1)
		hypothesis(1) = _samples[1](0) - _samples[0](0); //_samples[1](0) - x1
		hypothesis(2) = (_samples[0](0) - _samples[1](0)) * _samples[0](1) + (_samples[1](1) - _samples[0](1)) * _samples[0](0);
		
		if(m_debug)
			cout << "Exiting Line2DTLinkage::GenerateHypothesis()" << endl;

		return hypothesis;
	}

	double Line2DTLinkage::Distance(ArrayXf _dataPoint, ArrayXf _model) {
		return std::abs(_dataPoint(0) * _model(0) + _dataPoint(1) * _model(1) + _model(2)) / sqrt(std::pow(_model(0), 2) + std::pow(_model(1), 2)); //ax+by+c / sqrt(a^2 + b^2)
	}

	void Line2DTLinkage::FitModel(const ArrayXXf& _cluster, ArrayXf& _model) {
		//FIXME: This wouldn't fit lines appropriately if the least square line is kind of vertical.  We can add this functionality too at the cost of over double the computation
		if(m_debug)
			cout << "Entering Line2DTLinkage::FitModel()" << endl;

		//Find Least Squares Lines
		Matrix<float, Dynamic, Dynamic, RowMajor> A(_cluster.rows(), 2);
		A.col(0) = _cluster.col(0); //involves a copy :(
		A.col(1).setOnes();
		_model = A.bdcSvd(ComputeThinU | ComputeThinV).solve(_cluster.col(1).matrix()); 

		if(m_debug)
			cout << "Exiting Line2DTLinkage::FitModel()" << endl;
	}

}
