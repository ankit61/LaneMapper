#include"Line2DTLinkage.h"

namespace LD {

	void Line2DTLinkage::GenerateHypothesis(const ArrayXXf& _data, 
			const ArrayXXf& _sampleIndices, ArrayXXf& _hypotheses) {
		if(m_debug)
			cout << "Entering Line2DTLinkage::GenerateHypothesis()" << endl;

		_hypotheses = ArrayXXf(m_modelParams, _sampleIndices.cols()); 

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
