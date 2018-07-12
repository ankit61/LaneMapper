#include"Circle2DTLinkage.h"

namespace LD {

	void Circle2DTLinkage::GenerateHypothesis(const ArrayXXf& _data, const ArrayXXf& _sampleIndices, ArrayXXf& _hypotheses) {
		if(m_debug)
			cout << "Entering Circle2DTLinkageTLinkage::GenerateHypothesis()" << endl;

		_hypotheses = ArrayXXf(m_modelParams, _sampleIndices.cols()); 

		Array2f pt1, pt2, pt3;

		for(ulli i = 0; i < _sampleIndices.cols(); i++) {
			pt1 = _data.col(_sampleIndices(0, i));
			pt2 = _data.col(_sampleIndices(1, i));
			pt3 = _data.col(_sampleIndices(2, i));
			Matrix3f M;
			Matrix<float, 3, 1> b;

			M << pt1(0), pt1(1), 1, 
			  pt2(0), pt2(1), 1, 
			  pt3(0), pt3(1), 1;

			b << -pt1.matrix().squaredNorm(), 
			  -pt2.matrix().squaredNorm(), 
			  -pt3.matrix().squaredNorm();
			_hypotheses.col(i).matrix() = M.inverse() * b;
			_hypotheses(0, i) = - _hypotheses(0, i) / 2; //x center
			_hypotheses(1, i) = - _hypotheses(1, i) / 2; //y center
			_hypotheses(2, i) = std::sqrt(_hypotheses.col(i).head(2).matrix().squaredNorm() - _hypotheses(2, i)); //radius
		}

		if(m_debug)
			cout << "Exiting Circle2DTLinkageTLinkage::GenerateHypothesis()" << endl;

	}

	double Circle2DTLinkage::Distance(ArrayXf _dataPoint, ArrayXf _model) {
		return std::abs((_model.head(2) - _dataPoint).matrix().norm() - _model(2));
	}

	void Circle2DTLinkage::FitModel(const ArrayXXf& _cluster, ArrayXf& _model) {

		if(m_debug)
			cout << "Entering Circle2DTLinkage::FitModel()" << endl;

		MatrixXf A(_cluster.rows(), 3);
		A << _cluster.col(0), _cluster.col(1), MatrixXf::Ones(_cluster.rows(), 1);

		_model = A.bdcSvd(ComputeThinU | ComputeThinV).solve(-_cluster.matrix().rowwise().squaredNorm()); 

		_model(0) /= -2;
		_model(1) /= -2;
		_model(2) = std::sqrt(std::pow(_model(0), 2) + std::pow(_model(1), 2) - _model(2));

		if(m_debug)
			cout << "Exiting Circle2DTLinkage::FitModel()" << endl;
	}
}
