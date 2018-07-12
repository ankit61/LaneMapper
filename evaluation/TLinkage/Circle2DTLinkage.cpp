#include"Circle2DTLinkage.h"

namespace LD {

	ArrayXf Circle2DTLinkage::GenerateHypothesis(const vector<ArrayXf>& _samples) {
		if(m_debug)
			cout << "Entering Circle2DTLinkageTLinkage::GenerateHypothesis()" << endl;

		ArrayXf hypothesis(m_modelParams); 

		Matrix3f M;
		Matrix<float, 3, 1> b;

		M << _samples[0](0), _samples[0](1), 1, 
		  _samples[1](0), _samples[1](1), 1, 
		  _samples[2](0), _samples[2](1), 1;

		b << -_samples[0].matrix().squaredNorm(), 
		  -_samples[1].matrix().squaredNorm(), 
		  -_samples[2].matrix().squaredNorm();

		hypothesis.matrix() = M.inverse() * b;
		hypothesis(0) = - hypothesis(0) / 2; //x center
		hypothesis(1) = - hypothesis(1) / 2; //y center
		hypothesis(2) = std::sqrt(hypothesis.head(2).matrix().squaredNorm() - hypothesis(2)); //radius

		if(m_debug)
			cout << "Exiting Circle2DTLinkageTLinkage::GenerateHypothesis()" << endl;
		
		return hypothesis;
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
