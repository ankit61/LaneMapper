#include "PlaneTLinkage.h"
#include<Eigen/Geometry>
#include<Eigen/Eigenvalues>

namespace LD {

	ArrayXf PlaneTLinkage::GenerateHypothesis(const vector<ArrayXf>& _samples) {
		
		ArrayXf hypothesis(m_modelParams); 

		hypothesis.head(3).matrix() = (_samples[0].head<3>() - _samples[1].head<3>()).matrix().cross((_samples[2].head<3>() - _samples[0].head<3>()).matrix());
		hypothesis.head(3).matrix().normalize();
		hypothesis(3) = - hypothesis(0) * _samples[0](0) - hypothesis(1) * _samples[0](1) - hypothesis(2) * _samples[0](2);

		return hypothesis;
	}

	double PlaneTLinkage::Distance(ArrayXf _dataPoint, ArrayXf _model) {
		return std::abs(_model.head(3).matrix().dot(_dataPoint.matrix()) + _model(3));
	}

	void PlaneTLinkage::FitModel(const ArrayXXf& _cluster, ArrayXf& _model) {
		
		if(m_debug)
			cout << "Entering PlaneTLinkage::FitModel()" << endl;

		ArrayXf means = _cluster.colwise().mean();
		MatrixXf zeroCentered =  (_cluster.rowwise() - means.transpose()).matrix();
		EigenSolver<MatrixXf> solver(zeroCentered.transpose() * zeroCentered);
		ulli minEigenValueIndex;
		solver.eigenvalues().real().array().minCoeff(&minEigenValueIndex); //eigen values will be real as matrix is symmetric
		_model.resize(m_modelParams);
		_model.head(3) = solver.eigenvectors().col(minEigenValueIndex).real();
		_model.head(3).matrix().normalize();
		_model(3) = - _model.head(3).matrix().dot(means.matrix());
		
		if(m_debug)
			cout << "Exiting PlaneTLinkage::FitModel()" << endl;
	}

}
