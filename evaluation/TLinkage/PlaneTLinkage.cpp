#include "PlaneTLinkage.h"
#include<Eigen/Geometry>
#include<Eigen/Eigenvalues>

void PlaneTLinkage::GenerateHypothesis(const ArrayXXf& _data, const ArrayXXf& _sampleIndices, ArrayXXf& _hypotheses) {
	if(m_debug)
		cout << "Entering PlaneTLinkage::GenerateHypothesis()" << endl;
	
	_hypotheses = ArrayXXf(m_modelParams, _sampleIndices.cols()); 

	Array3f pt1, pt2, pt3;

	for(ulli i = 0; i < _sampleIndices.cols(); i++) {
		pt1 = _data.col(_sampleIndices(0, i));
		pt2 = _data.col(_sampleIndices(1, i));
		pt3 = _data.col(_sampleIndices(2, i));

		_hypotheses.col(i).topRows(3).matrix() = (pt1 - pt2).matrix().cross((pt3 - pt1).matrix());
		_hypotheses.col(i).topRows(3).matrix().normalize();
		_hypotheses(3, i) = - _hypotheses(0, i) * pt1(0) - _hypotheses(1, i) * pt1(1) - _hypotheses(2, i) * pt1(2);
	}

	if(m_debug)
		cout << "Exiting PlaneTLinkage::GenerateHypothesis()" << endl;
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
	_model = ArrayXf(m_modelParams);
	_model.head(3) = solver.eigenvectors().col(minEigenValueIndex).real();
	_model.head(3).matrix().normalize();
	_model(3) = - _model.head(3).matrix().dot(means.matrix());
	
	if(m_debug)
		cout << "Exiting PlaneTLinkage::FitModel()" << endl;
} 
