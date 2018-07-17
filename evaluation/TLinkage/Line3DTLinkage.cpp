#include"Line3DTLinkage.h"
#include<Eigen/Geometry>

namespace LD {

	ArrayXf Line3DTLinkage::GenerateHypothesis(const vector<ArrayXf>& _samples) {
		if(m_debug)
			cout << "Entering Line3DTLinkage::GenerateHypothesis()" << endl;

		ArrayXf hypothesis(m_modelParams); 

		hypothesis.head<3>() = _samples[0];
		hypothesis.tail<3>() = (_samples[1] - _samples[0]).matrix().normalized();
		

		if(m_debug)
			cout << "Exiting Line3DTLinkage::GenerateHypothesis()" << endl;

		return hypothesis;
	}

	double Line3DTLinkage::Distance(ArrayXf _dataPoint, ArrayXf _model) {
		Array3f ptOnLine = _model.head<3>() + _model.tail<3>();
		Array3f diff1	 = _dataPoint - ptOnLine; 
		Array3f diff2	 = _dataPoint - _model.head<3>();
		return (diff1).matrix().cross((diff2).matrix()).norm() / _model.tail<3>().matrix().norm();
	}

	void Line3DTLinkage::FitModel(const ArrayXXf& _cluster, ArrayXf& _model) {
		if(m_debug)
			cout << "Entering Line3DTLinkage::FitModel()" << endl;

		//Find Least Squares Lines
		ArrayXf means = _cluster.colwise().mean();
		MatrixXf zeroCentered =  (_cluster.rowwise() - means.transpose()).matrix();
		EigenSolver<MatrixXf> solver(zeroCentered.transpose() * zeroCentered);
		ulli maxEigenValueIndex;
		solver.eigenvalues().real().array().maxCoeff(&maxEigenValueIndex); //eigen values will be real as matrix is symmetric	
		
		_model.resize(m_modelParams);
		_model.head<3>() = means;
		_model.tail<3>() = solver.eigenvectors().col(maxEigenValueIndex).real().normalized();

		if(m_debug)
			cout << "Exiting Line3DTLinkage::FitModel()" << endl;
	}
}
