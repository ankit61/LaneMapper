#include"RoadTLinkage.h"

namespace LD {

	ArrayXf RoadTLinkage::GenerateHypothesis(const vector<ArrayXf>& _samples) {
		if(m_debug)
			cout << "Entering RoadTLinkage::GenerateHypothesis()" << endl;

		Matrix<float, 5, 5> A;
		Matrix<float, 5, 1> B;
		for(int r = 0; r < A.rows(); r++) {
			float x = _samples[r](0), y = _samples[r](1);
			A(r, 0) = 1;
			A(r, 1) = x;
			A(r, 2) = y;
			A(r, 3) = std::pow(x, 2);
			A(r, 4) = std::pow(y, 2);
			
			B(r) = _samples[r](2); //z
		}

		ArrayXf hypothesis = (A.inverse() * B).array();

		if(m_debug)
			cout << "Exiting RoadTLinkage::GenerateHypothesis()" << endl;

		return hypothesis;
	}

	double RoadTLinkage::Distance(ArrayXf _dataPoint, ArrayXf _model) {
		//absolute difference between zs
		Matrix<float, 5, 1> A;
		float x = _dataPoint(0), y = _dataPoint(1);
		A << 1, 
			 x,
			 y,
			 std::pow(x, 2),
			 std::pow(y, 2);

		return std::abs(_dataPoint(2) - A.dot(_model.matrix()));
	}

	void RoadTLinkage::FitModel(const ArrayXXf& _cluster, ArrayXf& _model) {
		if(m_debug)
			cout << "Entering RoadTLinkage::FitModel()" << endl;
		
		//Least squares approach
		MatrixXf A(_cluster.rows(), m_modelParams);
		
		A.col(0).setOnes();
		A.col(1) = _cluster.col(0);
		A.col(2) = _cluster.col(1);
		A.col(3) = _cluster.col(0).pow(2);
		A.col(4) = _cluster.col(1).pow(2);

		_model = A.bdcSvd(ComputeThinU | ComputeThinV).solve(_cluster.col(2).matrix());

		if(m_debug)
			cout << "Exiting RoadTLinkage::FitModel()" << endl;
	}
}
