#include"SurfaceTLinkage.h"

namespace LD {

	ArrayXf SurfaceTLinkage::GenerateHypothesis(const vector<ArrayXf>& _samples) {
		if(m_debug)
			cout << "Entering SurfaceTLinkage::GenerateHypothesis()" << endl;

		Matrix<float, 10, 10> A;
		Matrix<float, 10, 1> B;
		for(int r = 0; r < A.rows(); r++) {
			float x = _samples[r](0), y = _samples[r](1);
			A(r, 0) = 1;
			A(r, 1) = x;
			A(r, 2) = y;
			A(r, 3) = std::pow(x, 2);
			A(r, 4) = x * y;
			A(r, 5) = std::pow(y, 2);
			A(r, 6) = std::pow(x, 3);
			A(r, 7) = std::pow(x, 2) * y;
			A(r, 8) = std::pow(y, 2) * x;
			A(r, 9) = std::pow(y, 3);

			B(r) = _samples[r](2); //z
		}

		ArrayXf hypothesis = (A.inverse() * B).array();

		if(m_debug)
			cout << "Exiting SurfaceTLinkage::GenerateHypothesis()" << endl;

		return hypothesis;
	}

	double SurfaceTLinkage::Distance(ArrayXf _dataPoint, ArrayXf _model) {
		//absolute difference between zs
		Matrix<float, 10, 1> A;
		float x = _dataPoint(0), y = _dataPoint(1);
		A << 1, 
			 x,
			 y,
			 std::pow(x, 2),
			 x * y,
			 std::pow(y, 2),
			 std::pow(x, 3),	 
			 std::pow(x, 2) * y,
			 x * std::pow(y, 2),
			 std::pow(y, 3);

		return std::abs(_dataPoint(2) - A.dot(_model.matrix()));
	}

	void SurfaceTLinkage::FitModel(const ArrayXXf& _cluster, ArrayXf& _model) {
		if(m_debug)
			cout << "Entering SurfaceTLinkage::FitModel()" << endl;
		
		//Least squares approach
		MatrixXf A(_cluster.rows(), 10);
		
		A.col(0).setOnes();
		A.col(1) = _cluster.col(0);
		A.col(2) = _cluster.col(1);
		A.col(3) = _cluster.col(0).pow(2);
		A.col(4) = _cluster.col(0) * _cluster.col(1);
		A.col(5) = _cluster.col(1).pow(2);
		A.col(6) = _cluster.col(0).pow(3);		 
		A.col(7) = _cluster.col(0).pow(2) * _cluster.col(1);
		A.col(8) = _cluster.col(0) * _cluster.col(1).pow(2);
		A.col(9) = _cluster.col(1).pow(3);

		_model = A.bdcSvd(ComputeThinU | ComputeThinV).solve(_cluster.col(2).matrix());

		if(m_debug)
			cout << "Exiting SurfaceTLinkage::FitModel()" << endl;
	}
}
