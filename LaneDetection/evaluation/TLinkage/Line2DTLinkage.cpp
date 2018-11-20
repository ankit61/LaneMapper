#include"Line2DTLinkage.h"

namespace LD {

	ArrayXf Line2DTLinkage::GenerateHypothesis(const vector<ArrayXf>& _samples) {

		ArrayXf hypothesis(m_modelParams); 

		hypothesis(0) = _samples[0](1) - _samples[1](1); //y1 - y2
		hypothesis(1) = _samples[1](0) - _samples[0](0); //x2 - x1
		hypothesis(2) = (_samples[0](0) - _samples[1](0)) * _samples[0](1) + (_samples[1](1) - _samples[0](1)) * _samples[0](0);

		return hypothesis;
	}

	double Line2DTLinkage::Distance(ArrayXf _dataPoint, ArrayXf _model) {
		return std::abs(_dataPoint(0) * _model(0) + _dataPoint(1) * _model(1) + _model(2)) / sqrt(std::pow(_model(0), 2) + std::pow(_model(1), 2)); //ax+by+c / sqrt(a^2 + b^2)
	}

	void Line2DTLinkage::FitModel(const ArrayXXf& _cluster, ArrayXf& _model) {
		if(m_debug)
			cout << "Entering Line2DTLinkage::FitModel()" << endl;

		//Find Least Squares Lines
		Matrix<float, Dynamic, Dynamic, RowMajor> A(_cluster.rows(), 2);
		A.col(0) = _cluster.col(0);
		A.col(1).setOnes();
		ArrayXf model1 = A.bdcSvd(ComputeThinU | ComputeThinV).solve(_cluster.col(1).matrix()); 

		double error1 = (_cluster.col(1).matrix() - A * model1.matrix()).norm();

		A.col(0) = _cluster.col(1);
		ArrayXf model2 = A.bdcSvd(ComputeThinU | ComputeThinV).solve(_cluster.col(0).matrix());
		
		double error2 =  (_cluster.col(1).matrix() - A * model2.matrix()).norm();
		
		_model.resize(m_modelParams);
		if(error1 < error2) { //in case line is vertcal
			_model(0) = model1(0);
			_model(1) = -1;
			_model(2) = model1(1);
		}
		else {
			_model.tail<2>() = model2;
			_model(0) = -1;
		}

		if(m_debug)
			cout << "Exiting Line2DTLinkage::FitModel()" << endl;
	}

}
