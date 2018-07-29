#include"Line3DTLinkage.h"
#include<Eigen/Geometry>

namespace LD {

	void Line3DTLinkage::ParseXML() {
		if(m_debug)
			cout << "Entering Line3DTLinkage::ParseXML()" << endl;

		m_xml = m_xml.child("Models").child("Line3D");
		m_resolution = m_xml.attribute("resolution").as_float(0);
		m_maxShift 	 = m_xml.attribute("maxShift").as_float(0);
		m_minShift 	 = m_xml.attribute("minShift").as_float(0);
		m_errorThreshold = m_xml.attribute("errorThreshold").as_float(0);

		if(m_resolution <= 0 || m_minShift <= 0 || m_maxShift <= 0 || m_errorThreshold <= 0)
			throw runtime_error("at least one of the following attributes in Line3D node is missing/invalid: resolution, minShift, maxShift, m_errorThreshold");
		
		if(m_debug)
			cout << "Entering Line3DTLinkage::ParseXML()" << endl;
	};
	
	ArrayXf Line3DTLinkage::GenerateHypothesis(const vector<ArrayXf>& _samples) {

		ArrayXf hypothesis(m_modelParams); 

		hypothesis.head<3>() = _samples[0];
		hypothesis.tail<3>() = (_samples[1] - _samples[0]).matrix().normalized();
		
		return hypothesis;
	}

	double Line3DTLinkage::Distance(ArrayXf _dataPoint, ArrayXf _model) {
		Array3f ptOnLine = _model.head<3>() + _model.tail<3>();
		Array3f diff1	 = _dataPoint - ptOnLine; 
		Array3f diff2	 = _dataPoint - _model.head<3>();
		return diff1.matrix().cross(diff2.matrix()).norm() / _model.tail<3>().matrix().norm();
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


	void Line3DTLinkage::RefineModels(const vector<ArrayXf>& _models, const ArrayXXf& _data, const ArrayXf& _clusters, const std::unordered_map<int, ulli>& _clusterID2Index,
		const std::unordered_map<int, vector<ulli> >& _clusterID2PtIndices, vector<ArrayXf>& _refinedModels, const int& _noiseIndex) {
		if(m_debug)
			cout << "Entering Line3DTLinkage::RefineModels()" << endl;
	
		//Find total number of clusters
		if(_clusterID2PtIndices.size() > 2)
			cout << "-----------------WARNING: Too many clusters: " << _clusterID2PtIndices.size() << "----------------------------" << endl;

		//Find largest model

		vector<std::pair<int, ulli> > count;
		vector<ArrayXf> refModelsCopy; 
		
		for(auto it = _clusterID2PtIndices.begin(); it != _clusterID2PtIndices.end(); it++)
			count.push_back(std::make_pair(it->first, it->second.size()));

		int largestClusterID;
		int largestSize = 0;
		for(int i = 0; i < count.size(); i++)
			if(count[i].second > largestSize && count[i].first != _noiseIndex)	
				largestSize = count[i].second, largestClusterID = count[i].first;

		
		ArrayXf originalModel = _models[_clusterID2Index.find(largestClusterID)->second];
		refModelsCopy.push_back(originalModel);

		if(_clusterID2PtIndices.size() > 1) {
			
			//Find largest model on opposite side
			bool isOriginalModelOnRight = IsModelOnRight(originalModel);
			largestSize = 0;
			for(int i = 0; i < count.size(); i++)
				if(count[i].second > largestSize && IsModelOnRight(_models[_clusterID2Index.find(count[i].first)->second]) == !isOriginalModelOnRight && count[i].first != _noiseIndex)
					largestSize = count[i].second, largestClusterID = count[i].first;
			
			if(largestSize > 0) {
				ArrayXf shiftedModel;
				ShiftModel(originalModel, _clusterID2PtIndices.find(largestClusterID)->second, _data, isOriginalModelOnRight, shiftedModel);
				refModelsCopy.push_back(shiftedModel);
			}
			else 
				cout << "-----------------WARNING: No cluster on the opposite side ----------------------------" << endl;
		
		}
		else
			cout << "-----------------WARNING: Only 1 cluster found ----------------------------" << endl;

		_refinedModels = refModelsCopy;

		if(m_debug)
			cout << "Exiting Line3DTLinkage::RefineModels()" << endl;
	}

	void Line3DTLinkage::ShiftModel(ArrayXf& _originalModel, const vector<ulli>& _clusteredIndices, const ArrayXXf& _data, bool _isOriginalModelOnRight, ArrayXf& _shiftedModel) {
		
		_shiftedModel = _originalModel;
		
		//start shifting it to the right/left
		_shiftedModel(1) += (_isOriginalModelOnRight ? -m_minShift : m_minShift);
		float error = EvaluateModel(_shiftedModel, _clusteredIndices, _data);
		float minError = error, minY = _shiftedModel(1);
		while(std::abs(_shiftedModel(1) - _originalModel(1)) < m_maxShift) {
			_shiftedModel(1) += (_isOriginalModelOnRight ? -m_resolution : m_resolution);
			error = EvaluateModel(_shiftedModel, _clusteredIndices, _data);
			if(error < minError) {
				minError = error;
				minY = _shiftedModel(1); 
			}
		}

		_shiftedModel(1) = minY;
		
		if(minError > m_errorThreshold)
			cout << "-----------------WARNING: Error too big " << minError << "----------------------------" << endl;
	}

	bool Line3DTLinkage::IsModelOnRight(const ArrayXf& _model) {
		if(m_debug)
			cout << "Exiting Line3DTLinkage::IsModelOnRight()" << endl;

		float t = -_model(0) / _model(3); //for line <x, y, z> + t * <a, b, c>, x' = x + t * b.  So t = (x' - x) / b.  (here, x' = 0)

		float yIntercept = _model(1) + t * _model(4);

		return yIntercept > 0;

		if(m_debug)
			cout << "Exiting Line3DTLinkage::IsModelOnRight()" << endl;
		
	}

	float Line3DTLinkage::EvaluateModel(const ArrayXf& _model, const vector<ulli>& _clusterIndices, const ArrayXXf& _data) {
		float distance = 0;
		for(int i = 0; i < _clusterIndices.size(); i++)
			distance += Distance(_data.col(_clusterIndices[i]), _model);
		return distance / _clusterIndices.size();
	}
}
