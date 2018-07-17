#include "TLinkage.h"
#include"UnionFind.h"
#include<cstdlib>
#include<limits>
#include<fstream>
#include <boost/algorithm/string/predicate.hpp>

namespace LD {

	
	TLinkage::TLinkage(int _minSamples, int _modelParams, string _xmlFile) : 
		Solver(_xmlFile), m_minSamples(_minSamples), m_modelParams(_modelParams), m_sampler(std::make_unique<UniformSampler>(m_xmlFileName)), m_outlierRejector(std::make_unique<MaxDiffOR>(_minSamples, m_xmlFileName)) { 

			ParseXML(); 
			m_foutModels.open(m_modelFile);
			m_foutClusters.open(m_clusterFile);

		}

	void TLinkage::ParseXML() {
		m_xml = m_xml.child("TLinkage");

		pugi::xml_node solverInstanceNode = m_xml.child("SolverInstance");

		string sampler 			= solverInstanceNode.attribute("sampler").as_string();
		string preferenceFinder = solverInstanceNode.attribute("preferenceFinder").as_string();
		string outlierRejector 	= solverInstanceNode.attribute("outlierRejector").as_string();
		m_samplesPerDataPt 		= solverInstanceNode.attribute("samplesPerDataPt").as_int();
		m_dataFile 				= solverInstanceNode.attribute("dataFile").as_string();
		m_shouldTranspose 		= solverInstanceNode.attribute("shouldTranspose").as_bool();
		m_modelFile 			= solverInstanceNode.attribute("modelFile").as_string();
		m_saveClusters 			= solverInstanceNode.attribute("saveClusters").as_bool(true);
		m_clusterFile 			= solverInstanceNode.attribute("clusterFile").as_string();
	
		if(sampler.empty() || preferenceFinder.empty() || outlierRejector.empty() || m_dataFile.empty() || !m_samplesPerDataPt || m_modelFile.empty() || (m_saveClusters && m_clusterFile.empty())) 
			throw runtime_error("TLinkage SolverInstance node doesn't have one or more of the following attributes: sampler, preferenceFinder, outlierRejector, dataFile, samplesPerDataPt, modelFile, saveClusters, clusterFile");
			
		//Set Sampler
		if(boost::iequals(sampler, "Uniform"))
			SetSampler(UNIFORM);
		else if(boost::iequals(sampler, "PreferNear"))
			SetSampler(PREFER_NEAR);
		else
			throw runtime_error("No such sampler found: " + sampler);

		//Set Outlier Rejector
		if(boost::iequals(outlierRejector, "MaxSizeChange"))
			SetOutlierRejector(MAX_SIZE_CHANGE);
		else if(boost::iequals(outlierRejector, "KLargest"))
			SetOutlierRejector(K_LARGEST);
		else
			throw runtime_error("No such outlier rejector found: " + outlierRejector);

		//Set Preference Finder
		if(boost::iequals(preferenceFinder, "Exp"))
			SetPreferenceFinder(EXP);
		else if(boost::iequals(preferenceFinder, "Gauss"))
			SetPreferenceFinder(EXP);
		else if(boost::iequals(preferenceFinder, "Tukey"))
			SetPreferenceFinder(EXP);
		else if(boost::iequals(preferenceFinder, "Hard"))
			SetPreferenceFinder(EXP);
		else
			throw runtime_error("No such preference finder found: " + preferenceFinder);

	}

	void TLinkage::Run() {
		if(m_debug)
			cout << "Entering TLinkage::Run()" << endl;

		std::ifstream fin(m_dataFile.c_str());
		if(!fin)
			throw runtime_error("couldn't open " + m_dataFile);
		while(fin >> m_imgName) {
			ArrayXXf data, sampleIndices, hypotheses, residuals, pref;
			ArrayXf clusters;
			ReadDataFromFile(fin, data);
			Sample(data, sampleIndices, m_samplesPerDataPt * data.cols());
			GenerateHypotheses(data, sampleIndices, hypotheses);
			FindResiduals(data, hypotheses, residuals);
			FindPreferences(residuals, pref);
			Cluster(pref, clusters);
			RejectOutliers(clusters, clusters);
			if(m_saveClusters) {
				m_foutClusters << m_imgName << endl;
				m_foutClusters << clusters.size() << endl;
				m_foutClusters << clusters << endl;
			}
			FitModels(data, clusters, m_models);
			m_foutModels << m_imgName << endl;
			m_foutModels << m_models.size() << "\t" << m_modelParams << endl;
			for(int i = 0; i < m_models.size(); i++)
				m_foutModels << m_models[i] << endl << endl;
		}
		
		if(m_debug)
			cout << "Exiting TLinkage::Run()" << endl;
	}

	void TLinkage::ReadDataFromFile(std::ifstream& _fin, ArrayXXf& _data) {
		if(m_debug)
			cout << "Entering TLinkage::ReadDataFromFile()" << endl;

		ulli rows, cols;
		_fin >> rows >> cols;

		_data.resize(rows, cols);
		for(ulli r = 0; r < rows; r++)
			for(ulli c = 0; c < cols; c++)
				_fin >> _data(r, c);

		if(m_shouldTranspose)
			_data.transposeInPlace();
		
		if(m_debug) {
			cout << "Read matrix of size: " << _data.rows() << "x" << _data.cols() << endl;
			cout << "Exiting TLinkage::ReadDataFromFile()" << endl;
		}
	}

	void TLinkage::SetSampler(SamplingMethod _method) {
		switch(_method) {
			case PREFER_NEAR:
				m_sampler = std::make_unique<DistBasedSampler>(m_xmlFileName);
				break;
			case UNIFORM:
				m_sampler = std::make_unique<UniformSampler>(m_xmlFileName);
				break;
			default:
				throw runtime_error("Attempt to set unknown sampler");
				
		}
	}

	void TLinkage::SetOutlierRejector(OutlierRejectionMethod _method) {
		switch(_method) {
			case MAX_SIZE_CHANGE:
				m_outlierRejector = std::make_unique<MaxDiffOR>(m_minSamples, m_xmlFileName);
				break;
			case K_LARGEST:
				m_outlierRejector = std::make_unique<KLargestOR>(m_xmlFileName);
				break;
			default:
				throw runtime_error("Attempt to set unknown outlier rejector");
		}
	}

	void TLinkage::SetPreferenceFinder(VotingScheme _method) {

		switch(_method) {
			case EXP: 
				m_preferenceFinder = std::make_unique<ExpPreferenceFinder>(m_xmlFileName);
				break;
			case GAUSS:
				m_preferenceFinder = std::make_unique<GaussPreferenceFinder>(m_xmlFileName);
				break;
			case HARD:
				m_preferenceFinder = std::make_unique<HardPreferenceFinder>(m_xmlFileName);
				break;
			case TUKEY:
				m_preferenceFinder = std::make_unique<TukeyPreferenceFinder>(m_xmlFileName);
				break;
			default:
				throw runtime_error("Preference method not available");
		}

	}

	void TLinkage::Sample(const ArrayXXf& _data, ArrayXXf& _sampleIndices, 
			const unsigned long long int& _numSamples) {

		if(m_debug)
			cout << "Entering TLinkage::Sample()" << endl;

		if(!m_sampler)
			throw runtime_error("Sampler not set");

		(*m_sampler)(_data, _sampleIndices, _numSamples, m_minSamples);	

		if(m_debug)
			cout << "Exiting TLinkage::Sample()" << endl;
	}

	void TLinkage::FindResiduals(const ArrayXXf& _data, const ArrayXXf& _hypotheses, 
			ArrayXXf& _residuals) {

		if(m_debug)
			cout << "Entering TLinkage::FindResiduals()" << endl;

		_residuals = ArrayXXf(_hypotheses.cols(), _data.cols());  //column major memory considered

		for(ulli i = 0; i < _residuals.rows(); i++)
			for(ulli j = 0; j < _residuals.cols(); j++)
				_residuals(i, j) = Distance(_data.col(j), _hypotheses.col(i));

		if(m_debug)
			cout << "Exiting TLinkage::FindResiduals()" << endl;	
	}

	void TLinkage::FindPreferences(const ArrayXXf& _residuals, ArrayXXf& _preferences) {

		if(m_debug)
			cout << "Entering TLinkage::FindPreferences()" << endl;

		(*m_preferenceFinder)(_residuals, _preferences);

		if(m_debug)
			cout << "Exiting TLinkage::FindPreferences()" << endl;

	}

	void TLinkage::GenerateHypotheses(const ArrayXXf& _data, 
		const ArrayXXf& _sampleIndices, ArrayXXf& _hypotheses) {
		
		if(m_debug)
			cout << "Entering TLinkage::GenerateHypotheses()" << endl;
	
		_hypotheses.resize(m_modelParams, _sampleIndices.cols());
		vector<ArrayXf> samples(m_minSamples);
		
		for(ulli c = 0; c < _sampleIndices.cols(); c++) {
			for(ulli r = 0; r < _sampleIndices.rows(); r++)
				samples[r] = _data.col(_sampleIndices(r, c));
			_hypotheses.col(c) = GenerateHypothesis(samples);
		}

		if(m_debug)
			cout << "Exiting TLinkage::GenerateHypotheses()" << endl;

	}

	void TLinkage::Cluster(ArrayXXf& _preferences, ArrayXf& _clusters) {
		if(m_debug)
			cout << "Entering TLinkage::Cluster()" << endl;

		ArrayXXf distances;
		CalculateTanimotoDist(_preferences, distances);

		_clusters = ArrayXf(_preferences.cols());

		UnionFind uf(_preferences.cols());

		//find min dist
		ulli pt1, pt2;
		double minDist = distances.minCoeff(&pt1, &pt2);
		while(minDist < 1) {

			//remove other point
			if(pt2 > 0)
				distances.col(pt2).head(pt2) = std::numeric_limits<float>::max();
			distances.row(pt2).tail(distances.cols() - pt2) = std::numeric_limits<float>::max();

			//modify the preferences of the retained point so it expresses preferences of the union
			_preferences.col(pt1) =  _preferences.col(pt1).min(_preferences.col(pt2));

			//update the distances matrix
			for(ulli i = 0; i < pt1; i++)
				if(distances(i, pt1) != std::numeric_limits<float>::max())
					distances(i, pt1) = Tanimoto(_preferences.col(pt1), _preferences.col(i));

			for(ulli i = pt1 + 1; i < distances.cols(); i++)
				if(distances(pt1, i) != std::numeric_limits<float>::max())
					distances(pt1, i) = Tanimoto(_preferences.col(pt1), _preferences.col(i));
			uf.unionSet(pt1, pt2);
			minDist = distances.minCoeff(&pt1, &pt2);
		}

		for(ulli i = 0; i < _clusters.size(); i++)  //copy union find vector to Eigen::Array
			_clusters(i) = uf.findSet(i);

		if(m_debug)
			cout << "Exiting TLinkage::Cluster()" << endl;
	}

	void TLinkage::CalculateTanimotoDist(const ArrayXXf& _preferences, ArrayXXf& _distances) {
		if(m_debug)
			cout << "Entering TLinkage::CalculateTanimotoDist()" << endl;
		_distances = ArrayXXf(_preferences.cols(), _preferences.cols());
		//create symmetric matrix without storing two copies
		for(ulli r = 0; r < _distances.rows(); r++) {
			for(ulli c = 0; c < _distances.cols(); c++) {
				if(c <= r)
					_distances(r, c) = std::numeric_limits<float>::max(); //so it's not min ever
				else 
					_distances(r, c) = Tanimoto(_preferences.col(r), _preferences.col(c));
			}
		}

		if(m_debug)
			cout << "Exiting TLinkage::CalculateTanimotoDist()" << endl;
	}

	float TLinkage::Tanimoto(const ArrayXf& _a, const ArrayXf& _b) {
		float dotProd = _a.matrix().dot(_b.matrix());
		return 1 - dotProd / (_a.matrix().squaredNorm() + _b.matrix().squaredNorm() - dotProd);
	}

	void TLinkage::RejectOutliers(const ArrayXf& _clusters, ArrayXf& _out) {

		if(m_debug)
			cout << "Entering TLinkage::RejectOutliers()" << endl;

		if(!m_outlierRejector)
			throw runtime_error("OutlierRejector not set");

		(*m_outlierRejector)(_clusters, _out);

		if(m_debug)
			cout << "Exiting TLinkage::RejectOutliers()" << endl;
	}

	void TLinkage::FitModels(const ArrayXXf& _data, const ArrayXf& _clusters, vector<ArrayXf>& _models, const int& _noiseIndex) {
		if(m_debug)
			cout << "Entering TLinkage::FitModels()" << endl;

		vector<vector<float> > clusteredPoints;
		std::unordered_map<long long int, ulli> clusterID2Index;

		for(ulli i = 0; i < _clusters.size(); i++) {
			if(_clusters(i) != _noiseIndex) {
				if(clusterID2Index.find(_clusters(i)) == clusterID2Index.end()) {
					clusterID2Index[_clusters(i)] = clusteredPoints.size();
					clusteredPoints.push_back(vector<float>());
				}
				for(ulli j = 0; j < _data.rows(); j++)
					clusteredPoints[clusterID2Index[_clusters(i)]].push_back(_data(j, i));
			}
		}

		if(m_debug)
			cout << "Clubbed points of the same cluster" << endl;

		vector<ArrayXXf> clusters(clusteredPoints.size());
		_models.resize(clusteredPoints.size());

		for(ulli i = 0; i < clusters.size(); i++) {

			clusters[i] = Map<Matrix<float, Dynamic, Dynamic, RowMajor>, Unaligned, OuterStride<> >(clusteredPoints[i].data(),
					clusteredPoints[i].size() / _data.rows(), _data.rows(), 
					OuterStride<>(_data.rows())).array(); //this is a deep copy
			FitModel(clusters[i], _models[i]);
		}

		if(m_debug)
			cout << "Exiting TLinkage::FitModels()" << endl;
	}

}
