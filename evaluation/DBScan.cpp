#include"DBScan.h"
#include<fstream>
#include"Utilities.h"
namespace LD {


	void DBScan::ParseXML() {
		if(m_debug)	
			cout << "Entering DBScan::ParseXML()" << endl;

		m_xml = m_xml.child("DBScan");

		m_eps 				= m_xml.attribute("eps").as_float();
		m_minPts 			= m_xml.attribute("minPts").as_int();
		m_dataFile			= m_xml.attribute("dataFile").as_string();
		m_outputFile		= m_xml.attribute("outputFile").as_string();
		m_shouldTranspose 	= m_xml.attribute("shouldTranspose").as_bool();

		if(!m_eps || !m_minPts || m_dataFile.empty() || m_outputFile.empty())
			throw runtime_error("at least one of the following attributes missing in DBScan node: eps, minPts, dataFile, outputFile, shouldTranspose");
		
		if(m_debug)	
			cout << "Exiting DBScan::ParseXML()" << endl;
	}

	void DBScan::Run() {
		if(m_debug)
			cout << "Entering DBScan::Run()" << endl;

		std::ifstream fin(m_dataFile.c_str());
		std::ofstream fout(m_outputFile);
		string imageName;
		while(fin >> imageName) {
			Eigen::ArrayXXf _data;
			ReadEigenMatFromFile(fin, _data, m_shouldTranspose);
			vector<int> labels;
			Cluster(_data, labels);
			fout << imageName << endl;
			PrintOutputFile(fout, _data, labels);
		}
		
		if(m_debug)
			cout << "Exiting DBScan::Run()" << endl;
	}

	void DBScan::PrintOutputFile(std::ofstream& _fout, const Eigen::ArrayXXf& _data, const vector<int>& _labels) {
		if(m_debug)
			cout << "Entering DBScan::PrintOutputFile()" << endl;

		ulli inlierCount = 0;
		for(ulli c = 0; c < _data.cols(); c++)
			if(_labels[c] != NOISE)
				inlierCount++;
			else if(_labels[c] == UNDEFINED)
				throw runtime_error("undefined point");
		

		_fout << inlierCount << "\t" << _data.rows() << endl;
		
		for(ulli c = 0; c < _data.cols(); c++) {
			if(_labels[c] == NOISE)
				continue;
			for(ulli r = 0; r < _data.rows(); r++)
				_fout << _data(r, c) << "\t";
			_fout << endl;
		}
		
		if(m_debug)
			cout << "Exiting DBScan::PrintOutputFile()" << endl;
	}

	double DBScan::Distance(const Eigen::ArrayXf& _pt1, const Eigen::ArrayXf& _pt2) {
		return (_pt1 - _pt2).matrix().norm();	
	}

	DBScan::ulli DBScan::GetNeighbors(const Eigen::ArrayXXf& _data, const ulli& _ptIndex, vector<ulli>& _neighborIndices, const std::unordered_set<ulli>& _uniques) {
		
		if(m_debug)
			cout << "Entering DBScan::GetNeighbors()" << endl;
		ulli numNeighbors = 0;
		for(ulli c = 0; c < _data.cols(); c++)
			if(Distance(_data.col(c), _data.col(_ptIndex)) < m_eps && c != _ptIndex) {
				if(_uniques.find(c) == _uniques.end())
					_neighborIndices.push_back(c);
				numNeighbors++;
			}
		
		if(m_debug)
			cout << "Exiting DBScan::GetNeighbors()" << endl;

		return numNeighbors;
	}

	void DBScan::Cluster(const Eigen::ArrayXXf& _data, vector<int>& _labels) {
		if(m_debug)
			cout << "Entering DBScan::Cluster()" << endl;
			
		_labels.resize(_data.cols(), UNDEFINED);
		ulli numClusters = 0;
		for(ulli c = 0; c < _data.cols(); c++) {
			if(_labels[c] != UNDEFINED)
				continue;

			vector<ulli> neighborIndices;
			GetNeighbors(_data, c, neighborIndices);
			std::unordered_set<ulli> uniqueNeighbors(neighborIndices.begin(), neighborIndices.end());

			if(uniqueNeighbors.size() < m_minPts) {
				_labels[c] = NOISE;
				continue;
			}
			
			_labels[c] = ++numClusters;

			for(ulli i = 0; i < neighborIndices.size(); i++) {
				
				if(_labels[neighborIndices[i]] == NOISE)
					_labels[neighborIndices[i]] = numClusters;
				if(_labels[neighborIndices[i]] != UNDEFINED)
					continue;
				
				_labels[neighborIndices[i]] = numClusters;
				vector<ulli> newNeighborIndices;
				ulli numNeighbors = GetNeighbors(_data, neighborIndices[i], newNeighborIndices, uniqueNeighbors);
				if(numNeighbors >= m_minPts) {
					neighborIndices.insert(neighborIndices.end(), newNeighborIndices.begin(), newNeighborIndices.end());
					uniqueNeighbors.insert(newNeighborIndices.begin(), newNeighborIndices.end());
				}
			}		
		}

		if(m_debug)
			cout << "Exiting DBScan::Cluster()" << endl;
	}

}
