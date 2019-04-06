#include<fstream>
#include<sstream>
#include"Utilities.h"

namespace LD {
	
	bool isValid(long long int r, long long int c, long long int rows, long long int cols) {
		return r >= 0 && c >= 0 && r < rows && c < cols;
	}

	void ReadEigenMatFromFile(std::ifstream& _fin, Eigen::ArrayXXf& _data, bool _shouldTranspose) {

		unsigned long long int rows, cols;
		_fin >> rows >> cols;

		_data.resize(rows, cols);
		for(unsigned long long int r = 0; r < rows; r++)
			for(unsigned long long int c = 0; c < cols; c++)
				_fin >> _data(r, c);

		if(_shouldTranspose)
			_data.transposeInPlace();
		
	}


	void ReadEigenMatFromFile(const string& _fileName, Eigen::ArrayXXf& _data, bool _shouldTranspose) {
		typedef unsigned long long int ulli;
		std::ifstream fin(_fileName);
		if(!fin)
			throw runtime_error("Can't open " + _fileName);
		string line;
		vector<vector<float> > els;
		ulli rows = 0, cols = 0;
		while(std::getline(fin, line)) {
			std::istringstream is(line);
			float el;
			els.push_back(vector<float>());
			while(is >> el)
				els[rows].push_back(el);
			if(!cols)
				cols = els[rows].size();
			else if(cols != els[rows].size())
				throw runtime_error("Number of columns are inconsistent");
			rows++;
		}

		_data.resize(rows, cols);
		for(ulli r = 0; r < rows; r++)
			for(ulli c = 0; c < cols; c++)
				_data(r, c) = els[r][c];

		if(_shouldTranspose)
			_data.transposeInPlace();
	}

	void CreateAlglibArray(const vector<Eigen::ArrayXf>& _samples, vector<alglib::real_1d_array>& _coordinates) {
		if(!_samples.size())
			return;

		_coordinates.resize(_samples[0].size());
		for(int i = 0; i < _coordinates.size(); i++)
			_coordinates[i].setlength(_samples.size());

		for(int i = 0; i < _coordinates.size(); i++) 
			for(int j = 0; j < _samples.size(); j++) 
				_coordinates[i](j) = _samples[j](i);
	}


	void CreateAlglibArray(const Eigen::ArrayXXf _samples, vector<alglib::real_1d_array>& _coordinates) {
		if(!_samples.cols())
			return;

		_coordinates.resize(_samples.cols());
		for(int i = 0; i < _coordinates.size(); i++)
			_coordinates[i].setlength(_samples.rows());

		for(int i = 0; i < _coordinates.size(); i++) 
			for(int j = 0; j < _samples.rows(); j++) 
				_coordinates[i](j) = _samples(j, i);
	}

    int ImgFile2Int(const string &_imgFileName) {
        return std::stoi(_imgFileName.substr(0, _imgFileName.size() - 3));
    }
	
}
