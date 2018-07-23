#include<fstream>
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

}
