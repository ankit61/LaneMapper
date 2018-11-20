#ifndef UTILITIES_H_
#define UTILITIES_H_

#include<Eigen/Dense>
#include"BaseLD.h"
#include<interpolation.h>

namespace LD {

	bool isValid(long long int r, long long int c, long long int rows, long long int cols);

	void ReadEigenMatFromFile(std::ifstream& _fin, Eigen::ArrayXXf& _data, bool _shouldTranspose);
	
	void ReadEigenMatFromFile(const string& _fileName, Eigen::ArrayXXf& _data, bool _shouldTranspose);

	void CreateAlglibArray(const vector<Eigen::ArrayXf>& _samples, vector<alglib::real_1d_array>& _coordinates);
	
	void CreateAlglibArray(const Eigen::ArrayXXf _samples, vector<alglib::real_1d_array>& _coordinates);

}

#endif
