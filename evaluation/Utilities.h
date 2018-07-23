#ifndef UTILITIES_H_
#define UTILITIES_H_

#include<Eigen/Dense>
#include"BaseLD.h"

namespace LD {

	bool isValid(long long int r, long long int c, long long int rows, long long int cols);

	void ReadEigenMatFromFile(std::ifstream& _fin, Eigen::ArrayXXf& _data, bool _shouldTranspose);

}

#endif
