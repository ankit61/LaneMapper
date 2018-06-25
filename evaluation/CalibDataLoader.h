#include<fstream>
#include<string>
#include<iostream>
#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<stdio.h>
#include<stdlib.h>
#include<Eigen/Dense>

using std::string;

class CalibDataLoader {
	private:
		CalibDataLoader() {}
	
	public: 
		
		static bool ReadVariable(string _file, string _variable_name, int _rows, int _cols, Eigen::MatrixXf& output); 
		//static std::vector<cv::Mat> ReadVariables(string _file, vector<string> _var_names, vector<cv::Size> _sizes);
};
