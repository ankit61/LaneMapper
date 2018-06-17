#include<string>
#include<libgen.h>
#include<vector>
#include<iostream>
#include<unordered_set>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<stdio.h>
#include<stdlib.h>
#include "CalibDataLoader.h"

using namespace cv;
using std::string;
using std::vector;
using std::cout;
using std::endl;

class VeloProjector {
	private:
		
		bool debug_;
		Mat velo_points_;
		int cam_num_;
		int retention_frequency_;
		double min_x_;
		Mat P_rect_, Tr_, R_rect_;  
		string img_root_;
		string calib_root_;
		string output_root_;
		string velo_root_;
		string velo_to_cam_file_;
		string cam_to_cam_file_;
		string data_file_;
		string seg_root_;
		Mat reflectivity_;

		void ReadVeloData(string _bin_file);

		void Project(const Mat& _p_velo_to_img, Mat& _velo_img);
		
		void ComputeProjMat(Mat& _p_velo_to_img);

	public:


		VeloProjector(string _img_root, string _data_file, string _seg_root, string _velo_root, string _calib_root, string _output_root, int _ret_frequency = 1, double _min_x = 5);
		
		void Run();
		
};
