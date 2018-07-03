#include "VeloProjector.h"

VeloProjector::VeloProjector(string _img_root, string _data_file, string _seg_root, string _refined_root, string _velo_root, string _calib_root, string _velo_img_pts_root, int _ret_frequency, double _min_x) : 
	img_root_(_img_root), calib_root_(_calib_root), seg_root_(_seg_root), refined_root_(_refined_root), velo_root_(_velo_root), output_root_(_velo_img_pts_root), data_file_(_data_file), min_x_(_min_x), retention_frequency_(_ret_frequency), cam_num_(2), pts_file3D_((output_root_ + "/merged_3D_pts.txt").c_str()) {

		#ifdef DEBUG
			debug_ = true;
		#else
			debug_ = false;
		#endif
		if(debug_)
			cout << "Entering VeloProjector::VeloProjector() " << endl;
		
		string cam_calib_file = calib_root_ + "/calib_cam_to_cam.txt";
		string velo_calib_file = calib_root_ + "/calib_velo_to_cam.txt";

		bool isSuccess = true;
		
		Eigen::MatrixXf R, T;
		
		isSuccess &= CalibDataLoader::ReadVariable(cam_calib_file, "P_rect_0" + std::to_string(cam_num_), 3, 4, P_rect_) &&
		//TODO: Check correctness of using R_rect_00 instead of R_rect_ + cam_num_?
		//the results are same as MATLAB's KITTI code even with R_rect_00
		CalibDataLoader::ReadVariable(cam_calib_file, "R_rect_00", 3, 3, R_rect_) &&
		CalibDataLoader::ReadVariable(velo_calib_file, "R", 3, 3, R) &&
		CalibDataLoader::ReadVariable(velo_calib_file, "T", 3, 1, T);
		Tr_ = Eigen::MatrixXf(R.rows() + 1, R.cols() + T.cols());
		
		Tr_ << R, T,
			   0, 0, 0, 1;

		if(!isSuccess)
			throw std::runtime_error("Incorrect format of calibration files");
	
		if(debug_)
			cout << "Exiting VeloProjector::VeloProjector() " << endl;
	}


bool isValid(long long int r, long long int c, long long int rows, long long int cols) {
	return r >= 0 && c >= 0 && r < rows && c < cols;
}

void VeloProjector::Run() {
	if(debug_)
		cout << "Entering VeloProjector::Run() " << endl;

	std::ifstream fin(data_file_.c_str());
	string line;
	Eigen::MatrixXf projection_mat, velo_img_pts; 
	while(std::getline(fin, line)) {
	
		string img_base_name = string(basename(const_cast<char*>(line.c_str())));
		string input_img_name = img_root_ + "/" + line;
		string seg_img_name = seg_root_ + "/segmented_" + line;
		string ref_img_name = refined_root_ + "/thresholded_" + line;

		Mat input_img = imread(input_img_name);
		Mat segmented_img = imread(seg_img_name, IMREAD_GRAYSCALE);
		Mat refined_img = imread(ref_img_name, IMREAD_GRAYSCALE);

		if(input_img.empty())
			throw std::runtime_error("Can't open " + input_img_name);
		else 
			cout << "Successfully read input image: " << input_img_name << endl;

		if(segmented_img.empty())
			throw std::runtime_error("Can't open " + seg_img_name);
		else 
			cout << "Successfully read segmented image: " << seg_img_name << endl;
		
		if(refined_img.empty())
			throw std::runtime_error("Can't open " + ref_img_name);
		else 
			cout << "Successfully read refined image: " << ref_img_name << endl;
		
		ReadVeloData(velo_root_ + "/" + line.substr(0, line.size() - 3) + "bin");
		ComputeProjMat(projection_mat);
		Project(projection_mat, velo_img_pts);
		double thresh = OtsuThresholdRoad(velo_img_pts, segmented_img);
		pts_file3D_ << img_base_name << endl;
		IntersectIn3D(velo_img_pts, refined_img, thresh, input_img);

		if(debug_) {
			string saved_name = output_root_ + "/lidar_" + img_base_name;
			imwrite(saved_name, input_img);
			cout << "Threshold: " << thresh << endl;
			cout << "Saving " << saved_name << endl;
		}
	}

	if(debug_)
		cout << "Exiting VeloProjector::Run() " << endl;
}

void VeloProjector::IntersectIn3D(const Eigen::MatrixXf _velo_img, const Mat& _refined_img, double _thresh, Mat _img) {
	if(debug_)
		cout << "Entering VeloProjector::IntersectIn3D()" << endl;

	for(int i = 0; i < _velo_img.rows(); i++) {
		int x = _velo_img(i, 0), y = _velo_img(i, 1);
		int reflect = reflectivity_.at<unsigned char>(i, 0);
		if(isValid(y, x, _refined_img.rows, _refined_img.cols) && _refined_img.at<unsigned char>(y, x)
			&& reflect > _thresh) {
			pts_file3D_ << velo_points_.row(i).colRange(0, 3) << endl;
			if(debug_ && !_img.empty())
				circle(_img, Point(x, y), 5, Scalar(reflect, 0, 0), -1);
		}
	}
	pts_file3D_.flush();
	
	if(debug_)
		cout << "Exiting VeloProjector::IntersectIn3D()" << endl;
}

double VeloProjector::OtsuThresholdRoad(const Eigen::MatrixXf _velo_img, const Mat& _seg_img) {
	if(debug_)
		cout << "Entering VeloProjector::OtsuThresholdRoad()" << endl;
	
	//Find Otsu thresholding for points that are on road and have positive reflectivity
	reflectivity_ = 255 * reflectivity_;
	reflectivity_.convertTo(reflectivity_, CV_8UC1);
	vector<unsigned char> on_road_ref;
	on_road_ref.reserve(reflectivity_.rows);

	for(long long int i = 0; i < reflectivity_.rows; i++) {
		int x = _velo_img(i, 0), y = _velo_img(i, 1);
		int reflect = reflectivity_.at<unsigned char>(i, 0);
		if(isValid(y, x, _seg_img.rows, _seg_img.cols) &&
				_seg_img.at<unsigned char>(y, x) && reflect)
			on_road_ref.push_back(reflectivity_.at<unsigned char>(i, 0));
	}

	double thresh = threshold(on_road_ref, on_road_ref, 0, 255, THRESH_TOZERO | THRESH_OTSU);

	if(debug_)
		cout << "Exiting VeloProjector::OtsuThresholdRoad()" << endl;

	return thresh;
}

void VeloProjector::ReadVeloData(string _bin_file) {
	//taken from KITTI website	
	if(debug_)
		cout << "Entering VeloProjector::ReadVeloData() " << endl;

	velo_points_.release();
	// allocate 4 MB buffer (only ~130*4*4 KB are needed)
	int32_t num = 1000000;
	float *data = (float*)malloc(num*sizeof(float));

	// pointers
	float *px = data+0;

	// load point cloud
	FILE *stream;
	stream = fopen (_bin_file.c_str(),"rb");
	num = fread(data,sizeof(float),num,stream) / 4;
	for (int32_t i=0; i<num; i++) {
		if(*px >= min_x_ && (i % retention_frequency_) == 0) {
			Mat m(1, 4, CV_32F, px);
			velo_points_.push_back(m);   
		}
		px+=4;
	}
	fclose(stream);
	
	if(debug_)
		cout << "Exiting VeloProjector::ReadVeloData() " << endl; 
}

void VeloProjector::ComputeProjMat(Eigen::MatrixXf& _P_velo_to_img) {
	if(debug_)
		cout << "Entering VeloProjector::ComputeProjMat() " << endl;
	
	Eigen::MatrixXf R_cam_to_rect = Eigen::MatrixXf::Identity(4, 4);
	
	R_cam_to_rect.topLeftCorner<3, 3>() = R_rect_;
	
	_P_velo_to_img = P_rect_ * R_cam_to_rect * Tr_; 	
	if(debug_)
		cout << "Exiting VeloProjector::ComputeProjMat() " << endl;
}

void VeloProjector::Project(const Eigen::MatrixXf& _P_velo_to_img, Eigen::MatrixXf& _velo_img) {
	
	if(debug_)
		cout << "Entering VeloProjector::Project() " << endl;
	
	int dim_norm = _P_velo_to_img.rows();
	int dim_proj = _P_velo_to_img.cols();

	if(dim_proj != velo_points_.cols) 
		throw std::runtime_error("incorrect dimensions to multiply");

	if(!velo_points_.isContinuous())
		throw std::runtime_error("matrix is not continuous");

	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > velo_pts_eg(velo_points_.ptr<float>(), velo_points_.rows, velo_points_.cols);
	
	reflectivity_ = velo_points_.col(dim_proj - 1).clone();
	velo_pts_eg.col(dim_proj - 1) = Eigen::MatrixXf::Ones(velo_pts_eg.rows(), 1);

	Eigen::MatrixXf new_points = (_P_velo_to_img * velo_pts_eg.transpose()).transpose();
	
	for(int i = 0; i < dim_norm - 1; i++)
		new_points.col(i).array() = new_points.col(i).array() / new_points.col(dim_norm - 1).array();

	_velo_img = new_points.topLeftCorner(new_points.rows(), new_points.cols() - 1);

	if(debug_)
		cout << "Exiting VeloProjector::Project() " << endl;
}
