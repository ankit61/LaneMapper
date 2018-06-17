#include "VeloProjector.h"

VeloProjector::VeloProjector(string _img_root, string _data_file, string _seg_root, string _velo_root, string _calib_root, string _velo_img_pts_root, int _ret_frequency, double _min_x) : 
	img_root_(_img_root), calib_root_(_calib_root), seg_root_(_seg_root), velo_root_(_velo_root), output_root_(_velo_img_pts_root), data_file_(_data_file), min_x_(_min_x), retention_frequency_(_ret_frequency), cam_num_(2) { 

		#ifdef DEBUG
			debug_ = true;
		#else
			debug_ = false;
		#endif
		if(debug_)
			cout << "Entering VeloProjector::VeloProjector() " << endl;
	
		velo_points_.reserve(1000000);
		
		string cam_calib_file = calib_root_ + "/calib_cam_to_cam.txt";
		string velo_calib_file = calib_root_ + "/calib_velo_to_cam.txt";

		bool isSuccess = true;
		
		Mat R, T;
		
		isSuccess &= CalibDataLoader::ReadVariable(cam_calib_file, "P_rect_0" + std::to_string(cam_num_), 3, 4, P_rect_) &&
		//TODO: Check correctness of using R_rect_00 instead of R_rect_ + cam_num_?
		//the results are same as MATLAB's KITTI code even with R_rect_00
		CalibDataLoader::ReadVariable(cam_calib_file, "R_rect_00", 3, 3, R_rect_) &&
		CalibDataLoader::ReadVariable(velo_calib_file, "R", 3, 3, R) &&
		CalibDataLoader::ReadVariable(velo_calib_file, "T", 3, 1, T);
		hconcat(R, T, Tr_);
		float last_row_data[] = { 0, 0, 0, 1};
		Tr_.push_back(Mat(1, 4, CV_32F, last_row_data));

		if(!isSuccess) {
			cout << "Incorrect format of calibration files" << endl;
			exit(1);
		}

		namedWindow("show", WINDOW_AUTOSIZE);
		
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
	Mat projection_mat, velo_img_pts;
	while(std::getline(fin, line)) {
	
		string img_base_name = string(basename(const_cast<char*>(line.c_str())));
		string input_img_name = img_root_ + "/" + line;
		string seg_img_name = seg_root_ + "/segmented_" + line;

		Mat input_img = imread(input_img_name);
		Mat segmented_img = imread(seg_img_name, IMREAD_GRAYSCALE);

		if(input_img.empty()) {
			cout << "Can't read input image: " << input_img_name << endl;
			exit(1);
		}
		else 
			cout << "Successfully read input image: " << input_img_name << endl;

		if(segmented_img.empty()) {
			cout << "Can't read segmented image: " << seg_img_name << endl;
			exit(1);
		}
		else 
			cout << "Successfully read segmented image: " << seg_img_name << endl;

		ReadVeloData(velo_root_ + "/" + line.substr(0, line.size() - 3) + "bin");
		ComputeProjMat(projection_mat);
		Project(projection_mat, velo_img_pts);
		
		reflectivity_ = 255 * reflectivity_;
		reflectivity_.convertTo(reflectivity_, CV_8UC1);
		vector<unsigned char> on_road_ref;
		on_road_ref.reserve(reflectivity_.rows);
		double thresh;

		//Find Otsu thresholding for points that are on road and have positive reflectivity
		
		for(long long int i = 0; i < reflectivity_.rows; i++) {
			int x = velo_img_pts.at<float>(i, 0), y = velo_img_pts.at<float>(i, 1);
			int reflect = reflectivity_.at<unsigned char>(i, 0);
			if(isValid(y, x, segmented_img.rows, segmented_img.cols) && segmented_img.at<unsigned char>(y, x) && reflect) {
				on_road_ref.push_back(reflectivity_.at<unsigned char>(i, 0));
			}
		}
		
		thresh = threshold(on_road_ref, on_road_ref, 0, 255, THRESH_TOZERO | THRESH_OTSU);

		//Display points
		
		for(int i = 0; i < velo_img_pts.rows; i++) {
			int x = velo_img_pts.at<float>(i, 0), y = velo_img_pts.at<float>(i, 1);
			int reflect = reflectivity_.at<unsigned char>(i, 0);
			if(isValid(y, x, input_img.rows, input_img.cols) && segmented_img.at<unsigned char>(y, x) && reflect > thresh)
				circle(input_img, Point(x, y), 5, Scalar(reflect, 0, 0));
		}

		string saved_name = output_root_ + "/lidar_" + img_base_name;

		imwrite(saved_name, input_img);

		if(debug_) {
			cout << "Threshold: " << thresh << endl;
			cout << "Saving " << saved_name << endl;
		}
	}

	if(debug_)
		cout << "Exiting VeloProjector::Run() " << endl;
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
	float *py = data+1;
	float *pz = data+2;
	float *pr = data+3;

	// load point cloud
	FILE *stream;
	stream = fopen (_bin_file.c_str(),"rb");
	num = fread(data,sizeof(float),num,stream) / 4;
	for (int32_t i=0; i<num; i++) {
		if(*px >= min_x_ && (i % retention_frequency_) == 0) {
			Mat m(1, 4, CV_32F, px);
			velo_points_.push_back(m);   
		}
		px+=4; py+=4; pz+=4; pr+=4;
	}
	fclose(stream);
	
	if(debug_)
		cout << "Exiting VeloProjector::ReadVeloData() " << endl;
}

void VeloProjector::ComputeProjMat(Mat& _P_velo_to_img) {
	if(debug_)
		cout << "Entering VeloProjector::ComputeProjMat() " << endl;
	
	Mat R_cam_to_rect = Mat::eye(4, 4, CV_32F);
	
	Mat tmp = R_cam_to_rect(Rect(0,0,3,3));
	R_rect_.copyTo(tmp);	

	_P_velo_to_img = P_rect_ * R_cam_to_rect * Tr_; 
	
	if(debug_)
		cout << "Exiting VeloProjector::ComputeProjMat() " << endl;
}

void VeloProjector::Project(const Mat& _P_velo_to_img, Mat& _velo_img) {
	
	if(debug_)
		cout << "Entering VeloProjector::Project() " << endl;
	
	int dim_norm = _P_velo_to_img.rows;
	int dim_proj = _P_velo_to_img.cols;

	if(dim_proj != velo_points_.cols) {
		cout << "incorrect dimensions to multiply" << endl;
		exit(1);
	}
	
	reflectivity_ = velo_points_.col(dim_proj - 1).clone();
	Mat tmp = velo_points_.col(dim_proj - 1);
	Mat(velo_points_.rows, 1, CV_32F, Scalar(1)).copyTo(tmp);

	Mat new_points = (_P_velo_to_img * velo_points_.t()).t(); 
	
	for(int i = 0; i < dim_norm - 1; i++)
		divide(new_points.col(i), new_points.col(dim_norm - 1), new_points.col(i));

	_velo_img = new_points.colRange(0, dim_norm - 1);	

	if(debug_)
		cout << "Exiting VeloProjector::Project() " << endl;
}