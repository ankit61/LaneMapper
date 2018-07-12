#include "VeloProjector.h"

namespace LD {

	VeloProjector::VeloProjector(string _xmlFile) : Solver(_xmlFile), m_calibDataLoader(_xmlFile) {

			if(m_debug)
				cout << "Entering VeloProjector::VeloProjector() " << endl;
			
			ParseXML();
			string camCalibFile  = m_calibRoot + "/calib_cam_to_cam.txt";
			string veloCalibFile = m_calibRoot + "/calib_velo_to_cam.txt";

			bool isSuccess = true;
			
			Eigen::MatrixXf R, T;
			
			isSuccess &= m_calibDataLoader.ReadVariable(camCalibFile, "P_rect_0" + std::to_string(m_camNum), 3, 4, m_PRect) &&
			//TODO: Check correctness of using m_RRect00 instead of m_RRect + m_camNum?
			//the results are same as MATLAB's KITTI code even with m_RRect00
			m_calibDataLoader.ReadVariable(camCalibFile, "R_rect_00", 3, 3, m_RRect) &&
			m_calibDataLoader.ReadVariable(veloCalibFile, "R", 3, 3, R) &&
			m_calibDataLoader.ReadVariable(veloCalibFile, "T", 3, 1, T);
			
			if(!isSuccess)
				throw std::runtime_error("Incorrect format of calibration files");
			
			m_Tr = Eigen::MatrixXf(R.rows() + 1, R.cols() + T.cols());
			
			m_Tr << R, T,
				   0, 0, 0, 1;

			if(m_debug)
				cout << "Exiting VeloProjector::VeloProjector() " << endl;
	}

	void VeloProjector::ParseXML() {
		if(m_debug)
			cout << "Entering VeloProjector::ParseXML()" << endl;

		m_xml = m_xml.child("VeloProjector");

		pugi::xml_node solverInstance;
		solverInstance		 = m_xml.child("SolverInstance");
		m_dataRoot           = solverInstance.attribute("dataRoot").as_string();
		m_dataFile           = solverInstance.attribute("dataFile").as_string();
		m_segRoot            = solverInstance.attribute("segRoot").as_string();
		m_refinedRoot        = solverInstance.attribute("refinedRoot").as_string();
		m_veloRoot           = solverInstance.attribute("veloRoot").as_string();
		m_calibRoot          = solverInstance.attribute("calibRoot").as_string();
		m_outputRoot         = solverInstance.attribute("outputRoot").as_string();
		m_retentionFrequency = solverInstance.attribute("retentionFrequency").as_int();
		m_camNum             = solverInstance.attribute("camNum").as_int(-1);
		m_minX               = solverInstance.attribute("minX").as_int();

		if(m_dataRoot.empty() || m_dataFile.empty() || m_segRoot.empty() || m_refinedRoot.empty() || m_veloRoot.empty() || m_calibRoot.empty() ||  m_outputRoot.empty() || !m_retentionFrequency || (m_camNum == -1) || !m_minX)
			throw runtime_error("at least one of the following attributes are missing in SolverInstance node of VeloProject: dataRoot, dataFile, segRoot, refinedRoot, veloRoot, calibRoot, outputRoot, retentionFrequency, camNum, minX");

		if(m_debug)
			cout << "Exiting VeloProjector::ParseXML()" << endl;
	}

	bool isValid(long long int r, long long int c, long long int rows, long long int cols) {
		return r >= 0 && c >= 0 && r < rows && c < cols;
	}

	void VeloProjector::Run() {
		if(m_debug)
			cout << "Entering VeloProjector::Run() " << endl;

		std::ifstream fin(m_dataFile.c_str());
		string line;
		Eigen::MatrixXf projection_mat, veloImgPts; 
		while(std::getline(fin, line)) {
		
			string imgBaseName  = string(basename(const_cast<char*>(line.c_str())));
			string inputImgName = m_dataRoot + "/" + line;
			string segImgName   = m_segRoot + "/segmented_" + line;
			string refImgName   = m_refinedRoot + "/thresholded_" + line;

			Mat inputImg   = imread(inputImgName);
			Mat segImg     = imread(segImgName, IMREAD_GRAYSCALE);
			Mat refinedImg = imread(refImgName, IMREAD_GRAYSCALE);

			if(inputImg.empty())
				throw std::runtime_error("Can't open " + inputImgName);
			else 
				cout << "Successfully read input image: " << inputImgName << endl;

			if(segImg.empty())
				throw std::runtime_error("Can't open " + segImgName);
			else 
				cout << "Successfully read segmented image: " << segImgName << endl;
			
			if(refinedImg.empty())
				throw std::runtime_error("Can't open " + refImgName);
			else 
				cout << "Successfully read refined image: " << refImgName << endl;
			
			ReadVeloData(m_veloRoot + "/" + line.substr(0, line.size() - 3) + "bin");
			ComputeProjMat(projection_mat);
			Project(projection_mat, veloImgPts);
			double thresh = OtsuThresholdRoad(veloImgPts, segImg);
			m_ptsFile3D << imgBaseName << endl;
			IntersectIn3D(veloImgPts, refinedImg, thresh, inputImg);

			for(int i = 0; i < veloImgPts.rows(); i++) {
				int x = veloImgPts(i, 0), y = veloImgPts(i, 1);
				int reflect = m_reflectivity.at<unsigned char>(i, 0);
				if(isValid(y, x, inputImg.rows, inputImg.cols) && segImg.at<unsigned char>(y, x)
						&& reflect > thresh) {
					circle(inputImg, Point(x, y), 5, Scalar(reflect, 0, 0), -1);
				}
			}

			if(m_debug) {
				string savedName = m_outputRoot + "/lidar_" + imgBaseName;
				imwrite(savedName, inputImg);
				cout << "Threshold: " << thresh << endl;
				cout << "Saving " << savedName << endl;
			}
		}

		if(m_debug)
			cout << "Exiting VeloProjector::Run() " << endl;
	}

	void VeloProjector::IntersectIn3D(const Eigen::MatrixXf _veloImg, const Mat& _refinedImg, double _thresh, Mat _img) {
		if(m_debug)
			cout << "Entering VeloProjector::IntersectIn3D()" << endl;

		for(int i = 0; i < _veloImg.rows(); i++) {
			int x = _veloImg(i, 0), y = _veloImg(i, 1);
			int reflect = m_reflectivity.at<unsigned char>(i, 0);
			if(isValid(y, x, _refinedImg.rows, _refinedImg.cols) && _refinedImg.at<unsigned char>(y, x)
				&& reflect > _thresh) {
				m_ptsFile3D << m_veloPoints.row(i).colRange(0, 3) << endl;
				if(m_debug && !_img.empty())
					circle(_img, Point(x, y), 5, Scalar(reflect, 0, 0), -1);
			}
		}
		m_ptsFile3D.flush();
		
		if(m_debug)
			cout << "Exiting VeloProjector::IntersectIn3D()" << endl;
	}

	double VeloProjector::OtsuThresholdRoad(const Eigen::MatrixXf _veloImg, const Mat& _segImg) {
		if(m_debug)
			cout << "Entering VeloProjector::OtsuThresholdRoad()" << endl;
		
		//Find Otsu thresholding for points that are on road and have positive reflectivity
		m_reflectivity = 255 * m_reflectivity;
		m_reflectivity.convertTo(m_reflectivity, CV_8UC1);
		vector<unsigned char> onRoadRef;
		onRoadRef.reserve(m_reflectivity.rows);

		for(long long int i = 0; i < m_reflectivity.rows; i++) {
			int x = _veloImg(i, 0), y = _veloImg(i, 1);
			int reflect = m_reflectivity.at<unsigned char>(i, 0);
			if(isValid(y, x, _segImg.rows, _segImg.cols) &&
					_segImg.at<unsigned char>(y, x) && reflect) //FIXME: allow reflect to be 0
				onRoadRef.push_back(m_reflectivity.at<unsigned char>(i, 0));
		}

		double thresh = threshold(onRoadRef, onRoadRef, 0, 255, THRESH_TOZERO | THRESH_OTSU);

		if(m_debug)
			cout << "Exiting VeloProjector::OtsuThresholdRoad()" << endl;

		return thresh;
	}

	void VeloProjector::ReadVeloData(string _bin_file) {
		//taken from KITTI website	
		if(m_debug)
			cout << "Entering VeloProjector::ReadVeloData() " << endl;

		m_veloPoints.release();
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
			if(*px >= m_minX && (i % m_retentionFrequency) == 0) {
				Mat m(1, 4, CV_32F, px);
				m_veloPoints.push_back(m);   
			}
			px+=4;
		}
		fclose(stream);
		
		if(m_debug)
			cout << "Exiting VeloProjector::ReadVeloData() " << endl; 
	}

	void VeloProjector::ComputeProjMat(Eigen::MatrixXf& _PVeloToImg) {
		if(m_debug)
			cout << "Entering VeloProjector::ComputeProjMat() " << endl;
		
		Eigen::MatrixXf RCamToRect = Eigen::MatrixXf::Identity(4, 4);
		
		RCamToRect.topLeftCorner<3, 3>() = m_RRect;
		
		_PVeloToImg = m_PRect * RCamToRect * m_Tr; 	
		if(m_debug)
			cout << "Exiting VeloProjector::ComputeProjMat() " << endl;
	}

	void VeloProjector::Project(const Eigen::MatrixXf& _PVeloToImg, Eigen::MatrixXf& _veloImg) {
		
		if(m_debug)
			cout << "Entering VeloProjector::Project() " << endl;
		
		int dimNorm = _PVeloToImg.rows();
		int dimProj = _PVeloToImg.cols();

		if(dimProj != m_veloPoints.cols) 
			throw std::runtime_error("incorrect dimensions to multiply");

		if(!m_veloPoints.isContinuous())
			throw std::runtime_error("matrix is not continuous");

		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > veloPtsEg(m_veloPoints.ptr<float>(), m_veloPoints.rows, m_veloPoints.cols);
		
		m_reflectivity = m_veloPoints.col(dimProj - 1).clone();
		veloPtsEg.col(dimProj - 1) = Eigen::MatrixXf::Ones(veloPtsEg.rows(), 1);

		Eigen::MatrixXf newPoints = (_PVeloToImg * veloPtsEg.transpose()).transpose();
		
		for(int i = 0; i < dimNorm - 1; i++)
			newPoints.col(i).array() = newPoints.col(i).array() / newPoints.col(dimNorm - 1).array();

		_veloImg = newPoints.topLeftCorner(newPoints.rows(), newPoints.cols() - 1);

		if(m_debug)
			cout << "Exiting VeloProjector::Project() " << endl;
	}

}
