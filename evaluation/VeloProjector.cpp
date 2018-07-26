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
			//TODO: Check correctness of using m_RRect00 instead of R_rect_0 + m_camNum?
			//the results are same as MATLAB's KITTI code even with R_rect00
			m_calibDataLoader.ReadVariable(camCalibFile, "R_rect_00", 3, 3, m_RRect) &&
			m_calibDataLoader.ReadVariable(veloCalibFile, "R", 3, 3, R) &&
			m_calibDataLoader.ReadVariable(veloCalibFile, "T", 3, 1, T);

		if(!isSuccess)
			throw std::runtime_error("Incorrect format of calibration files");

		m_Tr = Eigen::MatrixXf(R.rows() + 1, R.cols() + T.cols());

		m_Tr << R, T,
			 0, 0, 0, 1;
		
		ComputeProjMat(m_projectionMat);

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
		m_veloRoot           = solverInstance.attribute("veloRoot").as_string();
		m_calibRoot          = solverInstance.attribute("calibRoot").as_string();
		m_outputRoot         = solverInstance.attribute("outputRoot").as_string();
		m_retentionFrequency = solverInstance.attribute("retentionFrequency").as_int();
		m_camNum             = solverInstance.attribute("camNum").as_int(-1);
		m_minX               = solverInstance.attribute("minX").as_int();

		if(m_dataRoot.empty() || m_dataFile.empty() || m_veloRoot.empty() || m_calibRoot.empty() ||  m_outputRoot.empty() || !m_retentionFrequency || (m_camNum == -1) || !m_minX)
			throw runtime_error("at least one of the following attributes are missing in SolverInstance node of VeloProject: dataRoot, dataFile, segRoot, refinedRoot, veloRoot, calibRoot, outputRoot, retentionFrequency, camNum, minX, outputFile, segImgPrefix, refImgPrefix");

		if(m_debug)
			cout << "Exiting VeloProjector::ParseXML()" << endl;
	}


	void VeloProjector::Run() {
		if(m_debug)
			cout << "Entering VeloProjector::Run() " << endl;

		std::ifstream fin(m_dataFile.c_str());
		string line;
		Eigen::MatrixXf projectionMat, veloImgPts; 
		Mat veloPoints, reflectivity;
		while(std::getline(fin, line)) {
		
			m_imgBaseName  = string(basename(const_cast<char*>(line.c_str())));
			string inputImgName = m_dataRoot + "/" + line;

			Mat inputImg = imread(inputImgName);

			if(inputImg.empty())
				throw std::runtime_error("Can't open " + inputImgName);
			else if(m_debug)
				cout << "Successfully read input image: " << inputImgName << endl;

			ReadVeloData(m_veloRoot + "/" + line.substr(0, line.size() - 3) + "bin", veloPoints);
			Project(m_projectionMat, veloPoints, veloImgPts, reflectivity);
			ProcessProjectedLidarPts(veloImgPts, veloPoints, reflectivity, inputImg);
		}

		if(m_debug)
			cout << "Exiting VeloProjector::Run() " << endl;
	}



	void VeloProjector::ReadVeloData(string _binFile, Mat& _veloPoints) {
		//taken from KITTI website	
		if(m_debug)
			cout << "Entering VeloProjector::ReadVeloData() " << endl;

		_veloPoints.release();
		// allocate 4 MB buffer (only ~130*4*4 KB are needed)
		int32_t num = 1000000;
		float *data = (float*)malloc(num*sizeof(float));

		// pointers
		float *px = data+0;

		// load point cloud
		FILE *stream;
		stream = fopen(_binFile.c_str(),"rb");
		num = fread(data,sizeof(float),num,stream) / 4;
		for (int32_t i=0; i<num; i++) {
			if(*px >= m_minX && (i % m_retentionFrequency) == 0) {
				Mat m(1, 4, CV_32F, px);
				_veloPoints.push_back(m);   
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

	void VeloProjector::Project(const Eigen::MatrixXf& _PVeloToImg, Mat& _veloPoints, Eigen::MatrixXf& _veloImg, Mat& _reflectivity) {
		
		if(m_debug)
			cout << "Entering VeloProjector::Project() " << endl;
		
		int dimNorm = _PVeloToImg.rows();
		int dimProj = _PVeloToImg.cols();

		if(dimProj != _veloPoints.cols) 
			throw std::runtime_error("incorrect dimensions to multiply");

		if(!_veloPoints.isContinuous())
			throw std::runtime_error("matrix is not continuous");

		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > veloPtsEg(_veloPoints.ptr<float>(), _veloPoints.rows, _veloPoints.cols);
		
		_reflectivity = _veloPoints.col(dimProj - 1).clone();	
		veloPtsEg.col(dimProj - 1) = Eigen::MatrixXf::Ones(veloPtsEg.rows(), 1);

		Eigen::MatrixXf newPoints = (_PVeloToImg * veloPtsEg.transpose()).transpose();
		
		for(int i = 0; i < dimNorm - 1; i++)
			newPoints.col(i).array() = newPoints.col(i).array() / newPoints.col(dimNorm - 1).array();

		_veloImg = newPoints.topLeftCorner(newPoints.rows(), newPoints.cols() - 1);

		if(m_debug)
			cout << "Exiting VeloProjector::Project() " << endl;
	}
}
