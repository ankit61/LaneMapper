#include "Segmenter.h"

namespace LD {

	Segmenter::Segmenter(string _xmlFile) : Solver(_xmlFile), m_cropSizeH(1025), m_cropSizeW(2049), m_meanR(123.68), 
			m_meanG(116.779), m_meanB(103.939), m_deployFile("prototxt/icnet_cityscapes.prototxt"), 
			m_weightsFile("model/icnet_cityscapes_train_30k.caffemodel") {
		//initializes relevant member variables
		
		if(m_debug)
			cout << "Entered Segmenter::Segmenter()" << endl;

		ParseXML();	
		m_labels = {"road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"};
		
		Caffe::set_mode(Caffe::CPU);
		m_net.reset(new Net<float>(m_deployFile, TEST));
		m_net->CopyTrainedLayersFrom(m_weightsFile);
		
		if(m_debug)
			cout << "net initialized with .prototxt and .caffemodel files" << endl;
		
		Blob<float>* inputLayer = m_net->input_blobs()[0];
		inputLayer->Reshape(1, 3, m_cropSizeH, m_cropSizeW); // numChannels=3
		m_net->Reshape();
		
		m_colormap = cv::imread("colormap.png", cv::IMREAD_GRAYSCALE);

		WrapInputLayer();
		
		if(m_debug)
			cout << "Exiting Segmenter::Segmenter()" << endl;
	}

	void Segmenter::ParseXML() {
		m_xml = m_xml.child("Segmenter");
		pugi::xml_node solverInstance = m_xml.child("SolverInstance");
		
		m_dataRoot 		= solverInstance.attribute("dataRoot").as_string();
		m_dataFile		= solverInstance.attribute("dataFile").as_string();
		m_segRoot 		= solverInstance.attribute("segRoot").as_string();
		m_overlayedRoot = solverInstance.attribute("overlayedRoot").as_string();
		m_saveVizImg	= solverInstance.attribute("saveVizImg").as_bool(true); 
		m_vizImgPrefix	= solverInstance.attribute("vizImgPrefix").as_string(); 
		m_segImgPrefix	= solverInstance.attribute("segImgPrefix").as_string();

		if(m_dataRoot.empty() || m_dataFile.empty() || m_segRoot.empty() || m_overlayedRoot.empty() || m_segImgPrefix.empty() || (m_vizImgPrefix.empty() && m_saveVizImg))
			throw runtime_error("One of the following attributes are missing in SolverInstance node of Segmenter: dataRoot, dataFile, segRoot, overlayedRoot, saveVizImg, vizImgPrefix, segImgPrefix");
	}

	void Segmenter::operator()(const cv::Mat& _inputImg, cv::Mat& _segImg) {
		if(m_debug)
			cout << "Entering Segmenter::()" << endl;
		
		m_originalW = _inputImg.cols, m_originalH = _inputImg.rows;
		Preprocess(_inputImg);
		Segment(_segImg);
		PostProcess(_segImg, _segImg);
		
		if(m_debug)	
			cout << "Exiting Segmenter::()" << endl;
	}

	void Segmenter::Run() {
		//coordinates calls to other member functions and saves the final input
		
		if(m_debug)
			cout << "Entering Segmenter::Run()" << endl;
		
		std::ifstream fin(m_dataFile.c_str());
		string line;
		
		while(std::getline(fin, line)) {
			m_imgBaseName = basename(const_cast<char*>(line.c_str()));
			cv::Mat inputImg, segImg;
			inputImg = cv::imread(m_dataRoot + "/" + line);
			if(!inputImg.empty())
				throw runtime_error("could not open or find " + m_dataRoot + "/" + line);
			if(m_debug)
				cout << "Successfully opened " << line << endl;
		
			this->operator()(inputImg, segImg);
			Save(inputImg, segImg);
		}
		
		if(m_debug)
			cout << "Exiting Segmenter::Run()" << endl;

	}

	void Segmenter::WrapInputLayer() {
		//makes elements of m_inputChannels point to input layer of network 
		
		if(m_debug)
			cout << "Entering Segmenter::WrapInputLayer()" << endl;
		
		m_inputChannels = vector<cv::Mat>(3, cv::Mat(cv::Size(m_cropSizeW, m_cropSizeH), CV_32FC3));
		Blob<float>* inputLayer = m_net->input_blobs()[0];
		int width = inputLayer->width();
		int height = inputLayer->height();
		float* inputData = inputLayer->mutable_cpu_data();

		for (int i = 0; i < inputLayer->channels(); ++i) {
			cv::Mat channel(height, width, CV_32FC1, inputData);
			m_inputChannels[i] = channel;
			inputData += width * height;
		}
		
		if(m_debug)
			cout << "Exiting Segmenter::WrapInputLayer()" << endl;

	}

	void Segmenter::Preprocess(const cv::Mat& _inputImg) {
		//preprocesses the input image (resizing, zero centering) so it can be fed into the model
		
		if(m_debug)
			cout << "Entering Segmenter::Preprocess()" << endl;
		
		//equalizing histogram gives better results
		cv::Mat equalizedImg;
		cvtColor(_inputImg, equalizedImg, CV_BGR2YUV);
		vector<cv::Mat> channels;
		split(equalizedImg, channels);
		equalizeHist(channels[0], channels[0]);
		merge(channels, equalizedImg);
		cvtColor(equalizedImg, equalizedImg, CV_YUV2BGR);
	
		cv::Mat img;
		equalizedImg.convertTo(img, CV_32FC3);
		cv::resize(img, img, cv::Size(m_cropSizeW, m_cropSizeH));
		cv::subtract(img, cv::Scalar(m_meanB, m_meanG, m_meanR), img);
		cv::split(img, m_inputChannels); //puting image in input layer
		
		if(m_debug) {
			cout << "Image set to be input of network" << endl;
			cout << "Exiting Segmenter::Preprocess()" << endl;
		}
	}

	void Segmenter::Segment(cv::Mat& _rawOp) {
		//outputs raw output of ICNet, where every pixel value stores 0-18 which is the 
		//index (in m_labels) of the class that that pixel belongs to
		
		if(m_debug)
			cout << "Entering Segmenter::Segment()" << endl;
		
		m_net->ForwardPrefilled();
		
		if(m_debug)
			cout << "Forward pass successfully finsihed " << endl;

		Blob<float>* outputLayer = m_net->output_blobs()[0];
		cv::Mat score(outputLayer->channels(), outputLayer->width() * outputLayer->height(), CV_32FC1, outputLayer->mutable_cpu_data());
		cv::transpose(score, score); //transpose to take advantage of row major nature
		_rawOp = cv::Mat(outputLayer->height(), outputLayer->width(), CV_8UC1);
		
		double maxVal;
		cv::Point maxIndex;
		std::unordered_set<int> thingsIdentified;
		for (int i = 0; i < score.rows; i++) {
			minMaxLoc(score.row(i), 0, &maxVal, 0, &maxIndex);
			_rawOp.at<uchar>(i) = maxIndex.x;
			if(m_debug) {
				if(thingsIdentified.find(maxIndex.x) == thingsIdentified.end()) {
					cout << m_labels[maxIndex.x] << " found" << endl;
					thingsIdentified.insert(maxIndex.x);
				}
			}	
		}
		
		cv::resize(_rawOp, _rawOp, cv::Size(m_originalW, m_originalH));
		
		if(m_debug) {
			cout << "Found segmented image" << endl;
			cout << "Exiting Segmenter::Segment()" << endl;
		}
	}

	void Segmenter::PostProcess(const cv::Mat& _rawOp, cv::Mat& _segImg) {
		//saves segmented and overlayed images at desired locations

		if(m_debug) {
			cout << "Entering Segmenter::PostProcess()" << endl;
			cout << "The number of channels and depths of the segmented image and the colormap must be the same" << endl;
			
			cout << "segmented image(rows, columns, channels, depth): " << 
			_segImg.rows << " " << _segImg.cols << " " << 
			_segImg.channels() << " " << _segImg.depth() << endl;
			
			cout << "Colormap(rows, columns, channels, depth): " << 
			m_colormap.rows << " " << m_colormap.cols << " " << 
			m_colormap.channels() << " " << m_colormap.depth() << endl;
		}

		cv::LUT(_rawOp, m_colormap, _segImg);

		if(m_debug)
			cout << "Exiting Segmenter::PostProcess()" << endl;
	}

	void Segmenter::Save(const cv::Mat& _inputImg, const cv::Mat& _segImg) {
		// saves segmented and overlayed images at desired locations
		
		//create Segmentation
		if(m_debug) 
			cout << "Entering Segmenter::Save()" << endl;	
		
		string segImgName = m_segImgPrefix + m_imgBaseName;
		vector<cv::Mat> channels;
		
		if(m_debug)
			cout << "The number of rows, columns, and depths must be the same for different matrices" << endl;

		imwrite(m_segRoot + "/" +  segImgName, _segImg);
		
		if(m_debug)
			cout << "Image " << segImgName << "saved in " << m_segRoot << endl;

		if(m_saveVizImg) {
			
			SaveOverlaidImg(_inputImg, _segImg, m_imgBaseName);
			
			if(m_debug)
				cout << "Exiting Segmenter::Save()" << endl;
		}
	}

	void Segmenter::SaveOverlaidImg(const cv::Mat& _inputImg, const cv::Mat& _segImg, string _baseName) {
		cv::Mat overlayed;
		CreateOverlay(_inputImg, _segImg, overlayed);
		string overlayedImgName = m_vizImgPrefix + _baseName;
		cv::imwrite(m_overlayedRoot + "/" +  overlayedImgName, overlayed);

		if(m_debug)
			cout << "Image " << overlayedImgName << "saved in " << m_overlayedRoot << endl;
	}

	void Segmenter::CreateOverlay(const cv::Mat& _inputImg, const cv::Mat& _segImg, cv::Mat& _overlayed) {
		if(m_debug)
			cout << "Entering Segmenter::CreateOverlay()" << endl;

		vector<cv::Mat> channels;
		cv::Mat black = cv::Mat::zeros(_segImg.rows, _segImg.cols, CV_8UC1);
		
		channels.push_back(black);
		channels.push_back(black);
		cv::Mat segImg3C;
		_segImg.convertTo(segImg3C, CV_8UC1);
		channels.push_back(segImg3C);
		
		cv::merge(channels, segImg3C);
		cv::addWeighted(segImg3C, 0.5, _inputImg, 0.5, 0.0, _overlayed);
		
		if(m_debug)
			cout << "Exiting Segmenter::CreateOverlay()" << endl;

	}
}
