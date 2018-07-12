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
		
		if(m_debug)
			cout << "Exiting Segmenter::Segmenter()" << endl;
	}

	void Segmenter::ParseXML() {
		m_xml = m_xml.child("Segmenter");
		m_dataRoot = m_xml.child("SolverInstance").attribute("dataRoot").as_string();
		m_dataFile = m_xml.child("SolverInstance").attribute("dataFile").as_string();
		m_segRoot = m_xml.child("SolverInstance").attribute("segRoot").as_string();
		m_overlayedRoot = m_xml.child("SolverInstance").attribute("overlayedRoot").as_string();

		if(m_dataRoot.empty() || m_dataFile.empty() || m_segRoot.empty() || m_overlayedRoot.empty())
			throw runtime_error("One of the following attributes are missing in SolverInstance node of Segmenter: dataRoot, dataFile, segRoot, overlayedRoot");
	}

	void Segmenter::Run() {
		//coordinates calls to other member functions and saves the final input
		
		if(m_debug)
			cout << "Entering Segmenter::Run()" << endl;
		
		std::ifstream fin(m_dataFile.c_str());
		string line;
		vector<cv::Mat> inputChannels;
		WrapInputLayer(&inputChannels);
		
		while(std::getline(fin, line)) {
			m_imgBaseName = basename(const_cast<char*>(line.c_str()));
			cv::Mat inputImg, segImg;
			inputImg = cv::imread(m_dataRoot + "/" + line);
			CHECK(!inputImg.empty()) << "could not open or find " << m_dataRoot + "/" + line;
			if(m_debug)
				cout << "Successfully opened " << line << endl;
			m_originalW = inputImg.cols, m_originalH = inputImg.rows;
		
			Preprocess(inputImg, &inputChannels);
			Segment(segImg);
			Save(inputImg, segImg);
		}
		
		if(m_debug)
			cout << "Exiting Segmenter::Run()" << endl;

	}

	void Segmenter::WrapInputLayer(vector<cv::Mat>* _inputChannels) {
		//makes elements of _inputChannels point to input layer of network 
		
		if(m_debug)
			cout << "Entering Segmenter::WrapInputLayer()" << endl;
		
		*_inputChannels = vector<cv::Mat>(3, cv::Mat(cv::Size(m_cropSizeW, m_cropSizeH), CV_32FC3));
		Blob<float>* inputLayer = m_net->input_blobs()[0];
		int width = inputLayer->width();
		int height = inputLayer->height();
		float* inputData = inputLayer->mutable_cpu_data();

		for (int i = 0; i < inputLayer->channels(); ++i) {
			cv::Mat channel(height, width, CV_32FC1, inputData);
			_inputChannels->at(i) = channel;
			inputData += width * height;
		}
		
		if(m_debug)
			cout << "Exiting Segmenter::WrapInputLayer()" << endl;

	}

	void Segmenter::Preprocess(const cv::Mat& _inputImg, vector<cv::Mat>* _inputChannels) {
		//preprocesses the input image (resizing, zero centering) so it can be fed into the model
		
		if(m_debug)
			cout << "Entering Segmenter::Preprocess()" << endl;
		
		cv::Mat img;
		_inputImg.convertTo(img, CV_32FC3);
		cv::resize(img, img, cv::Size(m_cropSizeW, m_cropSizeH));
		cv::subtract(img, cv::Scalar(m_meanB, m_meanG, m_meanR), img);
		cv::split(img, *_inputChannels); //puting image in input layer
		
		if(m_debug) {
			cout << "Image set to be input of network" << endl;
			cout << "Exiting Segmenter::Preprocess()" << endl;
		}
	}

	void Segmenter::Segment(cv::Mat& _segImg) {
		//outputs raw output of ICNet, where every pixel value stores 0-18 which is the 
		//index (in m_labels) of the class that that pixel belongs to
		
		if(m_debug)
			cout << "Entering Segmenter::Segment()" << endl;
		
		m_net->ForwardPrefilled();
		
		if(m_debug)
			cout << "Forward pass successfully finsihed " << endl;

		Blob<float>* outputLayer = m_net->output_blobs()[0];
		cv::Mat score(outputLayer->channels(), outputLayer->width() * outputLayer->height(), CV_32FC1, outputLayer->mutable_cpu_data());
		score = score.t(); //transpose to take advantage of row major nature
		_segImg = cv::Mat(outputLayer->height(), outputLayer->width(), CV_8UC1);
		
		double maxVal;
		cv::Point maxIndex;
		std::unordered_set<int> thingsIdentified;
		for (int i = 0; i < score.rows; i++) {
			minMaxLoc(score.row(i), 0, &maxVal, 0, &maxIndex);
			_segImg.at<uchar>(i) = maxIndex.x;
			if(m_debug) {
				if(thingsIdentified.find(maxIndex.x) == thingsIdentified.end()) {
					cout << m_labels[maxIndex.x] << " found" << endl;
					thingsIdentified.insert(maxIndex.x);
				}
			}	
		}
		
		cv::resize(_segImg, _segImg, cv::Size(m_originalW, m_originalH));
		
		if(m_debug) {
			cout << "Found segmented image" << endl;
			cout << "Exiting Segmenter::Segment()" << endl;
		}
	}

	void Segmenter::Save(cv::Mat& _inputImg, cv::Mat& _segImg) {
		//refines the _segImg into a more viewable form and finds the overlayed image
		//It saves both images at desired locations
		
		//create Segmentation
		if(m_debug) {
			cout << "Entering Segmenter::Save()" << endl;	
			
			cout << "The number of channels and depths of the segmented image and the colormap must be the same" << endl;
			
			cout << "segmented image(rows, columns, channels, depth): " << 
			_segImg.rows << " " << _segImg.cols << " " << 
			_segImg.channels() << " " << _segImg.depth() << endl;
			
			cout << "Colormap(rows, columns, channels, depth): " << 
			m_colormap.rows << " " << m_colormap.cols << " " << 
			m_colormap.channels() << " " << m_colormap.depth() << endl;
		}
		
		cv::LUT(_segImg, m_colormap, _segImg);
		string segImgName = "segmented_" + m_imgBaseName;
		vector<cv::Mat> channels;
		
		if(m_debug)
			cout << "The number of rows, columns, and depths must be the same for different matrices" << endl;
		

		imwrite(m_segRoot + "/" +  segImgName, _segImg);
		
		if(m_debug)
			cout << "Image " << segImgName << "saved in " << m_segRoot << endl;

		if(m_debug) {

			//create overlay
			cv::Mat overlayed;
			string overlayedImgName = "overlayed_" + m_imgBaseName;
			
			vector<cv::Mat> channels;
			cv::Mat black = cv::Mat::zeros(_segImg.rows, _segImg.cols, CV_8UC1);
			channels.push_back(black);
			channels.push_back(black);
			_segImg.convertTo(_segImg, CV_8UC1);
			channels.push_back(_segImg);
			cv::merge(channels, _segImg);

			cv::addWeighted(_segImg, 0.5, _inputImg, 0.5, 0.0, overlayed);
		
			imwrite(m_overlayedRoot + "/" +  overlayedImgName, overlayed);
			
			cout << "Image " << overlayedImgName << "saved in " << m_overlayedRoot << endl;		
			cout << "Exiting Segmenter::Save()" << endl;
		}
	}
}
