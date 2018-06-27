#include "Segmenter.h"

Segmenter::Segmenter(string _argv1, string _argv2, string _argv3, string _argv4) {
	//initializes relevant member variables
	
	#ifdef DEBUG
		debug_ = true;
	#else
		debug_ = false;
	#endif
	
	if(debug_)
		cout << "Entered Segmenter::Segmenter()" << endl;
	
	data_root_ = _argv1, data_file_ = _argv2, segmented_root_ = _argv3, overlayed_root_ = _argv4;
	crop_size_h_ = 1025, crop_size_w_ = 2049;
	mean_r_ = 123.68, mean_g_ = 116.779, mean_b_ = 103.939;
	labels_ = {"road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"};
	deploy_file_ = "prototxt/icnet_cityscapes.prototxt";
	weights_file_ = "model/icnet_cityscapes_train_30k.caffemodel";
	
	Caffe::set_mode(Caffe::CPU);
	net_.reset(new Net<float>(deploy_file_, TEST));
	net_->CopyTrainedLayersFrom(weights_file_);
	
	if(debug_)
		cout << "net initialized with .prototxt and .caffemodel files" << endl;
	
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, 3, crop_size_h_, crop_size_w_); // numChannels=3
	net_->Reshape();
	
	colormap_ = cv::imread("colormap.png", cv::IMREAD_GRAYSCALE);
	
	if(debug_)
		cout << "Exiting Segmenter::Segmenter()" << endl;
}

void Segmenter::Run() {
	//coordinates calls to other member functions and saves the final input
	
	if(debug_)
		cout << "Entering Segmenter::Run()" << endl;
	
	std::ifstream fin(data_file_.c_str());
	string line;
	vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);
	
	while(std::getline(fin, line)) {
		img_base_name_ = string(basename(const_cast<char*>(line.c_str())));
		cv::Mat input_img = cv::imread(data_root_ + "/" + line), segmented_img;
		CHECK(!input_img.empty()) << "could not open or find " << line;
		if(debug_)
			cout << "Successfully opened " << line << endl;
		original_w_ = input_img.cols, original_h_ = input_img.rows;
	
		Preprocess(input_img, &input_channels);
		Segment(segmented_img);
		Save(input_img, segmented_img);
	}
	
	if(debug_)
		cout << "Exiting Segmenter::Run()" << endl;

}

void Segmenter::WrapInputLayer(vector<cv::Mat>* _input_channels) {
	//makes elements of _input_channels point to input layer of network 
	
	if(debug_)
		cout << "Entering Segmenter::WrapInputLayer()" << endl;
	
	*_input_channels = vector<cv::Mat>(3, cv::Mat(cv::Size(crop_size_w_, crop_size_h_), CV_32FC3));
	Blob<float>* input_layer = net_->input_blobs()[0];
	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();

	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		_input_channels->at(i) = channel;
		input_data += width * height;
	}
	
	if(debug_)
		cout << "Exiting Segmenter::WrapInputLayer()" << endl;

}

void Segmenter::Preprocess(const cv::Mat& _input_img, vector<cv::Mat>* _input_channels) {
	//preprocesses the input image (resizing, zero centering) so it can be fed into the model
	
	if(debug_)
		cout << "Entering Segmenter::Preprocess()" << endl;
	
	cv::Mat img;
	_input_img.convertTo(img, CV_32FC3);
	cv::resize(img, img, cv::Size(crop_size_w_, crop_size_h_));
	cv::subtract(img, cv::Scalar(mean_b_, mean_g_, mean_r_), img);
	cv::split(img, *_input_channels); //puting image in input layer
	
	if(debug_) {
		cout << "Image set to be input of network" << endl;
		cout << "Exiting Segmenter::Preprocess()" << endl;
	}
}

void Segmenter::Segment(cv::Mat& _segmented_img) {
	//outputs raw output of ICNet, where every pixel value stores 0-18 which is the 
	//index (in labels_) of the class that that pixel belongs to
	
	if(debug_)
		cout << "Entering Segmenter::Segment()" << endl;
	
	net_->ForwardPrefilled();
	
	if(debug_)
		cout << "Forward pass successfully finsihed " << endl;

	Blob<float>* output_layer = net_->output_blobs()[0];
	cv::Mat score(output_layer->channels(), output_layer->width() * output_layer->height(), CV_32FC1, output_layer->mutable_cpu_data());
	score = score.t(); //transpose to take advantage of row major nature
	_segmented_img = cv::Mat(output_layer->height(), output_layer->width(), CV_8UC1);
	
	double max_val;
	cv::Point max_index;
	std::unordered_set<int> things_identified;
	for (int i = 0; i < score.rows; i++) {
		minMaxLoc(score.row(i), 0, &max_val, 0, &max_index);
		_segmented_img.at<uchar>(i) = max_index.x;
		if(debug_) {
			if(things_identified.find(max_index.x) == things_identified.end()) {
				cout << labels_[max_index.x] << " found" << endl;
				things_identified.insert(max_index.x);
			}
		}	
	}
	
	cv::resize(_segmented_img, _segmented_img, cv::Size(original_w_, original_h_));
	
	if(debug_) {
		cout << "Found segmented image" << endl;
		cout << "Exiting Segmenter::Segment()" << endl;
	}
}

void Segmenter::Save(cv::Mat& _input_img, cv::Mat& _segmented_img) {
	//refines the _segmented_img into a more viewable form and finds the overlayed image
	//It saves both images at desired locations
	
	//create Segmentation
	if(debug_) {
		cout << "Entering Segmenter::Save()" << endl;	
		
		cout << "The number of channels and depths of the segmented image and the colormap must be the same" << endl;
		
		cout << "segmented image(rows, columns, channels, depth): " << 
		_segmented_img.rows << " " << _segmented_img.cols << " " << 
		_segmented_img.channels() << " " << _segmented_img.depth() << endl;
		
		cout << "Colormap(rows, columns, channels, depth): " << 
		colormap_.rows << " " << colormap_.cols << " " << 
		colormap_.channels() << " " << colormap_.depth() << endl;
	}
	
	cv::LUT(_segmented_img, colormap_, _segmented_img);
	string segmented_img_name = "segmented_" + img_base_name_;
	vector<cv::Mat> channels;
	
	if(debug_)
		cout << "The number of rows, columns, and depths must be the same for different matrices" << endl;
	

	imwrite(segmented_root_ + "/" +  segmented_img_name, _segmented_img);
	
	if(debug_)
		cout << "Image " << segmented_img_name << "saved in " << segmented_root_ << endl;

	if(debug_) {

		//create overlay
		cv::Mat overlayed;
		string overlayed_img_name = "overlayed_" + img_base_name_;
		
		vector<cv::Mat> channels;
		cv::Mat black = cv::Mat::zeros(_segmented_img.rows, _segmented_img.cols, CV_8UC1);
		channels.push_back(black);
		channels.push_back(black);
		_segmented_img.convertTo(_segmented_img, CV_8UC1);
		channels.push_back(_segmented_img);
		cv::merge(channels, _segmented_img);

		cv::addWeighted(_segmented_img, 0.5, _input_img, 0.5, 0.0, overlayed);
	
		imwrite(overlayed_root_ + "/" +  overlayed_img_name, overlayed);
		
		cout << "Image " << overlayed_img_name << "saved in " << overlayed_root_ << endl;		
		cout << "Exiting Segmenter::Save()" << endl;
	}
}
