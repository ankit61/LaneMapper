/**
 * \brief C++ implementation of eval_all.m script of ICNet
 * \author Ankit Ramchandani
 * \date  06-06-2018
*/

#include<string>
//<libgen.h> is included in Linux; it's only used to extract base name from full file path
//user can easily code this functionality on other platforms manually
#include<libgen.h>		
#include<vector>
#include<iostream>
#include<unordered_set>
#include<caffe/caffe.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace caffe;
using std::string;
using std::vector;
using std::cout;
using std::endl;

class Evaluator {
	private:
		double mean_r_; 	/**< means in red channel */
		double mean_g_;		/**< means in green channel */
		double mean_b_; 	/**< means in blue channel */
		
		bool debug_;
		/**< corresponds to debug mode; can be set during compilation by adding -DDEBUG */
		
		vector<string> labels_;
		/**< stores 19 categories of cityscape on which the model is trained */
		
		int crop_size_w_;
		/**< corresponds to the width of image that model takes as input */
		
		int crop_size_h_;
		/**< corresponds to the height of image that model takes as input */	
		
		int original_w_;
		/**< corresponds to the width of original image given by user */
	
		int original_h_;
		/**< corresponds to the height of original image given by user */

		string img_base_name_;
		/**< name of current image file being processed	*/
		
		string weights_file_, deploy_file_;
		/**< names of .caffemodel and .prototxt files respectively */

		string data_file_; 
		/**< name of file which stores relative paths of all images to process */
		
		string data_root_;
		/**< directory in which images to segment are stored */
		
		string segmented_root_;
		/**< directory in which results of road segmentation should be stored */
		
		string overlayed_root_;
		/**< directory in which overlayed images should be stored */
		
		shared_ptr<Net<float> > net_; /**< the neural network used to process */
		
		cv::Mat colormap_; /**< the colormap which would make output more readable */

		/**
		 * \brief preprocesses the input image (resizing, zero centering) so 
		 * it can be fed into the model
		 * \pre assumes Evaluator::WrapInputLayer() has been called before 
		 * with _input_channels as input; also assumes that member variables 
		 * are intialized
		 * \param _input_img input image as given by user
		 * \param _input_channels output of Evaluator::WrapInputLayers() must be given here
		 */
		void Preprocess(const cv::Mat& _input_img, vector<cv::Mat>* _input_channels);
		
		/**
		 * \brief outputs raw output of ICNet, where every pixel value stores 0-18
		 *  which is the index (in labels_) of the class that that pixel belongs to
		 * \pre assumes Preprocess() has been called before
		 * \param _segmented_img stores segmented image after completion
		 */
		void Segment(cv::Mat& _segmented_img);
		
		/**
		 * \brief refines the _segmented_img into a more viewable form and finds
		 * the overlayed image.  It saves both images at desired locations
		 * \pre assumes Segment() is called and its output is the second parameter
		 * \param _input_img input image as given by user
		 * \param _segmented_img as outputted by Evaluator::Segment()
		 */
		void Save(cv::Mat& _input_img, cv::Mat& _segmented_img);
		
		/**
		 * \brief makes elements of _input_channels point to input layer of network 
		 * \param _input_channels blank vector should be passed
		 */
		void WrapInputLayer(vector<cv::Mat>* _input_channels);

	public:
		/**
		 * \brief initializes relevant member variables
		 * \param _argv1 root of directory where all images to segment are stored
		 * \param _argv2 name of file which lists relative paths (to _argv1) of all 
		 * images to be segmented
		 * \param _argv3 directory where segmented images should be stored
		 * \param _argv4 directory where overlayed images should be stored
		 */
		Evaluator(string _argv1, string _argv2, string _argv3, string _argv4);

		/** 
		 * \brief coordinates calls to other member functions and saves the final input
		 */
		void Run();

};

Evaluator::Evaluator(string _argv1, string _argv2, string _argv3, string _argv4) {
	//initializes relevant member variables
	
	#ifdef DEBUG
		debug_ = true;
	#else
		debug_ = false;
	#endif
	
	if(debug_)
		cout << "Entered Evaluator::Evaluator()" << endl;
	
	data_root_ = _argv1, data_file_ = _argv2, segmented_root_ = _argv3, overlayed_root_ = _argv4;
	crop_size_h_ = 1025, crop_size_w_ = 2049;
	mean_r_ = 123.68, mean_g_ = 116.779, mean_b_ = 103.939;
	labels_ = {"road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"};
	deploy_file_ = "prototxt/icnet_cityscapes.prototxt";
	weights_file_ = "model/icnet_cityscapes_trainval_90k.caffemodel";
	
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
		cout << "Exiting Evaluator::Evaluator()" << endl;
}

void Evaluator::Run() {
	//coordinates calls to other member functions and saves the final input
	
	if(debug_)
		cout << "Entering Evaluator::Run()" << endl;
	
	std::ifstream fin(data_file_.c_str());
	string line;
	vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);
	
	while(fin && std::getline(fin, line)) {
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
		cout << "Exiting Evaluator::Run()" << endl;

}

void Evaluator::WrapInputLayer(vector<cv::Mat>* _input_channels) {
	//makes elements of _input_channels point to input layer of network 
	
	if(debug_)
		cout << "Entering Evaluator::WrapInputLayer()" << endl;
	
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
		cout << "Exiting Evaluator::WrapInputLayer()" << endl;

}

void Evaluator::Preprocess(const cv::Mat& _input_img, vector<cv::Mat>* _input_channels) {
	//preprocesses the input image (resizing, zero centering) so it can be fed into the model
	
	if(debug_)
		cout << "Entering Evaluator::Preprocess()" << endl;
	
	cv::Mat img;
	_input_img.convertTo(img, CV_32FC3);
	cv::resize(img, img, cv::Size(crop_size_w_, crop_size_h_));
	cv::subtract(img, cv::Scalar(mean_b_, mean_g_, mean_r_), img);
	cv::split(img, *_input_channels); //puting image in input layer
	
	if(debug_) {
		cout << "Image set to be input of network" << endl;
		cout << "Exiting Evaluator::Preprocess()" << endl;
	}
}

void Evaluator::Segment(cv::Mat& _segmented_img) {
	//outputs raw output of ICNet, where every pixel value stores 0-18 which is the 
	//index (in labels_) of the class that that pixel belongs to
	
	if(debug_)
		cout << "Entering Evaluator::Segment()" << endl;
	
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
		cout << "Exiting Evaluator::Segment()" << endl;
	}
}

void Evaluator::Save(cv::Mat& _input_img, cv::Mat& _segmented_img) {
	//refines the _segmented_img into a more viewable form and finds the overlayed image
	//It saves both images at desired locations
	
	//create Segmentation
	if(debug_) {
		cout << "Entering Evaluator::Save()" << endl;	
		
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
	cv::Mat black = cv::Mat::zeros(_segmented_img.rows, _segmented_img.cols, CV_8UC1);
	channels.push_back(black);
	channels.push_back(black);
	_segmented_img.convertTo(_segmented_img, CV_8UC1);  
	channels.push_back(_segmented_img);
	
	if(debug_) {
		cout << "The number of rows, columns, and depths must be the same for different matrices" << endl;
		
		cout << "black(rows, columns, channels, depth): " << 
		black.rows << " " << black.cols << " " << 
		black.channels() << " " << black.depth() << endl;
		
		cout << "segmented image(rows, columns, channels, depth): " << 
		_segmented_img.rows << " " << _segmented_img.cols << " " << 
		_segmented_img.channels() << " " << _segmented_img.depth() << endl;
	}
	
	cv::merge(channels, _segmented_img);
	imwrite(segmented_root_ + "/" +  segmented_img_name, _segmented_img);
	
	if(debug_)
		cout << "Image " << segmented_img_name << "saved in " << segmented_root_ << endl;

	//create overlay
	cv::Mat overlayed;
	string overlayed_img_name = "overlayed_" + img_base_name_;
	cv::addWeighted(_segmented_img, 0.5, _input_img, 0.5, 0.0, overlayed);
	imwrite(overlayed_root_ + "/" +  overlayed_img_name, overlayed);
	
	if(debug_) {
		cout << "Image " << overlayed_img_name << "saved in " << overlayed_root_ << endl;
		
		cout << "Exiting Evaluator::Save()" << endl;
	}
}

int main(int argc, char* argv[]) {
	if(argc != 5) {
		cout << "4 additional command line arguments expected: root of directory where all images to segment are stored, name of file which lists relative paths of all images to be segmented, directory where segmented images should be stored and directory where overlayed images should be stored" << endl;
		return 1;
	}
	::google::InitGoogleLogging(argv[0]);
	Evaluator e = Evaluator(string(argv[1]), string(argv[2]), string(argv[3]), string(argv[4]));
	e.Run();
}
