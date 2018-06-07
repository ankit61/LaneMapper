#include<string>
#include<libgen.h>
#include<vector>
#include<iostream>
#include<unordered_set>
#include<caffe/caffe.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using std::string;
using std::vector;
using namespace caffe;
using std::cout;
using std::endl;


class eval {
	private:
		double meanR, meanG, meanB;
		bool inDebug;
		vector<string> labels;
		int cropSizeW, cropSizeH;
		int originalW, originalH;
		string imgBaseName;
		string modelWeightFile, modelDeployFile;
		string dataFile, segmentedRoot, dataRoot, overlayedRoot;
		shared_ptr<Net<float> > net;
		cv::Mat colormap;
		void preprocess(cv::Mat& img, vector<cv::Mat>* inputChannels);
		void segment(const cv::Mat& img, cv::Mat& segmentedImg);
		void save(cv::Mat& inputImg, cv::Mat& segmentedImg);
		void wrapInputLayer(vector<cv::Mat>* inputChannels);

	public:
		eval(string argv1, string argv2, string argv3, string argv4);
		void run();

};

eval::eval(string argv1, string argv2, string argv3, string argv4) {
	//initializes all data
	#ifdef DEBUG
		inDebug = true;
	#else
		inDebug = false;
	#endif
	if(inDebug)
		cout << "Entered eval::eval()" << endl;
	dataRoot = argv1, dataFile = argv2, segmentedRoot = argv3, overlayedRoot = argv4;
	cropSizeH = 1025, cropSizeW = 2049;
	meanR = 123.68, meanG = 116.779, meanB = 103.939;
	labels = {"road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"};
	modelDeployFile = "prototxt/icnet_cityscapes.prototxt";
	modelWeightFile = "model/icnet_cityscapes_trainval_90k.caffemodel";
	Caffe::set_mode(Caffe::CPU);
	net.reset(new Net<float>(modelDeployFile, TEST));
	net->CopyTrainedLayersFrom(modelWeightFile);
	if(inDebug)
		cout << "net initialized with .prototxt and .caffemodel files" << endl;
	Blob<float>* inputLayer = net->input_blobs()[0];
	inputLayer->Reshape(1, 3, cropSizeH, cropSizeW); // numChannels=3
	net->Reshape();
	colormap = cv::imread("colormap.png", cv::IMREAD_GRAYSCALE);
	//cv::cvtColor(colormap, colormap, CV_RGB2BGR);
	if(inDebug)
		cout << "Exiting eval::eval()" << endl;
}

void eval::run() {
	if(inDebug)
		cout << "Entering eval::run()" << endl;
	std::ifstream fin(dataFile.c_str());
	string line, imgbaseName;
	vector<cv::Mat> inputChannels(3, cv::Mat(cv::Size(cropSizeW, cropSizeH), CV_32FC3));
	wrapInputLayer(&inputChannels);
	while(fin && std::getline(fin, line)) {
		imgBaseName = string(basename(const_cast<char*>(line.c_str())));
		cv::Mat inputImg = cv::imread(dataRoot + "/" + line), segmentedImg, img;
		inputImg.convertTo(img, CV_32FC3);
		CHECK(!img.empty()) << "could not open or find " << line;
		if(inDebug)
			cout << "Successfully opened " << line << endl;
		originalW = img.cols, originalH = img.rows;
		preprocess(img, &inputChannels);
		segment(img, segmentedImg);
		save(inputImg, segmentedImg);
	}
	if(inDebug)
		cout << "Exiting eval::run()" << endl;

}

void eval::wrapInputLayer(vector<cv::Mat>* inputChannels) {
	if(inDebug)
		cout << "Entering eval::wrapInputLayer()" << endl;
	Blob<float>* inputLayer = net->input_blobs()[0];
	int width = inputLayer->width();
	int height = inputLayer->height();
	float* inputData = inputLayer->mutable_cpu_data();
	for (int i = 0; i < inputLayer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, inputData);
		inputChannels->at(i) = channel;
		inputData += width * height;
	}
	if(inDebug)
		cout << "Exiting eval::wrapInputLayer()" << endl;

}

void eval::preprocess(cv::Mat& img, vector<cv::Mat>* inputChannels) {
	if(inDebug)
		cout << "Entering eval::preprocess()" << endl;
	cv::resize(img, img, cv::Size(cropSizeW, cropSizeH));
	cv::subtract(img, cv::Scalar(meanB, meanG, meanR), img);
	cv::split(img, *inputChannels); //puting image in input layer
	//not rotaing image as in eval_sub.m  because c++ is row major unlike MATLAB
	if(inDebug) {
		cout << "Image set to be input of network" << endl;
		cout << "Exiting eval::preprocess()" << endl;
	}
}

void eval::segment(const cv::Mat& img, cv::Mat& segmentedImg) {
	if(inDebug)
		cout << "Entering eval::segment()" << endl;
	net->ForwardPrefilled();
	if(inDebug) {
		cout << "Forward pass computed " << endl;
	}
	Blob<float>* outputLayer = net->output_blobs()[0];
	//assuming output has only one channel
	cv::Mat score(outputLayer->channels(), outputLayer->width() * outputLayer->height(), CV_32FC1, outputLayer->mutable_cpu_data());
	score = score.t(); //transpose to take advantage of row major nature
	//max index should not change by these operations
	/*cv::exp(score, score);
	cv::Mat channelWiseSum;
	cv::reduce(score, channelWiseSum, 1, CV_REDUCE_SUM);
	vector<cv::Mat> channelWiseSumVec(score.cols);
	cv::Mat channelWiseSum2D;
	for(int i = 0; i < score.cols; i++) {
		channelWiseSumVec[i] = channelWiseSum;
	}
	cv::hconcat(channelWiseSumVec, channelWiseSum2D);
	cv::divide(score, channelWiseSum2D, score);*/
	segmentedImg = cv::Mat(outputLayer->height(), outputLayer->width(), CV_8UC1);
	double maxVal;
	cv::Point maxIndex;
	std::unordered_set<int> thingsIdentified;
	for (int i = 0; i < score.rows; i++) {
		minMaxLoc(score.row(i), 0, &maxVal, 0, &maxIndex);
		segmentedImg.at<uchar>(i) = maxIndex.x;
		if(thingsIdentified.find(maxIndex.x) == thingsIdentified.end()) {
			if(inDebug)
				cout << labels[maxIndex.x] << " found" << endl;
			thingsIdentified.insert(maxIndex.x);
		}
		
	}
	cv::resize(segmentedImg, segmentedImg, cv::Size(originalW, originalH));
	if(inDebug) {
		cout << "Found segmented image" << endl;
		cout << "Exiting eval::segment()" << endl;
	}
	//cv::cvtColor(prediction.clone(), prediction, CV_GRAY2BGR);

}

void eval::save(cv::Mat& inputImg, cv::Mat& segmentedImg) {
	//create segmentation
	if(inDebug) {
		cout << "Entering eval::save()" << endl;	
		cout << "The number of channels and depths of the segmented image and the colormap must be the same" << endl;
		cout << "Segmented image(rows, columns, channels, depth): " << segmentedImg.rows << " " << segmentedImg.cols << " " << segmentedImg.channels() << " " << segmentedImg.depth() << endl;
		cout << "Colormap(rows, columns, channels, depth): " << colormap.rows << " " << colormap.cols << " " << colormap.channels() << " " << colormap.depth() << endl;
	}
	cv::LUT(segmentedImg, colormap, segmentedImg);
	string segImgName = "segmented_" + imgBaseName;
	vector<cv::Mat> channels;
	cv::Mat empty = cv::Mat::zeros(segmentedImg.rows, segmentedImg.cols, CV_8UC1);
	channels.push_back(empty);
	channels.push_back(empty);
	segmentedImg.convertTo(segmentedImg, CV_8UC1);
	channels.push_back(segmentedImg);
	if(inDebug) {
		cout << "The number of rows, columns, and depths must be the same for different matrices" << endl;
		cout << "empty(rows, columns, channels, depth): " << empty.rows << " " << empty.cols << " " << empty.channels() << " " << empty.depth() << endl;
		cout << "Segmented image(rows, columns, channels, depth): " << segmentedImg.rows << " " << segmentedImg.cols << " " << segmentedImg.channels() << " " << segmentedImg.depth() << endl;
	}
	cv::merge(channels, segmentedImg);
	imwrite(segmentedRoot + "/" +  segImgName, segmentedImg);
	if(inDebug)
		cout << "Image " << segImgName << "saved in " << segmentedRoot << endl;

	//create overlay
	cv::Mat overlayed;
	string overlayedImgName = "overlayed_" + imgBaseName;
	cv::addWeighted(segmentedImg, 0.5, inputImg, 0.5, 0.0, overlayed);
	imwrite(overlayedRoot + "/" +  overlayedImgName, overlayed);
	if(inDebug) {
		cout << "Image " << overlayedImgName << "saved in " << overlayedRoot << endl;
		cout << "Exiting eval::save()" << endl;
	}
}

int main(int argc, char* argv[]) {
	if(argc != 5) {
		cout << "4 additional command line arguments expected: root of directory where all images to segment are stored, name of file which lists relative paths of all images to be segmented, directory where segmented images should be stored and directory where overlayed images should be stored" << endl;
		return 1;
	}
	::google::InitGoogleLogging(argv[0]);
	eval e = eval(string(argv[1]), string(argv[2]), string(argv[3]), string(argv[4]));
	e.run();
}
