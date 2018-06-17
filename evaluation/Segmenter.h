/**
 * \brief C++ implementation of evaluation scripts of ICNet
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

class Segmenter {
	private:
		double mean_r_; 	/**< means in red channel */
		double mean_g_;		/**< means in green channel */
		double mean_b_; 	/**< means in blue channel */
		
		/**< corresponds to debug mode; can be set during compilation by adding -DDEBUG */
		bool debug_;
		
		/**< stores 19 categories of cityscape on which the model is trained */
		vector<string> labels_;
		
		/**< corresponds to the width of image that model takes as input */
		int crop_size_w_;
		
		/**< corresponds to the height of image that model takes as input */	
		int crop_size_h_;
		
		/**< corresponds to the width of original image given by user */
		int original_w_;
	
		/**< corresponds to the height of original image given by user */
		int original_h_;

		/**< name of current image file being processed	*/
		string img_base_name_;
		
		/**< names of .caffemodel and .prototxt files respectively */
		string weights_file_, deploy_file_;

		/**< name of file which stores relative paths of all images to process */
		string data_file_; 
		
		/**< directory in which images to segment are stored */
		string data_root_;
		
		/**< directory in which results of road segmentation should be stored */
		string segmented_root_;
		
		/**< directory in which overlayed images should be stored */
		string overlayed_root_;
		
		/**< the neural network used to process */
		shared_ptr<Net<float> > net_;		
		
		/**< the colormap which would make output more readable */
		cv::Mat colormap_;
		
	
		/**
		 * \brief preprocesses the input image (resizing, zero centering) so 
		 * it can be fed into the model
		 * \pre assumes Segmenter::WrapInputLayer() has been called before 
		 * with _input_channels as input; also assumes that member variables 
		 * are intialized
		 * \param _input_img input image as given by user
		 * \param _input_channels output of Segmenter::WrapInputLayers() must be given here
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
		 * \param _segmented_img as outputted by Segmenter::Segment()
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
		Segmenter(string _argv1, string _argv2, string _argv3, string _argv4);

		/** 
		 * \brief coordinates calls to other member functions and saves the final input
		 */
		void Run();

};
