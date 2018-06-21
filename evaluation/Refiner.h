/**
 * \brief C++ implementation of eval_all.m script of ICNet
 * \author Ankit Ramchandani
 * \date  
*/

#include<string>
//<libgen.h> is included in Linux; it's only used to extract base name from full file path
//user can easily code this functionality on other platforms manually
#include<libgen.h>		
#include<math.h>
#include<vector>
#include<iostream>
#include<fstream>
#include<unordered_set>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using std::string;
using std::vector;
using std::cout;
using std::endl;

class Refiner {
	private:
		
		/**< corresponds to debug mode; can be set during compilation by adding -DDEBUG */
		bool debug_;
		
		/**< directory which stores original images (not segmented)  */
		string data_root_;
		
		/**< name of file which stores relative paths of all original images to process */
		string data_file_;
		
		/**< directory in which segmented images to refine are stored;
		     segmentation results of original images must have "segmented_" prepended to their names
			 Example: if uu_000000.png is name of original image, then segmented_uu_000000.png must be
			 the name of its segmented version stored in segmented_root_    */
		string segmented_root_;

		/**< directory in which refined image results should be stored */
		string refined_root_;

		/**< name of current image file being processed	*/
		string img_base_name_;

		/**< file stream of file where all stats would be stored*/
		std::ofstream stat_file_stream_;

		/**
		 *	\brief  
		 *
		 *
		 *
		*/

		
		void FindLaneScores(const cv::Mat& _extracted_img, cv::Mat& _lane_scores, int thresh = 50, int m = 10);

		/**
		 * \brief fits the image to a gaussian and thresholds the image based upon its intensity values
		 * \param _input_img image which is black except where there is road
		 * \param _thresholded_image stores output 
		 */
		void ThresholdImage(const cv::Mat& _input_img, cv::Mat& _thresholded_image);
		
		public:
		/**
		 * \brief initializes relevant member variables
		 * \param _data_root root of directory where all original images are stored
		 * \param _data_file name of file which lists relative paths (to _data_root) of all images original images
		 * \param _segmented_root directory where segmented images are stored
		 * \param _refined_root directory where refined images should be stored
		 */
		Refiner(string _data_root, string _data_file, string _segemented_root, string _refined_root);

		/** 
		 * \brief coordinates calls to other member functions and saves the final input
		 */
		void Run();
	
};
