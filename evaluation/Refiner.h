/**
 * \brief C++ implementation of eval_all.m script of ICNet
 * \author Ankit Ramchandani
 * \date  
*/

//TODO: Check pre conditions of functions. Some only except CV_8U type images

#ifndef REFINER_H_
#define REFINER_H_

#include<string>
//<libgen.h> is included in Linux; it's only used to extract base name from full file path
//user can easily code this functionality on other platforms manually
#include<libgen.h>		
#include<math.h>
#include<vector>
#include<queue>
#include<iostream>
#include<fstream>
#include<unordered_set>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<limits>
#include<algorithm>
#include "FloodFill.h"

#include"Solver.h"

namespace LD {

	using namespace cv;
	using std::string;
	using std::vector;
	using std::cout;
	using std::endl;

	class Refiner : public Solver {
		protected:
			
			/**< directory which stores original images (not segmented)  */
			string m_dataRoot;
			
			/**< name of file which stores relative paths of all original images to process */
			string m_dataFile;
			
			/**< directory in which segmented images to refine are stored;
				 segmentation results of original images must have "segmented_" prepended to their names
				 Example: if uu_000000.png is name of original image, then segmented_uu_000000.png must be
				 the name of its segmented version stored in segmented_root_    */
			string m_segRoot;

			/**< directory in which refined image results should be stored */
			string m_refinedRoot;

			/**< name of current image file being processed	*/
			string m_imgBaseName;

			/**< file stream of file where all stats would be stored*/
			std::ofstream m_statFileStream;

			/**
			 * \brief fits the image to a gaussian and thresholds the image based upon its intensity values
			 * \param _extractedImg image which is black except where there is road
			 * \param _thresholdedImg stores output 
			 */
			void ThresholdImage(const Mat& _extractedImg, Mat& _thresholdedImg);

			virtual void ParseXML() override;
			
			public:
			/**
			 * \brief initializes relevant member variables
			 * \param _dataRoot root of directory where all original images are stored
			 * \param _dataFile name of file which lists relative paths (to _dataRoot) of all images original images
			 * \param _segRoot directory where segmented images are stored
			 * \param _refinedRoot directory where refined images should be stored
			 */
			Refiner(string _xmlFile);


			/** 
			 * \brief coordinates calls to other member functions and saves the final input
			 */
			virtual void Run() override;
		
	};

}

#endif
