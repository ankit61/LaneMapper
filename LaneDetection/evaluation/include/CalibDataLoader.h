#ifndef CALIB_DATA_LOADER_H_
#define CALIB_DATA_LOADER_H_

#include<fstream>
#include<string>
#include<iostream>
#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<stdio.h>
#include<stdlib.h>
#include<Eigen/Dense>
#include"BaseLD.h"

namespace LD {

	class CalibDataLoader : BaseLD {
		protected:

			virtual void ParseXML() {}
		
		public: 
			
			CalibDataLoader(string _xmlFile) : BaseLD(_xmlFile) {}

			bool ReadVariable(string _file, string _varName, int _rows, int _cols, Eigen::MatrixXf& _output) {
				
				if(m_debug)
					cout << "Entering CalibDataLoader::ReadVariable()" << endl;
				
				FILE* stream = fopen(_file.c_str(), "r");
				
                if(stream == NULL) {
                    perror(("Failed to open " + _file).c_str());
                    exit(1);
                }
                char word[80] = "";
				while(!feof(stream) && !ferror(stream) && strcmp(word, (_varName + ":").c_str()) != 0)
					fscanf(stream, "%s", word);
				_output = Eigen::MatrixXf(_rows, _cols);

				if(feof(stream))
					return false;

				for(int r = 0; r < _rows; r++) {
					for(int c = 0; c < _cols; c++) {
						if(ferror(stream))
							return false;
						fscanf(stream, "%f", &_output(r,c));
					}
				}
				fclose(stream);

				if(m_debug)
					cout << "Exiting CalibDataLoader::ReadVariable()" << endl;

				return true;

			}; 
	};

}

#endif
