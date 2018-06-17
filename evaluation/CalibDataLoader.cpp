#include "CalibDataLoader.h"


bool CalibDataLoader::ReadVariable(string _file, string _var_name, int _rows, int _cols, cv::Mat& output) {
	FILE* stream = fopen(_file.c_str(), "r");
	char word[80] = "";
	while(!feof(stream) && !ferror(stream) && strcmp(word, (_var_name + ":").c_str()) != 0)
		fscanf(stream, "%s", word);
	output = cv::Mat(cv::Size(_cols, _rows), CV_32F);
	
	if(feof(stream))
		return false;

	float* fp;
	for(int r = 0; r < _rows; r++) {
		fp = output.ptr<float>(r);
		for(int c = 0; c < _cols; c++) {
			if(ferror(stream))
				return false;
			fscanf(stream, "%f", &fp[c]);
		}
	}
	fclose(stream);
	return true;
}
