#include "CalibDataLoader.h"


bool CalibDataLoader::ReadVariable(string _file, string _var_name, int _rows, int _cols, Eigen::MatrixXf& _output) {
	#ifdef DEBUG
		std::cout << "Entering CalibDataLoader::ReadVariable()" << std::endl;
	#endif
	FILE* stream = fopen(_file.c_str(), "r");
	char word[80] = "";
	while(!feof(stream) && !ferror(stream) && strcmp(word, (_var_name + ":").c_str()) != 0)
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
	
	#ifdef DEBUG
		std::cout << "Exiting CalibDataLoader::ReadVariable()" << std::endl;
	#endif
	
	return true;
}
