#include "Segmenter.h"

int main(int argc, char* argv[]) {
	if(argc != 5) {
		cout << "4 additional command line arguments expected: root of directory where all images to segment are stored, name of file which lists relative paths of all images to be segmented, directory where segmented images should be stored and directory where overlayed images should be stored" << endl;
		return 1;
	}
	::google::InitGoogleLogging(argv[0]);
	Segmenter e = Segmenter(string(argv[1]), string(argv[2]), string(argv[3]), string(argv[4]));
	e.Run();
}
