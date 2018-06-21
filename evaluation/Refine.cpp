#include "Refiner.h"

int main(int argc, char* argv[]) {
	if(argc != 5) {
		cout << "4 additional command line arguments expected: root of directory where all original images are stored, name of file which lists relative paths of all original images, directory where segmented images are stored and directory where refined images should be stored" << endl;
		return 1;
	}

	Refiner r(argv[1], argv[2], argv[3], argv[4]);
	r.Run();

}
