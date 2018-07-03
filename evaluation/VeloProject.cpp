#include<iostream>
#include"VeloProjector.h"

using namespace std;

int main(int argc, char* argv[]) {
	if(argc != 8) {
		cout << "Expecting 6 args: " << endl;
		cout << "directory-where-images-are-stored" << endl;
		cout << "file-with-relative-paths-to-images" << endl;
		cout << "directory-where-segmented-results-are-stored" << endl;
		cout << "directory-where-refined-results-are-stored" << endl;
		cout << "directory-where-velodyne-points-are-stored" << endl;
		cout << "directory-where-calibration-files-are-stored" << endl;
		cout << "directory-where-output-should-be-saved" << endl;
		return -1;
	}
	

	VeloProjector vp(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]);
	vp.Run();
}
