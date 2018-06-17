#include<iostream>
#include"VeloProjector.h"

using namespace std;

int main(int argc, char* argv[]) {
	if(argc != 7) {
		cout << "Expecting 6 args" << endl;
		return -1;
	}
	

	VeloProjector vp(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
	vp.Run();
}
