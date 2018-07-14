
#include"Utilities.h"

namespace LD {
	bool isValid(long long int r, long long int c, long long int rows, long long int cols) {
		return r >= 0 && c >= 0 && r < rows && c < cols;
	}
}
