#ifndef BaseLD_H_
#define BaseLD_H_

#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<exception>

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::runtime_error;

class BaseLD {
	protected:
		typedef unsigned long long int ulli;
		bool m_debug = true; //to be changed
		virtual void ParseXML() = 0;
};

#endif
