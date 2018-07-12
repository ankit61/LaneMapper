#ifndef BaseLD_H_
#define BaseLD_H_

#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<pugixml.hpp>
#include<exception>

namespace LD {
	
	using std::cout;
	using std::endl;
	using std::string;
	using std::vector;
	using std::runtime_error;

	class BaseLD {
		protected:
			typedef unsigned long long int ulli;
			string m_xmlFileName;
			bool m_debug; //to be changed
			pugi::xml_node m_xml;

			virtual void ParseXML(string _file) final {
				pugi::xml_document doc;
				pugi::xml_parse_result docStatus = doc.load_file(_file.c_str());

				if(!docStatus)
					throw runtime_error("Error with " + _file + ": "  + docStatus.description());

				m_xml = doc.document_element();

				m_debug = m_xml.child("Main").attribute("debug").as_bool();
			}
			
			virtual void ParseXML() = 0;

			BaseLD(string _file) : m_xmlFileName(_file) {
				ParseXML(_file); //will call above function
			}

	};

}

#endif
