#include"KPercentExtractor.h"

namespace LD {
	
	void KPercentExtractor::ParseXML() {
		m_xml = m_xml.child("KPercentExtractor");
		m_k = m_xml.attribute("k").as_int();

		if(!m_k)
			throw runtime_error("at least one of the following attribute is missing: k");
	}
	
	void KPercentExtractor::Refine(const Mat& _extractedImg, Mat& _refinedImg) {
		
		if(m_debug)	
			cout << "Entering KPercentExtractor::Refine()" << endl;
		
		//get top k% pixels
		Mat flattened = _extractedImg.reshape(1, 1).clone();
		if(flattened.isContinuous()) {
			std::sort(flattened.data, flattened.data + flattened.total());
			int numZeros = std::distance(flattened.datastart, std::upper_bound(flattened.datastart, flattened.dataend, 0));
			int threshIndex = ((flattened.total() - numZeros) * (100 - m_k)) / 100;
			int thresh = flattened.data[numZeros + threshIndex];
			
			if(m_debug)
				cout << "Threshold set to " << thresh << endl;

			threshold(_extractedImg, _refinedImg, thresh, 255, THRESH_BINARY);
		}
		else {
			//ideally it should always be continuous
			throw runtime_error("Matrix is not continuous");
		}

		if(m_debug)	
			cout << "Exiting KPercentExtractor::Refine()" << endl;
	}
}
