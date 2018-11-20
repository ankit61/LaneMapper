#include"KPercentExtractor.h"

namespace LD {
	
	void KPercentExtractor::ParseXML() {
		m_xml = m_xml.child("KPercentExtractor");
		m_k = m_xml.attribute("k").as_int();

		if(!m_k)
			throw runtime_error("at least one of the following attribute is missing: k");
	}
	
	void KPercentExtractor::Preprocess(const Mat& _original, const Mat& _segImg, Mat& _preprocessed) {
		if(m_debug)
			cout << "Entering KPercentExtractor::Preprocess()" << endl;

			cvtColor(_original, _preprocessed, COLOR_RGB2GRAY);
			bitwise_and(_preprocessed, _segImg, _preprocessed);
		
		if(m_debug)
			cout << "Exiting KPercentExtractor::Preprocess()" << endl;
	}
	
	void KPercentExtractor::Refine(const Mat& _original, const Mat& _segImg, Mat& _refinedImg) {
		
		if(m_debug)	
			cout << "Entering KPercentExtractor::Refine()" << endl;
		
		Mat extractedImg;
		Preprocess(_original, _segImg, extractedImg);
		//get top k% pixels
		Mat flattened = extractedImg.reshape(1, 1).clone();
		if(flattened.isContinuous()) {
			std::sort(flattened.data, flattened.data + flattened.total());
			int numZeros = std::distance(flattened.datastart, std::upper_bound(flattened.datastart, flattened.dataend, 0));
			int threshIndex = ((flattened.total() - numZeros) * (100 - m_k)) / 100;
			int thresh = flattened.data[numZeros + threshIndex] - 1;
			
			if(m_debug)
				cout << "Threshold set to " << thresh << endl;

			threshold(extractedImg, _refinedImg, thresh, 255, THRESH_BINARY);
		}
		else {
			//ideally it should always be continuous
			throw runtime_error("Matrix is not continuous");
		}

		if(m_debug)	
			cout << "Exiting KPercentExtractor::Refine()" << endl;
	}
}
