#ifndef K_PERCENT_EXTRACTOR_H_
#define K_PERCENT_EXTRACTOR_H_

#include"Refiner.h"

namespace LD {

	class KPercentExtractor : public Refiner {
		
		protected:

			int m_k;
			virtual void ParseXML() override;

		public:
			virtual void Preprocess(const Mat& _original, const Mat& _segImg, Mat& _preprocessed) override;

			virtual void Refine(const Mat& _extractedImg, Mat& _refinedImg) override;

			KPercentExtractor(string _xmlFile) : Refiner(_xmlFile) { ParseXML(); }
	};

}

#endif
