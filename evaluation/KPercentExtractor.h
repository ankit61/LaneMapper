#ifndef K_PERCENT_EXTRACTOR
#define K_PERCENT_EXTRACTOR

#include"Refiner.h"

namespace LD {

	class KPercentExtractor : public Refiner {
		
		protected:

			int m_k;
			virtual void ParseXML() override;

		public:

			virtual void Refine(const Mat& _extractedImg, Mat& _refinedImg) override;

			KPercentExtractor(string _xmlFile) : Refiner(_xmlFile) { ParseXML(); }
	};

}

#endif
