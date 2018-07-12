#ifndef OUTLIER_REJECTOR_H_
#define OUTLIER_REJECTOR_H_

#include<Eigen/Dense>
#include "../BaseLD.h"

namespace LD {
	using namespace Eigen;

	class OutlierRejector : public BaseLD {
		public:
			virtual void operator()(const ArrayXf& _clusters, ArrayXf& _out) = 0;

			virtual void ParseXML() { 
				m_xml = m_xml.child("Solvers").child("TLinkage").child("OutlierRejectors");  
			}

			OutlierRejector(string _xmlFile) : BaseLD(_xmlFile) { ParseXML(); }
	};
}
#endif
