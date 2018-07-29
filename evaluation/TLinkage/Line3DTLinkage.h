#ifndef LINE_3D_T_LINKAGE_H_
#define LINE_3D_T_LINKAGE_H_

#include"TLinkage.h"

namespace LD {

	class Line3DTLinkage : public TLinkage {

		protected:

			virtual ArrayXf GenerateHypothesis(const vector<ArrayXf>& _samples); 

			virtual double Distance(ArrayXf _dataPoint, ArrayXf _model) override;

			virtual void FitModel(const ArrayXXf& _clusters, ArrayXf& _model) override;
			
			virtual void RefineModels(const vector<ArrayXf>& _models, const ArrayXXf& _data, const ArrayXf& _clusters, const std::unordered_map<int, ulli>& _clusterID2Index, const std::unordered_map<int, vector<ulli> >& _clusterID2PtIndices, vector<ArrayXf>& _refinedModels, const int& _noiseIndex = -1);

			virtual void ParseXML() override;
			
			void ShiftModel(ArrayXf& _originalModel, const vector<ulli>& _clusteredIndices, const ArrayXXf& _data, bool _isOriginalModelOnRight, ArrayXf& _shiftedModel);

			
			bool IsModelOnRight(const ArrayXf& _model);
	
			float EvaluateModel(const ArrayXf& _model, const vector<ulli>& _clusterIndices, const ArrayXXf& _data);

			float m_resolution;
			float m_errorThreshold;
			float m_minShift, m_maxShift;

		public:

			Line3DTLinkage(string _xmlFile) : TLinkage(2, 6, _xmlFile) { ParseXML(); }
	};

}
#endif
