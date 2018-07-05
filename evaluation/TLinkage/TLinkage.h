#ifndef T_LINKAGE_H_
#define T_LINKAGE_H_

#include"BaseLD.h"
#include"Sampler.h"
#include"OutlierRejector.h"
#include<Eigen/Dense>
#include<memory>
#include"UniformSampler.h"
#include"MaxDiffOR.h"

using namespace Eigen;

//TODO: Define ParseXML.

class TLinkage : public BaseLD {
	public:

		enum VotingScheme {
			EXP,
			GAUSS,
			HARD,
			TUKEY
		};

		enum SamplingMethod {
			UNIFORM,
			PREFER_NEAR,
			NORMAL,
		};

		enum OutlierRejectionMethod {
			MAX_SIZE_CHANGE
		};

		virtual void Sample(const ArrayXXf& _data, ArrayXXf& _sampleIndices, 
			const ulli& _numSamples);

		virtual void GenerateHypothesis(const ArrayXXf& _data, 
			const ArrayXXf& _sampleIndices, ArrayXXf& _hypotheses) = 0;

		virtual void FitModel(const ArrayXXf& _data, const ArrayXf& _clusters, ArrayXXf& _models, const int& noiseIndex = -1) = 0;
		
		virtual void FindResiduals(const ArrayXXf& _data, const ArrayXXf& _hypotheses, 
			ArrayXXf& _residuals);

		virtual void FindPreferences(const ArrayXXf& _residuals, 
			ArrayXXf& _preferences, VotingScheme _method,
			const double& _eps);

		virtual void Cluster(ArrayXXf& _preferences, ArrayXf& _clusters);

		virtual void RejectOutliers(const ArrayXf& _clusters, ArrayXf& _out);

		virtual void SetSampler(SamplingMethod _method);
		
		virtual void SetOutlierRejector(OutlierRejectionMethod _method);

		void CalculateTanimotoDist(const ArrayXXf& _preferences, ArrayXXf& _distances);

		float Tanimoto(const ArrayXf& _a, const ArrayXf& _b);

		//TLinkage() : m_sampler(std::make_unique<UniformSampler>()), m_outlierRejector(std::make_unique<MaxDiffOR(0)>()), 
		//	m_minSamples(0), m_modelParams(0)  {}

		TLinkage(int _minSamples, int _modelParams) : m_sampler(std::make_unique<UniformSampler>()), 
			m_outlierRejector(std::make_unique<MaxDiffOR>(_minSamples)), m_minSamples(_minSamples), m_modelParams(_modelParams) {}

	protected:

		std::unique_ptr<Sampler> m_sampler;
		std::unique_ptr<OutlierRejector> m_outlierRejector;
		
		virtual double Distance(ArrayXf _dataPoint, ArrayXf _model) = 0;

		virtual void ParseXML() {}

		int m_minSamples, m_modelParams;
};

#endif
