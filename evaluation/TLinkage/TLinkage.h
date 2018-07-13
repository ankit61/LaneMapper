#ifndef T_LINKAGE_H_
#define T_LINKAGE_H_

#include"../Solver.h"
#include<Eigen/Dense>
#include<memory>

//Samplers

#include"Sampler.h"
#include"UniformSampler.h"
#include"DistBasedSampler.h"

//Outlier Rejectors

#include"OutlierRejector.h"
#include"MaxDiffOR.h"

//Preference Finders

#include"PreferenceFinder.h"
#include"ExpPreferenceFinder.h"
#include"HardPreferenceFinder.h"
#include"GaussPreferenceFinder.h"
#include"TukeyPreferenceFinder.h"

namespace LD {

	using namespace Eigen;

	//TODO: Define ParseXML.

	class TLinkage : public Solver {
		public:

			enum SamplingMethod {
				UNIFORM,
				PREFER_NEAR,
			};
			enum VotingScheme {
				EXP,
				GAUSS,
				HARD,
				TUKEY
			};

			enum OutlierRejectionMethod {
				MAX_SIZE_CHANGE
			};

			virtual void Run() override;

			void ReadDataFromFile(ArrayXXf& _data);

			virtual void Sample(const ArrayXXf& _data, ArrayXXf& _sampleIndices, 
					const ulli& _numSamples);

			virtual ArrayXf GenerateHypothesis(const vector<ArrayXf>& _samples) = 0;

			virtual void GenerateHypotheses(const ArrayXXf& _data, 
				const ArrayXXf& _sampleIndices, ArrayXXf& _hypotheses);

			virtual void FitModels(const ArrayXXf& _data, const ArrayXf& _clusters, vector<ArrayXf>& _models, const int& _noiseIndex = -1);

			virtual void FitModel(const ArrayXXf& _clusteredPoints, ArrayXf& _model) = 0;

			virtual void FindResiduals(const ArrayXXf& _data, const ArrayXXf& _hypotheses,
					ArrayXXf& _residuals);

			virtual void FindPreferences(const ArrayXXf& _residuals, ArrayXXf& _preferences);

			virtual void Cluster(ArrayXXf& _preferences, ArrayXf& _clusters);

			virtual void RejectOutliers(const ArrayXf& _clusters, ArrayXf& _out);

			virtual void SetSampler(SamplingMethod _method);

			void SetPreferenceFinder(VotingScheme _method);

			virtual void SetOutlierRejector(OutlierRejectionMethod _method);

			void CalculateTanimotoDist(const ArrayXXf& _preferences, ArrayXXf& _distances);

			float Tanimoto(const ArrayXf& _a, const ArrayXf& _b);

			TLinkage(int _minSamples, int _modelParams, string _xmlFile) : Solver(_xmlFile), 
			m_minSamples(_minSamples), m_modelParams(_modelParams),
			m_sampler(std::make_unique<UniformSampler>(m_xmlFileName)), 
			m_outlierRejector(std::make_unique<MaxDiffOR>(_minSamples, m_xmlFileName)) { ParseXML(); }

		protected:

			virtual double Distance(ArrayXf _dataPoint, ArrayXf _model) = 0;

			virtual void ParseXML();

			std::unique_ptr<Sampler> m_sampler;
			std::unique_ptr<OutlierRejector> m_outlierRejector;
			vector<ArrayXf> m_models;
			std::unique_ptr<PreferenceFinder> m_preferenceFinder;
			pugi::xml_node m_TLinkageNode;
			int m_minSamples, m_modelParams, m_samplesPerDataPt;
			string m_dataFile;
			bool m_shouldTranspose;
			string m_modelFile;
	};
}
#endif
