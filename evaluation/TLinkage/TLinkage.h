#ifndef T_LINKAGE_H_
#define T_LINKAGE_H_

#include"../Solver.h"
#include<Eigen/Dense>
#include<memory>
#include<fstream>
#include<unordered_map>

#include"Sampler.h"
#include"OutlierRejector.h"
#include"PreferenceFinder.h"

//TODO: Use more compile time processing

namespace LD {

	using namespace Eigen;

	class TLinkage : public Solver {
		public:

			enum class SamplingMethod {
				UNIFORM,
				PREFER_NEAR,
			};
			
			enum class PreferenceMethod {
				EXP,
				GAUSS,
				HARD,
				TUKEY
			};

			enum class OutlierRejectionMethod {
				MAX_SIZE_CHANGE,
				K_LARGEST,
				NONE
			};

			virtual void Run() override;
			
			virtual void operator()(const ArrayXXf& _data, ArrayXf& _clusters, vector<ArrayXf>& _models);

			virtual void SetSampler(SamplingMethod _method);

			void SetPreferenceFinder(PreferenceMethod _method);

			virtual void SetOutlierRejector(OutlierRejectionMethod _method);
			
			void PrintClustersToFile(const ArrayXf& _clusters, const string& _imgName);
			
			void PrintModelsToFile(vector<ArrayXf> _models, const string& _imgName);

			TLinkage(int _minSamples, int _modelParams, string _xmlFile);

		protected:

			virtual double Distance(ArrayXf _dataPoint, ArrayXf _model) = 0;
			
			void CalculateTanimotoDist(const ArrayXXf& _preferences, ArrayXXf& _distances);
			
			float Tanimoto(const ArrayXf& _a, const ArrayXf& _b);

			virtual void ParseXML();

			virtual void Sample(const ArrayXXf& _data, ArrayXXf& _sampleIndices, 
					const ulli& _numSamples);

			virtual ArrayXf GenerateHypothesis(const vector<ArrayXf>& _samples) = 0;

			virtual void GenerateHypotheses(const ArrayXXf& _data, 
				const ArrayXXf& _sampleIndices, ArrayXXf& _hypotheses);

			void FitModels(const ArrayXXf& _data, const ArrayXf& _clusters, vector<ArrayXf>& _models, std::unordered_map<int, ulli>& _clusterID2Index, const int& _noiseIndex = -1);

			virtual void FitModel(const ArrayXXf& _clusteredPoints, ArrayXf& _model) = 0;

			virtual void FindResiduals(const ArrayXXf& _data, const ArrayXXf& _hypotheses,
					ArrayXXf& _residuals);

			virtual void FindPreferences(const ArrayXXf& _residuals, ArrayXXf& _preferences);

			virtual void Cluster(ArrayXXf& _preferences, ArrayXf& _clusters, std::unordered_map<int, vector<ulli> >& _clusterID2PtIndices);

			virtual void RejectOutliers(const ArrayXf& _clusters, const std::unordered_map<int, vector<ulli> >& _clusterID2PtIndices, ArrayXf& _out, const int& _noiseIndex = -1);
			
			virtual void RefineModels(const vector<ArrayXf>& _models, const ArrayXXf& _data, const ArrayXf& _clusters, const std::unordered_map<int, ulli>& _clusterID2Index, const std::unordered_map<int, vector<ulli> >& _clusterID2PtIndices, vector<ArrayXf>& _refinedModels, const int& _noiseIndex = -1) {}

			std::unique_ptr<Sampler> m_sampler;
			std::unique_ptr<OutlierRejector> m_outlierRejector;
			std::unique_ptr<PreferenceFinder> m_preferenceFinder;
			pugi::xml_node m_TLinkageNode;
			int m_minSamples, m_modelParams;
			int m_samplesPerDataPt;
			string m_dataFile;
			string m_imgName;
			bool m_shouldTranspose;
			bool m_saveClusters;
			string m_modelFile;
			string m_clusterFile;
			std::ofstream m_foutModels;
			std::ofstream m_foutClusters;
	};
}
#endif
