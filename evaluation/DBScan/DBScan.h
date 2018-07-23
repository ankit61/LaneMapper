#ifndef DBSCAN_H_
#define DBSCAN_H_

#include<Eigen/Dense>
#include"../Solver.h"
#include<unordered_set>

namespace LD {

	class DBScan : public Solver {
		protected:
			
			void ParseXML() override;
			
			virtual double Distance(const Eigen::ArrayXf& _pt1, const Eigen::ArrayXf& _pt2);
			
			virtual ulli GetNeighbors(const Eigen::ArrayXXf& _data, const ulli& _ptIndex, vector<ulli>& _neighborIndices, const std::unordered_set<ulli>& _uniques = std::unordered_set<ulli>()) final;

			void PrintOutputFile(std::ofstream& _fout, const Eigen::ArrayXXf& _data, const vector<int>& _labels);

			float m_eps;
			ulli m_minPts;
			string m_dataFile, m_outputFile;
			bool m_shouldTranspose;

		public:
			
			enum {
				UNDEFINED = -2,
				NOISE
			};

			DBScan(const string& _xmlFile) : Solver(_xmlFile) { ParseXML(); }

			virtual void Cluster(const Eigen::ArrayXXf& _data, vector<int>& _labels);

			virtual void Run();
	};

}

#endif
