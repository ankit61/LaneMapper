#ifndef MAP_GENERATOR_H
#define MAP_GENERATOR_H

#include "BaseLD.h"
#include "BSplineTLinkage.h"
#include "Utilities.h"
#include "PointsVisualizer.h"

namespace LD {
    class MapGenerator : public BaseLD {

    public:

        enum LaneType {
            LEFT, RIGHT
        };

        MapGenerator(string _file);

        void Project(const Eigen::ArrayXXf &_veloPoints, const Eigen::MatrixXf &_rotation,
                        const Eigen::VectorXf &_translation, Eigen::ArrayXXf &_newVeloPoints);

        void Project(const Eigen::ArrayXf &_veloPoints, const Eigen::MatrixXf &_rotation,
                        const Eigen::VectorXf &_translation, Eigen::ArrayXf &_newVeloPoints);

        void GetWorldCtrlPts(const string& _imgName, const vector<ArrayXf>& _models, vector<ArrayXXf>& _worldCtrlPts);

        void TransformCtrlPtsToNext(const Eigen::ArrayXf& _model, const string& _curImgFileName, Eigen::ArrayXXf& _ctrlPts);

        void TransformCtrlPtsToNext(const Eigen::ArrayXXf& _curCtrlPts, const string& _curImgFileName, Eigen::ArrayXXf& _nextCtrlPts);

        void PrintPassedWorldCtrlPts(const ArrayXXf& _ctrlPts, const string& _imgBaseName, LaneType _type);

        void GetPassedWorldCtrlPts(const ArrayXXf& _curCtrlPts, const string& _curImgFileName, ArrayXXf& _passedWorldCtrlPts);

        void PrintWorldCtrlPts(const ArrayXXf& _ctrlPts, LaneType _type);

    protected:

        virtual void ParseXML();

        void ParseVioFile(vector<Eigen::ArrayXf> &_transformationInfo);

        void GetRotationTranslation(const string& _imgFileName, Eigen::MatrixXf& _rotation, Eigen::VectorXf& _translation);

        void GetRotationTranslation(const Eigen::ArrayXf& _trasformationVec, Eigen::MatrixXf& _rotation, Eigen::VectorXf& _translation);

        void GetWorldCtrlPts(const ArrayXf &_model, const MatrixXf &_rotation, const VectorXf &_translation, ArrayXXf &_worldCtrlPts);

        string m_vioFile;
        string m_leftCtrlPtsFile, m_rightCtrlPtsFile;
        string m_dataRoot, m_dataFile;
        int m_minX;

        vector<ArrayXf> m_transformationInfo;
        BSplineTLinkage m_bSplineTLinkage;
        PointsVisualizer m_visualizer;
        std::ofstream m_foutLeftCtrlPts, m_foutRightCtrlPts;
    };

}

#endif //EVALUTIONPROJECT_MAPGENERATOR_H
