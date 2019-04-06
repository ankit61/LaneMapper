#include "MapGenerator.h"
#include "Utilities.h"
#include <algorithm>

namespace LD {

    using namespace Eigen;

    MapGenerator::MapGenerator(string _file) : BaseLD(_file), m_bSplineTLinkage(_file), m_visualizer(_file) {
        ParseXML();
        ParseVioFile(m_transformationInfo);
    }

    void MapGenerator::ParseXML() {

        if (m_debug)
            cout << "Entering MapGenerator::ParseXML()" << endl;

        m_xml = m_xml.child("MapGenerator");
        
        m_dataRoot = m_xml.attribute("dataRoot").as_string();
        m_dataFile = m_xml.attribute("dataFile").as_string();
        m_vioFile = m_xml.attribute("vioFile").as_string();

        m_leftCtrlPtsFile = m_xml.attribute("leftCtrlPtsFile").as_string();
        m_rightCtrlPtsFile = m_xml.attribute("rightCtrlPtsFile").as_string();

        m_minX = m_xml.attribute("minX").as_double();

        if (m_vioFile.empty() || m_leftCtrlPtsFile.empty() || m_rightCtrlPtsFile.empty() || m_dataFile.empty() || m_dataRoot.empty())
            throw runtime_error(
                "missing values in one of the following attributes: vioFile, rightCtrlPtsFile, leftCtrlPtsFile, dataFile, dataRoot"
            );

        if (m_debug)
            cout << "Exiting MapGenerator::ParseXML()" << endl;

    }

    void MapGenerator::Project(const ArrayXXf &_veloPoints, const Eigen::MatrixXf &_rotation, const VectorXf &_translation,
                          ArrayXXf &_newVeloPoints) {
        if (m_debug)
            cout << "Entering MapGenerator::Project()" << endl;

        //_veloPoints expected to have shape of 3 x n, where  n are the number of points
        _newVeloPoints = (((_rotation) * _veloPoints.matrix()).colwise() + _translation).array();

        if (m_debug)
            cout << "Exiting MapGenerator::Project()" << endl;
    }

    void MapGenerator::Project(const ArrayXf &_veloPoint, const MatrixXf &_rotation, const VectorXf &_translation,
                               ArrayXf &_newVeloPoint) {
        if (m_debug)
            cout << "Entering MapGenerator::Project()" << endl;

        _newVeloPoint = ((_rotation * _veloPoint.matrix()) + _translation).array();

        if (m_debug)
            cout << "Exiting MapGenerator::Project()" << endl;
    }

    void MapGenerator::PrintPassedWorldCtrlPts(const ArrayXXf &_ctrlPts, const string &_imgBaseName, LaneType _type) {
        ArrayXXf worldCtrlPts;
        GetPassedWorldCtrlPts(_ctrlPts, _imgBaseName, worldCtrlPts);
        PrintWorldCtrlPts(worldCtrlPts.transpose(), _type);
    }

    void MapGenerator::GetWorldCtrlPts(const ArrayXf &_model, const MatrixXf &_rotation, const VectorXf &_translation,
                                       ArrayXXf &_worldCtrlPts) {
        if (m_debug)
            cout << "Entering MapGenerator::GetWorldCtrlPts()" << endl;

        ArrayXXf localCtrlPts;
        m_bSplineTLinkage.GetControlPts(_model, localCtrlPts);
        Project(localCtrlPts, _rotation, _translation, _worldCtrlPts);

        if (m_debug)
            cout << "Exiting MapGenerator::GetWorldCtrlPts()" << endl;
    }

    void MapGenerator::GetWorldCtrlPts(const string &_imgName, const vector<ArrayXf> &_models,
                                       vector<ArrayXXf> &_worldCtrlPts) {
        MatrixXf rotation;
        VectorXf translation;

        GetRotationTranslation(_imgName, rotation, translation);
        _worldCtrlPts = vector<ArrayXXf>(_models.size());
        for (int i = 0; i < _models.size(); i++)
            GetWorldCtrlPts(_models[i], rotation, translation, _worldCtrlPts[i]);

    }

    void MapGenerator::GetPassedWorldCtrlPts(const ArrayXXf &_curCtrlPts, const string &_curImgFileName,
                                             ArrayXXf &_passedWorldCtrlPts) {
        if (m_debug)
            cout << "Entering MapGenerator::GetPassedCtrlPts()" << endl;

        MatrixXf rotation;
        VectorXf translation;
        GetRotationTranslation(_curImgFileName, rotation, translation);

        if (ImgFile2Int(_curImgFileName) == m_transformationInfo.size() - 1) {
            _passedWorldCtrlPts = _curCtrlPts;
            Project(_passedWorldCtrlPts, rotation, translation, _passedWorldCtrlPts);
            return;
        }

        ArrayXXf nextPts;
        TransformCtrlPtsToNext(_curCtrlPts, _curImgFileName, nextPts);
        _passedWorldCtrlPts.resizeLike(_curCtrlPts);
        int passedPts = 0;

        for (int c = 0; c < nextPts.cols(); c++) {
            if (nextPts(0, c) < m_minX)
                _passedWorldCtrlPts.col(passedPts++) = _curCtrlPts.col(c);
        }

        _passedWorldCtrlPts.conservativeResize(NoChange, passedPts);

        Project(_passedWorldCtrlPts, rotation, translation, _passedWorldCtrlPts);

        if (m_debug)
            cout << "Exiting MapGenerator::GetPassedCtrlPts()" << endl;

    }

    void MapGenerator::TransformCtrlPtsToNext(const ArrayXf &_model, const string &_curImgFileName, ArrayXXf &_ctrlPts) {
        m_bSplineTLinkage.GetControlPts(_model, _ctrlPts);
        TransformCtrlPtsToNext(_ctrlPts, _curImgFileName, _ctrlPts);
    }

    void MapGenerator::TransformCtrlPtsToNext(const Eigen::ArrayXXf &_curCtrlPts, const string &_curImgFileName,
                                              Eigen::ArrayXXf &_nextCtrlPts) {
        int curImg = ImgFile2Int(_curImgFileName);
        if (curImg == m_transformationInfo.size() - 1)
            throw runtime_error("no next image.  Current image must not be the very last one");

        MatrixXf curRotation, nextRotation, relRotation;

        VectorXf curTranslation, nextTranslation, relTranslation;

        GetRotationTranslation(m_transformationInfo[curImg], curRotation, curTranslation);

        GetRotationTranslation(m_transformationInfo[curImg + 1], nextRotation, nextTranslation);

        relRotation = nextRotation.inverse() * curRotation;

        relTranslation = nextTranslation - relRotation * curTranslation;

        Project(_curCtrlPts, relRotation, relTranslation, _nextCtrlPts);

    }
    
    void MapGenerator::ParseVioFile(vector<ArrayXf> &_transformationInfo) {
        if (m_debug)
            cout << "Entering MapGenerator::ParseVioFile()" << endl;

        std::ifstream fin(m_vioFile);
        string line;

        if (!fin)
            throw runtime_error("couldn't open " + m_vioFile);

        while (std::getline(fin, line)) {
            std::istringstream iss(line);
            ArrayXf cur(12); //num points per line in vio file = 12
            for (int i = 0; i < cur.size(); i++)
                iss >> cur(i);
            _transformationInfo.push_back(cur);
        }

        fin.close();

        if (m_debug)
            cout << "Exiting MapGenerator::ParseVioFile()" << endl;

    }

    void MapGenerator::PrintWorldCtrlPts(const ArrayXXf &_ctrlPts, LaneType _type) {

        if (m_debug)
            cout << "Entering MapGenerator::PrintCtrlPts()" << endl;

        if(_ctrlPts.size() == 0)
			return;

		switch (_type) {
            case LEFT:
                if (!m_foutLeftCtrlPts.is_open())
                    m_foutLeftCtrlPts.open(m_leftCtrlPtsFile.c_str());

                m_foutLeftCtrlPts << _ctrlPts << endl;
                break;
            case RIGHT:
                if (!m_foutRightCtrlPts.is_open())
                    m_foutRightCtrlPts.open(m_rightCtrlPtsFile.c_str());

                m_foutRightCtrlPts << _ctrlPts << endl;
                break;
        }

        if (m_debug)
            cout << "Exiting MapGenerator::PrintCtrlPts()" << endl;
    }

    void MapGenerator::GetRotationTranslation(const string &_imgName, MatrixXf &_rotation, VectorXf &_translation) {

        if (m_debug)
            cout << "Entering MapGenerator::GetRotationTranslation()" << endl;

        //This function is highly questionable

        int imgNum = ImgFile2Int(_imgName);
        GetRotationTranslation(m_transformationInfo[imgNum], _rotation, _translation);

        if (m_debug)
            cout << "Exiting MapGenerator::GetRotationTranslation()" << endl;
    }

    void MapGenerator::GetRotationTranslation(const ArrayXf &_transformationVec, MatrixXf &_rotation,
                                              VectorXf &_translation) {

        if (m_debug)
            cout << "Entering MapGenerator::GetRotationTranslation()" << endl;

        assert(_transformationVec.size() == 12);
        _rotation = ArrayXXf(3, 3);
        _translation = ArrayXf(3);
        _rotation << _transformationVec(0), _transformationVec(1), _transformationVec(2),
                _transformationVec(4), _transformationVec(5), _transformationVec(6),
                _transformationVec(8), _transformationVec(9), _transformationVec(10);

        _translation << _transformationVec(11), _transformationVec(3), _transformationVec(7);

        if (m_debug)
            cout << "Exiting MapGenerator::GetRotationTranslation()" << endl;
    }

}
