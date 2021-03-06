cmake_minimum_required(VERSION 3.5)
project(LaneDetection)

find_package(OpenCV 2.4.9 REQUIRED)
#find_package( OpenCV REQUIRED )
find_package(Boost COMPONENTS regex system filesystem REQUIRED)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

set(ICNET_ROOT "/home/netbot/Research/ICNet/")
set(PUGIXML_ROOT "/home/netbot/Research/pugixml-1.9")
set(ALGLIB_ROOT "/home/netbot/Research/ALGLIB")

# include directory
include_directories(
        /usr/include/python2.7
        /usr/lib/python2.7/dist-packages/numpy/core/include
        /usr/local/include
        /usr/include/hdf5/serial
        ${ICNET_ROOT}/PSPNet/.build_release/src
        ${ICNET_ROOT}/PSPNet/src
        ${ICNET_ROOT}/PSPNet/include
		${PUGIXML_ROOT}/src
		/usr/include/eigen3
		${ALGLIB_ROOT}/src/
		./include/
		./TLinkage/include
		./DBScan/
)

#inlcude libs
link_directories(
	${ICNET_ROOT}/PSPNet/.build_release/lib/
)

# set complile flag
SET(LIBS "-lcaffe -lglog -lboost_system -lopencv_highgui -lopencv_imgproc -lopencv_core")

SET(GCC_COVERAGE_COMPILE_FLAGS "-Wall -Wno-sign-compare -MMD -MP -pthread -fPIC -g -O2")

add_definitions(
	-DCAFFE_VERSION=1.0.0-rc3 
#	-DNDEBUG 
	-DUSE_OPENCV 
	-DUSE_LEVELDB
	-DAE_CPU=AE_INTEL
	-DUSE_LMDB 
	-DCPU_ONLY
	-DAE_CPU=AE_INTEL
)

add_executable(
	LaneDetection 
	src/main.cpp
	src/LaneDetector.cpp
	src/Segmenter.cpp 
	src/Refiner.cpp 
	src/KPercentExtractor.cpp
	src/LaneExtractor.cpp
	src/VeloProjector.cpp 
	src/ResultIntersector.cpp 
#	SurfaceDataMaker.cpp
	src/LaneQualityChecker.cpp
	TLinkage/src/TLinkage.cpp
#	TLinkage/Line2DTLinkage.cpp
#	TLinkage/PlaneTLinkage.cpp
#	TLinkage/Circle2DTLinkage.cpp
#	TLinkage/SurfaceTLinkage.cpp
#	TLinkage/RoadTLinkage.cpp
#	TLinkage/Line3DTLinkage.cpp
	TLinkage/src/BSplineTLinkage.cpp
	TLinkage/src/KLargestOR.cpp
	TLinkage/src/MaxDiffOR.cpp
	TLinkage/src/UniformSampler.cpp
	TLinkage/src/DistBasedSampler.cpp
	src/PointsVisualizer.cpp
	src/FloodFill.cpp
	src/Utilities.cpp
	src/MapGenerator.cpp
	DBScan/DBScan.cpp
	${PUGIXML_ROOT}/src/pugixml.cpp
	${ALGLIB_ROOT}/src/alglibinternal.cpp
	${ALGLIB_ROOT}/src/alglibmisc.cpp
	${ALGLIB_ROOT}/src/ap.cpp
	${ALGLIB_ROOT}/src/dataanalysis.cpp
	${ALGLIB_ROOT}/src/diffequations.cpp
	${ALGLIB_ROOT}/src/fasttransforms.cpp
	${ALGLIB_ROOT}/src/integration.cpp
	${ALGLIB_ROOT}/src/interpolation.cpp
	${ALGLIB_ROOT}/src/linalg.cpp
	${ALGLIB_ROOT}/src/optimization.cpp
	${ALGLIB_ROOT}/src/solvers.cpp
	${ALGLIB_ROOT}/src/specialfunctions.cpp
	${ALGLIB_ROOT}/src/statistics.cpp
)

target_link_libraries(LaneDetection ${OpenCV_LIBS} ${LIBS})
