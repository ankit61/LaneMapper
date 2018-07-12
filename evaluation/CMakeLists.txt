cmake_minimum_required(VERSION 3.5)
project(LaneDetection)

find_package( OpenCV REQUIRED )
find_package(Boost COMPONENTS regex system filesystem REQUIRED)
set(CMAKE_CXX_STANDARD 14)

set(ICNET_ROOT "/home/ankit/Research/ICNet")
set(PUGIXML_ROOT "/home/ankit/Research/pugixml-1.9")

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
)

#inlcude libs
link_directories(
	${ICNET_ROOT}/PSPNet/.build_release/lib/
	${PUGIXML_ROOT}
)

# set complile flag
SET(LIBS "-lcaffe -lglog -lboost_system -lopencv_highgui -lopencv_imgproc -lopencv_core -lpugixml")

SET(GCC_COVERAGE_COMPILE_FLAGS "-Wall -Wno-sign-compare -MMD -MP -pthread -fPIC -g")

add_definitions(
	-DCAFFE_VERSION=1.0.0-rc3 
#	-DNDEBUG 
	-O2 
	-DUSE_OPENCV 
	-DUSE_LEVELDB 
	-DUSE_LMDB 
	-DCPU_ONLY
)

add_executable(
	LaneDetection 
	main.cpp 
	Segmenter.cpp 
	Refiner.cpp 
	VeloProjector.cpp 
	TLinkage/TLinkage.cpp
	TLinkage/Line2DTLinkage.cpp
	TLinkage/PlaneTLinkage.cpp
	TLinkage/Circle2DTLinkage.cpp
)

target_link_libraries(LaneDetection ${OpenCV_LIBS} ${LIBS})
