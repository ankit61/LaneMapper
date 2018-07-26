#include "Solver.h"

#include"LaneDetector.h"

#include"Segmenter.h"

#include"Refiner.h"
#include"KPercentExtractor.h"
#include"LaneExtractor.h"

#include"VeloProjector.h"
#include"ResultIntersector.h"
#include"SurfaceDataMaker.h"

#include"TLinkage/TLinkage.h"
#include"TLinkage/Line2DTLinkage.h"
#include"TLinkage/PlaneTLinkage.h"
#include"TLinkage/Circle2DTLinkage.h"
#include"TLinkage/SurfaceTLinkage.h"
#include"TLinkage/RoadTLinkage.h"
#include"TLinkage/Line3DTLinkage.h"

#include"DBScan/DBScan.h"

#include<pugixml.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include<memory>

using namespace LD;

int main(int argc, char* argv[]) {
	if(argc != 2)
		throw runtime_error("xml file name expected as first argument");

	pugi::xml_document xml;
	pugi::xml_parse_result status = xml.load_file(argv[1]);
	if(!status)
		throw runtime_error(string("xml error") + status.description());
	
	string solver = xml.document_element().child("Main").attribute("solver").as_string();
	std::unique_ptr<LD::Solver> solverPtr = nullptr;

	if(boost::iequals(solver, "Segmenter"))
		solverPtr = std::make_unique<Segmenter>(argv[1]);
	else if(boost::iequals(solver, "KPercentExtractor"))
		solverPtr = std::make_unique<KPercentExtractor>(argv[1]);
	else if(boost::iequals(solver, "LaneExtractor"))
		solverPtr = std::make_unique<LaneExtractor>(argv[1]);
	else if(boost::iequals(solver, "ResultIntersector"))
		solverPtr = std::make_unique<ResultIntersector>(argv[1]);
	else if(boost::iequals(solver, "SurfaceDataMaker"))
		solverPtr = std::make_unique<SurfaceDataMaker>(argv[1]);	
	else if(boost::iequals(solver, "Line2DTLinkage"))
		solverPtr = std::make_unique<Line2DTLinkage>(argv[1]);
	else if(boost::iequals(solver, "PlaneTLinkage"))
		solverPtr = std::make_unique<PlaneTLinkage>(argv[1]);
	else if(boost::iequals(solver, "Circle2DTLinkage"))
		solverPtr = std::make_unique<Circle2DTLinkage>(argv[1]);
	else if(boost::iequals(solver, "SurfaceTLinkage"))
		solverPtr = std::make_unique<SurfaceTLinkage>(argv[1]);
	else if(boost::iequals(solver, "RoadTLinkage"))
		solverPtr = std::make_unique<RoadTLinkage>(argv[1]);
	else if(boost::iequals(solver, "Line3DTLinkage"))
		solverPtr = std::make_unique<Line3DTLinkage>(argv[1]);
	else if(boost::iequals(solver, "DBScan"))
		solverPtr = std::make_unique<DBScan>(argv[1]);
	else if(boost::iequals(solver, "LaneDetector"))
		solverPtr = std::make_unique<LaneDetector>(argv[1]);
	else	
		throw runtime_error("No such solver implemented: " + solver);

	solverPtr->Run();
}
