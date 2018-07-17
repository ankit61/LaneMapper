#include "Solver.h"

#include"Segmenter.h"

#include"Refiner.h"
#include"KPercentExtractor.h"

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
	std::unique_ptr<LD::Solver> solverPtr;

	if(boost::iequals(solver, "Segmenter"))
		solverPtr = std::make_unique<Segmenter>(argv[1]);
	else if(boost::iequals(solver, "Refiner")) {
		string instance(xml.document_element().child("Solvers").child("Refiner").child("SolverInstance").attribute("instance").as_string());
		if(boost::iequals(instance, "KPercentExtractor"))
			solverPtr = std::make_unique<KPercentExtractor>(argv[1]);
	}
	else if(boost::iequals(solver, "VeloProjector")) {
		string instance(xml.document_element().child("Solvers").child("VeloProjector").child("SolverInstance").attribute("instance").as_string());
		if(boost::iequals(instance, "ResultIntersector"))
			solverPtr = std::make_unique<ResultIntersector>(argv[1]);
		else if(boost::iequals(instance, "SurfaceDataMaker"))
			solverPtr = std::make_unique<SurfaceDataMaker>(argv[1]);	
		else
			throw runtime_error("No such VeloProjector instance implemented: " + instance);
	}
	else if(boost::iequals(solver, "TLinkage")) {
		string model(xml.document_element().child("Solvers").child("TLinkage").child("SolverInstance").attribute("model").as_string());
		if(boost::iequals(model, "Line2D"))
			solverPtr = std::make_unique<Line2DTLinkage>(argv[1]);
		else if(boost::iequals(model, "Plane"))
			solverPtr = std::make_unique<PlaneTLinkage>(argv[1]);
		else if(boost::iequals(model, "Circle2D"))
			solverPtr = std::make_unique<Circle2DTLinkage>(argv[1]);
		else if(boost::iequals(model, "Surface"))
			solverPtr = std::make_unique<SurfaceTLinkage>(argv[1]);
		else if(boost::iequals(model, "Road"))
			solverPtr = std::make_unique<RoadTLinkage>(argv[1]);
		else if(boost::iequals(model, "Line3D"))
			solverPtr = std::make_unique<Line3DTLinkage>(argv[1]);
		else
			throw runtime_error("No such TLinkage model implemented: " + model);
	}
	else
		throw runtime_error("No such solver implemented: " + solver);

	solverPtr->Run();

}
