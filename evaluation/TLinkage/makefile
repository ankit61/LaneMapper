.PHONY: clean

Line: TLinkage.cpp TLinkage.h Line2DTLinkage.cpp Line2DTLinkage.h DemoLines.cpp
	make -f makefile.lines

Plane: TLinkage.cpp TLinkage.h PlaneTLinkage.cpp PlaneTLinkage.h DemoPlane.cpp
	make -f makefile.planes

Circle: TLinkage.cpp TLinkage.h Circle2DTLinkage.cpp Circle2DTLinkage.h DemoCircle.cpp
	make -f makefile.circles

clean:
	$(RM) *.o
	$(RM)  *.d
	$(RM) Segment
	$(RM) VeloProject
	$(RM) Refine
