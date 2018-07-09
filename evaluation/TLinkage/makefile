.PHONY: clean

Line: TLinkage.cpp TLinkage.h DemoLines.cpp Line2DLinkage.cpp Line2DLinkage.h
	make -f makefile.lines

Plane: TLinkage.cpp TLinkage.h DemoPlane.cpp PlaneTLinkage.cpp PlaneTLinkage.h
	make -f makefile.planes

Circle:	TLinkage.cpp TLinkage.h DemoCircle.cpp Circle2DTLinkage.cpp Circle2DTLinkage.h
	make -f makefile.circles

clean:
	$(RM) *.o
	$(RM)  *.d
	$(RM) Segment
	$(RM) VeloProject
	$(RM) Refine
