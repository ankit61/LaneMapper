.PHONY: clean

Segment: Segmenter.cpp Segmenter.h Segment.cpp
	make -f makefile.segment

Refine: Refiner.cpp Refiner.h Refine.cpp
	make -f makefile.refine

VeloProject: VeloProjector.cpp VeloProjector.h VeloProject.cpp CalibDataLoader.cpp CalibDataLoader.h
	make -f makefile.veloproject

clean:
	$(RM) *.o
	$(RM)  *.d
	$(RM) Segment
	$(RM) VeloProject
	$(RM) Refine
