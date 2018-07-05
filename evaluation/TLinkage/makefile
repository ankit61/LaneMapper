CC = g++

CFLAGS = -Wall -g -Wno-sign-compare -MD -Wreturn-type

SRCS = Demo.cpp TLinkage.cpp Line2DTLinkage.cpp

OBJS = $(SRCS:.cpp=.o)

INCLUDES = -I/usr/include/eigen3

MAIN = TLinkage

.PHONY:	depend

all:	$(MAIN)

$(MAIN): $(OBJS)
		$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS)

%.o : %.cpp  #Generates .o files from .cpp files
		$(CC) $(CFLAGS)  $(DEFINES) $(INCLUDES) -c $<  -o $@

depend: $(SRCS)
		makedepend $(INCLUDES) $^

-include $(OBJS:.o=.d)
