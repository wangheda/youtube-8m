
# gzstream.C gzstream.h is from gzstream
# annoylib.h mman.h kissrandom.h

CXX      = g++  
CPPFLAGS = -I. -O -std=c++11 -Wall -pthread
LDFLAGS  = -L. -lgzstream -lz -lpthread -lm
AR       = ar cr

all: libgzstream.a knn_tool

default: libgzstream.a

libgzstream.a : gzstream.o
	${AR} libgzstream.a gzstream.o

gzstream.o : gzstream.C gzstream.h
	${CXX} ${CPPFLAGS} -c -o gzstream.o gzstream.C

knn_tool.o : knn.cc gzstream.h
	${CXX} ${CPPFLAGS} -c -o knn_tool.o knn.cc

knn_tool : knn_tool.o libgzstream.a
	${CXX} -o knn_tool knn_tool.o ${LDFLAGS}

clean :
	rm *.o

cleanall :
	rm *.o libgzstream.a knn_tool
