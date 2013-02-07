CXXFLAGS =	-std=c++0x -fopenmp -O3 -g -Wall -fmessage-length=0 -I/home/buck/build/eigen3 -I/usr/include/eigen3/
LDFLAGS := $(LDFLAGS) -std=c++0x -fopenmp -O3 -largtable2 -static

OBJS =		trainNN.o predictNN.o
HEADERS =		*.h
LIBS =

default: all

trainNN : trainNN.o
	$(CXX) $(CXXFLAGS) trainNN.o $(LDFLAGS) -o trainNN

predictNN : predictNN.o
	$(CXX) $(CXXFLAGS) predictNN.o $(LDFLAGS) -o predictNN

%.o : %.cpp *.h Makefile
	@echo "***" $< "***"
	$(CXX) $(CXXFLAGS) -c $< -o $@  

.PHONY : all clean
all:	trainNN 

clean:
	rm -f $(OBJS) $(TARGET)
