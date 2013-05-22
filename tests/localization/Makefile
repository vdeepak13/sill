#SOURCES = $(wildcard $(PRL_TEST_DIR)/*.cpp)
SOURCES = ralign.cpp cov.cpp normcut.cpp

BINARIES = $(SOURCES:%.cpp=%)

PRL_DIR = ../..
PRL_INCLUDE_DIR = $(PRL_DIR)/include
PRL_TEST_DIR = $(PRL_DIR)/tests

LIBS = boost-custom boost-trunk boost-sandbox pstade
LIB_ROOT = ../../../libs
LIB_INCLUDES = $(addprefix -I$(LIB_ROOT)/,$(LIBS))
OTHER_INCLUDES = -I../../../../../pieskovisko/GeometricTools/WildMagic4/SDK/Include ../../../../../pieskovisko/GeometricTools/WildMagic4/SDK/Library/Debug/libWm4Foundation.a
#LIB_DIR = -L../../libs/boost-trunk/build/lib
LLIBS = -lblas -llapack 

# Use GCC:
#CXX = g++ -E -Wall
#CXX = g++ -O0 -Wall -lblas -llapack
CXX = g++ -g -ggdb -O0 -Wall -Winit-self -fkeep-inline-functions -DBOOST_UBLAS_ENABLE_PROXY_SHORTCUTS $(LLIBS)
#CXX = g++ -ggdb -O2 -Wall -Winit-self -fkeep-inline-functions $(LLIBS)
# -D_GLIBCXX_DEBUG
#CXX = g++ -Wall
# Use the Intel compiler:
# CXX = icpc -cxxlib-icc -O2 

all:	$(BINARIES)

clean:
	rm -f $(BINARIES)

% : %.cpp
	$(CXX) -I$(PRL_INCLUDE_DIR) $(LIB_INCLUDES) $(LIB_DIR) -o $@ $^ ../../../libs/boost-trunk/build/lib/libboost_serialization-d.a $(OTHER_INCLUDES)

