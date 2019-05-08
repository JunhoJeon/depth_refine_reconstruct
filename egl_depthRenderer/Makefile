.SUFFIXES:


CXX = g++
FLAGS = -g -std=c++11
FLAGS += -I "src"
FLAGS += -I "../mLib/include"
FLAGS += -I "../mLib/src"

OPENCV = `pkg-config opencv --cflags --libs`

LFLAGS = -g
LFLAGS += -lpthread -lglut -lGLU -lGL -lglfw -lGLEW -lEGL
LFLAGS += $(OPENCV)

SRC = main.cpp mLibSource.cpp shader.cpp
OBJS = $(SRC:.cpp=.o)
EXECUTABLE = depthRenderer

.PHONY:	all purge clean

all:	$(EXECUTABLE)

build/%.o:	src/%.cpp
	$(CXX) $(FLAGS) -MP -MD $(<,.o=.d) $< -c -o $@

build/%.o:	src/common/%.cpp
	$(CXX) $(FLAGS) -MP -MD $(<,.o=.d) $< -c -o $@

$(EXECUTABLE):        $(addprefix build/, $(OBJS))
	$(CXX) $^ -o $@ $(LFLAGS)

clean:
	rm -rf build/*.o build/*.d
	rm -rf $(EXECUTABLE)

purge: clean
	rm -rf build/*

# dependency rules
include $(wildcard build/*.d)
