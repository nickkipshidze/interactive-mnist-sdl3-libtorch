CXX = g++
CXXFLAGS = -std=c++23 -I/opt/libtorch/include/ -I/opt/libtorch/include/torch/csrc/api/include/
LDFLAGS = -L/opt/libtorch/lib/ -Wl,-rpath,/opt/libtorch/lib
LDLIBS = -lSDL3 -lSDL3_ttf -ltorch -ltorch_cpu -lc10

TARGET = main
SRC = ./main.cpp

.PHONY: run clean

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm ./main
