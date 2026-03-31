CXX = g++
CXXFLAGS = -O3 -std=c++17 -fopenmp

OBJS = main.o weno5_rk3.o field.o first_order_upwind.o

gsolver2d: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f *.o gsolver2d
