CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -pedantic -MMD -MP -O3 -march=native
LDFLAGS :=
TARGET := solve_cvrptw
ANGLE ?= 45
MODE := sequential
BUILD_MODE_FILE := .build_mode

ifeq ($(PARALLEL),1)
MODE := parallel
CXXFLAGS += -fopenmp -DUSE_PARALLEL
LDFLAGS += -fopenmp
endif

SRCS := \
	solve_cvrptw.cpp \
	lib/vrp.cpp \
	lib/route_utils.cpp \
	lib/cluster/clustering.cpp \
	lib/clark/clarke_wright.cpp \
	lib/optim/intra_route_optimization.cpp \
	lib/optim/inter_route_optimization.cpp \
	lib/optim/sa_optimization.cpp \
	lib/optim/route_minimization.cpp

OBJS := $(SRCS:.cpp=.o)
DEPS := $(OBJS:.o=.d)

.PHONY: all clean run par FORCE

all: $(BUILD_MODE_FILE) $(TARGET)

par:
	$(MAKE) PARALLEL=1 all

$(BUILD_MODE_FILE): FORCE
	@if [ ! -f $@ ] || [ "$$(cat $@)" != "$(MODE)" ]; then echo "$(MODE)" > $@; fi

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

%.o: %.cpp $(BUILD_MODE_FILE)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) r102.txt $(ANGLE)

clean:
	rm -f $(TARGET) $(OBJS) $(DEPS) $(BUILD_MODE_FILE)
	rm -rf outputs

-include $(DEPS)
