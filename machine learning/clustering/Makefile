.PHONY: all

default: all

CC=gcc
GPP=g++
CFLAGS = -Wall -O3
CFLAGS += -I "$(PWD)"
CFLAGS += -I "$(PWD)/../lib"
CFLAGS_END = -lGL -lGLU -lglut
CCCOMP = $(CC) $(CFLAGS) -c

all:
	@echo "Aviliable programs"
	@echo "- kmeans"
	@echo ""

kmeans: kmeans.cpp
	$(GPP) $(CFLAGS) kmeans.cpp -o kmeans.o

%.o: %.cpp %.h
	$(CCCOMP) $< -o $@

.PHONY: clean

clean:
	@-rm -f *.o