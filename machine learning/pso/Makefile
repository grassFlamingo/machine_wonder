
.PHONY: all

default: all

CC=gcc
GPP=g++
CFLAGS = -Wall -O3
CFLAGS += -I "$(PWD)"
CFLAGS_END = -lGL -lGLU -lglut
CCCOMP = $(CC) $(CFLAGS) -c

all: pso-cli
	@echo "Builded"
	
main: main.c
	$(CC) $(CFLAGS) -o main.o main.c $(CFLAGS_END)

pso-cli: pso-cli.cpp pso.hpp
	$(GPP) $(CFLAGS) -o pso-cli.o pso-cli.cpp 

grad-cli: grad-cli.cpp pso.hpp
	$(GPP) $(CFLAGS) -o grad-cli.o grad-cli.cpp 


%.o: %.cpp %.h
	$(CCCOMP) $< -o $@

.PHONY: clean

clean:
	@-rm -f *.o
	@-rm pso pso-cli