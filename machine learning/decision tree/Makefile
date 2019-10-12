
.PHONY: all

default: all

CC=g++
CFLAGS = -Wall -g
CFLAGS += -I "$(PWD)"
CCCOMP = $(CC) $(CFLAGS) -c

ALL_OBJECTS = CSVReader.o \
			theColumn.o \
			utils.o

all: main
	@echo "Builded"
	
main: main.cpp $(ALL_OBJECTS)
	$(CC) $(CFLAGS) -o main.o main.cpp $(ALL_OBJECTS)

%.o: %.cpp %.h

.PHONY: clean

clean:
	@-rm -f *.o