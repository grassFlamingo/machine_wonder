
.PHONY: all

default: all

CC=g++
CFLAGS = -Wall -O3
CFLAGS += -I "$(PWD)"
CCCOMP = $(CC) $(CFLAGS) -c

ALL_OBJECTS = CSVReader.o \
			ColumnTable.o \
			Utils.o

all: main
	@echo "Builded"
	
main: main.cpp $(ALL_OBJECTS)
	$(CC) $(CFLAGS) -o main.o main.cpp $(ALL_OBJECTS)

%.o: %.cpp %.h
	$(CCCOMP) $< -o $@

.PHONY: clean

clean:
	@-rm -f *.o