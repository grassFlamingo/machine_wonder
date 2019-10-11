
default: all

CC=g++
CFLAGS = -Wall -O3
CFLAGS += -I "$(PWD)"

CCCOMP = $(CC) $(CFLAGS) -c

all: main
	@echo "Building"
	

main: main.cpp CSV.o
	$(CC) $(CFLAGS) main.cpp CSV.o -o main.o

CSV.o: CSV.cpp CSV.h theColumn.o
	$(CCCOMP) CSV.cpp -o CSV.o

theColumn.o: theColumn.cpp theColumn.h
	$(CCCOMP) theColumn.cpp -o theColumn.o

clean:
	@rm -f *.o