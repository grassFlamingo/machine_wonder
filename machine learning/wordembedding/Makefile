.PHONY: all

default: all

GPP = g++
CFLAGS = -Wall -O3
CFLAGS += -I "$(PWD)"

ODIR = bin

# HEADER_LIST = $(wildcard *.h)    # get cpp file list  
# get corresponding target file
_ALL_OBJECTS = toolkit.o \
				matrix.o
				
ALL_OBJECTS = $(patsubst %,$(ODIR)/%,$(_ALL_OBJECTS))

# | means do not check timestamp
helpmessage: | $(ODIR)
	@echo "Hello, Welcome"
	@echo "targets $(ALL_OBJECTS)"
	@echo "Starting..."
	
$(ODIR):
	@echo "Creating \"$(ODIR)\""
	@-mkdir -p $(ODIR)

all: helpmessage main.cpp $(ALL_OBJECTS)
	$(GPP) $(CFLAGS) main.cpp $(ALL_OBJECTS) -o main
	@echo "End."

$(ODIR)/%.o: %.cpp %.h
	$(GPP) -c -o $@ $< 

.PHONY: clean

clean:
	@-rm -rf $(ODIR) main

