

PATH_BIN = ${PWD}/bin
PATH_BIN_SECFOW = ${PATH_BIN}/secfow

PATH_SRC = ${PWD}/src
PATH_SRC_SECFOW = ${PATH_SRC}/secfow

export GOPATH = ${PWD}

GOBUILD = go build -p 6 -race -v

all: ${PATH_BIN_SECFOW} secfow_client secfow_server

secfow_client: ${PATH_BIN_SECFOW}/client

secfow_server: ${PATH_BIN_SECFOW}/server

${PATH_BIN_SECFOW}/client: ${PATH_SRC_SECFOW}/client/main.go
	@echo "Build secfow client"
	${GOBUILD} -o ${PATH_BIN_SECFOW}/client secfow/client 
	cp ${PATH_SRC_SECFOW}/client/con*.json ${PATH_BIN_SECFOW}
	@echo "Builded sec client\n"

${PATH_BIN_SECFOW}/server: ${PATH_SRC_SECFOW}/server/main.go
	@echo "Build secfow server"
	${GOBUILD} -o ${PATH_BIN_SECFOW}/server secfow/server
	cp ${PATH_SRC_SECFOW}/server/con*.json ${PATH_BIN_SECFOW}
	@echo "Builded sec server\n"

${PATH_BIN_SECFOW}: 
	@mkdir -p ${PATH_BIN_SECFOW}

clean:
	@-rm ${PATH_BIN_SECFOW}/client ${PATH_BIN_SECFOW}/server