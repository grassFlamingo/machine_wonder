package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net"
	"os"
	"secfow"
	"strings"
)

type configapp struct {
	Localport  string `json:"localport"`
	Remoteaddr string `json:"remoteaddr"`
}

type config struct {
	Server string      `json:"server"`
	Port   string      `json:"port"`
	Token  string      `json:"token"`
	Apps   []configapp `json:"apps"`
}

func init() {
	secfow.InitLog()
}

func handshake(scnn *secfow.SConnect, remoteaddr string) (err error) {

	// handshake with conn server
	buff := make([]byte, 1024)

	prc := secfow.PackWithRandChars(secfow.CodeHello, true)

	// say hello
	n, err := scnn.Write(prc)
	if err != nil || n != len(prc) {
		return err
	}

	// wait hello
	n, err = scnn.Read(buff)
	if err != nil {
		return err
	}

	if !strings.Contains(string(buff), secfow.CodeHello) {
		return errors.New("Got Errot while checking passcode. Exit")
	}

	// send changekey
	randC1 := secfow.PackWithRandChars(secfow.CodeChangeKey, false)
	n, err = scnn.Write(randC1)
	if err != nil {
		return err
	}
	randC1 = randC1[len(secfow.CodeChangeKey):]

	// read changed key
	n, err = scnn.Read(buff)
	if err != nil {
		return err
	}

	// the new key
	scnn.UpdateKey(buff[0:secfow.CriperKeyLen])

	// log.Println(buff[0:secfow.CriperKeyLen])

	n = copy(buff, buff[secfow.CriperKeyLen:n])
	// log.Println("randc2", buff[0:n])

	copy(buff[n:], []byte(remoteaddr))

	scnn.Write(buff[0 : len(remoteaddr)+n]) // randC2 + remote address

	// wait response
	n, err = scnn.Read(buff)
	if err != nil {
		return err
	}

	if n < len(randC1) || !bytes.Equal(buff[0:len(randC1)], randC1) {
		return errors.New("Hint randC1 Not match")
	}
	// connect to remote fail
	if buff[n-1] != 1 && len(remoteaddr) > 0 {
		return errors.New("Remote Connect fail")
	}
	return nil
}

func serveLocalOnce(conn net.Conn, remoteaddr string, serveraddr string) {

	sonn, err := secfow.NewSconnect(serveraddr)

	if err != nil {
		log.Println("Connect to server fail", err)
		return
	}

	err = handshake(sonn, remoteaddr)
	if err != nil {
		log.Println("Handshake error", err)
		return
	}

	log.Println("[<->]", conn.LocalAddr(), "->", sonn.RemoteAddr())

	go secfow.BuildPipe(sonn, conn)
	go secfow.BuildPipe(conn, sonn)

}

func serveLocal(app configapp, serveraddr string) {
	lser, err := net.Listen("tcp", app.Localport)
	if err != nil {
		log.Println("Cannot Bind Port", app.Localport)
		return
	}
	for {
		conn, err := lser.Accept()

		if err != nil {
			log.Println("Listen Error", err)
			continue
		}
		serveLocalOnce(conn, app.Remoteaddr, serveraddr)
	}
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Useage:", os.Args[0], "<configurefile.json>")
		return
	}

	_fbyte, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		log.Fatal("Not a configure file", err)
	} else {
		log.Println("Read Configure file", os.Args[1])
	}

	conf := &config{}
	err = json.Unmarshal(_fbyte, conf)
	if err != nil {
		log.Fatal("Not a json file", err)
	}
	log.Printf("Server: %s\nPort: %s\nApps: %+v\n", conf.Server, conf.Port, conf.Apps)
	secfow.SetToken(conf.Token)

	// try connect to server
	serveraddr := fmt.Sprintf("%s:%s", conf.Server, conf.Port)
	cnn, err := secfow.NewSconnect(serveraddr)
	if err != nil {
		log.Fatal("Cannot Connect", err)
	}
	err = handshake(cnn, "")
	cnn.Close()

	if err != nil {
		// hand hake error
		log.Fatal("Hand shake error ", err)
	}

	// Listen Local ports
	for _, app := range conf.Apps {
		go serveLocal(app, serveraddr)
	}
	// this will wait forever
	select {}

}
