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

type appConf struct {
	Localport  string `json:"localport"`
	RemoteAddr string `json:"remoteaddr"`

	FowService secfow.FowService // service
}

type serviceConf struct {
	Server string    `json:"server"` // server address for this app only
	Port   string    `json:"port"`   // server port for this app only
	Token  string    `json:"token"`  // server token for this app only
	Apps   []appConf `json:"apps"`   // the app list
}

type config struct {
	Version  string        `json:"version"`
	Services []serviceConf `json:"services"`
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

func serveLocal(app appConf, serveraddr string) {
	ser, err := net.Listen("tcp", app.Localport)
	if err != nil {
		log.Println("Cannot Bind Port", app.Localport)
		return
	}
	for {
		conn, err := ser.Accept()

		if err != nil {
			log.Println("Listen Error", err)
			continue
		}

		sonn, err := app.FowService.NewSConnect(serveraddr)
		if err != nil {
			log.Println("Cannot connect to Server:", serveraddr)
			continue
		}

		// serveLocalOnce(conn, app.RemoteAddr, serveraddr)

		err = handshake(sonn, app.RemoteAddr)
		if err != nil {
			log.Println("Handshake error", err)
			conn.Close()
			sonn.Close()
			continue
		}

		log.Println("[<->]", conn.LocalAddr(), "->", sonn.RemoteAddr())

		go secfow.BuildPipe(sonn, conn)
		go secfow.BuildPipe(conn, sonn)
	}
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Useage:", os.Args[0], "<configurefile.json>")
		return
	}

	_jsonbyte, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		log.Fatal("Not a configure file", err)
	} else {
		log.Println("Read Configure file", os.Args[1])
	}

	conf := &config{}
	err = json.Unmarshal(_jsonbyte, conf)
	if err != nil {
		log.Fatal("Not a json file", err)
	}

	// Listen Local ports
	for _, ser := range conf.Services {

		log.Printf("Service: %s:%s\n", ser.Server, ser.Port)

		// try connect to server
		serveraddr := fmt.Sprintf("%s:%s", ser.Server, ser.Port)

		fowService := secfow.NewFowService(ser.Token)

		cnn, err := fowService.NewSConnect(serveraddr)

		if err != nil {
			log.Println("Cannot Connect", err)
			continue
		}
		err = handshake(cnn, "")
		cnn.Close()

		if err != nil {
			// hand hake error
			log.Println("Hand shake error ", err)
			continue
		}

		for _, app := range ser.Apps {
			log.Printf("%s -> (%s) -> %s", app.Localport, serveraddr, app.RemoteAddr)
			app.FowService = fowService
			go serveLocal(app, serveraddr)
		}

	}
	// this will wait forever
	select {}

}
