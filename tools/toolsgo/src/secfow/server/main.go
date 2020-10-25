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

type serverConfig struct {
	Bind  string `json:"bind"`
	Port  string `json:"port"`
	Token string `json:"Token"`
}

func init() {
	secfow.InitLog()
}

func handshake(conn *secfow.SConnect) (err error) {
	buff := make([]byte, 1024)

	// wait hello
	n, err := conn.Read(buff)
	if err != nil {
		return err
	}

	if !bytes.Contains(buff, secfow.CodeHelloB) {
		return errors.New("Not a hello")
	}

	// send hello back
	n, err = conn.Write(secfow.PackWithRandChars(secfow.CodeHello, true))
	if err != nil {
		return err
	}

	// wait changekey
	n, err = conn.Read(buff)
	if err != nil {
		return err
	}

	if strings.Index(string(buff), secfow.CodeChangeKey) != 0 {
		log.Println("Not Change a key")
		return errors.New("Not Change a key")
	}

	randC1 := make([]byte, n-len(secfow.CodeChangeKey))
	copy(randC1, buff[len(secfow.CodeChangeKey):n])
	randC2 := secfow.PackWithRandChars("", false)

	// random generate key
	key := secfow.GenKey()
	// log.Println("key", key)
	// log.Println("randC2", randC2)

	n = copy(buff, key)
	copy(buff[n:], randC2)
	conn.Write(buff[0 : len(randC2)+n])

	conn.UpdateKey(key)

	// read randC2 + remote address
	n, err = conn.Read(buff)
	if err != nil {
		return
	}

	if n < len(randC2) || !bytes.Equal(buff[0:len(randC2)], randC2) {
		return errors.New("Not RanC2")
	}

	if n == len(randC2) {
		// no remote address
		// just send randC1 back using new key
		conn.Write(randC1)
		conn.Close()
		return nil
	}

	rmaddr := string(buff[len(randC2):n])
	rmConn, err := net.Dial("tcp", rmaddr)

	// log.Println("randC1", randC1)
	n = copy(buff, randC1)
	if err != nil {
		buff[n] = 0
		conn.Write(buff[0 : n+1])
		log.Println("[>-<]", err)
		return err
	}

	buff[n] = 1
	conn.Write(buff[0 : n+1])
	log.Println("[<->]", rmaddr)
	go secfow.BuildPipe(conn, rmConn)
	go secfow.BuildPipe(rmConn, conn)

	return nil
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
	conf := &serverConfig{}
	err = json.Unmarshal(_fbyte, conf)
	if err != nil {
		log.Fatal("Not a json file", err)
	}

	secfow.SetToken(conf.Token)

	bindaddr := fmt.Sprintf("%s:%s", conf.Bind, conf.Port)
	// open connection
	server, err := net.Listen("tcp", bindaddr)
	if err != nil {
		log.Fatalln("could not listen to tcp", bindaddr)
		return
	}
	defer server.Close()

	log.Println("Listen tcp", bindaddr)

	for {
		conn, err := server.Accept()
		if err != nil {
			// handle error
			continue
		}
		err = handshake(secfow.NewSconnectC(conn))
		if err != nil {
			log.Println("[error]", err)
			conn.Close()
		}
	}
}
