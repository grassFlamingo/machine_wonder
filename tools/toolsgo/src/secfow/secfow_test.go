package secfow

import (
	"bytes"
	"net"
	"strings"
	"testing"
)

func TestPackWithRandChars(t *testing.T) {
	const ts = "dndsakfjnknfkjnfnfnfkdsnfkdsnfknfknfknfk"

	prc := PackWithRandChars(ts, true)
	if !strings.Contains(string(prc), ts) {
		t.Fail()
	}

	prc = PackWithRandChars(CodeHello, false)
	sprc := string(prc)
	if !strings.Contains(sprc, CodeHello) {
		t.Fail()
	}
	t.Log(sprc)
}

func TestGenKey(t *testing.T) {
	t.Log(GenKey())
}

func TestToken2Hash(t *testing.T) {
	token := "dhfkjdajfdskfjljfdfj"
	key := tokenToKey(token)
	if len(key) != 256 {
		t.Fatal("Not A Key")
	}
	t.Log(key)
}

func testrwListener(t *testing.T, ser net.Listener) {

	con, err := ser.Accept()

	if err != nil {
		t.Fatal(err)
	}

	scn := NewSconnectC(con)
	buff := make([]byte, 1024)
	defer scn.Close()

	for i := 0; i < 10; i++ {
		scn.Write([]byte("hello there"))
		n, _ := scn.Read(buff)
		if !bytes.Equal(buff[0:n], []byte("eps 1024 oto")) {
			t.Fatal("bytes not equal", string(buff))
		}
	}
}

func TestReadWrite(t *testing.T) {
	SetToken("abcdefg")
	ser, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatal(err)
	}

	defer ser.Close()

	go testrwListener(t, ser)

	cli, err := net.Dial("tcp", ser.Addr().String())
	if err != nil {
		t.Fatal(err)
	}

	scli := NewSconnectC(cli)
	defer scli.Close()

	buff := make([]byte, 1024)

	for i := 0; i < 10; i++ {
		n, _ := scli.Read(buff)
		if !bytes.Equal(buff[0:n], []byte("hello there")) {
			t.Fatal("bytes not equal", string(buff[0:n]))
		}
		scli.Write([]byte("eps 1024 oto"))
	}

}
