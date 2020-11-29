package secfow

import (
	"crypto/rc4"
	"crypto/sha512"
	"log"
	"math/rand"
	"net"
	"time"
)

// FowService the configure for secfow
type FowService struct {
	ConnList []SConnect
	key      []byte
}

// SConnect secfow connection
type SConnect struct {
	conn   net.Conn
	iocrip *iocripher
}

type iocripher struct {
	icrip *rc4.Cipher
	ocrip *rc4.Cipher
}

// CodeHello hello code for handshake
const CodeHello string = "10101010101010 secloc Hello"

// CodeHelloB hello cold in bytes
var CodeHelloB []byte = []byte(CodeHello)

// CodeChangeKey code for change key
const CodeChangeKey string = "Change Key"

// CodeChangeKeyB code change key in byte
var CodeChangeKeyB []byte = []byte(CodeChangeKey)

// CriperKeyLen length os cripe key
const CriperKeyLen int = 256

// InitLog init golog system
func InitLog() {
	log.SetPrefix("SecLocal: ")
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
}

// GenKey generate random key
func GenKey() []byte {
	key := make([]byte, CriperKeyLen)
	for i := 0; i < CriperKeyLen; i++ {
		key[i] = byte(rand.Int31() % 0xff)
	}
	return key
}

// PackWithRandChars generate random chars
func PackWithRandChars(mids string, insertleft bool) []byte {
	// ascii code from ' ' to '}'; 0x20 to 0x7D
	const cmin = 0x20
	const cmax = 0x7D
	const cdif = cmax - cmin

	lenL := 0
	if insertleft {
		lenL = rand.Int()%60 + 4 // length form [3 -> 63]
	}

	lenR := rand.Int()%60 + 4
	out := make([]byte, lenL+len(mids)+lenR)
	i := 0
	for ; i < lenL; i++ {
		out[i] = byte(rand.Int()%cdif + cmin)
	}
	for j := 0; j < len(mids); j++ {
		out[i] = mids[j]
		i++
	}

	for j := 0; j < lenR; j++ {
		out[i] = byte(rand.Int()%cdif + cmin)
		i++
	}

	return out
}

func tokenToKey(token string) []byte {
	hash := make([]byte, 0, 256)

	const p = 256 / sha512.Size // 4
	tlen := len(token) / p

	for i := p - 1; i >= 0; i-- {
		_h := sha512.Sum512([]byte("secloc" + token[i*tlen:] + "secloc"))
		for _, _ha := range _h {
			hash = append(hash, _ha)
		}
	}
	return hash
}

// BuildPipe just netin -> netout
func BuildPipe(netin net.Conn, netout net.Conn) {
	buff := make([]byte, 1024)
	var err error = nil
	var n int = 0

	defer func() {
		log.Println("[xxx]", netin.RemoteAddr(), "->", netout.RemoteAddr())
		netin.Close()
		netout.Close()
	}()

	for {
		n, err = netin.Read(buff)
		if err != nil {
			return
		}
		n, err = netout.Write(buff[0:n])
		if err != nil {
			return
		}
	}

}

// NewFowService create a app config by token
func NewFowService(token string) (conf FowService) {
	conf = FowService{}
	conf.key = tokenToKey(token)
	return conf
}

// NewSConnect get a new SConnection
func (conf *FowService) NewSConnect(addr string) (scnn *SConnect, err error) {
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		return nil, err
	}
	return &SConnect{conn: conn, iocrip: newiocrip(conf.key)}, nil
}

// NewSConnectC new SConnect using net.Conn
func (conf *FowService) NewSConnectC(conn net.Conn) (scnn *SConnect) {
	return &SConnect{
		conn:   conn,
		iocrip: newiocrip(conf.key),
	}
}

// UpdateKey Update the key
func (scnn *SConnect) UpdateKey(key []byte) {
	scnn.iocrip.Reset()
	scnn.iocrip = newiocrip(key)
}

// Read reads data from the connection.
// Read can be made to time out and return an error after a fixed
// time limit; see SetDeadline and SetReadDeadline.
func (scnn *SConnect) Read(b []byte) (n int, err error) {
	n, err = scnn.conn.Read(b)
	if err != nil {
		return 0, err
	}
	scnn.iocrip.istream(b, b[0:n])
	return n, err
}

// Write writes data to the connection.
// Write can be made to time out and return an error after a fixed
// time limit; see SetDeadline and SetWriteDeadline.
func (scnn *SConnect) Write(b []byte) (n int, err error) {
	tb := make([]byte, len(b))
	scnn.iocrip.ostream(tb, b)
	n, err = scnn.conn.Write(tb)
	if err != nil {
		return 0, err
	}
	return n, err
}

// Close closes the connection.
// Any blocked Read or Write operations will be unblocked and return errors.
func (scnn *SConnect) Close() error {
	scnn.iocrip.Reset()
	return scnn.conn.Close()
}

// LocalAddr returns the local network address.
func (scnn *SConnect) LocalAddr() net.Addr {
	return scnn.conn.LocalAddr()
}

// RemoteAddr returns the remote network address.
func (scnn *SConnect) RemoteAddr() net.Addr {
	return scnn.conn.LocalAddr()
}

// SetDeadline sets the read and write deadlines associated
// with the connection. It is equivalent to calling both
// SetReadDeadline and SetWriteDeadline.
//
// A deadline is an absolute time after which I/O operations
// fail instead of blocking. The deadline applies to all future
// and pending I/O, not just the immediately following call to
// Read or Write. After a deadline has been exceeded, the
// connection can be refreshed by setting a deadline in the future.
//
// If the deadline is exceeded a call to Read or Write or to other
// I/O methods will return an error that wraps os.ErrDeadlineExceeded.
// This can be tested using errors.Is(err, os.ErrDeadlineExceeded).
// The error's Timeout method will return true, but note that there
// are other possible errors for which the Timeout method will
// return true even if the deadline has not been exceeded.
//
// An idle timeout can be implemented by repeatedly extending
// the deadline after successful Read or Write calls.
//
// A zero value for t means I/O operations will not time out.
func (scnn *SConnect) SetDeadline(t time.Time) error {
	return scnn.conn.SetDeadline(t)
}

// SetReadDeadline sets the deadline for future Read calls
// and any currently-blocked Read call.
// A zero value for t means Read will not time out.
func (scnn *SConnect) SetReadDeadline(t time.Time) error {
	return scnn.conn.SetReadDeadline(t)
}

// SetWriteDeadline sets the deadline for future Write calls
// and any currently-blocked Write call.
// Even if write times out, it may return n > 0, indicating that
// some of the data was successfully written.
// A zero value for t means Write will not time out.
func (scnn *SConnect) SetWriteDeadline(t time.Time) error {
	return scnn.conn.SetWriteDeadline(t)
}

//----------------------
func newiocrip(key []byte) *iocripher {
	icrip, err := rc4.NewCipher(key) // skip this err
	ocrip, err := rc4.NewCipher(key)
	if err != nil {
		log.Fatal("Not A Key")
	}
	return &iocripher{
		icrip: icrip, ocrip: ocrip,
	}
}

func (iocrip *iocripher) istream(dst []byte, src []byte) {
	iocrip.icrip.XORKeyStream(dst, src)
}

func (iocrip *iocripher) ostream(dst []byte, src []byte) {
	iocrip.ocrip.XORKeyStream(dst, src)
}

func (iocrip *iocripher) Reset() {
	iocrip.icrip.Reset()
	iocrip.ocrip.Reset()
}
