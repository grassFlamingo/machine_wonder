package duplicheck_test

import (
	"duplicheck"
	"encoding/binary"
	"testing"
)

func TestSearch(t *testing.T) {
	dict := duplicheck.ConflictDict{}
	dict.Append(duplicheck.KeySumString("abcfefg"), "abcdefg")
	dict.Append(duplicheck.KeySumString("abcfefg"), "abcdefg3434")

	dict.Append(duplicheck.KeySumString("1234"), "1234")
	dict.Append(duplicheck.KeySumString("1234"), "1234a")

	dict.Append(duplicheck.KeySumString("1"), "1")
	dict.Append(duplicheck.KeySumString("2"), "2")

	dict.Dump()

	if !dict.IsOrdered() {
		t.Fatal("not ordered")
	}

}

func BenchmarkSearch(b *testing.B) {
	dict := duplicheck.ConflictDict{}
	for i := 0; i < 1<<14; i++ {
		buf := make([]byte, 8)
		binary.BigEndian.PutUint32(buf, uint32(i))
		dict.Append(duplicheck.KeySumBytes(buf), "abcdef")
	}
	if !dict.IsOrdered() {
		b.Fatal("not ordered")
	}
}
