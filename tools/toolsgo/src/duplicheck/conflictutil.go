package duplicheck

import (
	"crypto/sha512"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"os"
)

// KeySum the key sum
type KeySum [sha512.Size]byte

// KeySumBytes get keysum from sha sum of src
func KeySumBytes(src []byte) KeySum {
	return sha512.Sum512(src)
}

// KeySumFile get KeySum from a given file
func KeySumFile(value string) (KeySum, error) {
	file, err := os.Open(value)
	if err != nil {
		return KeySum{}, err
	}
	defer file.Close()
	return KeySumReader(file)
}

// KeySumReader get key sum from a reader
func KeySumReader(file io.Reader) (KeySum, error) {
	ks := KeySum{}
	hasher := sha512.New()
	buf := make([]byte, 1024)
	for {
		l, err := file.Read(buf)
		if l == 0 || err == io.EOF {
			// EOF
			break
		}
		if err != nil {
			return ks, nil
		}
		hasher.Write(buf[0:l])
	}
	ks.FromBytes(hasher.Sum(nil))
	return ks, nil
}

// KeySumString get keysum from sha sum of src
func KeySumString(src string) KeySum {
	return sha512.Sum512([]byte(src))
}

// HexEncode the hex string
func (key *KeySum) HexEncode() string {
	return hex.EncodeToString(key[:])
}

// FromBytes converts bytes to this key
func (key *KeySum) FromBytes(src []byte) {
	copy(key[:], src[0:sha512.Size])
}

// Prefix uint16 prefix value
func (key *KeySum) Prefix() uint32 {
	return binary.BigEndian.Uint32(key[:])
}

// KeySumCompare a > b: + 1; a == b: 0; a < b: -1
func KeySumCompare(a KeySum, b KeySum) int {
	for i := 0; i < sha512.Size; i++ {
		ai := uint8(a[i])
		bi := uint8(b[i])
		if ai < bi {
			return -1
		} else if ai > bi {
			return 1
		}
	}
	return 0
}

type conflictItem struct {
	value string
	next  *conflictItem
}

// ConflictBucket conflict bucket
type ConflictBucket struct {
	Key   KeySum
	items conflictItem
}

// IsConflicted check whether this bucket is conflicted
func (bucket *ConflictBucket) IsConflicted() bool {
	return bucket.items.next != nil
}

// HasNext Items has next
func (bucket *ConflictBucket) HasNext() bool {
	return bucket.items.next != nil
}

// Append append a value to bucket
func (bucket *ConflictBucket) append(value string) {
	p := &bucket.items
	for {
		if p.next == nil {
			p.next = &conflictItem{value, nil}
			break
		}
		p = p.next
	}
}

// IterItems iterator for items
func (bucket *ConflictBucket) IterItems(foo func(value string)) {
	p := &bucket.items
	for {
		if p == nil {
			break
		}
		foo(p.value)
		p = p.next
	}
}

// ConflictDict an ordered dict contains the hash message and value
type ConflictDict struct {
	Bucket []ConflictBucket
}

// KeyIndex search the index from bucket
// return (index, nearest smaller element)
// where index < 0 means not found
func (dict *ConflictDict) KeyIndex(key KeySum) (int, int) {
	// using binary search
	i, j := 0, len(dict.Bucket)

	for i < j {
		h := int(uint(i+j) >> 1) // avoid overflow
		ph := KeySumCompare(dict.Bucket[h].Key, key)
		if ph > 0 {
			j = h
		} else if ph < 0 {
			i = h + 1
		} else {
			// 2 keys are equal
			return h, h
		}
	}
	return -1, i
}

// Append append a key to this dict
func (dict *ConflictDict) Append(key KeySum, value string) {
	i, j := dict.KeyIndex(key)
	if i < 0 {
		// key not found
		dict.Bucket = append(dict.Bucket, ConflictBucket{})
		copy(dict.Bucket[j+1:], dict.Bucket[j:])
		dict.Bucket[j] = ConflictBucket{
			Key:   key,
			items: conflictItem{value, nil},
		}
	} else {
		// key found
		dict.Bucket[i].append(value)
	}
}

// IsOrdered (debug) Check dict is ordered
func (dict *ConflictDict) IsOrdered() bool {
	key := dict.Bucket[0].Key
	for _, i := range dict.Bucket {
		t := KeySumCompare(key, i.Key)
		if t > 0 {
			return false
		}
		key = i.Key
	}
	return true
}

// Dump debug only
func (dict *ConflictDict) Dump() {
	fmt.Println("ConflictDict")
	for i, buck := range dict.Bucket {
		fmt.Printf("[%d] %s\n", i, buck.Key.HexEncode())
		buck.IterItems(func(value string) {
			fmt.Println("   ", value)
		})
	}
}
