package main

import (
	"duplicheck"
	"flag"
	"fmt"
	"os"
	"path/filepath"
)

var (
	mainpath string
	maindict = duplicheck.ConflictDict{}
	mainchnn = make(chan int, 0)
)

func init() {
	// called before main
	flag.StringVar(&mainpath, "main_path", ".", "the main path to check")
	flag.Parse()

	absmain, _ := filepath.Abs(mainpath)
	fmt.Println("Duplicates Check For", absmain)
}

func walkPath(path string, info os.FileInfo, err error) error {

	if info.IsDir() {
		return nil
	}

	ks, err := duplicheck.KeySumFile(path)
	if err != nil {
		// do not break
		fmt.Println(err)
		return nil
	}
	maindict.Append(ks, path)
	mainchnn <- 1

	return nil
}

func main() {
	go func() {
		count := 0
		for {
			cnt := <-mainchnn
			if cnt == -1 {
				fmt.Printf("\rcounts %d\n", count)
				break
			}
			count += cnt
			fmt.Printf("\rcounts %d\t", count)
		}
	}()

	err := filepath.Walk(mainpath, walkPath)
	mainchnn <- -1
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println("Conflict Found:")
	i := 0
	for _, buck := range maindict.Bucket {
		if !buck.HasNext() {
			continue
		}
		fmt.Printf("[%d] %x\n", i, buck.Key.Prefix())
		i++
		buck.IterItems(func(value string) {
			info, _ := os.Stat(value)
			fmt.Println(" ", info.ModTime().Format("2006-01-02 15:04:05"), value)
		})
	}
}
