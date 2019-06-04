#! /bin/bash


echo "Entering " $1

cd $1

echo $PWD


for i in ./*.jpg; do
        md5sum "$i" | awk -v FName="$i" '
            {cmd="mv " "\""FName"\"" " " $1 ".jpg"}
            {system(cmd)}
        '
done


