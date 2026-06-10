#!/bin/bash
set -e 

src=$1
des=$2
ext=-norm.wav

[ ! -d $src ] && echo "No such file: $src" && exit 1;

mkdir -p des
for name in `find $src -name *$ext`; do 
  echo `basename $name $ext` $name
done > $des/wav.scp

