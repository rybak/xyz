#!/bin/bash


for f in *.bmp
do
    nf=${f/bmp/png}
    echo "converting $f → $nf"
    if [[ ! -f "$nf" ]]
    then
        convert "$f" "$nf"
    else
        echo "$nf already exists"
    fi
done
