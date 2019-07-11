#!/bin/bash

rm -r ~/pv/apgas/bin/*
cd ~/pv/apgas/src
javac -nowarn -cp .:../lib/\* groupP/BuddyLcm.java -d ../bin
