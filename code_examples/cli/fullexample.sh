#! /usr/bin/env zsh

./prepare_fullexample.py
D47crunch fullexample_rawdata.csv
mv output/D47_correl.csv ./fullexample_D47.csv
rm -r output
D47calib -o fullexample_output.csv -j '>' fullexample_D47.csv
D47calib -o fullexample_output2.csv -j '>' -g fullexample_D47.csv