#!/bin/bash

testplans=(./tests/*)
for ((i=0; i<${#testplans[@]}; i++)); do
    jmeter -n -t ${testplans[$i]}
done

echo "" > statlog.txt
echo "ResultFile,CompareFile,Column,Correlation" > correlations.csv
csvs=(./results/*)
for ((i=0; i<${#csvs[@]}; i++)); do
    ./jmeter-plots.py -r ${csvs[$i]} -p histogram -o "${csvs[$i]}-histogram";
    for ((j=i+1 ; j<${#csvs[@]}; j++)); do
        name="${csvs[$i]##*/}-vs-${csvs[$j]##*/}";
        echo "$name {" 2>&1 | tee -a statlog.txt;
        ./jmeter-plots.py -r ${csvs[$i]} -c ${csvs[$j]} -o "$name-candlestick" 2>&1 | tee -a statlog.txt;
        ./jmeter-plots.py -r ${csvs[$i]} -c ${csvs[$j]} -p Q-Q -o "$name-Q-Q" 2>&1 | tee -a correlations.csv;
        echo "}" 2>&1 | tee -a statlog.txt
    done
done

echo "Correlations {" 2>&1 | tee -a statlog.txt;
./jmeter-plots.py -r correlations.csv -p CorrelationBPlot -o Summary 2>&1 | tee -a statlog.txt;
echo "}" 2>&1 | tee -a statlog.txt;
