@echo off

rem Iterate through all test plans in the ./tests/ directory
for %%f in (.\tests*) do (
    rem Run JMeter for each test plan
    jmeter -n -t "%%f"
)

rem Clear the statlog.txt file
echo. > statlog.txt
echo ResultFile,CompareFile,Column,Correlation > correlations.csv
rem Iterate through all CSV files in the ./results/ directory
for %%f in (.\results*) do (
    python jmeter-plots.py -r %%f -p histogram -o "%name%-hisogram"
    rem Iterate through the remaining CSV files
    for %%g in (.\results*) do (
        rem Construct the file name for the comparison
        set "name=%%~nf-vs-%%~ng"
        rem Append the comparison details to the statlog.txt file
        echo %name% { >> statlog.txt
        rem Generate the candlestick plot
        python jmeter-plots.py -r "%%f" -c "%%g" -o "%name%-candlestick" >> statlog.txt
        rem Generate the Q-Q plot
        python jmeter-plots.py -r "%%f" -c "%%g" -p Q-Q -o "%name%-Q-Q" >> correlations.csv
        echo } >> statlog.txt
    )
)

echo Correlations { >> statlog.txt
python jmeter-plots.py -r correlations.csv -p CorrelationBPlot -o Summary >> statlog.txt
echo } >> statlog.txt
