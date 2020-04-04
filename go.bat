python .\run_greedygrid.py ^
       --log_dir "greedy_temp" ^
       --recipe_file "./recipes/1 cm test.csv" ^
       --resolutions "5"

python .\run_banditgrid.py ^
       --log_dir "bandit_temp" ^
       --recipe_file "./recipes/1 cm test.csv" ^
       --resolutions "5"

rem python .\run_solargrid.py ^
rem        --log_dir "redo_1_2_5_10" ^
rem        --recipe_file "./recipes/5 cm test.csv" ^
rem        --resolutions "117"

rem python .\run_solargrid.py ^
rem        --log_dir "redo_1_2_5_10" ^
rem        --recipe_file "./recipes/10 cm test.csv" ^
rem        --resolutions "136"
