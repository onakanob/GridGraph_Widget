python .\run_greedygrid.py ^
       --log_dir "TEST" ^
       --recipe_file "./recipes/1 cm test.csv" ^
       --resolutions "27"

python .\run_greedygrid.py ^
       --log_dir "TEST" ^
       --recipe_file "./recipes/2 cm test.csv" ^
       --resolutions "46" ^
       --save_video

rem python .\run_solargrid.py ^
rem        --log_dir "redo_1_2_5_10" ^
rem        --recipe_file "./recipes/5 cm test.csv" ^
rem        --resolutions "117"

rem python .\run_solargrid.py ^
rem        --log_dir "redo_1_2_5_10" ^
rem        --recipe_file "./recipes/10 cm test.csv" ^
rem        --resolutions "136"
