rem python .\run_solargrid.py ^
rem        --log_dir "redo_1_2_5_10" ^
rem        --recipe_file "./recipes/1 cm test.csv" ^
rem        --resolutions "27"

rem python .\run_solargrid.py ^
rem        --log_dir "redo_1_2_5_10" ^
rem        --recipe_file "./recipes/2 cm test.csv" ^
rem        --resolutions "46"
rem        --save_video

python .\run_solargrid.py ^
       --log_dir "redo_1_2_5_10" ^
       --recipe_file "./recipes/5 cm test.csv" ^
       --resolutions "117"

python .\run_solargrid.py ^
       --log_dir "redo_1_2_5_10" ^
       --recipe_file "./recipes/10 cm test.csv" ^
       --resolutions "136"
