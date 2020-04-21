rem python .\run_greedy_2diode.py ^
rem        --log_dir "2diode_1_2_5_10" ^
rem        --recipe_file "../recipes/1 cm test.csv" ^
rem        --resolutions "27" ^
rem        --save_video

python .\run_greedy_2diode.py ^
       --log_dir "2diode_1_2_5_10" ^
       --recipe_file "../recipes/2 cm test.csv" ^
       --resolutions "46" ^
       --save_video

python .\run_greedy_2diode.py ^
       --log_dir "2diode_1_2_5_10" ^
       --recipe_file "../recipes/5 cm test.csv" ^
       --resolutions "117" ^
       --save_video

python .\run_greedy_2diode.py ^
       --log_dir "2diode_1_2_5_10" ^
       --recipe_file "../recipes/10 cm test.csv" ^
       --resolutions "136"
