set log_dir="2diode_1_2_5_10"
set max_iters=2000

python .\run_greedy_2diode.py ^
       --log_dir %log_dir% ^
       --recipe_file "../recipes/1 cm test.csv" ^
       --resolutions "27" ^
       --max_iters %max_iters% ^n
       --save_video

python .\run_greedy_2diode.py ^
       --log_dir %log_dir% ^
       --recipe_file "../recipes/2 cm test.csv" ^
       --resolutions "46" ^
       --max_iters %max_iters% ^
       --save_video

python .\run_greedy_2diode.py ^
       --log_dir %log_dir% ^
       --recipe_file "../recipes/5 cm test.csv" ^
       --resolutions "117" ^
       --max_iters %max_iters% ^
       --save_video

python .\run_greedy_2diode.py ^
       --log_dir %log_dir% ^
       --recipe_file "../recipes/10 cm test.csv" ^
       --resolutions "136" ^
       --max_iters %max_iters% ^
       --save_video
