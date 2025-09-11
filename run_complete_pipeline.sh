python main.py \
 --db-path /home/cosmo/net/malaria/Cosmo/testaMINT/segMINT/malaria_simulations.duckdb \
 --num-workers 4 \
 --parallel-trials 4\
 --device cuda \
 --predictor cases \
 --param-limit 4096 \
 --sim-limit 8 \
 --use-cyclical-time \
 --window-size 14 \
 --min-cases 0.1 \
 --batch-size 4096 \
 --run-tuning \
 --tuning-trials 32 \
 --tuning-timeout 43200 \
 --tuning-output-dir results_tuned_long_final \
 --output-dir results_long_final\
 --epochs 64 \
 --patience 16 \
 --seed 42

python main.py \
 --db-path /home/cosmo/net/malaria/Cosmo/testaMINT/segMINT/malaria_simulations.duckdb \
 --num-workers 4 \
 --parallel-trials 4\
 --device cuda \
 --predictor prevalence \
 --param-limit 4096 \
 --sim-limit 8 \
 --use-cyclical-time \
 --window-size 14 \
 --min-prevalence 0.02 \
 --batch-size 4096 \
 --run-tuning \
 --tuning-trials 32 \
 --tuning-timeout 43200 \
 --tuning-output-dir results_tuned_long_final \
 --output-dir results_long_final \
 --epochs 64 \
 --patience 16 \
 --seed 42

 nohup bash -lc 'python main.py --db-path /home/cosmo/net/malaria/Cosmo/testaMINT/segMINT/malaria_simulations.duckdb --num-workers 4 --parallel-trials 4 --device cuda --predictor cases --param-limit 4096 --sim-limit 8 --use-cyclical-time --window-size 14 --min-cases 0.1 --batch-size 4096 --run-tuning --tuning-trials 32 --tuning-timeout 43200 --tuning-output-dir results_tuned_long_final --output-dir results_long_final --epochs 64 --patience 16 --seed 42 && python main.py --db-path /home/cosmo/net/malaria/Cosmo/testaMINT/segMINT/malaria_simulations.duckdb --num-workers 4 --parallel-trials 4 --device cuda --predictor prevalence --param-limit 4096 --sim-limit 8 --use-cyclical-time --window-size 14 --min-prevalence 0.02 --batch-size 4096 --run-tuning --tuning-trials 32 --tuning-timeout 43200 --tuning-output-dir results_tuned_long_final --output-dir results_long_final --epochs 64 --patience 16 --seed 42' > night_run_$(date +%F_%H%M).log 2>&1 &


