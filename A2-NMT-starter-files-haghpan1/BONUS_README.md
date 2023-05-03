
## CSC401/2511 - W23 README file for bonus (transformers) 

These files were made available based on a few students' request to avail some starter/
boiler-plate code to allow solving the NMT (Hansards) using 'transformers' architecture.

There are other acceptable and qualitatively equal bonus work approaches available and suggested.

This is just to aid students who would like to explore the transformers architecture as bonus work.

### Rules of usage
Bonus marks are optional, these are just some quick, experimental, hacky, buggy, (`<insert -ive qualifiers>`) 
boiler-plate code to speed up your work (if applicable). 

So, please use at your own risk, you have full freedom to modify or desecrate this code. We only ask that you:

1. Do not ask "how to" questions, the instructor+TAs team won't have the bandwidth to answer. 
2. If you have a better implementation in mind (which you should) - please (submit) *bring your own code* (**BYOC**)
3. The code doesn't work - BYOC
4. For all *subsequent questions* regarding bonus code, use:
```
q_idx = 4
while q_idx < sys.maxsize:
    print(f"Answer to Question_${q_idx} is: please BYOC.")
    q_idx += 1
```

### Usage of bonus files
Three files provided: `bonus_[a2_run, a2_training_and_testing].py, a2_transformer.py`.

Steps:

*Disclaimer: Tested on local GPU, but not on teach cluster GPUs* 
1. `bonus_a2_training_and_testing.py`: you only need to implement/fill in the method 
`def compute_average_bleu_over_dataset_using_transformer(...)`.

2. Run `bonus_a2_run.py` as you would run `a2_run.py` with the following params for **train**:

```
srun -p csc401 --gres gpu python3 bonus_a2_run.py train \
        model_w_transformer.pt \
        --training-dir "/path/to/data/Hansard/Training" \
        --with-transformer --batch-size 32 --device cuda \
        --viz-wandb $WB_USERNAME
```
Skip the last line to skip visualization.

2. Run `bonus_a2_run.py` as you would run `a2_run.py` with the following params for **test**:

```
srun -p csc401 --gres gpu python3 bonus_a2_run.py test \
        model_w_transformer.pt \
        --testing-dir "/path/to/data/Hansard/Testing" \
        --with-transformer \ 
        --batch-size 1 \
        --device cuda
```
Batch size > 1 may not be working - fix it in your submission.