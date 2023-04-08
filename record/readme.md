This folder is to save all records of experiments.

The folder structure is as follows: 
```
eg.
record/
    attack1/
        bd_test_dataset/
        bd_train_dataset/
        defense/ (all defense results following this attack)
            abl/
            ac/
            ...
        xxxx.log
        attack_df.csv
        attack_df_summary.csv
        attack_result.pt (attack result pt file used in following defenses)
        ...
    attack2/
    ...
```