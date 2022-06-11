#!/bin/bash

# this script run all attack script for minimal epochs and save the output.
# Also, run the load module to load the attack_result.
# All log for this test can be found in debug folder start with start-time of the WHOLE test file.

NOW=`date '+%F_%H:%M:%S'` # get the startTime
echo "\n test start at $NOW"

cd ..

mkdir -p ./sh/debug/${NOW}

git status > ./sh/debug/${NOW}/git_status_at_${NOW}.log 2>&1

git log > ./sh/debug/${NOW}/git_log_at_${NOW}.log 2>&1

#cat ./utils/save_load_attack.py > ./sh/debug/${NOW}/save_load_attack_at_${NOW}.log

rm -rf ./record/one_epochs_debug_badnet_attack
python ./attack/badnet_attack.py --save_folder_name one_epochs_debug_badnet_attack --epochs 1 >> ./sh/debug/${NOW}/one_epochs_debug_badnet_attack.log 2>&1
#cat ./record/one_epochs_debug_badnet_attack/*.log >> ./sh/debug/${NOW}/one_epochs_debug_badnet_attack.log 2>&1
python ./sh/load_for_test.py --attack_result_file_path ../record/one_epochs_debug_badnet_attack/attack_result.pt >> ./sh/debug/${NOW}/one_epochs_debug_badnet_attack.log 2>&1

rm -rf ./record/one_epochs_debug_blended_attack
python ./attack/blended_attack.py --save_folder_name one_epochs_debug_blended_attack --epochs 1 >> ./sh/debug/${NOW}/one_epochs_debug_blended_attack.log 2>&1
#cat ./record/one_epochs_debug_blended_attack/*.log >> ./sh/debug/${NOW}/one_epochs_debug_blended_attack.log 2>&1
python ./sh/load_for_test.py --attack_result_file_path ../record/one_epochs_debug_blended_attack/attack_result.pt >> ./sh/debug/${NOW}/one_epochs_debug_blended_attack.log 2>&1

rm -rf ./record/inputaware_attack_two_epochs_debug #this is TWO epochs!!!
python ./attack/inputaware_attack.py --save_folder_name inputaware_attack_two_epochs_debug --epochs 2 --clean_train_epochs 1 >> ./sh/debug/${NOW}/inputaware_attack_two_epochs_debug.log 2>&1
#cat ./record/inputaware_attack_two_epochs_debug/*.log >> ./sh/debug/${NOW}/inputaware_attack_two_epochs_debug.log 2>&1
python ./sh/load_for_test.py --attack_result_file_path ../record/inputaware_attack_two_epochs_debug/attack_result.pt >> ./sh/debug/${NOW}/inputaware_attack_two_epochs_debug.log 2>&1

rm -rf ./record/one_epochs_debug_lc_attack
python ./attack/lc_attack.py --save_folder_name one_epochs_debug_lc_attack --epochs 1 >> ./sh/debug/${NOW}/one_epochs_debug_lc_attack.log 2>&1
#cat ./record/one_epochs_debug_lc_attack/*.log >> ./sh/debug/${NOW}/one_epochs_debug_lc_attack.log 2>&1
python ./sh/load_for_test.py --attack_result_file_path ../record/one_epochs_debug_lc_attack/attack_result.pt >> ./sh/debug/${NOW}/one_epochs_debug_lc_attack.log 2>&1

rm -rf ./record/one_epochs_debug_lf_attack
python ./attack/lf_attack.py --save_folder_name one_epochs_debug_lf_attack --epochs 1 >> ./sh/debug/${NOW}/one_epochs_debug_lf_attack.log 2>&1
#cat ./record/one_epochs_debug_lf_attack/*.log >> ./sh/debug/${NOW}/one_epochs_debug_lf_attack.log 2>&1
python ./sh/load_for_test.py --attack_result_file_path ../record/one_epochs_debug_lf_attack/attack_result.pt >> ./sh/debug/${NOW}/one_epochs_debug_lf_attack.log 2>&1

rm -rf ./record/one_epochs_debug_sig_attack
python ./attack/sig_attack.py --save_folder_name one_epochs_debug_sig_attack --epochs 1 >> ./sh/debug/${NOW}/one_epochs_debug_sig_attack.log 2>&1
#cat ./record/one_epochs_debug_sig_attack/*.log >> ./sh/debug/${NOW}/one_epochs_debug_sig_attack.log 2>&1
python ./sh/load_for_test.py --attack_result_file_path ../record/one_epochs_debug_sig_attack/attack_result.pt >> ./sh/debug/${NOW}/one_epochs_debug_sig_attack.log 2>&1

rm -rf ./record/one_epochs_debug_ssba_attack
python ./attack/ssba_attack.py --save_folder_name one_epochs_debug_ssba_attack --epochs 1 >> ./sh/debug/${NOW}/one_epochs_debug_ssba_attack.log 2>&1
#cat ./record/one_epochs_debug_ssba_attack/*.log >> ./sh/debug/${NOW}/one_epochs_debug_ssba_attack.log 2>&1
python ./sh/load_for_test.py --attack_result_file_path ../record/one_epochs_debug_ssba_attack/attack_result.pt >> ./sh/debug/${NOW}/one_epochs_debug_ssba_attack.log 2>&1

rm -rf ./record/one_epochs_debug_wanet_attack
python ./attack/wanet_attack.py --save_folder_name one_epochs_debug_wanet_attack --epochs 1 >> ./sh/debug/${NOW}/one_epochs_debug_wanet_attack.log 2>&1
#cat ./record/one_epochs_debug_wanet_attack/*.log >> ./sh/debug/${NOW}/one_epochs_debug_wanet_attack.log 2>&1
python ./sh/load_for_test.py --attack_result_file_path ../record/one_epochs_debug_wanet_attack/attack_result.pt >> ./sh/debug/${NOW}/one_epochs_debug_wanet_attack.log 2>&1

echo "test end, go to debug to see the result"

NOW=`date '+%F_%H:%M:%S'` # get the endTime
echo "\n test end at $NOW"


