# idea: generate the save folder name with some random generate string, in order to avoid potential name comflicts in repeat experiments

import sys, logging
sys.path.append('../../')

import time,random, string,  os, sys
from datetime import datetime
from typing import *



def generate_save_folder(
        run_info : Optional[str] = '',
        given_load_file_path : Optional[str] = None,
        recover: Optional[bool] = False,
        all_record_folder_path : str = '../record',
) -> str:

    # idea:  This function helps to generate save path for experiment.
    # if you do not want to set the name, this function will set it for experiment.
    # Note that by using the randomly generate str, replication of experiment may not overwrite the folder of each other

    def inside_generate(
            all_record_folder_path: str,
            startTimeStr: str,
            run_info: str,
    ) -> str:
        random_code = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(4)])
        save_path = all_record_folder_path + '/' + startTimeStr + '_' + os.path.basename(sys.argv[0]).split('.')[0] + '_' + run_info + '_' + random_code
        return save_path

    startTimeStr = str(datetime.now().strftime('%Y%m%d_%H%M%S'))

    if given_load_file_path is None:
        # default None
        # no save_folder and no load_path, so can only random generate one

        save_path = inside_generate(
            all_record_folder_path,
            startTimeStr,
            run_info,
        )

        while os.path.isdir(save_path):

            save_path = inside_generate(
                all_record_folder_path,
                startTimeStr,
                run_info,
            )

        # os.mkdir(save_path)

    elif given_load_file_path is not None and recover is True:

        given_load_file_path = given_load_file_path.rstrip('/')

        if os.path.isfile(os.path.abspath(given_load_file_path)):
            load_folder_name = os.path.dirname(given_load_file_path)
        else:
            load_folder_name = given_load_file_path

        print(load_folder_name)

    else:
        '''
        os.path.basename(os.path.dirname('dir/sub_dir/other_sub_dir/file_name.txt'))
        Out[8]: 'other_sub_dir'
        os.path.basename(os.path.dirname('dir/sub_dir/other_sub_dir/'))
        Out[9]: 'other_sub_dir'
        '''
        given_load_file_path = given_load_file_path.rstrip('/')

        # isfile need abs path otherwise seems to be false anyway.
        if os.path.isfile(os.path.abspath(given_load_file_path)):
            load_folder_name = os.path.basename(os.path.dirname(given_load_file_path))
        else:
            load_folder_name = os.path.basename(given_load_file_path)

        # save_path = inside_generate(
        #     all_record_folder_path,
        #     startTimeStr,
        #     run_info,
        # ) + '_baseOn_' + load_folder_name

        generate_base = inside_generate(
            all_record_folder_path,
            startTimeStr,
            run_info,
        )

        save_path = generate_base.split('_baseOn_')[0] + '_baseOn_' + load_folder_name
        # if not contains "_baseOn_", then no split, 0 position is the original
        # else, then keep only the part before the first _baseOn_, rest of str drop

        while os.path.isdir(save_path):

            # save_path = inside_generate(
            #     all_record_folder_path,
            #     startTimeStr,
            #     run_info,
            # ) + '_baseOn_' + load_folder_name

            generate_base = inside_generate(
                all_record_folder_path,
                startTimeStr,
                run_info,
            )

            save_path = generate_base.split('_baseOn_')[0] + '_baseOn_' + load_folder_name

    os.mkdir(save_path)

    return save_path
