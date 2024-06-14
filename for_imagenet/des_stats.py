'''
To inspect the folder structure for data generation.
Make sure the poison ratio is accurate.
'''

import os, sys

def stats(given_fodler_path):
    info_list = []
    for subfodler_name in os.listdir(given_fodler_path):
        if not os.path.isdir(f"{given_fodler_path}/{subfodler_name}"):
            continue
        info = f"{given_fodler_path}/{subfodler_name} : " +\
               str(len(os.listdir(f"{given_fodler_path}/{subfodler_name}"))) +\
                "\n"
        info_list.append(
            info
        )
        print(info)

    with open(f"{given_fodler_path}/stats.txt", "w") as f:
        f.writelines(info_list)



