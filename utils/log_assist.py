import subprocess

def get_git_info():

    info = {}

    info['status'] = (subprocess.check_output(['git', 'status']).decode('ascii').strip())

    info['last 3 log'] = (subprocess.check_output(['git', 'log', '-n', '3']).decode('ascii').strip())

    if 'modified:' in subprocess.check_output(['git', 'status']).decode('ascii').strip():
        info['git hash'] = None
    else:
        info['git hash'] = (f"--git_hash {subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()}")

    return info