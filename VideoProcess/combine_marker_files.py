import os
import argparse


def folder_contents(base_dir: str, recursive: bool = False, files_filter_lambda=None, dirs_filter_lambda=None):
    """
    Returns directories and files that lie under @base_dir
    
    - Provide lambda for file and directory filtering (e.g.: lambda x: x.endswith('.json'))
    - Optionally search recursively
    """
    contents = os.listdir(base_dir)
    if len(contents) > 0:
        dirs, files = contents, contents

        if dirs_filter_lambda:
            dirs = list(filter(dirs_filter_lambda, dirs))
        dirs = [os.path.join(base_dir, d) for d in dirs]
        dirs = list(filter(lambda fn: os.path.isdir(fn), dirs))

        if files_filter_lambda:
            files = list(filter(files_filter_lambda, files))
        files = [os.path.join(base_dir, f) for f in files]
        files = list(filter(lambda fn: os.path.isfile(fn), files))
        sorted(files, key=lambda fn: os.path.basename(fn))

        if recursive:
            for dir in dirs:
                d, f = folder_contents(dir, recursive, files_filter_lambda, dirs_filter_lambda)
                dirs += d
                files += f
        return dirs, files
    return [], []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='', type=str, required=True)
    parser.add_argument('--recursive', default=True, type=bool)
    args = parser.parse_args()

    args.folder = os.path.join('', args.folder)

    if os.path.isdir(args.folder):
        dirs, files = folder_contents(os.path.abspath(args.folder), True, lambda x: x.endswith('.json'))
        for fn in files:
            with open(fn) as f:
                import json
                ob = json.load(f)
                for frame in ob:
                    print(len(frame['markers']))


if __name__ == '__main__':
    main()
