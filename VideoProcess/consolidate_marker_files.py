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
    parser.add_argument('--input', default='', type=str, required=True)
    parser.add_argument('--output', default='', type=str, required=True)
    parser.add_argument('--refined', default=True, type=bool, help='Whether to use refined markers files. Look for "_refined_markers.json" instead of "_markers.json"')
    parser.add_argument('--recursive', default=True, type=bool)
    args = parser.parse_args()

    args.input = os.path.join('', args.input)
    args.output = os.path.join('', args.output)

    ending = "_markers"

    if args.refined:
        ending = "_refined_markers"
        
    if os.path.isdir(args.input):
        _, files = folder_contents(os.path.abspath(args.input), True, lambda x: x.endswith(f'{ending}.json'))
        all = {}
        for fn in files:
            with open(fn) as f:
                import json
                ob = json.load(f)
                field = os.path.basename(fn)
                field = field.split(f'{ending}')[0]
                all[field] = ob
                total_markers = [len(o['markers']) for o in ob]
            print(f'> Added markers from "{field}"\n\tFrames: {len(total_markers)}\n\tTotal markers: {sum(total_markers)}\n\tAverage: {int(sum(total_markers) / len(total_markers))}')

        if len(all) > 0:
            json_string = json.dumps(all, separators=(',', ":"))  # Compact JSON structure
            open(args.output, "w+", 1).write(json_string)
        print('---')
        print(f'> Consolidated {len(all)} marker files to "{args.output}"')
        print('---')


if __name__ == '__main__':
    main()
