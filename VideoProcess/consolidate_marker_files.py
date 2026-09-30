import os
import argparse

from .utils import folder_contents


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
