from argparse import ArgumentParser
import pathlib


def main():
    parser = ArgumentParser(prog='imgsort')

    subparsers = parser.add_subparsers(required=True, dest="action")
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('input_folder', type=pathlib.Path, help='Path to folder containing the subfolders of images')
    parser_train.add_argument('model', type=pathlib.Path, help='Path to write the model data')
    parser_train.add_argument('-y', action='store_true', required=False, help='Overwrite model if path already exists. (Default: false)')
    parser_train.add_argument('-e', type=int, default=5, help='Number of epochs to train. Recommended between 2 and 10. (Default: 5)')
    parser_train.add_argument('-n', type=int, required=True, help='Number of classes to train. Must match the number of subfolders.') 


    parser_sort = subparsers.add_parser('sort')
    parser_sort.add_argument('unsorted_folder', type=pathlib.Path, help='Path to folder all unsorted images')
    parser_sort.add_argument('model', type=pathlib.Path, help='Path to read the model data')
    parser_sort.add_argument('-t', type=int, default=0, required=False, help='If all category prediction is lower than this threshold, do not move image (Default: 0)')
    parser_sort.add_argument('-y', action='store_true', required=False, help='Overwrite files with same name already present in subfolders')

    args = parser.parse_args()

    print(args)

    if args.action == 'train':
        print('Training...')
        from train import main_train
        main_train(args.input_folder, args.n, args.model, args.e)
    elif args.action == 'sort':
        print('Sorting...')
        from sort import main_sort
        main_sort(args.unsorted_folder, args.model, args.t, args.y)

if __name__ == '__main__':
    main()