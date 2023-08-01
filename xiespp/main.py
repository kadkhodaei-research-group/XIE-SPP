import argparse
import xiespp


def main_synthesizability():
    parser = argparse.ArgumentParser(description='Crystal synthesizability predictor')
    parser.add_argument('--test', action='store_true', help='Run on test samples')
    parser.add_argument('-f', '--file', type=str, help='CIF file to predict')
    parser.add_argument('--format', type=str, default='cif', help='Format of the file. '
                                                                  'Default is CIF. Use ASE format types')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size. Default is 8')
    parser.add_argument('-c', '--classifier', type=str, default='cnn',
                        help='Classifier to use (cnn [NEW], cnn-v1 or cae-mlp-v1). Default is cae-mlp')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble of classifiers. Default is 100. '
                                                                'This only works with the new cnn',
                        default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbosity')
    parser.add_argument('-m', '--multiprocessing', action='store_true', help='Use multiprocessing. Default is False')
    parser.add_argument('-w', '--workers', type=int, default=1, help='Number of workers. Default is 1')
    parser.add_argument('-o', '--output', type=str, help='Output file name.')

    args = parser.parse_args()
    files = get_files(args)

    if args.classifier not in ['cnn', 'cnn-v1', 'cae-mlp-v1']:
        print('Invalid classifier. Use -h or --help to see available classifiers.')
        exit(1)

    if args.format != 'cif' and args.classifier in ['cnn-v1', 'cae-mlp-v1']:
        print('Invalid format. Only CIF format is supported for this classifier.')
        exit(1)

    output = None
    if args.classifier in ['cnn-v1', 'cae-mlp-v1']:
        output = xiespp.synthesizability_1.synthesizability_predictor(
            files,
            classifier=args.classifier,
            verbose=args.verbose,
            use_multiprocessing=args.multiprocessing,
            workers=args.workers
        )
    if args.classifier == 'cnn':
        model = xiespp.synthesizability_2.SynthesizabilityPredictor(
            ensemble=args.ensemble, batch_size=args.batch_size
        )
        output = model.predict(files, verbose=args.verbose, input_format=args.format)

    output = list(output)
    print(output)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(str(output))
    exit(0)


def main_formation_energy():
    parser = argparse.ArgumentParser(description='Crystal formation energy predictor')
    parser.add_argument('--test', action='store_true', help='Run on test samples')
    parser.add_argument('-f', '--file', type=str, help='(CIF) file to predict')
    parser.add_argument('--format', type=str, default='cif', help='Format of the file. '
                                                                  'Default is CIF. Use ASE format types')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size. Default is 32')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble of classifiers. Default is 50.', default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbosity')
    parser.add_argument('-o', '--output', type=str, help='Output file name.')

    args = parser.parse_args()
    files = get_files(args)

    model = xiespp.formation_energy.FormationEnergyPredictor(
        ensemble=args.ensemble, batch_size=args.batch_size
    )
    output = model.predict(files, verbose=args.verbose, input_format=args.format)

    output = list(output)
    print(output)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(str(output))


def get_files(args):
    files = None
    if args.test:
        files = xiespp.get_test_samples()
        if len(files) == 0:
            print('No test samples found.')
            exit(1)
        print(f'Running on test samples: {files}')
    if args.file:
        files = [args.file]
    if files is None or len(files) == 0:
        print('No CIF file was provided. Use -f or --file to provide a CIF file.')
        exit(1)
    return files


if __name__ == '__main__':
    # main_synthesizability()
    # main_formation_energy()
    pass
