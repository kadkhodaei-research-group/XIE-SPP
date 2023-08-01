import argparse


def parse_args(description, batch_size_default, classifier=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--test', action='store_true', help='Run on test samples')
    parser.add_argument('-f', '--file', type=str, help='(CIF) file to predict')
    parser.add_argument('--format', type=str, default='cif',
                        help='Format of the file. Default is CIF. Use ASE format types.')
    parser.add_argument('-b', '--batch_size', type=int, default=batch_size_default,
                        help=f'Batch size. Default is {batch_size_default}')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble of classifiers.', default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbosity')
    parser.add_argument('-o', '--output', type=str, help='Output file name.')
    parser.add_argument('--device', type=str, default='/device:CPU:0',
                        help='Device to use (cpu or gpu). Default is cpu. To use gpu, set to /device:GPU:0')

    if classifier:
        parser.add_argument('-c', '--classifier', type=str, default='cnn',
                            help='Classifier to use (cnn [NEW], cnn-v1 or cae-mlp-v1). Default is cae-mlp')
    return parser.parse_args()


def main_synthesizability():
    args = parse_args('Crystal synthesizability predictor', 8, classifier=True)
    files = get_files(args)

    if args.classifier not in ['cnn', 'cnn-v1', 'cae-mlp-v1']:
        print('Invalid classifier. Use -h or --help to see available classifiers.')
        exit(1)

    if args.format != 'cif' and args.classifier in ['cnn-v1', 'cae-mlp-v1']:
        print('Invalid format. Only CIF format is supported for this classifier.')
        exit(1)

    import xiespp
    output = None
    if args.classifier in ['cnn-v1', 'cae-mlp-v1']:
        output = xiespp.synthesizability_1.synthesizability_predictor(
            files,
            classifier=args.classifier[:-3],
            verbose=args.verbose,
            # use_multiprocessing=args.multiprocessing,
            # workers=args.workers
        )
    if args.classifier == 'cnn':
        model = xiespp.synthesizability_2.SynthesizabilityPredictor(
            ensemble=args.ensemble, batch_size=args.batch_size, device=args.device
        )
        output = model.predict(files, verbose=args.verbose, input_format=args.format)

    output = list(output)
    print(output)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(str(output))
    exit(0)


def main_formation_energy():
    args = parse_args('Crystal formation energy predictor', 32)
    files = get_files(args)

    import xiespp
    model = xiespp.formation_energy.FormationEnergyPredictor(
        ensemble=args.ensemble, batch_size=args.batch_size, device=args.device
    )
    output = model.predict(files, verbose=args.verbose, input_format=args.format)

    output = list(output)
    print(output)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(str(output))
    exit(0)


def get_files(args):
    import xiespp
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
