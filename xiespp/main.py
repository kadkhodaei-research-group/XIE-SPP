"""
XIE-SPP command line interface.

    xiespp synthesizability  [FILES ...] [--model v2|v1-cnn|v1-cae-mlp] [-o out.csv]
    xiespp formation-energy  [FILES ...] [-o out.csv]

Run `xiespp <command> -h` for all options.
For creating CVR 3D images see example_cvr_images.ipynb.
"""
import argparse
import sys
from pathlib import Path

SYN_MODELS = ['v2', 'v1-cnn', 'v1-cae-mlp']
# Names used by the previous versions of the CLI (-c/--classifier)
LEGACY_SYN_MODELS = {'cnn': 'v2', 'cnn-v1': 'v1-cnn', 'cae-mlp-v1': 'v1-cae-mlp'}
TEST_SAMPLES = ['GaN', 'CSi', 'MoS2']


def add_input_args(parser):
    parser.add_argument('files', nargs='*', help='Structure files (CIF by default)')
    parser.add_argument('-f', '--file', action='append', default=[], help=argparse.SUPPRESS)  # Legacy
    parser.add_argument('--test-samples', nargs='?', const='GaN', choices=TEST_SAMPLES, metavar='NAME',
                        help=f'Run on the test samples shipped with the package: {", ".join(TEST_SAMPLES)} '
                             f'(default: GaN)')
    parser.add_argument('--test', dest='test_samples', action='store_const', const='GaN',
                        help=argparse.SUPPRESS)  # Legacy
    parser.add_argument('--format', type=str, default=None,
                        help='File format (ASE format names, e.g. vasp). Default: detected by ASE / CIF')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print progress')


def add_model_args(parser, batch_size_default, ensemble_default):
    parser.add_argument('-o', '--output', type=str, help='Save predictions to this CSV file')
    parser.add_argument('-b', '--batch-size', type=int, default=batch_size_default,
                        help=f'Batch size (default: {batch_size_default})')
    parser.add_argument('-e', '--ensemble', type=int, default=ensemble_default,
                        help=f'Number of rotational ensembles (default: {ensemble_default})')
    parser.add_argument('--device', type=str, default='/device:CPU:0',
                        help='TensorFlow device (default: /device:CPU:0; e.g. /device:GPU:0)')


def build_parser():
    parser = argparse.ArgumentParser(
        prog='xiespp',
        description='XIE-SPP: Crystal Image Encoder for Synthesis & Property Prediction',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('synthesizability', help='Predict synthesizability likelihood')
    add_input_args(p)
    add_model_args(p, batch_size_default=8, ensemble_default=100)
    p.add_argument('-m', '--model', '-c', '--classifier', dest='model', default='v2',
                   choices=SYN_MODELS + list(LEGACY_SYN_MODELS),
                   help='v2 (default, recommended), v1-cnn or v1-cae-mlp. v1 models only accept CIF files '
                        'and ignore --ensemble/--batch-size/--device')
    p.set_defaults(func=run_synthesizability)

    p = sub.add_parser('formation-energy', help='Predict formation energy (eV/atom)')
    add_input_args(p)
    add_model_args(p, batch_size_default=32, ensemble_default=50)
    p.set_defaults(func=run_formation_energy)
    return parser


def get_files(args):
    import xiespp
    files = list(args.files) + list(args.file)
    if args.test_samples:
        files += sorted(xiespp.get_test_samples(args.test_samples))
    if not files:
        sys.exit('No input files. Pass structure files or use --test-samples.')
    missing = [f for f in files if not Path(f).is_file()]
    if missing:
        sys.exit(f'File(s) not found: {missing}')
    return files


def write_predictions(files, predictions, output, column):
    import pandas as pd
    df = pd.DataFrame({'file': files, column: list(predictions)})
    print(df.to_csv(index=False), end='')
    if output:
        df.to_csv(output, index=False)
        print(f'Saved: {output}', file=sys.stderr)


def run_synthesizability(args):
    files = get_files(args)
    model_name = LEGACY_SYN_MODELS.get(args.model, args.model)
    import xiespp
    if model_name == 'v2':
        model = xiespp.synthesizability_2.SynthesizabilityPredictor(
            ensemble=args.ensemble, batch_size=args.batch_size, device=args.device)
        yp = model.predict(files, verbose=int(args.verbose), input_format=args.format)
    else:
        if args.format not in (None, 'cif'):
            sys.exit('v1 models only accept CIF files.')
        yp = xiespp.synthesizability_1.synthesizability_predictor(
            files, classifier=model_name[len('v1-'):], verbose=int(args.verbose))
    write_predictions(files, yp, args.output, 'synthesizability')


def run_formation_energy(args):
    files = get_files(args)
    import xiespp
    model = xiespp.formation_energy.FormationEnergyPredictor(
        ensemble=args.ensemble, batch_size=args.batch_size, device=args.device)
    yp = model.predict(files, verbose=int(args.verbose), input_format=args.format)
    write_predictions(files, yp, args.output, 'formation_energy')


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


# Legacy entry points (kept for backwards compatibility)
def main_synthesizability():
    main(['synthesizability'] + sys.argv[1:])


def main_formation_energy():
    main(['formation-energy'] + sys.argv[1:])


if __name__ == '__main__':
    main()
