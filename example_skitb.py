import json
import argparse
from siamfc_pytorch.siamfc.siamfc import TrackerSiamFC
from toolkit.experiments import ExperimentSkiTB

parser = argparse.ArgumentParser()
parser.add_argument('--discipline', help='Disciplines to run, either alpine skiing (AL), ski jumping (JP), or freestyle skiing (FS)', type=str, default='AL')
parser.add_argument('--mode', help='Multi-camera (MC) or single-camera (SC)', type=str, default='MC')
parser.add_argument('--split', help='Split on which to execute the tracker, either train, val, or test, split_file must also be given', default='test')
parser.add_argument('--split_file', help='Path to split file containing trin, val, or test videos', default=None)
parser.add_argument('--visualize', help='Visualize the running performance of the tracker', action='store_true')
args = parser.parse_args()

tracker = TrackerSiamFC(return_conf=True)

root_dir = './SkiTB'

if args.split_file is not None:
    # getting split video IDs
    f = open(args.split_file)
    split_dict = json.load(f)
    test_sequences = split_dict[args.split]
    split_type = args.split_file.split('.')[0].split('_')[-1]
    note = f'{args.split}_{split_type}'
else:
    test_sequences = None
    note = None

exp = ExperimentSkiTB(root_dir, 
                      discipline=args.discipline, 
                      mode=args.mode, 
                      test_videos=test_sequences,
                      result_dir='./', 
                      report_dir='./')

# Run an experiment and save results
exp.run(tracker, visualize=False)

# Generate a report 
exp.report([tracker.name], fn_note=note)