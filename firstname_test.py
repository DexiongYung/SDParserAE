import argparse

from firstname import NameParser
from util.config import load_json

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='filepath to config json', type=str)
parser.add_argument('--name', help='name to parse', nargs='?', default='jason', type=str)
parser.add_argument('--num_samples', help='# samples to generate from the model', nargs='?', default=0, type=int)
args = parser.parse_args()

config = load_json(args.config)

model = NameParser(2, config['hidden_size'], config['peak_probs'])
model.load_checkpoint(filename=f"{config['session_name']}.pth.tar")
for _ in range(args.num_samples):
    print(f"Sample Name: {model.generate()}")

print(f"Parsing: {args.name}")
for _ in range(1):
    print(model.infer([args.name]))
