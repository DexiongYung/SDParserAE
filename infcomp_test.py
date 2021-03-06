import argparse
import pyro

from const import *
from infcomp import NameParser, TITLE, SUFFIX
from util.config import load_json

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='filepath to config json', type=str, default='config/UNNAMED_SESSION.json')
parser.add_argument('--name', help='name to parse', nargs='?', default='Dr. Frank Donald Wood', type=str)
parser.add_argument('--true_posterior', help='whether to sample from p(z|x) or q(z|x)', nargs='?', default=False,
                    type=bool)
parser.add_argument('--num_particles', help='# of particles to use for SIS', nargs='?', default=10, type=int)

args = parser.parse_args()

# config = load_json(args.config)
config = {"session_name": "UNNAMED_SESSION", "rnn_hidden_size": 256, "rnn_num_layers": 4, "char_error_rate": 0.01,
          "lr": 0.001, "num_steps": 50000, "num_particles": 25}

name_parser = NameParser(config['rnn_num_layers'], config['rnn_hidden_size'])
name_parser.load_checkpoint(filename=f"{config['session_name']}.pth.tar")

# print(name_parser.generate(3))

print(f"Parsing Name: {args.name}")
if args.true_posterior:
    csis = pyro.infer.CSIS(name_parser.model, name_parser.guide, pyro.optim.Adam({'lr': 0.001}),
                           num_inference_samples=args.num_particles)
    posterior = csis.run(observations={'output': name_parser.get_observes(args.name)})
    csis_samples = [posterior() for _ in range(10)]
    for sample in csis_samples:
        rv_names = sample.stochastic_nodes

        title = None
        firstname = ''
        middlename = ''
        middlename1 = ''
        middlename2 = ''
        lastname = ''
        suffix = None

        if "title" in rv_names:
            title = TITLE[sample.nodes[f"title"]['value'].item()]
        if "suffix" in rv_names:
            suffix = SUFFIX[sample.nodes[f"suffix"]['value'].item()]
        for i in range(MAX_OUTPUT_LEN):
            if f"first_name_{i}" in rv_names:
                firstname += MODEL_CHARS[sample.nodes[f"first_name_{i}"]['value'].item()]
            if f"middle_name_1_{i}" in rv_names:
                middlename1 += MODEL_CHARS[sample.nodes[f"middle_name_1_{i}"]['value'].item()]
            if f"middle_name_2_{i}" in rv_names:
                middlename2 += MODEL_CHARS[sample.nodes[f"middle_name_2_{i}"]['value'].item()]
            if f"last_name_{i}" in rv_names:
                lastname += MODEL_CHARS[sample.nodes[f"last_name_{i}"]['value'].item()]
        if f"middle_initial_1" in rv_names:
            middlename1 += MODEL_CHARS[sample.nodes["middle_initial_1"]['value'].item()]
        if f"middle_initial_2" in rv_names:
            middlename2 += MODEL_CHARS[sample.nodes["middle_initial_2"]['value'].item()]

        if middlename1 != '' and middlename2 != '':
            middlename = middlename1 + ' ' + middlename2
        if middlename1 != '':
            middlename = middlename1

        result = {'title': title, 'firstname': firstname, 'middlename': middlename, 'lastname': lastname,
                  'suffix': suffix}
else:
    for _ in range(MAX_STRING_LEN):
        result = name_parser.infer([args.name])
        print(f"Parse Result: {result[0][1]}")
