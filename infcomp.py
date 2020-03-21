import os
import pyro
import pyro.distributions as dist
import string
import torch

from NameGenerator import NameGenerator
from AutoEncoderHandler import AutoEncoderHandler
from TransformerHandler import TransformerHandler
from const import *
from model.FormatModel import NameFormatModel
from model.ChararacterClassifierModel import CharacterClassifierModel
from util.config import *
from util.infcomp_utilities import *


class NameParser():
    """
    Generates names using a separate LSTM for first, middle, last name and a neural net
    using ELBO to parameterize NN for format classification.

    input_size: Should be the number of letters to allow
    hidden_size: Size of the hidden dimension in LSTM
    num_layers: Number of hidden layers in LSTM
    hidden_sz: Hidden layer size for LSTM RNN
    peak_prob: The max expected probability
    """

    def __init__(self, num_layers: int = 2, hidden_sz: int = 64, peak_prob: float = 0.999, format_hidd_sz: int = 64):
        super().__init__()
        # Load up BART output vocab to correlate with name generative models.
        config = load_json('config/first.json')
        self.output_chars = config['output']
        self.num_output_chars = len(self.output_chars)
        # Model neural nets instantiation
        self.model_fn = NameGenerator('config/first.json', 'nn_model/first.path.tar')
        self.model_ln = NameGenerator('config/last.json', 'nn_model/last.path.tar')
        # Guide neural nets instantiation
        """
        Output for pretrained LSTMS doesn't have SOS so just use PAD for SOS
        """
        self.guide_fn = TransformerHandler(PRINTABLE, self.output_chars, self.output_chars.index(PAD))
        self.guide_mn = TransformerHandler(PRINTABLE, self.output_chars, self.output_chars.index(PAD))
        self.guide_ln = TransformerHandler(PRINTABLE, self.output_chars, self.output_chars.index(PAD))
        # Format classifier neural networks
        self.guide_format = CharacterClassifierModel(PRINTABLE, hidden_sz, FORMAT_CLASS)
        self.guide_noise = AutoEncoderHandler(PRINTABLE, hidden_sz, NOISE_CLASS)
        # Title / Suffix classifier neural networks
        self.title_rnn = NameFormatModel(input_sz=NUM_PRINTABLE, hidden_sz=hidden_sz, output_sz=len(TITLE))
        self.suffix_rnn = NameFormatModel(input_sz=NUM_PRINTABLE, hidden_sz=hidden_sz, output_sz=len(SUFFIX))
        self.aux_format_rnn = NameFormatModel(input_sz=NUM_PRINTABLE, hidden_sz=format_hidd_sz,
                                              output_sz=AUX_FORMAT_DIM)
        self.main_format_rnn = NameFormatModel(input_sz=NUM_PRINTABLE, hidden_sz=format_hidd_sz,
                                               output_sz=MAIN_FORMAT_DIM)
        self.middle_name_format_rnn = NameFormatModel(input_sz=NUM_PRINTABLE, hidden_sz=hidden_sz,
                                                      output_sz=MIDDLE_NAME_FORMAT_DIM)
        # Hyperparameters
        self.peak_prob = peak_prob
        self.num_layers = num_layers
        self.hidden_sz = hidden_sz

    def model(self, observations={"output": 0}):
        with torch.no_grad():
            # Sample format
            aux_format_id = int(pyro.sample("aux_format_id", dist.Categorical(AUX_FORMAT_PROBS)).item())
            main_format_id = int(pyro.sample("main_format_id", dist.Categorical(MAIN_FORMAT_PROBS)).item())

            # Sample title, first name, middle name, last name, and/or suffix
            title, suffix = None, None
            firstname, middlename, lastname = '', '', ''

            if has_title(aux_format_id):
                title = TITLE[int(pyro.sample("title", dist.Categorical(TITLE_PROBS)).item())]

            if has_suffix(aux_format_id):
                suffix = SUFFIX[int(pyro.sample("suffix", dist.Categorical(SUFFIX_PROBS)).item())]

            # first & last name generation
            firstname = sample_name(self.model_fn, FIRST_NAME_ADD)
            lastname = sample_name(self.model_ln, LAST_NAME_ADD)

            # Middle Name generation
            char_format = None
            if has_middle_name(main_format_id):
                middle_name_format_id = int(
                    pyro.sample("middle_name_format_id", dist.Categorical(MIDDLE_NAME_FORMAT_PROBS)).item())
                middlenames = []

                for i in range(num_middle_name(middle_name_format_id)):
                    if has_initial(middle_name_format_id):
                        initial_probs = [1 / 26] * 26
                        letter_idx = int(pyro.sample(f"{MIDDLE_NAME_ADD}_{i}_0",
                                                     dist.Categorical(torch.tensor(initial_probs).to(DEVICE))).item())
                        pyro.sample(f"{MIDDLE_NAME_ADD}_{i}_1",
                                    dist.Categorical(torch.tensor([0.] * 54 + [1.] + [0.]).to(DEVICE)))
                        middlename = string.ascii_uppercase[letter_idx]  # For capital names
                    else:
                        middlename = sample_name(self.model_fn, f"{MIDDLE_NAME_ADD}_{i}")
                    middlenames.append(middlename)

                middlename, char_format = middle_name_format(middlenames, middle_name_format_id)

            # Combine sampled outputs
            main_name, char_format = main_format(firstname, middlename, lastname, main_format_id, char_format)
            unpadded_full_name, char_format = aux_format(title, main_name, suffix, aux_format_id, char_format)
            # Pad full name to a fixed length
            full_name = [char for char in unpadded_full_name] + [PAD] * (MAX_STRING_LEN - len(unpadded_full_name))

            """
            Sample Noise Class Per Joined Character
            - Possible Classes: no change, add, replace, remove
            - 0.99 for correct class and 0.01/3 for the rest
            """
            noise_classes = []
            for i in range(MAX_STRING_LEN):
                noise_class = pyro.sample(f"char_noise_{i}",
                                          dist.Categorical(torch.tensor(CHAR_NOISE_PROBS).to(DEVICE))).item()
                noise_classes.append(noise_class)

            """
            Construct the likelihood probability with an appropriate noising scheme
            char_noise of 0, 1, 2, 3 entail no noise, add a character, replace a character, and remove a character
            char_noise is applied to each character index
            """
            observation_prob = []
            for i in range(MAX_STRING_LEN):
                def insert_peaked_probs(char):
                    char_prob = [(1 - self.peak_prob) / (NUM_PRINTABLE - 1)] * NUM_PRINTABLE
                    char_prob[PRINTABLE.index(char)] = self.peak_prob
                    observation_prob.append(char_prob)

                def insert_uniform_probs():
                    char_prob = [1 / NUM_PRINTABLE] * NUM_PRINTABLE
                    observation_prob.append(char_prob)

                character = full_name[i]
                char_noise = noise_classes[i]
                if char_noise == 0:
                    insert_peaked_probs(character)
                elif char_noise == 1:
                    insert_peaked_probs(character)
                    insert_uniform_probs()
                elif char_noise == 2:
                    insert_uniform_probs()
                else:
                    continue

            while len(observation_prob) < MAX_STRING_LEN:
                char_prob = [(1 - self.peak_prob) / (NUM_PRINTABLE - 1)] * NUM_PRINTABLE
                char_prob[PRINTABLE.index('PAD')] = self.peak_prob
                observation_prob.append(char_prob)
            """
            Sample Format Class Per Joined Character
            - Possible Classes: fname, mname, lname, separator, title, suffix, SOS, EOS, PAD
            - 0.999 for correct class and 0.001/7 for rest
            """
            char_format_probs = []
            for i in range(MAX_STRING_LEN):
                def insert_format_probs():
                    curr_format_probs = [(1 - self.peak_prob) / (len(FORMAT_CLASS) - 1)] * len(FORMAT_CLASS)
                    curr_format_value = char_format[i] if i < len(unpadded_full_name) else 'PAD'
                    curr_format_probs[FORMAT_CLASS.index(curr_format_value)] = self.peak_prob
                    char_format_probs.append(curr_format_probs)

                char_noise = noise_classes[i]
                if char_noise == 0 or char_noise == 2:
                    # No noise or replacement of character: insert format for current index
                    insert_format_probs()
                elif char_noise == 1:
                    # Insertion of character: insert format for current and next index
                    insert_format_probs()
                    insert_format_probs()
                else:
                    # Removal of character: do not insert format
                    continue

            while len(char_format_probs) < MAX_STRING_LEN:
                curr_format_probs = [(1 - self.peak_prob) / (len(FORMAT_CLASS) - 1)] * len(FORMAT_CLASS)
                curr_format_probs[FORMAT_CLASS.index('PAD')] = self.peak_prob
                char_format_probs.append(curr_format_probs)
            for i in range(MAX_STRING_LEN):
                curr_format = pyro.sample(f"char_format_{i}",
                                          dist.Categorical(torch.tensor(char_format_probs[i]).to(DEVICE)))

            pyro.sample("output", dist.Categorical(torch.tensor(observation_prob[:MAX_STRING_LEN]).to(DEVICE)),
                        obs=observations["output"])

        parse = {'firstname': firstname, 'middlename': middlename, 'lastname': lastname, 'title': title,
                 'suffix': suffix}
        # print(f"MODEL Fullname: {full_name}")
        # print(f"MODEL Parse: {parse}")
        return full_name, parse

    def guide(self, observations=None):
        X = observations['output']
        X_len = len(X)

        # Infer formats and parse
        pyro.module("format_forward_lstm", self.guide_format.forward_lstm)
        pyro.module("format_backward_lstm", self.guide_format.backward_lstm)
        char_class_samples = self.guide_format.forward(X, "char_format")

        # Infer noise class
        pyro.module("noise_encoder_lstm", self.guide_noise.encoder.lstm)
        pyro.module("noise_decoder_lstm", self.guide_noise.decoder.lstm)
        pyro.module("noise_decoder_fc1", self.guide_noise.decoder.fc1)
        noise_samples, _ = self.guide_noise.encode_decode(X, "char_noise", break_on_eos = False)

        title, first, middles, last, suffix = parse_name(X, char_class_samples)
        # print(f"GUIDE Parse: {parse_name(X, char_class_samples)}")

        pyro.module("aux_format_rnn", self.aux_format_rnn.lstm)
        pyro.module("aux_format_fc1", self.aux_format_rnn.fc1)
        classify_using_transformer(self.aux_format_rnn, X, "aux_format_id")

        pyro.module("main_format_rnn", self.main_format_rnn.lstm)
        pyro.module("main_format_fc1", self.main_format_rnn.fc1)
        classify_using_transformer(self.main_format_rnn, X, "main_format_id")

        if len(title) > 0:
            pyro.module("title_rnn", self.title_rnn.lstm)
            pyro.module("title_fc1", self.title_rnn.fc1)
            title = classify_title_or_suffix(self.title_rnn, title, TITLE, "title")

        if len(suffix) > 0:
            pyro.module("suffix_rnn", self.suffix_rnn.lstm)
            pyro.module("suffix_fc1", self.suffix_rnn.fc1)
            suffix = classify_title_or_suffix(self.suffix_rnn, suffix, SUFFIX, "suffix")

        if len(first) > 0:
            pyro.module("fn_transformer", self.guide_fn.transformer.transformer)
            pyro.module("fn_fc1", self.guide_fn.transformer.fc1)
            input = name_to_idx_tensor(first[0], self.guide_fn.input)
            samples, _ = sample_from_transformer(self.guide_fn, input, FIRST_NAME_ADD, MAX_OUTPUT_LEN)
            first = ''.join(self.output_chars[s] for s in samples)

        if len(middles) > 0:
            pyro.module("middle_name_format_rnn", self.middle_name_format_rnn.lstm)
            pyro.module("middle_name_format_fc1", self.middle_name_format_rnn.fc1)
            classify_using_transformer(self.middle_name_format_rnn, X, "middle_name_format_id")

            pyro.module("mn_transformer", self.guide_mn.transformer.transformer)
            pyro.module("mn_fc1", self.guide_mn.transformer.fc1)
            _, middles = denoise_names(self.guide_mn, middles, MIDDLE_NAME_ADD, self.output_chars)

        if len(last) > 0:
            pyro.module("ln_transformer", self.guide_ln.transformer.transformer)
            pyro.module("ln_fc1", self.guide_ln.transformer.fc1)
            input = name_to_idx_tensor(last[0], self.guide_ln.input)
            samples, _ = sample_from_transformer(self.guide_ln, input, LAST_NAME_ADD, MAX_OUTPUT_LEN)
            last = ''.join(self.output_chars[s] for s in samples)

        # TODO!!! Have to add full name reconstruction

        return {'firstname': first, 'middlename': middles, 'lastname': last, 'title': title, 'suffix': suffix}

    def infer(self, names: list):
        # Infer using q(z|x)
        results = []
        for name in names:
            encoded_name = self.get_observes(name)
            result = self.guide(observations={'output': encoded_name})
            results.append(result)
        return results

    def generate(self, num_samples: int = 1):
        # Generate samples from p(x,z)
        results = []
        for _ in range(num_samples):
            results.append(self.model()[0])
        return results

    def get_observes(self, name_string: str):
        if len(name_string) > MAX_STRING_LEN: raise Exception(f"Name string length cannot exceed {MAX_STRING_LEN}.")
        name_as_list = [c for c in name_string]
        return name_to_idx_tensor(name_as_list, PRINTABLE)

    def load_checkpoint(self, folder="nn_model", filename="checkpoint.pth.tar"):
        name_fp = os.path.join(folder, "name_" + filename)
        format_fp = os.path.join(folder, "format_" + filename)
        noise_fp = os.path.join(folder, "noise_" + filename)
        title_suffix_fp = os.path.join(folder, "title_suffix" + filename)
        if not os.path.exists(name_fp) or not os.path.exists(format_fp) or not os.path.exists(noise_fp):
            raise Exception(f"No model in path {folder}")
        name_content = torch.load(name_fp, map_location=DEVICE)
        format_content = torch.load(format_fp, map_location=DEVICE)
        noise_content = torch.load(noise_fp, map_location=DEVICE)
        title_suffix_content = torch.load(title_suffix_fp, map_location=DEVICE)
        self.guide_fn.transformer.load_state_dict(name_content['guide_fn'])
        self.guide_mn.transformer.load_state_dict(name_content['guide_mn'])
        self.guide_ln.transformer.load_state_dict(name_content['guide_ln'])
        self.title_rnn.load_state_dict(title_suffix_content['title_rnn'])
        self.suffix_rnn.load_state_dict(title_suffix_content['suffix_rnn'])
        self.guide_format.load_state_dict(format_content['guide_format'])
        self.aux_format_rnn.load_state_dict(format_content['aux_format_rnn'])
        self.main_format_rnn.load_state_dict(format_content['main_format_rnn'])
        self.middle_name_format_rnn.load_state_dict(format_content['middle_name_format_rnn'])
        self.guide_noise.encoder.load_state_dict(noise_content['guide_noise_encoder'])
        self.guide_noise.decoder.load_state_dict(noise_content['guide_noise_decoder'])

    def save_checkpoint(self, folder="nn_model", filename="checkpoint.pth.tar"):
        name_fp = os.path.join(folder, "name_" + filename)
        title_suffix_fp = os.path.join(folder, "title_suffix" + filename)
        format_fp = os.path.join(folder, "format_" + filename)
        noise_fp = os.path.join(folder, "noise_" + filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        name_content = {
            'guide_fn': self.guide_fn.transformer.state_dict(),
            'guide_mn': self.guide_mn.transformer.state_dict(),
            'guide_ln': self.guide_ln.transformer.state_dict(),
        }
        title_suffix_content = {
            'title_rnn': self.title_rnn.state_dict(),
            'suffix_rnn': self.suffix_rnn.state_dict(),
        }
        format_content = {
            'guide_format': self.guide_format.state_dict(),
            'aux_format_rnn': self.aux_format_rnn.state_dict(),
            'main_format_rnn': self.main_format_rnn.state_dict(),
            'middle_name_format_rnn': self.middle_name_format_rnn.state_dict(),
        }
        noise_content = {
            'guide_noise_encoder': self.guide_noise.encoder.state_dict(),
            'guide_noise_decoder': self.guide_noise.decoder.state_dict()
        }
        torch.save(name_content, name_fp)
        torch.save(format_content, format_fp)
        torch.save(noise_content, noise_fp)
        torch.save(title_suffix_content, title_suffix_fp)
