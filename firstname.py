import os
import pyro
import pyro.distributions as dist
import torch

from const import *
from model.seq2seq import Encoder, Decoder
from util.convert import strings_to_tensor, strings_to_probs, pad_string, letter_to_index, printable_to_index

FIRST_NAME_ADD = "first_name"
LAST_NAME_ADD = "last_name"
MIDDLE_NAME_ADD = "middle_name"


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

    def __init__(self, num_layers: int = 2, hidden_sz: int = 64, peak_prob: float = 0.9):
        super().__init__()
        # Model neural nets instantiation
        self.model_fn_lstm = Decoder(LETTERS_COUNT, hidden_sz, LETTERS_COUNT, num_layers=num_layers)
        # Guide neural nets instantiation
        self.guide_fn_lstm = Decoder(LETTERS_COUNT, hidden_sz, LETTERS_COUNT, num_layers=num_layers)
        # Instantiate encoder
        self.encoder_lstm = Encoder(PRINTABLES_COUNT, hidden_sz, num_layers=num_layers)

        # Hyperparameters
        self.peak_prob = peak_prob
        self.num_layers = num_layers
        self.hidden_sz = hidden_sz

    def model(self, X_u: list, X_s: list, Z_s: dict, observations=None):
        """
        Model for generating names representing p(x,z)
        x: Training data (name string)
        z: Optionally supervised latent values (dictionary of name/format values)
        """
        pyro.module("model_fn_lstm", self.model_fn_lstm)

        formatted_X_u = strings_to_tensor(X_u, MAX_NAME_LENGTH, printable_to_index)
        formatted_X_s = strings_to_tensor(X_s, MAX_NAME_LENGTH, printable_to_index)

        with pyro.plate("sup_batch", len(X_s)):
            _, first_names = self.generate_name_supervised(self.model_fn_lstm, FIRST_NAME_ADD, len(X_s),
                                                           observed=Z_s[FIRST_NAME_ADD])
            full_names = list(map(lambda name: pad_string(name, MAX_NAME_LENGTH), first_names))
            probs = strings_to_probs(full_names, MAX_NAME_LENGTH, printable_to_index, true_index_prob=self.peak_prob)
            pyro.sample("sup_output", dist.OneHotCategorical(probs.transpose(0, 1)).to_event(1),
                        obs=formatted_X_s.transpose(0, 1))

        with pyro.plate("unsup_batch", len(X_u)):
            _, first_names = self.generate_name(self.model_fn_lstm, FIRST_NAME_ADD, len(X_u))
            full_names = list(map(lambda name: pad_string(name, MAX_NAME_LENGTH), first_names))
            probs = strings_to_probs(full_names, MAX_NAME_LENGTH, printable_to_index, true_index_prob=self.peak_prob)
            pyro.sample("unsup_output", dist.OneHotCategorical(probs.transpose(0, 1)).to_event(1),
                        obs=formatted_X_u.transpose(0, 1))

        return full_names

    def guide(self, X_u: list, X_s: list, Z_s: dict, observations=None):
        """
        Guide for approximation of the posterior q(z|x)
        x: Training data (name string)
        z: Optionally supervised latent values (dictionary of name/format values)
        """

        pyro.module("guide_fn_lstm", self.guide_fn_lstm)
        pyro.module("encoder_lstm", self.encoder_lstm)

        if observations is None:
            formatted_X_u = strings_to_tensor(X_u, MAX_NAME_LENGTH, printable_to_index)
        else:
            formatted_X_u = observations['unsup_output'].transpose(0, 1)

        hidd_cell_states = self.encoder_lstm.init_hidden(batch_size=len(X_u))
        for i in range(formatted_X_u.shape[0]):
            _, hidd_cell_states = self.encoder_lstm.forward(formatted_X_u[i].unsqueeze(0), hidd_cell_states)

        with pyro.plate("unsup_batch", len(X_u)):
            _, first_names = self.generate_name(self.guide_fn_lstm, FIRST_NAME_ADD, len(X_u),
                                                hidd_cell_states=hidd_cell_states, sample=False)

        return first_names

    def infer(self, X_u: list):
        formatted_X_u = strings_to_tensor(X_u, MAX_NAME_LENGTH, printable_to_index)
        hidd_cell_states = self.encoder_lstm.init_hidden(batch_size=len(X_u))
        for i in range(formatted_X_u.shape[0]):
            _, hidd_cell_states = self.encoder_lstm.forward(formatted_X_u[i].unsqueeze(0), hidd_cell_states)
        _, first_names = self.generate_name(self.guide_fn_lstm, FIRST_NAME_ADD, len(X_u),
                                            hidd_cell_states=hidd_cell_states, sample=False)
        return first_names

    def generate(self):
        _, name = self.generate_name(self.model_fn_lstm, FIRST_NAME_ADD, 1)
        return name

    def generate_name(self, lstm: Decoder, address: str, batch_size: int, hidd_cell_states: tuple = None,
                      sample: bool = True):
        """
        lstm: Decoder associated with name being generated
        address: The address to correlate pyro distribution with latent variables
        hidd_cell_states: Previous LSTM hidden state or empty hidden state
        max_name_length: The max name length allowed
        """
        # If no hidden state is provided, initialize it with all 0s
        if hidd_cell_states == None:
            hidd_cell_states = lstm.init_hidden(batch_size=batch_size)

        input_tensor = strings_to_tensor([SOS] * batch_size, 1, letter_to_index)
        names = [''] * batch_size

        for index in range(MAX_NAME_LENGTH):
            char_dist, hidd_cell_states = lstm.forward(input_tensor, hidd_cell_states)

            if sample:
                # Next LSTM input is the sampled character
                input_tensor = pyro.sample(f"unsup_{address}_{index}", dist.OneHotCategorical(char_dist))
                chars_at_indexes = list(
                    map(lambda index: MODEL_CHARS[int(index.item())], torch.argmax(input_tensor, dim=2).squeeze(0)))
            else:
                # Next LSTM input is the character with the highest probability of occurring
                pyro.sample(f"unsup_{address}_{index}", dist.OneHotCategorical(char_dist))
                chars_at_indexes = list(
                    map(lambda index: MODEL_CHARS[int(index.item())], torch.argmax(char_dist, dim=2).squeeze(0)))
                input_tensor = strings_to_tensor(chars_at_indexes, 1, letter_to_index)

            # Add sampled characters to names
            for i, char in enumerate(chars_at_indexes):
                names[i] += char

        # Discard everything after EOS character
        # names = list(map(lambda name: name[:name.find(EOS)] if name.find(EOS) > -1 else name, names))
        return hidd_cell_states, names

    def generate_name_supervised(self, lstm: Decoder, address: str, batch_size: int, observed: list = None):
        """
        lstm: Decoder associated with name being generated
        address: The address to correlate pyro distribution with latent variables
        observed: Dictionary of name/format values
        """
        hidd_cell_states = lstm.init_hidden(batch_size=batch_size)
        observed_tensor = strings_to_tensor(observed, MAX_NAME_LENGTH, letter_to_index)
        input_tensor = strings_to_tensor([SOS] * batch_size, 1, letter_to_index)
        names = [''] * batch_size

        for index in range(MAX_NAME_LENGTH):
            char_dist, hidd_cell_states = lstm.forward(input_tensor, hidd_cell_states)
            input_tensor = pyro.sample(f"sup_{address}_{index}", dist.OneHotCategorical(char_dist),
                                       obs=observed_tensor[index].unsqueeze(0))
            # Sampled char should be an index not a one-hot
            chars_at_indexes = list(
                map(lambda index: MODEL_CHARS[int(index.item())], torch.argmax(input_tensor, dim=2).squeeze(0)))
            # Add sampled characters to names
            for i, char in enumerate(chars_at_indexes):
                names[i] += char

        # Discard everything after EOS character
        names = list(map(lambda name: name[:name.find(EOS)] if name.find(EOS) > -1 else name, names))
        return hidd_cell_states, names

    def load_checkpoint(self, folder="nn_model", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception(f"No model in path {folder}")
        save_content = torch.load(filepath, map_location=DEVICE)
        self.model_fn_lstm.load_state_dict(save_content['model_fn_lstm'])
        self.guide_fn_lstm.load_state_dict(save_content['guide_fn_lstm'])
        self.encoder_lstm.load_state_dict(save_content['encoder_lstm'])

    def save_checkpoint(self, folder="nn_model", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        save_content = {
            'model_fn_lstm': self.model_fn_lstm.state_dict(),
            'guide_fn_lstm': self.guide_fn_lstm.state_dict(),
            'encoder_lstm': self.encoder_lstm.state_dict()
        }
        torch.save(save_content, filepath)
