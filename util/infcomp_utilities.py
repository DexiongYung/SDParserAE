import pyro
import pyro.distributions as dist
import torch
from typing import Tuple

from TransformerHandler import TransformerHandler
from const import *

FIRST_NAME_ADD = "first_name"
MIDDLE_NAME_ADD = "middle_name"
MIDDLE_INITIAL_ADD = "middle_initial"
LAST_NAME_ADD = "last_name"

TITLE = ['Mr', 'Mr.', 'Ms', 'Ms.', 'Mrs', 'Mrs.', 'Dr', 'Dr.', 'Sir', "Ma'am", 'Madam']
SUFFIX = ['Sr', 'Sr.', 'Snr', 'Jr', 'Jr.', 'Jnr']
FORMAT_CLASS = ['t', 'f', 'm', 'l', 's', 'sep', PAD, SOS]
# title, first, middle, last, suffix, 'separator', pad, SOS, EOS just for consistency. SOS is required for Transformer
NOISE_CLASS = ['n', 'a', 'r', 'd', PAD, SOS, EOS]  # none, add, replace, delete

AUX_FORMAT_DIM = 4
MAIN_FORMAT_DIM = 4
MIDDLE_NAME_FORMAT_DIM = 6

AUX_FORMAT_PROBS = torch.tensor([9 / 10, 1 / 30, 1 / 30, 1 / 30]).to(DEVICE)
MAIN_FORMAT_PROBS = torch.tensor([4 / 10, 4 / 10, 1 / 10, 1 / 10]).to(DEVICE)
MIDDLE_NAME_FORMAT_PROBS = torch.tensor([1 / MIDDLE_NAME_FORMAT_DIM] * MIDDLE_NAME_FORMAT_DIM).to(DEVICE)
TITLE_PROBS = torch.tensor([4 / 22, 4 / 22, 3 / 22, 3 / 22, 3 / 22, 3 / 22, 1 / 44, 1 / 44, 1 / 66, 1 / 66, 1 / 66]).to(
    DEVICE)
SUFFIX_PROBS = torch.tensor([1 / len(SUFFIX)] * len(SUFFIX)).to(DEVICE)
CHAR_NOISE_PROBS = [0.99, 0.01 / 3, 0.01 / 3, 0.01 / 3]


def sample_name(lstm, pyro_address_prefix) -> str:
    # Given a LSTM, generate a name
    name = ''
    input = lstm.indexTensor([[SOS]], 1)
    hidden = None
    for i in range(MAX_OUTPUT_LEN):
        output, hidden = lstm.forward(input[0], hidden)

        # This ensures no blank names are generated and first char is always capitalized
        if i == 0:
            output[0][0][lstm.output.index(EOS)] = -10000

            for c in string.ascii_lowercase:
                output[0][0][lstm.output.index(c)] = -10000

        output = torch.softmax(output, dim=2)[0][0]

        char_idx = int(pyro.sample(f"{pyro_address_prefix}_{i}", dist.Categorical(output)).item())
        character = lstm.output[char_idx]
        if char_idx is lstm.output.index(lstm.EOS):
            break
        input = lstm.indexTensor([[character]], 1)
        name += character
    return name


def main_format(firstname, middlename, lastname, main_format_id, middlename_char_format=None) -> Tuple[str, list]:
    if main_format_id == 0:
        char_format = ['f'] * len(firstname) + ['sep'] + ['l'] * len(lastname)
        return f"{firstname} {lastname}", char_format
    elif main_format_id == 1:
        char_format = ['l'] * len(lastname) + ['sep', 'sep'] + ['f'] * len(firstname)
        return f"{lastname}, {firstname}", char_format
    elif main_format_id == 2:
        char_format = ['f'] * len(firstname) + ['sep'] + middlename_char_format + ['sep'] + ['l'] * len(lastname)
        return f"{firstname} {middlename} {lastname}", char_format
    else:
        char_format = ['l'] * len(lastname) + ['sep', 'sep'] + ['f'] * len(firstname) + ['sep'] + middlename_char_format
        return f"{lastname}, {firstname} {middlename}", char_format


def aux_format(title, name, suffix, aux_format_id, main_char_format) -> Tuple[str, list]:
    if aux_format_id == 0:
        return name, main_char_format
    elif aux_format_id == 1:
        return f"{title} {name}", ['t'] * len(title) + ['sep'] + main_char_format
    elif aux_format_id == 2:
        return f"{name} {suffix}", main_char_format + ['sep'] + ['s'] * len(suffix)
    else:
        char_format = ['t'] * len(title) + ['sep'] + main_char_format + ['sep'] + ['s'] * len(suffix)
        return f"{title} {name} {suffix}", char_format


def parse_name(obs: torch.Tensor, classification: list):
    '''
    Parse name into components based on classification list, which classifies each obs
    index as first, middle, last, title, suffix, sep or pad
    '''
    class_str = ''.join(str(c) for c in classification)
    '''
    Separators can be noised to be non space characters so change them to
    space so can use .split in case there are multiple first, middle or last
    names
    '''
    for i in range(len(obs)):
        if classification[i] == 5:
            obs[i] = PRINTABLE.index(' ')

    titles, firsts, middles, lasts, suffixes = [], [], [], [], []

    for i in range(6):
        start = class_str.find(str(i))
        end = class_str.rfind(str(i))

        if start < 0 or end < 0:
            continue
        elif i == 0:
            titles = create_name_list(start, end, obs, classification, i)
        elif i == 1:
            firsts = create_name_list(start, end, obs, classification, i)
        elif i == 2:
            middles = create_name_list(start, end, obs, classification, i, True)
        elif i == 3:
            lasts = create_name_list(start, end, obs, classification, i)
        elif i == 4:
            suffixes = create_name_list(start, end, obs, classification, i)

    return titles, firsts, middles, lasts, suffixes


def create_name_list(start: int, end: int, obs: torch.Tensor, classification: list, class_idx: int,
                     multiple_allowed: bool = False):
    """
    Converts obs tensor to list of names(names are in list format so that PAD is considered 1 char in case of misclassification)
    Args:
        start: starting index of class
        end: ending index class
        obs: Observation tensor
        classification: list of each obs index classification
        class_idx: The classification id the index must have to append
        multiple_allowed: If multiple names should be allowed
    """
    ret = []
    name = []

    for n in range(start, end + 1):
        format_class = classification[n]
        if multiple_allowed and format_class == 5 and len(name) > 0:
            ret.append(name)
            name = []
        elif format_class == class_idx:
            index = obs[n].item()

            if index == PRINTABLE.index('PAD'):
                continue

            name.append(PRINTABLE[index])

    if len(name) > 0:
        ret.append(name)

    return ret


def has_title(aux_format_id) -> bool:
    return aux_format_id == 1 or aux_format_id == 3


def has_suffix(aux_format_id) -> bool:
    return aux_format_id == 2 or aux_format_id == 3


def has_middle_name(main_format_id) -> bool:
    return main_format_id == 2 or main_format_id == 3


def has_initial(middle_name_format_id) -> bool:
    return 0 <= middle_name_format_id <= 3


def num_middle_name(middle_name_format_id) -> int:
    if middle_name_format_id == 0 or middle_name_format_id == 2 or middle_name_format_id == 4:
        return 1
    else:
        return 2


def middle_name_format(middlenames, middle_name_format_id) -> Tuple[str, list]:
    if middle_name_format_id == 0:
        full_middlename = middlenames[0]
        char_format = ['m']
    elif middle_name_format_id == 1:
        full_middlename = f"{middlenames[0]} {middlenames[1]}"
        char_format = ['m', 'sep', 'm']
    elif middle_name_format_id == 2:
        full_middlename = f"{middlenames[0]}."
        char_format = ['m', 'sep']
    elif middle_name_format_id == 3:
        full_middlename = f"{middlenames[0]}. {middlenames[1]}."
        char_format = ['m', 'sep', 'sep', 'm', 'sep']
    elif middle_name_format_id == 4:
        full_middlename = middlenames[0]
        char_format = ['m'] * len(middlenames[0])
    else:
        full_middlename = f"{middlenames[0]} {middlenames[1]}"
        char_format = ['m'] * len(middlenames[0]) + ['sep'] + ['m'] * len(middlenames[1])
    return full_middlename, char_format


def name_to_idx_tensor(name: list, allowed_chars: list):
    '''
    Convert name in list where each index is a char to tensor form
    '''
    tensor = torch.zeros(len(name)).type(torch.LongTensor)

    for i in range(len(name)):
        tensor[i] = allowed_chars.index(name[i])

    return tensor.to(DEVICE)


def denoise_names(model: TransformerHandler, names: list, address: str, char_list: list):
    """
    Puts multiple first, middle or last names through Transformer for denoising
    """
    samples_list = []
    clean_names = []
    for idx in range(len(names)):
        name_tensor = name_to_idx_tensor(names[idx], PRINTABLE)
        samples, _ = sample_from_transformer(model, name_tensor, f"{address}_{idx}", MAX_OUTPUT_LEN)
        samples_list.append(samples)
        clean_names.append(''.join(char_list[idx] for idx in samples))
    return samples_list, clean_names


def sample_from_transformer(model: TransformerHandler, input: list, address: str, max_len: int,
                            break_on_eos: bool = True):
    """
    Iteratively samples from tranformer using input for the encoder and samples till max_len or
    EOS reached
    """
    trg_list = [model.decoder_sos_idx]
    samples, probs = [], []
    for i in range(max_len):
        trg = torch.LongTensor(trg_list).to(DEVICE)
        all_prob = model.forward(input, trg)
        prob = all_prob[i]
        sample = pyro.sample(f"{address}_{i}", dist.Categorical(prob.to(DEVICE))).item()

        if break_on_eos and sample == model.decoder_eos_idx:
            break

        samples.append(sample)
        probs.append(prob)
        trg_list.append(sample)

    return samples, probs


def classify_title_or_suffix(model, inputs: list, category: list, address: str):
    '''
    Takes a list of lists classified as titles in input then classifies to 
    title or suffix in 'category' list
    '''
    if len(inputs) > 1:
        raise Exception("We only allow one title or suffix currently")

    classes = []
    for idx in range(len(inputs)):
        input = name_to_idx_tensor(inputs[idx], PRINTABLE)
        probs = model(input.unsqueeze(1))
        classes.append(category[int(pyro.sample(f"{address}", dist.Categorical(probs.to(DEVICE))).item())])
    return classes


def classify_using_transformer(model, input: torch.tensor, address: str):
    '''
    Takes a list of lists classified as titles in input then classifies to 
    title or suffix in 'category' list
    '''
    probs = model(input.unsqueeze(1))
    return pyro.sample(address, dist.Categorical(probs.to(DEVICE))).item()
