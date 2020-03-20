import math
import torch
from random import randint
from torch import distributions

character_replacement = dict()

character_replacement['a'] = 'qwsz'
character_replacement['b'] = 'nhgv '
character_replacement['c'] = 'vfdx '
character_replacement['d'] = 'fresxc'
character_replacement['e'] = 'sdfr43ws'
character_replacement['f'] = 'gtrdcv'
character_replacement['g'] = 'hytfvb'
character_replacement['h'] = 'juytgbn'
character_replacement['i'] = 'ujklo98'
character_replacement['j'] = 'mkiuyhn'
character_replacement['k'] = 'jm,loij'
character_replacement['l'] = 'k,.;pok'
character_replacement['m'] = 'njk, '
character_replacement['n'] = 'bhjm '
character_replacement['o'] = 'plki90p'
character_replacement['p'] = 'ol;[-0o'
character_replacement['q'] = 'asw21'
character_replacement['r'] = 'tfde45'
character_replacement['s'] = 'dxzawe'
character_replacement['t'] = 'ygfr56'
character_replacement['u'] = 'ijhy78'
character_replacement['v'] = 'cfgb '
character_replacement['w'] = 'saq23e'
character_replacement['x'] = 'zsdc'
character_replacement['y'] = 'uhgt67'
character_replacement['z'] = 'xsa'
character_replacement['1'] = '2q'
character_replacement['2'] = '3wq1'
character_replacement['3'] = '4ew2'
character_replacement['4'] = '5re3'
character_replacement['5'] = '6tr4'
character_replacement['6'] = '7yt5'
character_replacement['7'] = '8uy6'
character_replacement['8'] = '9iu7'
character_replacement['9'] = '0oi8'
character_replacement['0'] = '-po9'


def noise_name(x: str, allowed_chars: str, max_length: int, max_noise: int = 2):
    noise_type = randint(0, 4)

    if noise_type == 0:
        return add_chars(x, allowed_chars, max_length, max_add=max_noise)
    elif noise_type == 1:
        return switch_chars(x, allowed_chars, max_switch=max_noise)
    elif 2:
        return remove_chars(x, max_remove=max_noise)
    elif 3:
        return replacement_chars(x, max_replace=max_noise)
    else:
        x = remove_chars(x, max_remove=max_noise)
        return add_chars(x, allowed_chars, max_add=max_noise)


def add_chars(x: str, allowed_chars: str, max_length: int, max_add: int):
    if max_add + len(x) > max_length:
        raise Exception(f"{max_add + len(x)} is greater than max length:{max_length}")

    ret = x
    num_to_add = randint(0, max_add)

    for i in range(num_to_add):
        random_char = allowed_chars[randint(0, len(allowed_chars) - 1)]
        pos = randint(0, len(ret) - 1)
        ret = "".join((ret[:pos], random_char, ret[pos:]))

    return ret


def switch_chars(x: str, allowed_chars: str, max_switch: int):
    ret = x
    num_to_switch = randint(0, min(math.floor(len(x) / 2), max_switch))

    for i in range(num_to_switch):
        random_char = allowed_chars[randint(0, len(allowed_chars) - 1)]
        pos = randint(0, len(ret) - 1)
        ret = "".join((ret[:pos], random_char, ret[pos + 1:]))

    return ret


def remove_chars(x: str, max_remove: int):
    ret = x
    num_to_remove = randint(0, min(math.floor(len(x) / 2), max_remove))

    for i in range(num_to_remove):
        pos = randint(0, len(ret) - 1)
        ret = "".join((ret[:pos], ret[pos + 1:]))

    return ret


def replacement_chars(x: str, max_replace: int):
    ret = x
    num_to_replace = randint(0, min(math.floor(len(x) / 2), max_replace))

    for i in range(num_to_replace):
        pos = randint(0, len(ret) - 1)
        char = x[pos]
        replace_chars = character_replacement[char]
        sample = torch.distributions.categorical.Categorical(
            torch.ones(len(replace_chars)) * (1 / len(replace_chars))).sample()
        replacement = replace_chars[sample[0]]

        ret = "".join((ret[:pos], replacement, ret[pos + 1:]))

    return ret
