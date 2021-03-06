import string
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: REPLACED THIS WITH 0 1 2 to run the code for now
SOS = 'SOS'  # '<SOS>'
EOS = 'EOS'  # '<EOS>'
PAD = 'PAD'  # '<PAD>'

# for infcomp.py
MAX_OUTPUT_LEN = 10
MAX_STRING_LEN = 35
PRINTABLE = [char for char in string.printable] + [SOS, PAD, EOS]
NUM_PRINTABLE = len(PRINTABLE)

# Formats:
# 0 = first last
# 1 = first middle last
# 2 = last, first
# 3 = last, first middle
# 4 = first middle_initial. last
# 5 = last, first middle_initial.
FORMAT_REGEX = ["^(\w[A-Za-z'-]+\s\w[A-Za-z'-]+)$", "^(\w[A-Za-z'-]+\s\w[A-Za-z'-]+\s\w[A-Za-z'-]+)$",
                "^(\w[a-zA-Z-']+,\s\w[a-zA-Z-']+)$", "^(\w[a-zA-Z-']+,\s\w[a-zA-Z-']+\s\w[a-zA-Z-']+)$",
                "^(\w[A-Za-z'-]+\s[A-Z]+.\s\w[A-Za-z'-]+)$", "^(\w[A-Za-z'-]+,\s\w[A-Za-z'-]+\s[A-Z]+\.)$"]
