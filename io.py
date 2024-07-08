import csv
import string
import random
from collections import Counter

alphabet = list(string.ascii_letters + string.digits + string.punctuation + ' ')

def raw_to_csv(infile, outfile):
    """
    Converts raw password file to csv file with columns 'pwd' and 'freq', representing unique passwords and their frequencies in the plain text file.
    """
    with open(infile, 'r', encoding='latin-1') as f:
        lines = f.read().splitlines()
        good_lines = [pwd for pwd in lines if all(c in alphabet for c in pwd)]
    freq = Counter(lines)
    data = freq.items()
    with open(outfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for pwd, freq in data:
            writer.writerow([pwd, freq])

def head(infile, outfile, n):
    """
    Writes the first n lines of infile to outfile.
    """
    with open(infile, 'r', encoding='latin-1') as f:
        lines = f.read().splitlines()
    with open(outfile, 'w') as f:
        for i in range(n):
            f.write(lines[i] + '\n')

def sample(infile, outfile, n):
    """
    Randomly samples (with replacement) n lines from infile and writes them to outfile.
    """
    with open(infile, 'r', encoding='latin-1') as f:
        lines = f.read().splitlines()
    with open(outfile, 'w') as f:
        for i in range(n):
            f.write(random.choice(lines) + '\n')

