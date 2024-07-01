import string
global alphabet
global encode
global decode

alphabet = list(string.ascii_letters + string.digits + string.punctuation + ' ')
encode = {c:i + 2 for i, c in enumerate(alphabet)}
decode = {v: k for k, v in encode.items()}
decode[0] = 'start'
decode[1] = 'end'


def encoding(x, maxlen):
    return [0] + [encode[c] for c in x] + [1]*(maxlen + 1 - len(x))

def count_upper_lower(string, case):
    if case == 'upper':
        return sum(letter.isupper() for letter in string)
    elif case == 'lower':
        return sum(letter.islower() for letter in string)


def is1class8(pwd):
    return (len(pwd) >= 8)

def is1class16(pwd):
    return len(pwd) >= 16
    
def is3class8(pwd):
    def contains_lower(pwd):
        return any([x.islower() for x in pwd])
    
    def contains_upper(pwd):
        return any([x.isupper() for x in pwd])
    
    def contains_digit(pwd):
        return any([x.isnumeric() for x in pwd])
    
    def contains_special(pwd):
        return any([not x.isalnum() for x in pwd])
    
    if is1class8(pwd):
        return sum([contains_lower(pwd), contains_upper(pwd), contains_digit(pwd), contains_special(pwd)]) >= 3
    else:
        return False