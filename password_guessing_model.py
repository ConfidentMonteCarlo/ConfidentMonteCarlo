from typing import Tuple, List
from abc import ABC, abstractmethod
import math
import random

from models import pcfg, ngram_chain, backoff
from model import Model
# from transformer import architecture

class SimpleModel(Model):
    """
    A simple model that generates passwords with 6 lowercase letters uniformly random.
    """
    def generate(self) -> Tuple[str, float]:
        pwd = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
        return (pwd, -6 * math.log2(26))

    def logprob(self, pwd: str) -> float:
        if all(ch.isalpha() and ch.islower() for ch in pwd) and len(pwd) == 6:
            return -6 * math.log2(26)
        else:
            return -float('inf')


class SimpleModel2(Model):
    """
    A simple model that generates "123456" with 0.5 probability and passwords with 6 lowercase letters uniformly random on the remaining probability mass.
    """
    def generate(self) -> Tuple[str, float]:
        pwd = ''.join(random.choices("abcdefghijklmnopqrstuvwxyz", k=6))
        coin = random.randint(0, 1)
        if coin == 0:
            return ("123456", math.log2(0.5))
        else:
            return (pwd, -6 * math.log2(26))

    def logprob(self, pwd: str) -> float:
        if pwd == "123456":
            return 0.5
        if all(ch.isalpha() and ch.islower() for ch in pwd) and len(pwd) == 6:
            return -(6 * math.log2(26) + math.log2(2))
        else:
            return -float('inf')

class GeometricModel(Model):
    """
    Generats passwords: 001, 002, 003, 004, ..., 099, 100 with probabilities 2^-1, 2^-2, 2^-3, ..., 2^-99, 2^-99
    """
    def generate(self) -> Tuple[str, float]:
        rand = random.random()
        num = math.ceil(-math.log2(1 - rand))
        # print(rand, num)
        if num > 100:
            num = 100
        return (str(int(num)).zfill(3), -num)

    def logprob(self, pwd: str) -> float:
        if not pwd.isdigit():
            return -float('inf')
        elif pwd == "100":
            return -99
        else:
            return -(int(pwd))

class PCFGModel(Model):
    def __init__(self, filename):
        with open(filename, 'rt', encoding="latin-1") as f:
            training = [w.strip('\r\n') for w in f]
        self.mod = pcfg.PCFG(training)
    def generate(self) -> Tuple[str, float]:
        neg2lp, pw = self.mod.generate()
        return (pw, -neg2lp)
    def logprob(self, pwd: str):
        neg2lp = self.mod.logprob(pwd)
        return -neg2lp


class NGramModel(Model):
    def __init__(self, filename: str, n: int):
        with open(filename, 'rt', encoding="latin-1") as f:
            training = [w.strip('\r\n') for w in f]
        self.mod = ngram_chain.NGramModel(training, n)
    def generate(self) -> Tuple[str, float]:
        neg2lp, pw = self.mod.generate()
        return (pw, -neg2lp)
    def logprob(self, pwd: str):
        neg2lp = self.mod.logprob(pwd)
        return -neg2lp


class BackoffModel(Model):
    def __init__(self, filename: str, n: int):
        with open(filename, 'rt', encoding="latin-1") as f:
            training = [w.strip('\r\n') for w in f]
        self.mod = backoff.BackoffModel(training, n)
    def generate(self) -> Tuple[str, float]:
        neg2lp, pw = self.mod.generate()
        return (pw, -neg2lp)
    def logprob(self, pwd: str):
        neg2lp = self.mod.logprob(pwd)
        return -neg2lp

class TransformerModel(Model):
    def __init__(self, modelpath: str):
        self.mod = architecture.transformer_inference(modelpath)
    def generate(self) -> Tuple[str, float]:
        pw, neg2lp = self.mod.generate()
        return (pw, -neg2lp)
    def logprob(self, pwd: str):
        neg2lp = self.mod.logprob(pwd)
        return -neg2lp

