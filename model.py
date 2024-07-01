from abc import ABC, abstractmethod

class Model(ABC):
    """
    An abstract base class representing a password generating model.
    """

    @abstractmethod
    def generate(self):
        """
        Generate a password sample from the model.
        
        Returns:
            Tuple[str, float]:
                A tuple containing the generated password and its log probability.
        """
        pass

    def sample(self, n):
        """
        Generate a n iid samples from the model.

        Paremeters:
            n (int):
                Number of samples to generate.

        Returns:
            List[Tuple[str, float]]:
                n samples in the form (password, log probability).
        """
        return [self.generate() for _ in range(n)]

    @abstractmethod
    def logprob(self, pwd):
        """
        Calculate the log (base 2) probability of a given password under the model.
        Please make sure this function returns log2(p) (not log_e(p) or -log2(p)) to ensure correctness.

        Paremeters:
            pwd (str):
                The password for which to calculate the log probability.

        Returns:
            float: 
                The log probability of the password under the model.
        """
        pass
