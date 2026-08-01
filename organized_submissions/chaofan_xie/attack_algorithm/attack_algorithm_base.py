from attack_problem.one_image_problem import SingleImageProblem

from abc import ABC, abstractmethod

class AttackAlgorithmBase(ABC):

    def __init__(self, config):
        """
        Base class for all black-box evolutionary attack algorithms.

        Parameters
        ----------
        config : dict
            Configuration parameters for the algorithm.
            Must contain:
                - 'rnd': a random.Random() instance
        """
        self.rnd = config["rnd"]

    @abstractmethod
    def evolve(self, problem):
        """
        Execute the attack algorithm.

        Parameters
        ----------
        problem : SingleImageProblem
            The attack problem object, which must provide:
                - problem.evaluate(x)
                - problem.image
                - problem.epsilon
                - problem.max_evaluation
                - etc.

        Returns
        -------
        best_solution : numpy.ndarray
            The best adversarial perturbation found.
        """
        pass

