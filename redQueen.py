from multiprocessing import Pool
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import uuid
import cPickle as pickle
import itertools
from scipy.special import lambertw
from scipy.optimize import brentq

RED = "#EB6231"
YELLOW = "#E29D26"
BLUE = "#5D80B4"
LIGHTGREEN = "#6ABD9B"
GREEN = "#8FB03E"


def execute(x):
    """
    :param x: 'SimulationsAlongParameter' or 'Simulation' instance.
    :return: 'SimulationsAlongParameter' or 'Simulation', run the simulation.
    """
    return x.run()


def id_generator(number_of_char):
    """
    :param number_of_char: 'Integer'.
    :return: 'String', a random string of 'number_of_char' characters.
    """
    return str(uuid.uuid4().get_hex().upper()[0:number_of_char])


def set_dir(path):
    """
    :param path: 'String', the relative path to a directory.
    :return: 'String', equivalent to the bash ${cd path}.
    """
    path = os.getcwd() + path
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    os.chdir(path)


# Normalize an array to 1
def sum_to_one(x):
    """
    Normalize an array such that the L1 norm equals to 1.
    :param x: 'Array' (or 'List'), must be a non null array of Float or Integer.
    :return: 'Array'.
    """
    return np.divide(x, np.sum(x))


# Flatten an array
def flatten(array):
    """
    Return the list of all leaves of the nested arrays, in natural traversal order.
    :param array: 'Array' (or list), of nested arrays (or lists).
    :return: 'List'.
    """
    list(itertools.chain.from_iterable(array))


class Model(object):
    """
    The class model implement the Prdm9 Red-Queen dynamic in a panmictic Wright-Fisher population.
    Each time-step of the Wright-Fisher simulation consists of 3 steps : mutation, erosion and selection.
    - Mutation: each mutation (drawn randomly) generate a new Prdm9 allele in one individual of the population,
        the hotspots activity of this new Prdm9 allele equals to 1.
    - Erosion: the hotspots are eroded and the hotspots activity decreases for each Prdm9 allele.
    - Selection: Prdm9 allele frequencies are updated and Prdm9 allele with null frequency are removed from the model.
    """
    def __init__(self, mutation_rate_prdm9=1.0 * 10 ** -5, mutation_rate_hotspot=1.0 * 10 ** -7,
                 population_size=10 ** 4, recombination_rate=1.0 * 10 ** -3, alpha=1.,
                 fitness='linear', drift=True, linearized=True):
        """
        :param mutation_rate_prdm9: 'Float', the mutation rate of Prdm9 (mutation per base*number of bases).
        :param mutation_rate_hotspot: 'Float', the mutation rate of hotspots (mutation per base*number of bases).
        :param population_size: 'Float' or 'Integer', the effective population size.
        :param recombination_rate: 'Float', the recombination rate at the hotspots loci.
        :param fitness: 'String', either 'linear', 'sigmoid' or 'polynomial'.
        :param alpha: 'Float'.
            - 'alpha' is not used if the fitness is 'linear'.
            - 'alpha' is the exponent of the fitness function if the fitness is 'polynomial'.
            - 'alpha' is the inflexion point if the fitness is 'sigmoid'.
        :param drift: 'Boolean', True if genetic drift is taken into account.
        :param linearized: 'Boolean', True if the fitness function is linearized.
        """
        # Initialisation of the parameters for approximation
        # If drift == True, a multinomial random drift is taken into account
        self.drift = drift
        # If linearized == True, the fitness function is linearized around the mean activity
        self.linearized = linearized

        # Initialisation of the parameters of the model
        self.mutation_rate_prdm9 = mutation_rate_prdm9
        self.mutation_rate_hotspot = mutation_rate_hotspot
        self.recombination_rate = recombination_rate
        self.population_size = float(population_size)

        # Initialisation of the fitness function
        fitness_hash = {"linear": 1, "polynomial": 2, "sigmoid": 3}
        assert fitness in fitness_hash.keys(), "Parameter 'fitness' must be a string: ['linear','sigmoid','polynomial']"
        self.fitness_family = fitness_hash[fitness]
        self.alpha = alpha
        if self.fitness_family == 1:
            self.alpha = 1.
        elif self.fitness_family == 3:
            self.sigmoid_slope = 2

        # Initialisation of the scaled parameters
        self.mu, self.rho, self.epsilon = 0, 0, 0
        self.scaling_parameters()

        # Initialisation of the array of initial alleles 
        nbr_of_alleles = 10
        self.t = 0
        self.id = nbr_of_alleles
        self.ids = np.array(range(nbr_of_alleles))
        self.prdm9_polymorphism = sum_to_one(np.random.sample(nbr_of_alleles))
        self.prdm9_polymorphism *= self.population_size
        self.prdm9_fitness = np.ones(nbr_of_alleles)
        self.prdm9_longevity = np.zeros(nbr_of_alleles)
        self.hotspots_activity = np.random.sample(nbr_of_alleles)

    def scaling_parameters(self):
        """
        Compute the mutation rate of Prdm9 and erosion rate of hotspots scaled by the population size.
        :return: None.
        """
        # Mu is the scaled mutation rate
        self.mu = 4 * self.population_size * self.mutation_rate_prdm9
        # Rho is the scaled erosion rate
        self.rho = 4 * self.population_size * self.mutation_rate_hotspot * self.recombination_rate
        self.epsilon = np.sqrt(self.rho / (self.mu * self.alpha))
        assert self.rho < 0.5, "The scaled erosion rate is too large, decrease either the " \
                               "recombination rate, the erosion rate or the population size"
        assert self.rho > 0.0000000001, "The scaled erosion rate is too low, increase either the " \
                                        "recombination rate, the erosion rate or the population size"

    def forward(self):
        """
        One step of the Wright-Fisher simulation : mutation, erosion and selection.
        :return: None.
        """
        self.mutation()
        self.erosion_and_selection()

    def mutation(self):
        """
        Draw randomly new Prdm9 alleles (with hotspots activity equal to 1) according to a Poisson distribution.
        :return: None.
        """
        # The number of new alleles is poisson distributed
        new_alleles = np.random.poisson(2 * self.population_size * self.mutation_rate_prdm9)
        # Initialize new alleles in the population only if necessary
        if new_alleles > 0:
            self.prdm9_polymorphism -= np.random.multinomial(new_alleles, sum_to_one(self.prdm9_polymorphism))
            self.prdm9_polymorphism = np.append(self.prdm9_polymorphism, np.ones(new_alleles))
            self.hotspots_activity = np.append(self.hotspots_activity, np.ones(new_alleles))
            self.prdm9_longevity = np.append(self.prdm9_longevity, np.zeros(new_alleles))
            self.ids = np.append(self.ids, range(self.id, self.id + new_alleles))
            self.id += new_alleles

    def erosion_and_selection(self):
        """
        - The hotspots are eroded and the hotspots activity decreases for each Prdm9 allele.
        - Prdm9 allele frequencies are updated and Prdm9 allele with null frequency are removed.
        :return: None.
        """
        prdm9_frequencies = sum_to_one(self.prdm9_polymorphism)

        # Erosion of the hotspots
        self.hotspots_activity *= np.exp(-self.rho * prdm9_frequencies)

        # Compute the fitness for each allele
        if self.fitness_family == 0:
            distribution_vector = prdm9_frequencies
        else:
            if self.linearized:
                mean_activity = np.sum(prdm9_frequencies * self.hotspots_activity)
                self.prdm9_fitness = self.derivative_log_fitness(mean_activity) * (self.hotspots_activity - mean_activity) / 2
                distribution_vector = prdm9_frequencies + self.prdm9_fitness * prdm9_frequencies
                if np.max(distribution_vector) > 1.:
                    distribution_vector = sum_to_one(np.clip(distribution_vector, 0., 1.))
                elif np.min(distribution_vector) < 0.:
                    distribution_vector = sum_to_one(np.clip(distribution_vector, 0., 1.))
            else:
                fitness_matrix = self.fitness(
                    np.add.outer(self.hotspots_activity, self.hotspots_activity) / 2)
                self.prdm9_fitness = np.dot(fitness_matrix, prdm9_frequencies)
                distribution_vector = sum_to_one(self.prdm9_fitness * prdm9_frequencies)

        if self.drift:
            self.prdm9_polymorphism = np.random.multinomial(int(self.population_size), distribution_vector).astype(
                float)
        else:
            self.prdm9_polymorphism = distribution_vector * self.population_size

        self.prdm9_longevity += 1

        # Remove the extincted alleles from the population
        remove_extincted = np.array(map(lambda x: x > 0, self.prdm9_polymorphism), dtype=bool)
        if not remove_extincted.all():
            self.prdm9_polymorphism = self.prdm9_polymorphism[remove_extincted]
            self.hotspots_activity = self.hotspots_activity[remove_extincted]
            self.prdm9_fitness = self.prdm9_fitness[remove_extincted]
            self.ids = self.ids[remove_extincted]
            self.prdm9_longevity = self.prdm9_longevity[remove_extincted]

    def fitness(self, x):
        """
        Compute f(x) where f is the fitness function, either 'linear', 'polynomial' or 'sigmoid'
        :param x: numpy 'Matrix', matrix hotspots activity where x_{i,j}=(L_i+L_j)/2 where L_i is the hotspots
        activity of allele i.
        :return: numpy 'Matrix', f(x)
            - f(x)=x if the fitness is 'linear'.
            - f(x)=x^alpha if the fitness is 'polynomial'.
            - f(x)=(x^k)/(x^k + alpha^k) if the fitness is 'sigmoid', where k is the 'slope' (sharpness) of the sigmoid.
        """
        if self.fitness_family == 3:
            return np.power(x, self.sigmoid_slope) / (
                np.power(x, self.sigmoid_slope) + np.power(self.alpha, self.sigmoid_slope))
        elif self.fitness_family == 2:
            return np.power(x, self.alpha)
        else:
            return x

    def derivative_log_fitness(self, x):
        """
        Compute f'(x)/f(x) where f is the fitness function, either 'linear', 'polynomial' or 'sigmoid'.
        :param x: 'Float', usually the mean activity of hotspots (R).
        :return: 'Float', f'(x)/f(x), where f is the fitness function.
        """
        if x == 0:
            return float("inf")
        else:
            if self.fitness_family == 3:
                return self.sigmoid_slope * np.power(self.alpha, self.sigmoid_slope) / (
                    x * (np.power(x, self.sigmoid_slope) + np.power(self.alpha, self.sigmoid_slope)))
            elif self.fitness_family == 2:
                return self.alpha * 1.0 / x
            else:
                return 1.0 / x

    @staticmethod
    def activity_limit(mean_activity):
        """
        Mean-field derivation of the activity of the hotspots when the Prdm9 allele goes extinct.
        :param mean_activity: 'Float', mean activity of hotspots (R).
        :return: 'Float', the activity limit of hotspots (R_{\infty}).
        """
        if mean_activity == 0.:
            return 0.
        else:
            return -1 * mean_activity * np.real(lambertw(-np.exp(- 1. / mean_activity) / mean_activity))

    def mean_activity_estimation(self):
        """
        Mean-field derivation of the mean activity of hotspots by solving the self consistent mean field equation.
        :return: 'Float', mean activity of hotspots (R).
        """
        return brentq(lambda x: self.self_consistent_equation(x), 0, 1)

    def self_consistent_equation(self, x):
        """
        Self-consistent mean-field equation for the mean activity of hotspots.
        :param x: 'Float', in the interval [0,1].
        :return: 'Float', g(x).
        """
        return self.derivative_log_fitness(x) * (1 - x) * (1 - self.activity_limit(x)) - (2. * x * self.rho / self.mu)

    def mean_activity_small_load(self):
        """
        Mean-field derivation of the mean activity of hotspots, using the small-load development (low erosion).
        :return: 'Float', mean activity of hotspots (R).
        """
        return 1 - self.epsilon

    def activity_limit_small_load(self):
        """
        Mean-field derivation of the activity of the hotspots when the Prdm9 allele goes extinct, using the
        small-load development (low erosion).
        :return: 'Float', the activity limit of hotspots (R_{\infty}).
        """
        return 1 - 2. * self.epsilon

    def frequencies_wrt_activity(self, mean_activity, l_limit):
        """
        Mean-field trajectory of the frequency of a Prdm9 allele, as a function of the activity of its hotspots.
        :param mean_activity: 'Float', mean activity of hotspots (R).
        :param l_limit: 'Float', the activity limit of hotspots.
        :return: Tuple('Array', 'Array'), array of hotspots activity and array of Prdm9 frequency (same length).
        """
        l = np.linspace(l_limit, 1)
        x = 1. - l + mean_activity * np.log(l)
        x *= self.derivative_log_fitness(mean_activity) / (2 * self.rho)
        return l, np.clip(x, 0., 1.)

    def frequencies_wrt_activity_estimation(self):
        """
        Mean-field trajectory of the frequency of a Prdm9 allele, as a function of the activity of its hotspots.
        :return: Tuple('Array', 'Array'), array of hotspots activity and array of Prdm9 frequency (same length).
        """
        mean_activity = self.mean_activity_estimation()
        l_limit = self.activity_limit(mean_activity)
        return self.frequencies_wrt_activity(mean_activity, l_limit)

    def frequencies_wrt_activity_small_load(self):
        """
        Mean-field trajectory of the frequency of a Prdm9 allele, as a function of the activity of its hotspots,
        using the small-load development (low erosion).
        :return: Tuple('Array', 'Array'), array of hotspots activity and array of Prdm9 frequency (same length).
        """
        mean_activity = self.mean_activity_small_load()
        l_limit = self.activity_limit_small_load()
        return self.frequencies_wrt_activity(mean_activity, l_limit)

    def prdm9_diversity_estimation(self):
        """
        Mean-field derivation of the Prdm9 diversity computed as \sum_i x_i^{-2} where x_i is the frequency of allele i.
        :return: 'Float', Prdm9 diversity (D).
        """
        mean_activity = self.mean_activity_estimation()
        if mean_activity == 1.:
            return self.population_size
        else:
            diff = 1 - 2 * mean_activity + self.activity_limit(mean_activity)
            diversity = 4 * self.rho / (self.derivative_log_fitness(mean_activity) * diff)
            return max(1., diversity)

    def prdm9_diversity_small_load(self):
        """
        Mean-field derivation of the Prdm9 diversity computed as \sum_i x_i^{-2} where x_i is the frequency of allele i,
        using the small-load development (low erosion).
        :return: 'Float', Prdm9 diversity (D).
        """
        return max(1., 6 * self.rho / (self.alpha * self.epsilon**2))

    def landscape_variance_estimation(self):
        """
        Mean-field derivation of the hotspots landscape variance computed as \sum_i x_i^{-2} l_i, where x_i is the
        frequency of allele i and l_i is the hotspots activity of allele i.
        :return: 'Float', landscape variance (V).
        """
        mean_activity = self.mean_activity_estimation()
        diff = 1 - 2 * mean_activity + self.activity_limit(mean_activity)
        landscape = mean_activity * diff * self.derivative_log_fitness(mean_activity) / (4 * self.rho)
        return min(1., landscape)

    def landscape_variance_small_load(self):
        """
        Mean-field derivation of the hotspots landscape variance computed as \sum_i x_i^{-2} l_i, where x_i is the
        frequency of allele i and l_i is the hotspots activity of allele i.
        :return: 'Float', landscape variance (V).
        """
        return 1. / self.prdm9_diversity_small_load()

    def probability_fixation(self, mean_activity):
        """
        Probability of fixation of a new Prdm9 allele in the population, using the Kimura's equation (1962).
        :param mean_activity: 'Float', mean activity of hotspots (R).
        :return: 'Float', the probability of fixation.
        """
        if mean_activity == 0.:
            return 1.
        elif mean_activity == 1.:
            return 1. / self.population_size
        else:
            selection = self.derivative_log_fitness(mean_activity) * (1. - mean_activity) / 2.
            return (1. - np.exp(-selection)) / (1. - np.exp(-2. * self.population_size * selection))

    def turn_over_estimation(self):
        """
        Mean-field derivation of the turn-over time defined as the decorrelation time of the relative cross-homozygosity
        :return: 'Float', turn-over time (T).
        """
        mean_activity = self.mean_activity_estimation()
        return self.prdm9_diversity_estimation() / (self.mu * self.probability_fixation(mean_activity))

    def turn_over_small_load(self):
        """
        Mean-field derivation of the turn-over time defined as the decorrelation time of the relative cross-homozygosity,
        using the small-load development (low erosion).
        :return: 'Float', turn-over time (T).
        """
        return self.prdm9_diversity_small_load() / (self.mu * self.alpha * (1. - self.mean_activity_small_load()) / 2.)

    def __str__(self):
        """
        Print the parameters of the model.
        :return: 'String'.
        """
        name = "u=%.1e" % self.mutation_rate_prdm9 + \
               "_v=%.1e" % self.mutation_rate_hotspot + \
               "_r=%.1e" % self.recombination_rate + \
               "_n=%.1e" % self.population_size
        if self.fitness_family == 3 or self.fitness_family == 2:
            name += "_f=%.1e" % self.alpha
        name += "_drift=%s" % self.drift + \
                "_linear=%s" % self.linearized
        return name

    def caption(self):
        """
        Return a meaningful description of the model.
        :return: 'String'.
        """
        caption = "Mutation rate of PRDM9: %.1e. \n" % self.mutation_rate_prdm9 + \
                  "Mutation rate of the hotspots: %.1e. \n" % self.mutation_rate_hotspot + \
                  "Recombination rate at the hotspots: %.1e. \n" % self.recombination_rate + \
                  "Population size: %.1e. \n" % self.population_size
        if self.fitness_family == 3:
            caption += "Inflexion point of the fitness function PRDM9=%.1e. \n" % self.alpha
        if self.fitness_family == 2:
            caption += "Exponent of the polynomial fitness function PRDM9=%.1e. \n" % self.alpha
        if self.drift:
            caption += "  The simulation take into account DRIFT. \n"
        else:
            caption += "  The simulation DOESN'T take into account DRIFT. \n"
        if self.linearized:
            caption += "  The simulation use a LINEARIZED APPROXIMATION for the fitness. \n"
        else:
            caption += "  The simulation DOESN'T APPROXIMATE the fitness. \n"
        return caption

    def copy(self):
        return copy.copy(self)


# this is a series of samples, at regular times
# each sample specifies number of alleles (K), and their associated frequencies and activities
class Trace(object):
    """
    During a simulation, 'Trace' store the data contains in a 'Model' instance at several time points.
    'Trace' instance can be used to compute several summary statistics over the simulation.
    """
    def __init__(self):
        """
        Initialize empty lists.
        """
        self.prdm9_frequencies, self.hotspots_activity, self.prdm9_fitness = [], [], []

        self.ids, self.prdm9_longevities = [], []

    def store(self, step):
        """
        Store the Prdm9 frequencies, fitnesses and longevities. Also store the activity of hotspots associated to each
        Prdm9 alleles.
        :param step: 'Model' instance.
        :return: None.
        """
        self.prdm9_frequencies.append(sum_to_one(step.prdm9_polymorphism))
        self.prdm9_fitness.append(step.prdm9_fitness)
        self.hotspots_activity.append(step.hotspots_activity)
        self.ids.append(step.ids)
        self.prdm9_longevities.append(step.prdm9_longevity)

    # mean over the population for each sample
    def mean_activity_array(self):
        """
        For each time point of the trace, compute the mean activity of hotspots (R).
        :return: 'Array', mean activity of hotspots (R) for each time point.
        """
        return np.array(map(lambda erosion, freq: self.n_moment(erosion, freq, 1), self.hotspots_activity,
                            self.prdm9_frequencies))

    # mean over the samples (mean over the simulation trajectory)
    def mean_activity(self):
        """
        The mean activity of hotspots (R) overaged over the trace.
        :return: 'Float', mean activity of hotspots (R).
        """
        return np.mean(self.mean_activity_array())

    def prdm9_diversity_array(self):
        """
        For each time point of the trace, compute the Prdm9 diversity (D).
        :return: 'Array', Prdm9 diversity (D) for each time point.
        """
        return np.array(map(lambda frequencies: 1. / np.sum(np.power(frequencies, 2)), self.prdm9_frequencies))

    # mean over the samples (mean over the simulation trajectory)
    def prdm9_diversity(self):
        """
        The Prdm9 diversity (D) overaged over the trace.
        :return: 'Float', Prdm9 diversity (D).
        """
        return np.mean(self.prdm9_diversity_array())

    def landscape_variance_array(self):
        """
        For each time point of the trace, compute the hotspots landscape variance (V).
        :return: 'Array', landscape variance (V) for each time point.
        """
        return np.array(map(lambda erosion, freq: self.n_moment(freq, erosion, 2), self.hotspots_activity,
                            self.prdm9_frequencies))

    # mean over the samples (mean over the simulation trajectory)
    def landscape_variance(self):
        """
        The hotspots landscape variance (V) overaged over the trace.
        :return: 'Float', landscape variance (V).
        """
        return np.mean(self.landscape_variance_array())

    def turn_over(self):
        """
        The turn-over time (T) defined as the time for which the cross-heterozygosity is equal to half of the
        instant homozygosity.
        :return: 'Float', turn-over time (T).
        """
        return self.dichotomic_search(0.5)

    def dichotomic_search(self, percent):
        """
        Return the time for which the cross-heterozygosity is equal to a given fraction (parameter 'percent') of the
        instant homozygosity.
        :param percent: 'Float', in the interval [0,1].
        :return:, 'Float', time.
        """
        lower_lag = 0
        upper_lag = len(self.ids) - 1
        precision = 2
        ch_0 = self.cross_homozygosity(0)
        if self.cross_homozygosity(upper_lag) / ch_0 >= percent:
            return upper_lag
        else:
            middle_lag = (lower_lag + upper_lag) / 2
            while upper_lag - lower_lag >= precision:
                middle_lag = (lower_lag + upper_lag) / 2
                if self.cross_homozygosity(middle_lag) / ch_0 >= percent:
                    lower_lag = middle_lag
                else:
                    upper_lag = middle_lag
            return middle_lag

    def cross_homozygosity(self, lag):
        """
        Cross homozygosity is defined as the fraction of homozygotes in a population that would be obtained by
        hybrdizing populations at time t and t+'lag' in equal proportions. The cross-homozygosity reduces to the
        regular (or instant) homozygosity for 'lag' = 0, and it drops to 0 for large 'lag'.
        :param lag: 'Integer', in the range [0..T_{max}], where T_{max} is the number of time points in the trace.
        :return: 'Float', Cross-homozygosity.
        """
        cross_homozygosity = []
        for index in range(0, len(self.ids) - lag):
            cross_homozygosity.append(0.)
            slice_dict = dict(zip(self.ids[index], self.prdm9_frequencies[index]))
            lag_dict = dict(zip(self.ids[index + lag], self.prdm9_frequencies[index + lag]))
            for key in (set(slice_dict.keys()) & set(lag_dict.keys())):
                cross_homozygosity[index] += slice_dict[key] * lag_dict[key]

        assert len(cross_homozygosity) > 0, "Cross Homozygosity is empty"
        return np.mean(cross_homozygosity)

    @staticmethod
    def n_moment(x, probabilities, n):
        """
        Compute E(x^n), the n^{th} moment of variable x with probability measure given by the vector 'probabilities'
        :param x: 'Array', vector of Float or Integer.
        :param probabilities: 'Array', vector of probabilities.
        :param n: 'Integer', n^{th} moment to be computed.
        :return: 'Float', E(x^n).
        """
        assert len(probabilities) == len(x), "x and probabilities should be of the same length"
        if n > 1:
            return np.sum((x ** n) * probabilities)
        else:
            return np.sum(x * probabilities)


# a simulation takes an instance of model as a parameter
# creates a Trace object
# run the model and stores states of the model at regular time intervals into snapshot object

class Simulation(object):
    """
    'Simulation' will first run a 'Model' instance during the burn-in phase. Once the burn-in is completed,
    meaning the initial alleles have been all replaced, it starts recording the trace into the attribute 'self.trace'.
    Also implements method 'save_figure' to display the result of the simulation.
    """
    def __init__(self, model, loops=10):
        """
        :param model: 'Model' instance.
        :param loops: Integer, the number of cycle (complete replacement of all Prdm9 alleles) simulated.
        """
        self.t_max = 0
        self.nbr_of_steps = 0.
        self.loops = loops
        self.model = model
        self.trace = Trace()
        self.generations = []

    def __str__(self):
        """
        Print the parameters of the model and the duration (nbr of steps) of simulation.
        :return: 'String'.
        """
        return "tmax=%.1e_" % self.t_max + str(self.model)

    def caption(self):
        """
        Return a meaningful description of the simulation.
        :return: 'String'.
        """
        return "%.1e generations computed \n \n" % self.t_max + \
               "Model parameters : \n" + self.model.caption()

    def burn_in(self):
        """
        Run the self.model instance until all initial alleles have been all replaced without recording the trace.
        :return: None.
        """
        t = 0
        initial_variants = set(self.model.ids)
        while len(initial_variants & set(self.model.ids)) > 0:
            self.model.forward()
            t += 1

        self.nbr_of_steps = max(100, 10000 / len(self.model.ids))
        self.t_max = 10 * (int(max(int(self.loops * t), self.nbr_of_steps)) / 10 + 1)

        print "Burn-in Completed"

    def run(self):
        """
        Burn-in phase and then run the self.model instance and record the trace.
        :return: self.
        """
        self.burn_in()

        step_t = 0.

        step = float(self.t_max) / self.nbr_of_steps
        for t in range(self.t_max):
            self.model.forward()

            step_t += 1
            if step_t > step:
                step_t -= step

                self.generations.append(t)
                self.trace.store(self.model)

            if int(10 * t) % self.t_max == 0:
                print "Computation at {0}%".format(float(100 * t) / self.t_max)

        return self

    def save_trajectory(self):
        """
        Plot the trace of the Prdm9 frequencies and activity of hotspots.
        Save the figure in the files 'trajectory.svg' and 'trajectory.png'.
        :return: 'String', str(self).
        """
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1440 / my_dpi), dpi=my_dpi)

        generations = list(itertools.chain.from_iterable(map(lambda x, y: [x] * len(y),
                                                             self.generations, self.trace.prdm9_frequencies)))
        xlim = [min(generations), max(generations)]

        plt.subplot(311)
        for my_id in range(self.model.id):
            array = np.zeros(len(self.generations))
            for t in range(len(self.generations)):
                if my_id in self.trace.ids[t]:
                    array[t] = self.trace.prdm9_frequencies[t][self.trace.ids[t].tolist().index(my_id)]

            plt.plot(self.generations, array, color=BLUE, lw=3)
        plt.xlabel('Generations')
        plt.ylim([0, 1])
        plt.xlim(xlim)
        plt.ylabel('PRDM9 frequencies')

        plt.subplot(312)
        plt.scatter(generations, flatten(self.trace.hotspots_activity), color=YELLOW, lw=0)
        plt.plot(self.generations, self.trace.mean_activity_array(), color=GREEN, lw=3)
        plt.xlabel('Generations')
        plt.ylabel('Hotspot activity')
        plt.ylim([0, 1])
        plt.xlim(xlim)

        plt.tight_layout()

        plt.savefig('trajectory.png', format="png")
        plt.savefig('trajectory.svg', format="svg")
        print "Trajectory computed"
        plt.clf()
        plt.close('all')
        return str(self)

    def save_figure(self):
        """
        Plot and save (in svg format) self-consistent equation and phase plan (Prdm9 frequencies vs activity of hotspots).
        :return: None.
        """
        self.save_estimation()
        self.save_phase_plan()
        plt.close('all')
        print str(self)

    def save_estimation(self):
        """
        Plot x and g(x), where g(x)=x is the self-consistent equation for the mean activity of hotspots, for x in [0,1].
        Save the figure in the file ending with 'estimation.svg'.
        :return: None.
        """
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1920 / my_dpi), dpi=my_dpi)
        mean_activity = self.model.mean_activity_estimation()
        theta = np.linspace(0, 1, 100)
        plt.plot(theta, np.array(map(lambda x: self.model.self_consistent_equation(x) + x, theta)), color=BLUE, linewidth=2)
        plt.plot(theta, theta, color=RED, linewidth=2)
        plt.plot((mean_activity, mean_activity), (0., 1.), 'k-', linewidth=3)
        plt.title('The self-consistent estimation of theta')
        plt.xlabel('theta')
        plt.ylabel('g(theta)')
        plt.ylim((0., 1.))
        plt.tight_layout()

        plt.savefig(str(self) + 'estimation.svg', format="svg")
        plt.clf()

    def save_phase_plan(self):
        """
        Scatter plot of the Prdm9 frequencies as a function of the activity of hotspots.
        The color of the dots is the age of the allele (blue is new allele and red is old allele).
        Save the figure in the file ending with 'phase.svg'.
        :return: None.
        """
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1920 / my_dpi), dpi=my_dpi)

        a = list(itertools.chain.from_iterable(self.trace.hotspots_activity))
        b = list(itertools.chain.from_iterable(self.trace.prdm9_frequencies))
        c = list(itertools.chain.from_iterable(self.trace.prdm9_longevities))
        plt.hexbin(a, b, c, gridsize=200, bins='log')
        plt.plot(*self.model.frequencies_wrt_activity_estimation(), color=YELLOW, linewidth=3)
        plt.plot(*self.model.frequencies_wrt_activity_small_load(), color=GREEN, linewidth=3)
        plt.title('PRMD9 frequency vs hotspot activity')
        plt.xlabel('hotspot activity')
        plt.ylabel('PRMD9 frequency')

        plt.tight_layout()

        plt.savefig(str(self) + 'phase.svg', format="svg")
        plt.clf()

    def pickle(self):
        pickle.dump(self, open(str(self) + ".p", "wb"))


# a more complex simulation object
# makes a series of simulations for regularly spaced (in log) values of key parameters of the model
class SimulationsAlongParameter(object):
    """
    'SimulationsAlongParameter' will run a series of independent simulations for regularly spaced (in log) values
    of only one parameter, while all the other parameters of the simulations are kept constant.
    """
    def __init__(self, model, parameter="null", nbr_of_simulations=20, scale=10 ** 2, loops=10):
        """

        :param model: 'Model' instance.
        :param parameter: 'String'.
            - 'population': Vary the population size.
            - 'mutation': Vary the Prdm9 mutation rate.
            - 'erosion': Vary the hotspots erosion rate.
            - 'recombination': Vary the hotspots recombination rate.
            - 'fitness: Vary the fitness parameter (alpha).
        :param nbr_of_simulations: 'Integer', the number of the independent simulations.
        :param scale: 'Integer', the range of variation of the parameter (can span several orders of magnitude).
        :param loops: Integer, the number of cycle (complete replacement of all Prdm9 alleles) in each 'Simulation'.
        """
        parameter_name_dict = {"fitness": "alpha",
                               "mutation": "mutation_rate_prdm9",
                               "erosion": "mutation_rate_hotspot",
                               "population": "population_size",
                               "recombination": "recombination_rate"}
        assert parameter in parameter_name_dict.keys(), "Axis must be either 'population', 'mutation', 'erosion'," \
                                                        "'recombination' or 'fitness'"
        assert scale > 1, "The scale parameter must be greater than one"
        self.parameter = parameter
        self.parameter_name = parameter_name_dict[parameter]
        self.scale = scale
        self.nbr_of_simulations = nbr_of_simulations
        self.loops = loops
        self.model = model.copy()

        self.simulations = []

        if self.parameter == "fitness" and self.model.fitness_family == 3:
            self.parameter_range = np.linspace(0.05, 0.95, nbr_of_simulations)
        else:
            self.parameter_range = np.logspace(
                np.log10(float(getattr(self.model, self.parameter_name)) / np.sqrt(scale)),
                np.log10(float(getattr(self.model, self.parameter_name)) * np.sqrt(scale)),
                nbr_of_simulations)
        for parameter_focal in self.parameter_range:
            model_copy = self.model.copy()
            setattr(model_copy, self.parameter_name, parameter_focal)
            model_copy.scaling_parameters()
            self.simulations.append(Simulation(model_copy, loops=self.loops))

    def caption(self):
        """
        Return a meaningful description of the simulations.
        :return: 'String'.
        """
        return "Batch of %s simulations. \n" % self.nbr_of_simulations + self.parameter_name + \
               "is scaled %.1e times.\n" % self.scale + self.model.caption()

    def run(self, nbr_of_cpu=7, directory_id=None):
        """
        Run all simulations. Runs in different CPU if nbr_of_cpu is strictly greater than 1.
        :param nbr_of_cpu: 'Integer', the number of CPU
        :param directory_id: 'String', the relative path to store the files
        :return: self.
        """
        if directory_id is None:
            directory_id = id_generator(8)
        set_dir("/" + directory_id + " " + self.parameter_name)
        if nbr_of_cpu > 1:
            pool = Pool(nbr_of_cpu)
            self.simulations = pool.map(execute, self.simulations)
            pool.close()
        else:
            map(lambda x: x.run(), self.simulations)

        self.pickle()
        self.save_figure()
        os.chdir('..')
        print 'Simulation computed'
        return self

    def save_figure(self):
        """
        For all simulations, plot and save (in svg format) the figures for self-consistent equation and
        phase plan (Prdm9 frequencies vs activity of hotspots).
        :return: None.
        """
        map(lambda x: x.save_figure(), self.simulations)

    def pickle(self):
        """
        Save the instance using Pickle in a .p file.
        :return: None
        """
        pickle.dump(self, open(self.parameter_name + ".p", "wb"))

    def plot_series(self, series, color, y_label):
        """
        Plot the mean and variance
        :param series: 'String',
        :param color: 'String', color in hex format (e.g. #6ABD9B)
        :param y_label: 'String', The label of the y-axis
        :return: None
        """
        parameter_caption = {"fitness": "The fitness parameter",
                             "mutation": "Mutation rate of PRDM9",
                             "erosion": "Mutation rate of the hotspots",
                             "population": "The population size",
                             "recombination": "The recombination rate of the hotspots"}[self.parameter]
        mean = map(lambda serie: np.mean(serie), series)
        plt.plot(self.parameter_range, mean, color=color, linewidth=2)
        sigma = map(lambda serie: 1.96*np.sqrt(np.var(serie)), series)
        y_max = np.add(mean, sigma)
        y_min = np.subtract(mean, sigma)
        plt.fill_between(self.parameter_range, y_max, y_min, color=color, alpha=0.3)
        plt.xlabel(parameter_caption)
        plt.ylabel(y_label)
        if self.parameter == "fitness" and self.model.fitness_family == 3:
            plt.xscale('linear')
        else:
            plt.xscale('log')


class Batch(list):
    """
    A list of 'SimulationsAlongParameter' instances.
    """
    def save_figures(self):
        """
        Plot and save (in svg format) all the summary statistics (R, D, V or R) as a function of parameters of the
        simulations (one plot per summary statistic). The parameter of the simulation are contained in the
        'SimulationsAlongParameter' instances and each plot contains one subplot for each instance of 'SimulationsAlongParameter'
        :return: None.
        """
        self.save_figure('mean_activity', 'linear')
        self.save_figure('prdm9_diversity', 'log')
        self.save_figure('landscape_variance', 'log')
        self.save_figure('turn_over', 'log')

    def save_figure(self, summary_statistic="mean_activity", yscale='log'):
        """
        Plot and save (in svg format) one of the summary statistics (R, D, V or R) as a function of parameters of the
        simulations. The parameter of the simulation are contained in the 'SimulationsAlongParameter' instances and
        the plot contains one subplot for each instance of 'SimulationsAlongParameter'
        :param summary_statistic: 'String', either 'mean_activity', 'prdm9_diversity', 'turn_over' or 'landscape_variance'.
            - 'mean_activity': mean activity of hotspots (R).
            - 'prdm9_diversity': Prdm9 diversity (D).
            - 'landscape_variance': landscape variance (V).
            - 'turn_over': turn-over time (T).
        :param yscale: 'String', must be 'log' or 'linear'.
        :return: self.
        """
        y_label = {"mean_activity": "The mean activity of the hotspots",
                   "prdm9_diversity": "The diversity of PRDM9",
                   "landscape_variance": "The hotspots landscape variance",
                   "turn_over": "The turn-over time"}[summary_statistic]
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        for j, batch in enumerate(self):
            plt.subplot(2, 2, j + 1)

            models = map(lambda sim: sim.model, batch.simulations)

            if summary_statistic == 'turn_over':
                lag = map(lambda sim: sim.generations[sim.trace.dichotomic_search(0.5)], batch.simulations)
                plt.plot(batch.parameter_range, lag, color=BLUE)
                plt.xscale('log')
            else:
                batch.plot_series(
                    map(lambda sim: np.array(getattr(sim.trace, summary_statistic + "_array")()), batch.simulations),
                    BLUE, y_label)
            for method, color in (("estimation", YELLOW), ("small_load", GREEN)):
                array = map(lambda model: getattr(model, summary_statistic + "_" + method)(), models)
                plt.plot(batch.parameter_range, array, color=color, linewidth=3)
                plt.yscale(yscale)

        plt.tight_layout()

        plt.savefig("%s-batch" % summary_statistic + '.svg', format="svg")
        plt.savefig("%s-batch" % summary_statistic + '.png', format="png")
        plt.clf()
        plt.close('all')
        print summary_statistic + ' computed'
        return self

    def pickle(self):
        """
        Save the instance using Pickle in a .p file.
        :return: None.
        """
        pickle.dump(self, open("Batch.p", "wb"))


def load_batch(dir_id):
    """
    :param dir_id: a string.
    :return: Given a directory ('dir_id') containing the results of a batch simulation,
        load the pickle file (Batch.p) and save the figure again.
        Could be used if changes have been made to the figure, without running the simulations again.
    """
    set_dir("/tmp/" + dir_id)
    simulation_batch = pickle.load(open("Batch.p", "rb"))
    simulation_batch.save_figures()
    map(lambda x: x.save_figure(), simulation_batch)


def make_batch():
    """
    Run simulations, then plot and save (in svg format) all the summary statistics (R, D, V or R) as a function of
    parameters of the simulations (effective population size, erosion rate, Prdm9 mutation rate, fitness parameter).
    :return: None.
    """
    set_dir("/tmp/" + id_generator(8))
    model = Model(mutation_rate_prdm9=1.0 * 10 ** -6,
                  mutation_rate_hotspot=1.0 * 10 ** -6,
                  population_size=10 ** 5,
                  recombination_rate=1.0 * 10 ** -3,
                  alpha=0.1,
                  fitness='polynomial', drift=True, linearized=True)
    batch = Batch()
    for parameter in ["population", "erosion", "mutation", "fitness"]:
        batch.append(SimulationsAlongParameter(model.copy(), parameter=parameter,
                                               nbr_of_simulations=64, scale=10 ** 4, loops=30))
    for simulation_along_parameter in batch:
        simulation_along_parameter.run(nbr_of_cpu=8)
    batch.pickle()
    batch.save_figures()

if __name__ == '__main__':
    # load_batch("0E1F0873")
    make_batch()
