from scipy.stats.kde import gaussian_kde
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
    return x.run()


def id_generator(number_of_char):
    return str(uuid.uuid4().get_hex().upper()[0:number_of_char])


def set_dir(path):
    path = os.getcwd() + path
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    os.chdir(path)


# The fitness function, given the erosion of the hotspots
def sum_to_one(array):
    return np.divide(array, np.sum(array))


class Model(object):
    def __init__(self, nbr_of_alleles, model_params, simu_params):
        self.t = 0
        self.id = nbr_of_alleles
        self.ids = np.array(range(nbr_of_alleles))
        self.prdm9_polymorphism = sum_to_one(np.random.sample(nbr_of_alleles))
        self.hotspots_erosion = np.random.sample(nbr_of_alleles)
        self.prdm9_longevity = np.zeros(nbr_of_alleles)
        self.prdm9_fitness = np.ones(nbr_of_alleles)
        self.model_params = model_params
        self.population_size = model_params.population_size
        self.prdm9_polymorphism *= self.population_size
        self.mutation_rate = model_params.population_size * model_params.mutation_rate_prdm9
        self.erosion_rate = model_params.erosion_rate_hotspot * model_params.population_size * \
                            model_params.recombination_rate
        assert self.erosion_rate < 0.5, "The scaled erosion rate is too large, decrease either the " \
                                        "recombination rate, the erosion rate or the population size"
        assert self.erosion_rate > 0.0000000001, "The scaled erosion rate is too low, increase either the " \
                                                 "recombination rate, the erosion rate or the population size"
        self.linearized = simu_params.linearized
        self.drift = simu_params.drift

    def forward(self):
        new_alleles = np.random.poisson(self.mutation_rate)
        if new_alleles > 0:
            self.new_alleles(new_alleles)
        prdm9_frequencies = sum_to_one(self.prdm9_polymorphism)

        self.hotspots_erosion *= np.exp(- self.erosion_rate * prdm9_frequencies)

        # Compute the fitness for each allele
        if self.model_params.fitness_family == 0:
            distribution_vector = prdm9_frequencies
        else:
            if self.linearized:
                l_bar = np.sum(prdm9_frequencies * self.hotspots_erosion)
                self.prdm9_fitness = self.model_params.coefficient_fitness(l_bar) * (self.hotspots_erosion - l_bar)
                distribution_vector = prdm9_frequencies + self.prdm9_fitness * prdm9_frequencies
                if np.max(distribution_vector) > 1.:
                    distribution_vector = sum_to_one(np.clip(distribution_vector, 0., 1.))
                elif np.min(distribution_vector) < 0.:
                    distribution_vector = sum_to_one(np.clip(distribution_vector, 0., 1.))
            else:
                fitness_matrix = self.model_params.fitness(
                    np.add.outer(self.hotspots_erosion, self.hotspots_erosion) / 2)
                self.prdm9_fitness = np.dot(fitness_matrix, prdm9_frequencies)
                distribution_vector = sum_to_one(self.prdm9_fitness * prdm9_frequencies)

        if self.drift:
            self.prdm9_polymorphism = np.random.multinomial(int(self.population_size), distribution_vector).astype(
                float)
        else:
            self.prdm9_polymorphism = distribution_vector * self.population_size

        self.prdm9_longevity += prdm9_frequencies
        self.remove_dead_prdm9()

    def new_alleles(self, new_alleles):
        self.prdm9_polymorphism -= np.random.multinomial(new_alleles, sum_to_one(self.prdm9_polymorphism))
        self.prdm9_polymorphism = np.append(self.prdm9_polymorphism, np.ones(new_alleles))
        self.prdm9_longevity = np.append(self.prdm9_longevity, np.zeros(new_alleles))
        self.hotspots_erosion = np.append(self.hotspots_erosion, np.ones(new_alleles))
        self.ids = np.append(self.ids, range(self.id, self.id + new_alleles))
        self.id += new_alleles

    def remove_dead_prdm9(self):
        remove_extincted = np.array(map(lambda x: x > 0, self.prdm9_polymorphism), dtype=bool)
        if not remove_extincted.all():
            self.prdm9_polymorphism = self.prdm9_polymorphism[remove_extincted]
            self.hotspots_erosion = self.hotspots_erosion[remove_extincted]
            self.prdm9_longevity = self.prdm9_longevity[remove_extincted]
            self.prdm9_fitness = self.prdm9_fitness[remove_extincted]
            self.ids = self.ids[remove_extincted]

    def __repr__(self):
        return "The polymorphism of PRDM9: %s" % self.prdm9_polymorphism + \
               "\nThe strength of the hotspots : %s" % self.hotspots_erosion + \
               "\nThe longevity hotspots : %s" % self.prdm9_longevity


class SimulationData(object):
    def __init__(self):
        self.prdm9_frequencies, self.hotspots_erosion = [], []
        self.prdm9_fitness, self.prdm9_longevity = [], []

        self.nbr_prdm9, self.ids = [], []

    def store(self, step):
        self.prdm9_frequencies.append(sum_to_one(step.prdm9_polymorphism))
        self.hotspots_erosion.append(step.hotspots_erosion)
        self.prdm9_fitness.append(step.prdm9_fitness)
        self.prdm9_longevity.append(step.prdm9_longevity)
        self.nbr_prdm9.append(step.prdm9_polymorphism)
        self.ids.append(step.ids)

    def flat_frequencies(self):
        return list(itertools.chain.from_iterable(self.prdm9_frequencies))

    def flat_erosion(self):
        return list(itertools.chain.from_iterable(self.hotspots_erosion))

    def flat_fitness(self):
        return list(itertools.chain.from_iterable(self.prdm9_fitness))

    def flat_longevity(self):
        return list(itertools.chain.from_iterable(self.prdm9_longevity))

    def shannon_entropy_prdm9(self):
        return map(lambda freq: self.hill_diversity(freq, 1), self.prdm9_frequencies)

    def simpson_entropy_prdm9(self):
        return map(lambda freq: self.hill_diversity(freq, 2), self.prdm9_frequencies)

    def hotspots_erosion_mean(self):
        return map(lambda erosion, freq: self.n_moment(erosion, freq, 1), self.hotspots_erosion,
                   self.prdm9_frequencies)

    def hotspots_erosion_array(self):
        return np.array(self.hotspots_erosion_mean())

    def hotspots_landscape(self):
        return map(lambda erosion, freq: self.n_moment(freq, erosion, 2), self.hotspots_erosion,
                   self.prdm9_frequencies)

    def hotspots_erosion_var(self):
        return map(lambda erosion, freq: self.var(erosion, freq), self.hotspots_erosion,
                   self.prdm9_frequencies)

    def hotspots_erosion_normalized_var(self):
        return map(lambda erosion, freq: self.normalized_var(erosion, freq), self.hotspots_erosion,
                   self.prdm9_frequencies)

    def prdm9_longevity_mean(self):
        return map(lambda longevity, freq: self.n_moment(longevity, freq, 1), self.prdm9_longevity,
                   self.prdm9_frequencies)

    def prdm9_longevity_var(self):
        return map(lambda longevity, freq: self.normalized_var(longevity, freq), self.prdm9_longevity,
                   self.prdm9_frequencies)

    def var(self, x, frequencies):
        return self.n_moment(x, frequencies, 2) - self.n_moment(x, frequencies, 1) ** 2

    def normalized_var(self, x, frequencies):
        squared_first_moment = self.n_moment(x, frequencies, 1) ** 2
        return (self.n_moment(x, frequencies, 2) - squared_first_moment) / squared_first_moment

    def hill_diversity(self, frequencies, q):
        assert q >= 0, "q must be positive"
        if q == 1:
            return self.entropy(frequencies)
        elif q == 0:
            return len(frequencies)
        elif q == 2:
            return 1. / np.sum(np.power(frequencies, 2))
        else:
            return np.power(np.sum(np.power(frequencies, q)), 1. / (1. - q))

    def mean_erosion(self):
        return np.mean(self.hotspots_erosion_mean())

    def mean_simpson_entropy(self):
        return np.mean(self.simpson_entropy_prdm9())

    def normed_cross_homozygosity(self, lag, homozygosity=0):
        if homozygosity == 0:
            homozygosity = self.cross_homozygosity(0)
        return self.cross_homozygosity(lag) / homozygosity

    def cross_homozygosity(self, lag):
        cross_homozygosity = []
        for index in range(0, len(self.ids) - lag):
            cross_homozygosity.append(0.)
            slice_dict = dict(zip(self.ids[index], self.prdm9_frequencies[index]))
            lag_dict = dict(zip(self.ids[index + lag], self.prdm9_frequencies[index + lag]))
            for key in (set(slice_dict.keys()) & set(lag_dict.keys())):
                cross_homozygosity[index] += slice_dict[key] * lag_dict[key]

        assert len(cross_homozygosity) > 0, "Cross Homozygosity is empty"
        return np.mean(cross_homozygosity)

    def dichotomic_search(self, percent):
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

    @staticmethod
    def entropy(frequencies):
        return np.exp(np.sum(map(lambda x: (-x * np.log(x) if x > 0 else 0), frequencies)))

    @staticmethod
    def n_moment(x, frequencies, n):
        if n > 1:
            return np.sum((x ** n) * frequencies)
        else:
            return np.sum(x * frequencies)


class ModelParams(object):
    def __init__(self, mutation_rate_prdm9=1.0 * 10 ** -5, erosion_rate_hotspot=1.0 * 10 ** -7, population_size=10 ** 4,
                 recombination_rate=1.0 * 10 ** -3, fitness_param=1., fitness='linear'):
        self.mutation_rate_prdm9 = mutation_rate_prdm9  # The rate at which new allele for PRDM9 are created
        self.erosion_rate_hotspot = erosion_rate_hotspot  # The rate at which the hotspots are eroded
        self.recombination_rate = recombination_rate
        self.population_size = float(population_size)  # population size

        fitness_hash = {"linear": 1, "polynomial": 2, "sigmoid": 3}
        assert fitness in fitness_hash.keys(), "Parameter 'fitness' must be a string: ['linear','sigmoid','polynomial']"
        self.fitness_family = fitness_hash[fitness]
        self.fitness_param = fitness_param
        if self.fitness_family == 1:
            self.fitness_param = 1.
        self.k = 2

    def fitness(self, x):
        if self.fitness_family == 3:
            return np.power(x, self.k) / (np.power(x, self.k) + np.power(self.fitness_param, self.k))
        elif self.fitness_family == 2:
            return np.power(x, self.fitness_param)
        else:
            return x

    def coefficient_fitness(self, x):
        if x == 0:
            return float("inf")
        else:
            if self.fitness_family == 3:
                return (self.k / 2) * np.power(self.fitness_param, self.k) / (
                    x * (np.power(x, self.k) + np.power(self.fitness_param, self.k)))
            elif self.fitness_family == 2:
                return self.fitness_param * 1.0 / (2. * x)
            else:
                return 1.0 / (2 * x)

    def ratio_parameter(self):
        return 2 * self.erosion_rate_hotspot * self.recombination_rate

    def params_mean_erosion(self, approximated=False):
        if approximated:
            param = self.erosion_rate_hotspot * self.recombination_rate / (
                self.mutation_rate_prdm9 * self.fitness_param * 2)
            return 1 - np.sqrt(param)
        else:
            return brentq(lambda x: self.variation_mean_erosion(x), 0, 1)

    def lx_bar(self, x):
        if x == 0.:
            return 0.
        else:
            b = self.coefficient_fitness(x) / self.ratio_parameter()
            return x * b * (1 - 2 * x + self.erosion_limit(x)) / 2

    def variation_mean_erosion(self, x):
        rate_in = self.mutation_rate_prdm9 * (1 - x) * (1 - self.erosion_limit(x)) * self.coefficient_fitness(x) * 2
        rate_out = self.erosion_rate_hotspot * self.recombination_rate * x
        return rate_in - rate_out

    @staticmethod
    def erosion_limit(mean_erosion):
        if mean_erosion == 0.:
            return 0.
        else:
            return -1 * mean_erosion * np.real(lambertw(-np.exp(- 1. / mean_erosion) / mean_erosion))

    def estimated_landscape(self, mean_erosion):
        b = self.coefficient_fitness(mean_erosion)
        a = self.erosion_rate_hotspot * self.recombination_rate * self.population_size
        landscape = mean_erosion * b * (1 - 2 * mean_erosion + self.erosion_limit(mean_erosion)) / (2 * a)
        return min(1., landscape)

    def estimated_params_landscape(self):
        return self.estimated_landscape(self.params_mean_erosion())

    def series_landscape(self):
        return 1. / self.series_simpson()

    def estimated_simpson(self, mean_erosion):
        if mean_erosion == 1.:
            return self.population_size
        else:
            denom = 1 - 2 * mean_erosion + self.erosion_limit(mean_erosion)
            simpson = (2 * self.erosion_rate_hotspot * self.recombination_rate * self.population_size) / (
                self.coefficient_fitness(mean_erosion) * denom)
            return max(1., simpson)

    def estimated_params_simpson(self):
        return self.estimated_simpson(self.params_mean_erosion())

    def series_simpson(self):
        return max(1., 4 * 3. * self.mutation_rate_prdm9 * self.population_size)

    def estimated_erosion_var(self, mean_erosion):
        return mean_erosion * (1 - 2 * mean_erosion + self.erosion_limit(mean_erosion)) / 2

    def turn_over_neutral(self):
        return 1 / self.mutation_rate_prdm9

    def fixation_new_variant(self, mean_erosion):
        if mean_erosion == 0.:
            return 1.
        elif mean_erosion == 1.:
            return 1. / self.population_size
        else:
            selection = self.coefficient_fitness(mean_erosion) * (1. - mean_erosion)
            return (1. - np.exp(-selection)) / (1. - np.exp(-2. * self.population_size * selection))

    def estimated_turn_over(self, mean_erosion):
        return self.estimated_simpson(mean_erosion) / (2 * self.fixation_new_variant(mean_erosion) *
                                                       self.mutation_rate_prdm9 * self.population_size)

    def estimated_params_turn_over(self):
        return self.estimated_turn_over(self.params_mean_erosion())

    def series_turn_over(self):
        param = self.mutation_rate_prdm9 / (self.fitness_param * self.recombination_rate * self.erosion_rate_hotspot)
        return 12 * np.sqrt(2 * param)

    def frequencies_wrt_erosion(self, mean_erosion):
        l_limit = self.erosion_limit(mean_erosion)
        l = np.linspace(l_limit, 1)
        x = 1 - l + mean_erosion * np.log(l)
        x *= self.coefficient_fitness(mean_erosion) / (
            self.erosion_rate_hotspot * self.recombination_rate * self.population_size)
        return l, np.clip(x, 0., 1.)

    def __str__(self):
        name = "u=%.1e" % self.mutation_rate_prdm9 + \
               "_v=%.1e" % self.erosion_rate_hotspot + \
               "_r=%.1e" % self.recombination_rate + \
               "_n=%.1e" % self.population_size
        if self.fitness_family == 3 or self.fitness_family == 2:
            name += "_f=%.1e" % self.fitness_param
        return name

    def caption(self):
        caption = "Mutation rate of PRDM9: %.1e. \n" % self.mutation_rate_prdm9 + \
                  "Mutation rate of the hotspots: %.1e. \n" % self.erosion_rate_hotspot + \
                  "Recombination rate at the hotspots: %.1e. \n" % self.recombination_rate + \
                  "Population size: %.1e. \n" % self.population_size
        if self.fitness_family == 3:
            caption += "Inflexion point of the fitness function PRDM9=%.1e. \n" % self.fitness_param
        if self.fitness_family == 2:
            caption += "Exponent of the polynomial fitness function PRDM9=%.1e. \n" % self.fitness_param
        return caption

    def copy(self):
        return copy.copy(self)


class SimulationParams(object):
    def __init__(self, drift=True, linearized=True, scaling=10):
        self.scaling = scaling
        self.drift = drift
        self.linearized = linearized

    def __str__(self):
        return "scale=%.1e" % self.scaling + \
               "_drift=%s" % self.drift + \
               "_linear=%s" % self.linearized

    def caption(self):
        caption = self.caption_for_attr('drift')
        caption += self.caption_for_attr('linearized')
        return caption

    def caption_for_attr(self, attr):
        if attr == 'drift':
            if getattr(self, attr):
                return "  The simulation take into account DRIFT. \n"
            else:
                return "  The simulation DOESN'T take into account DRIFT. \n"
        elif attr == 'linearized':
            if getattr(self, attr):
                return "  The simulation use a LINEARIZED APPROXIMATION for the fitness. \n"
            else:
                return "  The simulation DOESN'T APPROXIMATE the fitness. \n"

    def copy(self):
        return copy.copy(self)


class Simulation(object):
    def __init__(self, model_params, simu_params):
        self.model_params = model_params
        self.simu_params = simu_params
        self.t_max = 0
        self.nbr_of_steps = 0.
        self.model = Model(10, model_params, simu_params)
        self.data = SimulationData()
        self.generations = []

    def __str__(self):
        return "tmax=%.1e_" % self.t_max + str(self.model_params) + "_" + str(self.simu_params)

    def caption(self):
        return "%.1e generations computed \n \n" % self.t_max + \
               "Model parameters : \n" + self.model_params.caption() + "\n" + \
               "Simulation parameters : \n" + self.simu_params.caption()

    def burn_in(self):
        print "Entering burn-in"
        t = 0
        initial_variants = set(self.model.ids)
        while len(initial_variants & set(self.model.ids)) > 0:
            self.model.forward()
            t += 1
        self.nbr_of_steps = 10000 / len(self.model.ids)
        self.t_max = 10 * (int(max(int(self.simu_params.scaling * t), self.nbr_of_steps)) / 10 + 1)
        print "Burn-in Completed"

    def run(self):
        self.burn_in()

        step_t = 0.

        step = float(self.t_max) / self.nbr_of_steps
        for t in range(self.t_max):
            self.model.forward()

            step_t += 1
            if step_t > step:
                step_t -= step

                self.generations.append(t)
                self.data.store(self.model)

            if int(10 * t) % self.t_max == 0:
                print "Computation at {0}%".format(float(100 * t) / self.t_max)

        return self

    def save_trajectory(self):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1440 / my_dpi), dpi=my_dpi)

        generations = list(itertools.chain.from_iterable(map(lambda x, y: [x] * len(y),
                                                             self.generations, self.data.prdm9_frequencies)))
        xlim = [min(generations), max(generations)]

        plt.subplot(311)
        plt.plot(self.generations, self.data.simpson_entropy_prdm9(), color=YELLOW, lw=3)

        plt.xlabel('Generations')
        plt.ylim([1, np.max(self.data.simpson_entropy_prdm9())])
        plt.xlim(xlim)
        plt.ylabel('PRDM9 diversity')

        plt.subplot(312)
        for my_id in range(self.model.id):
            array = np.zeros(len(self.generations))
            for t in range(len(self.generations)):
                if my_id in self.data.ids[t]:
                    array[t] = self.data.prdm9_frequencies[t][self.data.ids[t].tolist().index(my_id)]

            plt.plot(self.generations, array, color=BLUE, lw=3)
        plt.xlabel('Generations')
        plt.ylim([0, 1])
        plt.xlim(xlim)
        plt.ylabel('PRDM9 frequencies')

        plt.subplot(313)
        plt.scatter(generations, self.data.flat_erosion(), color=BLUE, lw=0)
        plt.plot(self.generations, self.data.hotspots_erosion_mean(), color=YELLOW, lw=3)
        plt.xlabel('Generations')
        plt.ylabel('Hotspot activity')
        plt.ylim([0, 1])
        plt.xlim(xlim)

        plt.tight_layout()

        plt.savefig('trajectory.png', format="png")
        plt.savefig('trajectory.svg', format="svg")
        print "Trajectory computed"
        plt.clf()
        return str(self)

    def pickle(self):
        pickle.dump(self, open(str(self) + ".p", "wb"))

    def estimated_simpson(self):
        return self.model_params.estimated_simpson(self.data.mean_erosion())

    def estimated_params_simpson(self):
        return self.model_params.estimated_params_simpson()

    def series_simpson(self):
        return self.model_params.series_simpson()

    def estimated_landscape(self):
        return self.model_params.estimated_landscape(self.data.mean_erosion())

    def estimated_params_landscape(self):
        return self.model_params.estimated_params_landscape()

    def series_landscape(self):
        return self.model_params.series_landscape()

    def estimated_erosion_var(self):
        return self.model_params.estimated_erosion_var(self.data.mean_erosion())

    def estimated_turn_over(self):
        return self.model_params.estimated_turn_over(self.data.mean_erosion())

    def estimated_params_turn_over(self):
        return self.model_params.estimated_params_turn_over()

    def series_turn_over(self):
        return self.model_params.series_turn_over()


class BatchSimulation(object):
    def __init__(self,
                 model_params,
                 simu_params,
                 axis="null",
                 nbr_of_simulations=20,
                 scale=10 ** 2):
        axis_hash = {"fitness": "The fitness parameter",
                     "mutation": "Mutation rate of PRDM9",
                     "erosion": "Mutation rate of the hotspots",
                     "population": "The population size",
                     "recombination": "The recombination rate of the hotspots",
                     "scaling": "The scaling factor"}
        assert axis in axis_hash.keys(), "Axis must be either 'population', 'mutation', 'erosion'," \
                                         "'recombination', 'fitness', or 'scaling'"
        assert scale > 1, "The scale parameter must be greater than one"
        self.axis = axis
        self.axis_str = axis_hash[axis]
        self.scale = scale
        self.nbr_of_simulations = nbr_of_simulations
        self.model_params = model_params.copy()
        self.simu_params = simu_params.copy()

        self.simulations = []

        if self.axis == "scaling":
            self.axis_range = np.logspace(np.log10(1. / np.sqrt(scale)),
                                          np.log10(1. * np.sqrt(scale)), nbr_of_simulations)
        elif self.axis == "fitness" and self.model_params.fitness_family == 3:
            self.axis_range = np.linspace(0.05, 0.95, nbr_of_simulations)
        else:
            self.axis_range = np.logspace(
                np.log10(float(getattr(self.model_params, self.variable_hash())) / np.sqrt(scale)),
                np.log10(float(getattr(self.model_params, self.variable_hash())) * np.sqrt(scale)),
                nbr_of_simulations)
        for axis_current in self.axis_range:
            model_params_copy = self.model_params.copy()
            setattr(model_params_copy, self.variable_hash(), axis_current)
            self.simulations.append(Simulation(model_params_copy, simu_params.copy()))

    def variable_hash(self):
        return {"fitness": "fitness_param",
                "mutation": "mutation_rate_prdm9",
                "erosion": "erosion_rate_hotspot",
                "population": "population_size",
                "recombination": "recombination_rate",
                "scaling": "scaling"}[self.axis]

    def caption(self):
        return "Batch of %s simulations. \n" % self.nbr_of_simulations + self.axis_str + \
               "is scaled %.1e times.\n" % self.scale + self.simu_params.caption() + self.model_params.caption()

    def run(self, nbr_of_cpu=7, directory_id=None):
        if directory_id is None:
            directory_id = id_generator(8)
        set_dir("/" + directory_id + " " + self.axis_str)
        if nbr_of_cpu > 1:
            pool = Pool(nbr_of_cpu)
            self.simulations = pool.map(execute, self.simulations)
            pool.close()
        else:
            map(lambda x: x.run(), self.simulations)

        self.pickle()
        os.chdir('..')
        print 'Simulation computed'

    def pickle(self):
        pickle.dump(self, open(self.axis_str + ".p", "wb"))

    def plot_series(self, series, color, caption, title=True):
        mean = map(lambda serie: np.mean(serie), series)
        plt.plot(self.axis_range, mean, color=color, linewidth=2)
        sigma = map(lambda serie: np.sqrt(np.var(serie)), series)
        y_max = np.add(mean, sigma)
        y_min = np.subtract(mean, sigma)
        plt.fill_between(self.axis_range, y_max, y_min, color=color, alpha=0.3)
        if title:
            plt.title('{0} for different {1}'.format(caption, self.axis_str))
        plt.xlabel(self.axis_str)
        plt.ylabel(caption)
        if self.axis == "fitness" and self.model_params.fitness_family == 3:
            plt.xscale('linear')
        else:
            plt.xscale('log')


class Batches(list):
    def save_figure(self):
        self.save_erosion()
        self.save_simpson()
        self.save_turn_over()
        self.save_landscape()

    def save_erosion(self):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        for j, batch in enumerate(self):
            plt.subplot(2, 2, j + 1)

            models_params = map(lambda sim: sim.model_params, batch.simulations)

            batch.plot_series(
                map(lambda sim: np.array(sim.data.hotspots_erosion_array()), batch.simulations),
                    BLUE, 'Mean activity of the hotspots',
                    title=False)
            mean_erosion = map(lambda model_param: model_param.params_mean_erosion(), models_params)
            plt.plot(batch.axis_range, mean_erosion, color=YELLOW, linewidth=3)
            plt.yscale('linear')

        plt.tight_layout()

        plt.savefig('batch-erosion-mean.svg', format="svg")
        plt.clf()
        print 'Erosion Mean computed'
        return self

    def save_simpson(self):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        for j, batch in enumerate(self):
            plt.subplot(2, 2, j + 1)

            batch.plot_series(map(lambda sim: sim.data.simpson_entropy_prdm9(), batch.simulations),
                              BLUE, 'Diversity of PRDM9', title=False)

            simu_simpson = map(lambda sim: sim.estimated_params_simpson(), batch.simulations)
            plt.plot(batch.axis_range, simu_simpson, color=YELLOW, linewidth=3)

            plt.yscale('log')

        plt.tight_layout()

        plt.savefig('batch-simpson.svg', format="svg")
        plt.clf()
        print 'Simpson computed'
        return self

    def save_landscape(self):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        for j, batch in enumerate(self):
            plt.subplot(2, 2, j + 1)

            batch.plot_series(map(lambda sim: sim.data.hotspots_landscape(), batch.simulations),
                              BLUE, 'Landscape of hotspots', title=False)
            simu_simpson = map(lambda sim: sim.estimated_params_landscape(), batch.simulations)
            plt.plot(batch.axis_range, simu_simpson, color=YELLOW, linewidth=3)

            plt.yscale('log')

        plt.tight_layout()

        plt.savefig('batch-landscape.svg', format="svg")
        plt.clf()
        print 'Lanscape computed'
        return self

    def save_turn_over(self):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        for j, batch in enumerate(self):
            plt.subplot(2, 2, j + 1)

            lag = map(lambda sim: sim.generations[sim.data.dichotomic_search(0.5)], batch.simulations)
            plt.plot(batch.axis_range, lag, color=BLUE)

            estimated_turn_over = map(lambda sim: sim.estimated_params_turn_over(), batch.simulations)
            plt.plot(batch.axis_range, estimated_turn_over, color=YELLOW, linewidth=3)

            plt.yscale('linear')

            plt.xlabel(batch.axis_str)
            plt.ylabel('Turn-over of PRDM9')
            plt.yscale('log')
            if batch.axis == "fitness" and batch.model_params.fitness_family == 3:
                plt.xscale('linear')
            else:
                plt.xscale('log')

        plt.tight_layout()

        plt.savefig('batch-turn-over.svg', format="svg")
        plt.clf()
        print 'Turn-over computed'
        return self

    def pickle(self):
        pickle.dump(self, open("batches.p", "wb"))


def load_batches(dir_id):
    set_dir("/tmp/" + dir_id)
    batches = pickle.load(open("batches.p", "rb"))
    batches.save_figure()


def make_batches():
    set_dir("/tmp/" + id_generator(8))
    model_parameters = ModelParams(mutation_rate_prdm9=1.0 * 10 ** -6,
                                   erosion_rate_hotspot=1.0 * 10 ** -8,
                                   population_size=10 ** 5,
                                   recombination_rate=1.0 * 10 ** -3,
                                   fitness_param=0.1,
                                   fitness='polynomial')
    simu_params = SimulationParams(drift=True, linearized=True, scaling=30)
    batches = Batches()
    for axis in ["population", "erosion", "mutation", "fitness"]:
        batches.append(BatchSimulation(model_parameters.copy(), simu_params.copy(), axis=axis,
                                       nbr_of_simulations=16, scale=10 ** 4))
    for batch_simulation in batches:
        batch_simulation.run(nbr_of_cpu=8)
    batches.pickle()
    batches.save_figure()


def make_trajectory():
    set_dir("/tmp/")
    model_parameters = ModelParams(mutation_rate_prdm9=1.0 * 10 ** -5,
                                   erosion_rate_hotspot=1.0 * 10 ** -4,
                                   population_size=10 ** 4,
                                   recombination_rate=1.0 * 10 ** -3,
                                   fitness_param=0.1,
                                   fitness='polynomial')
    simulation_params = SimulationParams(drift=True, linearized=False, scaling=10)
    simulation = Simulation(model_parameters, simulation_params, )
    simulation.run()
    simulation.save_trajectory()


if __name__ == '__main__':
    # load_batches("931B53EB")
    # load_batches("AEA2940F")
    make_batches()
