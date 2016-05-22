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
from scipy import interpolate


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

    def hotspots_erosion_bound(self):
        return np.array(self.hotspots_erosion_mean())

    def hotspots_erosion_unbound(self):
        l = np.array(self.hotspots_erosion_mean())
        return (1 - l) / l

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

    def ratio_parameter(self, balance=1.):
        return balance * 2 * self.erosion_rate_hotspot * self.recombination_rate

    def bound_mean_erosion(self, balance=1.):
        return brentq(lambda x: self.variation_mean_erosion(x, balance), 0, 1)

    def lx_bar(self, x, balance=1):
        if x == 0.:
            return 0.
        else:
            b = self.coefficient_fitness(x) / self.ratio_parameter(balance)
            return x * b * (1 - 2 * x + self.erosion_limit(x)) / 2

    def variation_mean_erosion(self, x, balance=1.):
        rate_in = self.mutation_rate_prdm9 * (1 - x) * self.fixation_new_variant(x) * 4
        rate_out = self.ratio_parameter(0.5) * x
        return rate_in - rate_out

    @staticmethod
    def erosion_limit(mean_erosion):
        if mean_erosion == 0.:
            return 0.
        else:
            return -1 * mean_erosion * np.real(lambertw(-np.exp(- 1. / mean_erosion) / mean_erosion))

    def estimated_simpson(self, mean_erosion):
        if mean_erosion == 1.:
            return self.population_size
        else:
            denom = 1 - 2 * mean_erosion + self.erosion_limit(mean_erosion)
            simpson = (2 * self.erosion_rate_hotspot * self.recombination_rate * self.population_size) / (
                self.coefficient_fitness(mean_erosion) * denom)
            return max(1., simpson)

    def estimated_params_simpson(self):
        return self.estimated_simpson(self.bound_mean_erosion())

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
            selection = self.coefficient_fitness(mean_erosion) * (1 - mean_erosion)
            return (1 - np.exp(-selection)) / (1 - np.exp(-2 * self.population_size * selection))

    def turn_over_selection(self, mean_erosion):
        return 1. / self.fixation_new_variant(mean_erosion)

    def turn_over_derivative(self, mean_erosion):
        return -1. / (self.coefficient_fitness(mean_erosion) * (mean_erosion - 1 + mean_erosion * np.log(mean_erosion)))

    def turn_over_params_derivative(self):
        return self.turn_over_derivative(self.bound_mean_erosion())

    def turn_over_max(self):
        return max(1, 1 / (self.mutation_rate_prdm9 * self.population_size))

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
    def __init__(self, drift=True, linearized=True, color="blue", scaling=10):
        self.scaling = scaling
        self.drift = drift
        self.linearized = linearized
        self.color = color

    def __str__(self):
        return "scale=%.1e" % self.scaling + \
               "_drift=%s" % self.drift + \
               "_linear=%s" % self.linearized + \
               "_color=%s" % self.color

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


class BatchParams(list):
    def __init__(self, **kwargs):
        super(BatchParams, self).__init__()
        simu_params = SimulationParams(**kwargs)
        self.append(simu_params)
        self.caption_str = simu_params.color.upper() + " : " + simu_params.caption()

    def append_simu_params(self, switch_paremeters_dico):
        for color, params in switch_paremeters_dico.iteritems():
            assert color not in [i.color for i in self], "The color is already used"
            copy_self = copy.copy(self[0])
            setattr(copy_self, params, not getattr(self[0], params))
            copy_self.color = color
            self.append(copy_self)
            self.caption_str += copy_self.color.upper() + " : " + copy_self.caption_for_attr(params)

    def __str__(self):
        return ", ".join(map(lambda x: x.color, self))

    def caption(self):
        return self.caption_str

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

        self.save_figure()
        return self

    def plot_histogram(self, lst, x):
        assert len(lst) > 0, "Lst is empty"
        if (not lst) or lst.count(lst[0]) == len(lst):
            plt.hist(lst, color=self.simu_params.color, alpha=0.3)
        else:
            s_density = gaussian_kde(lst)(x)
            plt.plot(x, s_density, color=self.simu_params.color)
            plt.fill_between(x, s_density, np.zeros(100), color=self.simu_params.color, alpha=0.3)
        plt.text(0.5, 0.5, "Mean = {0:.1e}  \nVar = {1:.1e}".format(np.mean(lst), np.var(lst)), fontsize=12,
                 verticalalignment='top')

    def save_trajectory(self):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        generations = list(itertools.chain.from_iterable(map(lambda x, y: [x] * len(y),
                                                             self.generations, self.data.prdm9_frequencies)))
        xlim = [min(generations), max(generations)]
        plt.subplot(211)
        plt.scatter(generations, self.data.flat_frequencies())
        plt.title('PRDM9 frequencies over time. \n')
        plt.xlabel('Generations')
        plt.ylim([0, 1])
        plt.xlim(xlim)
        plt.ylabel('PRDM9 frequency')

        plt.subplot(212)
        plt.scatter(generations, self.data.flat_erosion())
        plt.title('Hotspots erosion over time. \n')
        plt.xlabel('Generations')
        plt.ylabel('Hotspot erosion')
        plt.ylim([0, 1])
        plt.xlim(xlim)

        plt.tight_layout()

        plt.savefig('trajectory.png')
        print "Trajectory computed"
        plt.clf()
        return str(self)

    def save_figure(self):
        mean_erosion = self.data.mean_erosion()
        params_mean_erosion = self.model_params.bound_mean_erosion()
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
        plt.subplot(331)
        plt.text(0.05, 0.98, self.caption(), fontsize=10, verticalalignment='top')
        theta = np.arange(0.0, 1.0, 0.01)
        plt.plot(theta, self.model_params.fitness(theta), color=self.simu_params.color)
        plt.title('The fitness function')
        plt.xlabel('x')
        plt.ylabel('w(x)')

        plt.subplot(332)
        plt.plot(self.generations, self.data.simpson_entropy_prdm9(), color=self.simu_params.color)
        simpson = self.model_params.estimated_simpson(mean_erosion)
        params_simpson = self.model_params.estimated_simpson(params_mean_erosion)
        plt.plot((self.generations[0], self.generations[-1]), (simpson, simpson), 'k-', color="green")
        plt.plot((self.generations[0], self.generations[-1]), (params_simpson, params_simpson), 'k-', color="orange")
        plt.title('Efficient nbr of PRDM9 alleles over time. \n')
        plt.xlabel('Generation')
        plt.ylabel('Number of alleles')
        plt.yscale('log')

        plt.subplot(333)
        plt.plot(self.generations, self.data.hotspots_erosion_mean(), color=self.simu_params.color)
        plt.plot((self.generations[0], self.generations[-1]), (params_mean_erosion, params_mean_erosion), 'k-',
                 color="orange")
        plt.title('Mean erosion of the hotspots over time. \n')
        plt.xlabel('Generation')
        plt.ylabel('Mean erosion')
        plt.yscale('linear')

        plt.subplot(334)
        plt.hexbin(self.data.flat_erosion(), self.data.flat_fitness(), self.data.flat_longevity(),
                   gridsize=200, bins='log')
        plt.title('PRDM9 fitness vs hotspot erosion')
        plt.xlabel('hotspot erosion')
        plt.ylabel('PRMD9 fitness')

        plt.subplot(335)
        plt.hexbin(self.data.flat_erosion(), self.data.flat_frequencies(), self.data.flat_longevity(),
                   gridsize=200, bins='log')
        plt.plot(*self.model_params.frequencies_wrt_erosion(mean_erosion), color="green")
        plt.plot(*self.model_params.frequencies_wrt_erosion(params_mean_erosion), color="orange")
        plt.title('PRMD9 frequency vs hotspot erosion')
        plt.xlabel('hotspot erosion')
        plt.ylabel('PRMD9 frequency')

        plt.subplot(336)
        lag_max = self.data.dichotomic_search(0.01)
        num = min(30, lag_max)
        longevities = np.linspace(0, self.generations[lag_max], num)
        lags = np.linspace(0, lag_max, num)
        ch_0 = self.data.cross_homozygosity(0)
        plt.plot(longevities, map(lambda lag: self.data.cross_homozygosity(int(lag)) / ch_0, lags))
        generation_5 = self.generations[self.data.dichotomic_search(0.05)]
        plt.plot((generation_5, generation_5), (0, 1), 'k-', color="black")
        plt.title('Cross correlation of homozygosity')
        plt.xlabel('Generations')
        plt.ylabel('Cross correlation of homozygosity')

        plt.subplot(337)
        x = np.linspace(0, 1, 100)
        self.plot_histogram(self.data.flat_frequencies(), x)
        plt.title('PRDM9 frequencies histogram')
        plt.xlabel('PRDM9 Frequencies')
        plt.ylabel('Frequency')

        plt.subplot(338)
        x = np.linspace(0, 1, 100)
        self.plot_histogram(self.data.flat_erosion(), x)
        plt.title('Erosion of the hotspots histogram')
        plt.xlabel('Erosion of the hotspots')
        plt.ylabel('Frequency')

        plt.subplot(339)
        x = np.linspace(0, np.max(self.data.flat_longevity()), 100)
        self.plot_histogram(self.data.flat_longevity(), x)
        plt.title('Longevity of PRDM9 alleles histogram')
        plt.xlabel('Longevity')
        plt.ylabel('Frequency')

        plt.tight_layout()

        plt.savefig(str(self) + '.png')
        plt.clf()
        print str(self)
        return str(self)

    def pickle(self):
        pickle.dump(self, open(str(self) + ".p", "wb"))

    def estimated_simpson(self):
        return self.model_params.estimated_simpson(self.data.mean_erosion())

    def estimated_params_simpson(self):
        return self.model_params.estimated_params_simpson()

    def estimated_erosion_var(self):
        return self.model_params.estimated_erosion_var(self.data.mean_erosion())

    def turn_over_selection(self):
        return self.model_params.turn_over_selection(self.data.mean_erosion())

    def turn_over_derivative(self):
        return self.model_params.turn_over_derivative(self.data.mean_erosion())

    def turn_over_params_derivative(self):
        return self.model_params.turn_over_params_derivative()


class BatchSimulation(object):
    def __init__(self,
                 model_params,
                 batch_params,
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
        self.batch_params = batch_params.copy()

        self.simulations = {}
        for simu_params in self.batch_params:
            self.simulations[simu_params.color] = []

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
            for simu_params in self.batch_params:
                model_params_copy = self.model_params.copy()
                if self.axis == "scaling":
                    model_params_copy.mutation_rate_prdm9 /= axis_current
                    model_params_copy.erosion_rate_hotspot /= axis_current
                    model_params_copy.population_size *= axis_current
                    model_params_copy.recombination_rate /= axis_current
                else:
                    setattr(model_params_copy, self.variable_hash(), axis_current)
                self.simulations[simu_params.color].append(Simulation(model_params_copy, simu_params.copy()))

    def variable_hash(self):
        return {"fitness": "fitness_param",
                "mutation": "mutation_rate_prdm9",
                "erosion": "erosion_rate_hotspot",
                "population": "population_size",
                "recombination": "recombination_rate",
                "scaling": "scaling"}[self.axis]

    def caption(self):
        return "Batch of %s simulations. \n" % self.nbr_of_simulations + self.axis_str + \
               "is scaled %.1e times.\n" % self.scale + self.batch_params.caption() + self.model_params.caption()

    def run(self, nbr_of_cpu=7, directory_id=None):
        if directory_id is None:
            directory_id = id_generator(8)
        set_dir("/" + directory_id + " " + self.axis_str)
        for key, simulations in self.simulations.iteritems():
            if nbr_of_cpu > 1:

                pool = Pool(nbr_of_cpu)
                self.simulations[key] = pool.map(execute, simulations)
                pool.close()
            else:
                map(lambda x: x.run(), simulations)
        self.pickle()
        os.chdir('..')
        print 'Simulation computed'

    def pickle(self):
        pickle.dump(self, open(self.axis_str + ".p", "wb"))

    def plot_series(self, series, color, caption, title=True):
        mean = map(lambda serie: np.mean(serie), series)
        plt.plot(self.axis_range, mean, color=color)
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

    def save_figure(self, k=1, directory_id=None):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        plt.subplot(321)
        plt.text(0.05, 0.98, self.caption(), fontsize=10, verticalalignment='top')

        for simu_params in self.batch_params:
            simulations = self.simulations[simu_params.color]

            plt.subplot(321)
            theta = np.arange(0.0, 1.0, 0.01)
            for simulation in simulations:
                plt.plot(theta, simulation.model_params.fitness(theta), color=simu_params.color)

            plt.title('The fitness function')
            plt.xlabel('x')
            plt.ylabel('w(x)')
            models_params = map(lambda sim: sim.model_params, simulations)
            mean_erosion = map(lambda model_param: model_param.bound_mean_erosion(k), models_params)

            plt.subplot(322)
            self.plot_series(map(lambda sim: sim.data.simpson_entropy_prdm9(), simulations), simu_params.color,
                             'Number of PRDM9 alleles')
            params_simpson = map(
                lambda model_param, l: model_param.estimated_simpson(l), models_params, mean_erosion)
            simu_simpson = map(lambda sim: sim.estimated_simpson(), simulations)
            plt.plot(self.axis_range, params_simpson, color='orange')
            plt.plot(self.axis_range, simu_simpson, color='green')
            plt.yscale('linear')
            plt.yscale('log')

            plt.subplot(323)

            self.plot_series(
                map(lambda sim: np.array(sim.data.hotspots_erosion_bound()), simulations), simu_params.color,
                'Mean Hotspot Erosion')
            plt.plot(self.axis_range, mean_erosion, color='orange')
            plt.yscale('linear')

            plt.subplot(324)
            self.plot_series(map(lambda sim: sim.data.hotspots_erosion_var(), simulations), simu_params.color,
                             'Variance Hotspot Erosion')
            params_erosion_var = map(
                lambda model_param, l: model_param.estimated_erosion_var(l), models_params, mean_erosion)
            simu_erosion_var = map(lambda sim: sim.estimated_erosion_var(), simulations)
            plt.plot(self.axis_range, params_erosion_var, color='orange')
            plt.plot(self.axis_range, simu_erosion_var, color='green')
            plt.yscale('linear')

            plt.subplot(325)
            self.plot_series(map(lambda sim: sim.data.prdm9_longevity_mean(), simulations), simu_params.color,
                             'Mean PRDM9 longevity')
            plt.yscale('log')

            plt.subplot(326)
            for percent in np.linspace(0.01, 0.1, 10):
                lag = map(lambda sim: sim.generations[sim.data.dichotomic_search(percent)], simulations)
                plt.plot(self.axis_range, lag, color=simu_params.color, alpha=10 * percent)

            turn_over_selection = map(lambda sim: sim.turn_over_selection(), simulations)
            plt.plot(self.axis_range, turn_over_selection, color='lightgreen')
            turn_over_derivative = map(lambda sim: sim.turn_over_selection(), simulations)
            plt.plot(self.axis_range, turn_over_derivative, color='darkgreen')
            # turn_over_neutral = map(lambda model_param: model_param.turn_over_neutral(), models_params)
            # plt.plot(self.axis_range, turn_over_neutral, color='grey')
            # turn_over_max = map(lambda model_param: model_param.turn_over_max(), models_params)
            # plt.plot(self.axis_range, turn_over_max, color='black')
            plt.title('{0} for different {1}'.format('Cross-homozygosity', self.axis_str))
            plt.xlabel(self.axis_str)
            plt.ylabel('Cross-homozygosity')
            plt.yscale('log')
            if self.axis == "fitness" and self.model_params.fitness_family == 3:
                plt.xscale('linear')
            else:
                plt.xscale('log')

        plt.tight_layout()

        if directory_id is None:
            directory_id = id_generator(8)
        plt.savefig(directory_id + " " + self.axis_str + " " + str(k) + '.png')
        plt.clf()
        return self

    def plot_polymorphism(self):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        simulations = self.simulations["blue"]
        self.plot_series(map(lambda sim: sim.data.simpson_entropy_prdm9(), simulations), "blue",
                         'PRDM9 diversity')
        plt.yscale('log')
        plt.ylim([1, plt.ylim()[1]])

        plt.tight_layout()

        plt.savefig('polymorphism.png')
        plt.clf()

    def plot_mean_erosion(self):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        simulations = self.simulations["blue"]
        self.plot_series(map(lambda sim: sim.data.hotspots_erosion_bound(), simulations), "blue",
                         'PRDM9 diversity')
        plt.yscale('linear')

        plt.tight_layout()

        plt.savefig('mean-erosion.png')
        plt.clf()

    def plot_cross_homozygosity(self):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        simulations = self.simulations["blue"]
        for percent in np.linspace(0.01, 0.1, 10):
            lag = map(lambda sim: sim.generations[sim.data.dichotomic_search(percent)], simulations)
            plt.plot(self.axis_range, lag, color="blue", alpha=10 * percent)

        plt.title('{0} for different {1}'.format('Cross-homozygosity', self.axis_str))
        plt.xlabel(self.axis_str)
        plt.ylabel('Cross-homozygosity')
        plt.yscale('log')
        plt.xscale('log')

        plt.tight_layout()

        plt.savefig('cross-homozygosity.png')
        plt.clf()


class PhaseDiagram(object):
    def __init__(self,
                 model_params,
                 batch_params,
                 axis1="null",
                 axis2="null",
                 nbr_of_simulations=5,
                 scale=10 ** 2):
        batch = BatchSimulation(model_params, batch_params, axis1, nbr_of_simulations, scale)
        self.axis_x = axis2
        self.axis_y = axis1
        self.batch_params = batch_params
        self.axis1 = batch.axis_range
        self.axis2 = BatchSimulation(model_params, batch_params, axis2, nbr_of_simulations, scale).axis_range
        self.batches = []
        for axis_current in self.axis1:
            model_params_copy = model_params.copy()
            setattr(model_params_copy, batch.variable_hash(), axis_current)
            self.batches.append(BatchSimulation(model_params_copy, batch_params, axis2, nbr_of_simulations, scale))

    def __str__(self):
        return "PhaseDiagram"

    def run(self, nbr_of_cpu=4, directory_id=None):
        for batch in self.batches:
            batch.run(nbr_of_cpu=nbr_of_cpu, directory_id=directory_id)
        self.pickle()

    def pickle(self):
        pickle.dump(self, open(str(self) + ".p", "wb"))

    def save_figure(self, interpolation, num=30):
        self.save_pcolor(lambda sim: np.mean(sim.data.simpson_entropy_prdm9()),
                         "Simpson entropy of PRDM9 (polymorphism)", interpolation, num)
        self.save_pcolor(lambda sim: np.mean(sim.data.hotspots_erosion_mean()),
                         "Mean Erosion of the hotspots", interpolation, num)
        self.save_pcolor(lambda sim: sim.generations[sim.data.dichotomic_search(0.01)],
                         "Cross-homozygosity of PRDM9 (turn-over)", interpolation, num)

    def save_pcolor(self, function, name, interpolation=True, num=30):
        for simu_params in self.batch_params:
            my_dpi = 96
            plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
            intensities = []
            for batch in self.batches:
                simulations = batch.simulations[simu_params.color]
                intensity = map(function, simulations)
                intensities.append(intensity)
                # batch.save_figure()

            intensities = np.array(intensities)
            if interpolation:
                f = interpolate.interp2d(self.axis2, self.axis1, intensities, kind='cubic')
                axis2 = np.logspace(np.log10((np.min(self.axis2))), np.log10((np.max(self.axis2))), num)
                axis1 = np.logspace(np.log10((np.min(self.axis1))), np.log10((np.max(self.axis1))), num)
                intensities = f(axis2, axis1)
                axis_x, axis_y = np.meshgrid(axis2, axis1)
            else:
                axis_x, axis_y = np.meshgrid(self.axis2, self.axis1)
            plt.pcolor(axis_x, axis_y, intensities, cmap='RdBu', shading='faceted', snap=False)
            plt.colorbar()
            plt.xscale('log')
            plt.xlabel(self.label(self.axis_x))
            plt.ylabel(self.label(self.axis_y))
            plt.yscale('log')
            plt.title(name)
            plt.tight_layout()

            plt.savefig(str(self) + " " + name + str(interpolation) + " " + simu_params.color + '.png')
            plt.clf()
        print name + ' computed'
        return self

    @staticmethod
    def label(axis):
        return {"fitness": "The fitness parameter",
                "mutation": "Mutation rate of PRDM9",
                "erosion": "Mutation rate of the hotspots",
                "population": "The population size",
                "recombination": "The recombination rate of the hotspots",
                "scaling": "The scaling factor"}[axis]


class Batches(list):
    def save_figure(self, k_range=np.logspace(-1, 1, 5)):
        self.save_erosion(k_range)
        self.save_simpson(False)
        self.save_cross_homozygosity(False)
        self.save_simpson(True)
        self.save_cross_homozygosity(True)

    def save_erosion(self, k_range=np.logspace(-1, 1, 5)):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 540 / my_dpi), dpi=my_dpi)

        for j, batch in enumerate(self):
            plt.subplot(1, 3, j + 1)
            for simu_params in batch.batch_params:
                simulations = batch.simulations[simu_params.color]

                models_params = map(lambda sim: sim.model_params, simulations)

                batch.plot_series(
                    map(lambda sim: np.array(sim.data.hotspots_erosion_bound()), simulations), simu_params.color,
                    'Mean Hotspot Erosion', title=False)
                for k_value in k_range:
                    plt.text(0.5, 0.5, "k = %s" % k_value, fontsize=12, verticalalignment='top')
                    mean_erosion = map(lambda model_param: model_param.bound_mean_erosion(k_value), models_params)
                    plt.plot(batch.axis_range, mean_erosion, color='orange', linewidth=3)
                plt.yscale('linear')

        plt.tight_layout()

        plt.savefig(' batch erosion mean.png')
        plt.clf()
        print 'Erosion Mean computed'
        return self

    def save_simpson(self, estimated_mean_erosion=True):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 540 / my_dpi), dpi=my_dpi)

        for j, batch in enumerate(self):
            plt.subplot(1, 3, j + 1)
            for simu_params in batch.batch_params:
                simulations = batch.simulations[simu_params.color]

                batch.plot_series(map(lambda sim: sim.data.simpson_entropy_prdm9(), simulations),
                                  simu_params.color, 'Diversity', title=False)
                if estimated_mean_erosion:
                    simu_simpson = map(lambda sim: sim.estimated_params_simpson(), simulations)
                    plt.plot(batch.axis_range, simu_simpson, color='orange', linewidth=3)
                else:
                    simu_simpson = map(lambda sim: sim.estimated_simpson(), simulations)
                    plt.plot(batch.axis_range, simu_simpson, color='green', linewidth=3)
                plt.yscale('log')

        plt.tight_layout()

        plt.savefig(str(estimated_mean_erosion) + ' batch simpson.png')
        plt.clf()
        print 'Simpson computed'
        return self

    def save_cross_homozygosity(self, estimated_mean_erosion=True):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 540 / my_dpi), dpi=my_dpi)

        for j, batch in enumerate(self):
            plt.subplot(1, 3, j + 1)
            for simu_params in batch.batch_params:
                simulations = batch.simulations[simu_params.color]

                for percent in np.linspace(0.01, 0.1, 10):
                    lag = map(lambda sim: sim.generations[sim.data.dichotomic_search(percent)], simulations)
                    plt.plot(batch.axis_range, lag, color=simu_params.color, alpha=10 * percent)

                if estimated_mean_erosion:
                    turn_over_derivative = map(lambda sim: sim.turn_over_params_derivative(), simulations)
                    plt.plot(batch.axis_range, turn_over_derivative, color='orange', linewidth=3)
                else:
                    turn_over_derivative = map(lambda sim: sim.turn_over_derivative(), simulations)
                    plt.plot(batch.axis_range, turn_over_derivative, color='green', linewidth=3)

                plt.yscale('linear')

            plt.xlabel(batch.axis_str)
            plt.ylabel('Cross-homozygosity')
            plt.yscale('log')
            if batch.axis == "fitness" and batch.model_params.fitness_family == 3:
                plt.xscale('linear')
            else:
                plt.xscale('log')

        plt.tight_layout()

        plt.savefig(str(estimated_mean_erosion) + 'batch cross homozygosity.png')
        plt.clf()
        print 'Cross homozygosity computed'
        return self

    def pickle(self):
        pickle.dump(self, open("batches.p", "wb"))


if __name__ == '__main__':
    '''
    dir_id = "67FB8030"
    set_dir("/tmp/" + dir_id)
    batches = pickle.load(open("batches.p", "rb"))
    batches.pop()
    batches.save_figure([1])
    model_parameters = ModelParams(mutation_rate_prdm9=1.0 * 10 ** -5,
                                   erosion_rate_hotspot=1.0 * 10 ** -4,
                                   population_size=10 ** 5,
                                   recombination_rate=1.0 * 10 ** -3,
                                   fitness_param=0.1,
                                   fitness='polynomial')
    batch_parameters = BatchParams(drift=True, linearized=False, color="blue", scaling=20)
    batches = Batches()
    for axis in ["population", "erosion", "mutation"]:
        batches.append(BatchSimulation(model_parameters.copy(), batch_parameters.copy(), axis=axis,
                                       nbr_of_simulations=16, scale=10 ** 2))
    for batch_simulation in batches:
        batch_simulation.run(nbr_of_cpu=8, directory_id=dir_id)
    for batch_simulation in batches:
        batch_simulation.save_figure(directory_id=dir_id)
    batches.pickle()
    batches.save_figure(np.logspace(-0.5, 0.5, 5), directory_id=dir_id)
    phasediagram = PhaseDiagram(model_parameters, batch_parameters, "population", "mutation", nbr_of_simulations=14, scale=10 ** 2)
    phasediagram.run(nbr_of_cpu=7)
    phasediagram.save_figure(interpolation=True)
    phasediagram.save_figure(interpolation=False)
    '''
    dir_id = "1F0F8599"
    set_dir("/tmp/" + dir_id)
    phasediagram = pickle.load(open("PhaseDiagram.p", "rb"))
    phasediagram.save_figure(interpolation=True, num=100)
    phasediagram.save_figure(interpolation=False)
    '''
    dir_id = "3FCA8AF6"
    set_dir("/tmp/" + dir_id)
    lst = os.listdir(os.getcwd())
    batches = Batches()
    for pickle_file in lst:
        if pickle_file[-2:] == ".p":
            batch_simulation = pickle.load(open(pickle_file, "rb"))
            batch_simulation.save_figure(directory_id=dir_id)
            # set_dir("/" + dir_id + " " + batch_simulation.axis_str)
            # for sims in batch_simulation.simulations.values():
            #     map(lambda sim: sim.save_figure(), sims)
            # os.chdir('..')
            batches.append(batch_simulation)
    batches.save_figure(np.logspace(-0.5, 0.5, 5), directory_id=dir_id)
    '''
