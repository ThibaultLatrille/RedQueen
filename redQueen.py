from scipy.stats.kde import gaussian_kde
from multiprocessing import Pool
import numpy as np
import copy
import matplotlib.pyplot as plt
import time
import os
import uuid
import cPickle as pickle
import itertools
from scipy.special import lambertw


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
    def __init__(self, nbr_of_alleles, model_params):
        self.prdm9_polymorphism = sum_to_one(np.random.sample(nbr_of_alleles))
        self.hotspots_erosion = np.random.sample(nbr_of_alleles)
        self.prdm9_longevity = np.zeros(nbr_of_alleles)
        self.prdm9_fitness = np.ones(nbr_of_alleles)
        self.fitness_family = model_params.fitness_family
        self.fitness_param = model_params.fitness_param
        self.alpha_zero = model_params.alpha_zero
        self.alpha = model_params.alpha_zero
        self.t = 0
        self.id = nbr_of_alleles
        self.ids = np.array(range(nbr_of_alleles))

    def fitness(self, x):
        if self.fitness_family == 3:
            k = 4
            matrix = x ** k / (x ** k + self.fitness_param ** k)
        elif self.fitness_family == 2:
            matrix = np.power(x, self.fitness_param)
        else:
            matrix = x
        if self.alpha == 1.:
            return matrix
        else:
            return np.power(matrix, self.alpha)

    def coefficient_fitness(self, x):
        if self.fitness_family == 3:
            k = 4
            return (self.alpha * k / 2) * self.fitness_param ** 4 / (x * (x ** k + self.fitness_param ** k))
        elif self.fitness_family == 2:
            return self.alpha * self.fitness_param * 1.0 / (2. * x)
        else:
            return self.alpha * 1.0 / (2 * x)

    def remove_dead_prdm9(self, cut_off):
        remove_extincted = np.array(map(lambda x: x > cut_off, self.prdm9_polymorphism), dtype=bool)
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


class ModelDiscrete(Model):
    def __init__(self, nbr_of_alleles, model_params, simu_params):
        super(ModelDiscrete, self).__init__(nbr_of_alleles, model_params)
        self.population_size = model_params.population_size
        self.prdm9_polymorphism *= self.population_size
        self.mutation_rate = model_params.population_size * model_params.mutation_rate_prdm9
        self.erosion_rate = model_params.erosion_rate_hotspot * model_params.population_size * \
                            model_params.recombination_rate
        if model_params.scaled:
            self.erosion_rate *= model_params.population_size
        assert self.erosion_rate < 0.1, "The scaled erosion rate is too large, decrease either the " \
                                        "recombination rate, the erosion rate or the population size"
        assert self.erosion_rate > 0.000000001, "The scaled erosion rate is too low, increase either the " \
                                                "recombination rate, the erosion rate or the population size"
        self.alpha = 1.
        self.linearized = simu_params.linearized
        self.drift = simu_params.drift

    def forward(self, t=1):
        new_alleles = np.random.poisson(self.mutation_rate)
        if new_alleles > 0:
            self.new_alleles(new_alleles)
        prdm9_frequencies = sum_to_one(self.prdm9_polymorphism)

        self.hotspots_erosion *= np.exp(- self.erosion_rate * prdm9_frequencies)

        # Compute the fitness for each allele
        if self.fitness_family == 0:
            distribution_vector = prdm9_frequencies
        else:
            if self.linearized:
                l_bar = np.sum(prdm9_frequencies * self.hotspots_erosion)
                self.prdm9_fitness = self.coefficient_fitness(l_bar) * (self.hotspots_erosion - l_bar)
                distribution_vector = prdm9_frequencies + self.prdm9_fitness * prdm9_frequencies
                if np.max(distribution_vector) > 1.:
                    distribution_vector = sum_to_one(np.clip(distribution_vector, 0., 1.))
                elif np.min(distribution_vector) < 0.:
                    distribution_vector = sum_to_one(np.clip(distribution_vector, 0., 1.))
            else:
                fitness_matrix = self.fitness(np.add.outer(self.hotspots_erosion, self.hotspots_erosion) / 2)
                self.prdm9_fitness = np.dot(fitness_matrix, prdm9_frequencies)
                distribution_vector = sum_to_one(self.prdm9_fitness * prdm9_frequencies)

        if self.drift:
            self.prdm9_polymorphism = np.random.multinomial(int(self.population_size), distribution_vector).astype(
                float)
        else:
            self.prdm9_polymorphism = distribution_vector * self.population_size

        self.prdm9_longevity += 1. / self.population_size
        self.remove_dead_prdm9(cut_off=0)

    def new_alleles(self, new_alleles):
        self.prdm9_polymorphism -= np.random.multinomial(new_alleles, sum_to_one(self.prdm9_polymorphism))
        self.prdm9_polymorphism = np.append(self.prdm9_polymorphism, np.ones(new_alleles))
        self.prdm9_longevity = np.append(self.prdm9_longevity, np.zeros(new_alleles))
        self.hotspots_erosion = np.append(self.hotspots_erosion, np.ones(new_alleles))
        self.ids = np.append(self.ids, range(self.id, self.id + new_alleles))
        self.id += new_alleles


class ModelContinuous(Model):
    def __init__(self, nbr_of_alleles, model_params, simu_params):
        super(ModelContinuous, self).__init__(nbr_of_alleles, model_params)

        self.mutation_rate = model_params.population_size * model_params.mutation_rate_prdm9
        self.erosion_rate = model_params.population_size * model_params.erosion_rate_hotspot
        self.recombination_rate = model_params.population_size * model_params.recombination_rate
        self.alpha = model_params.alpha_zero

        self.cut_off = simu_params.cut_off
        self.drift = simu_params.drift

    def forward(self, t):
        dt = min(1. / (max(self.erosion_rate * self.recombination_rate,
                           self.alpha, self.mutation_rate * self.alpha) * 1), t / 10)
        self.t += t
        current_t = 0
        while current_t < t:
            current_t += dt
            self.new_alleles(dt)
            prdm9_frequencies = sum_to_one(self.prdm9_polymorphism)
            self.hotspots_erosion -= (self.erosion_rate * self.recombination_rate * dt) * (
                prdm9_frequencies * self.hotspots_erosion)
            # Compute the fitness for each allele
            if self.fitness_family == 0:
                distribution_vector = prdm9_frequencies
            else:
                l_bar = np.sum(prdm9_frequencies * self.hotspots_erosion)
                self.prdm9_fitness = self.coefficient_fitness(l_bar) * (self.hotspots_erosion - l_bar)
                if self.drift:
                    brownian_matrix = np.add.outer(prdm9_frequencies, prdm9_frequencies)
                    for index, frequency in enumerate(prdm9_frequencies):
                        brownian_matrix[index, index] = frequency * (1 - frequency)
                    brownian = np.random.multivariate_normal(np.zeros(prdm9_frequencies.size), brownian_matrix)
                else:
                    brownian = 0.
                distribution_vector = prdm9_frequencies + (self.prdm9_fitness * prdm9_frequencies + brownian) * dt
                if np.max(distribution_vector) > 1.:
                    distribution_vector = sum_to_one(np.clip(distribution_vector, 0., 1.))
                if np.min(distribution_vector) < 0.:
                    distribution_vector = sum_to_one(np.clip(distribution_vector, 0., 1.))

            self.prdm9_polymorphism = distribution_vector
            self.remove_dead_prdm9(cut_off=self.cut_off * 0.1)

        self.prdm9_longevity += t

    def new_alleles(self, dt):
        prdm9_frequencies = sum_to_one(self.prdm9_polymorphism)
        l_bar = float(np.sum(prdm9_frequencies * self.hotspots_erosion))
        s = 2 * self.coefficient_fitness(l_bar) * (1 - l_bar)
        fixed = np.random.poisson(self.mutation_rate * s * dt)
        if fixed > 0:
            self.prdm9_polymorphism *= (1 - float(fixed) * self.cut_off)
            self.prdm9_polymorphism = np.append(self.prdm9_polymorphism, np.ones(fixed) * 0.001)
            self.hotspots_erosion = np.append(self.hotspots_erosion, np.ones(fixed))
            self.prdm9_longevity = np.append(self.prdm9_longevity, np.zeros(fixed))
            self.ids = np.append(self.ids, range(self.id, self.id + fixed))
            self.id += fixed


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
        return map(lambda erosion, freq: self.normalized_var(erosion, freq), self.hotspots_erosion,
                   self.prdm9_frequencies)

    def prdm9_longevity_mean(self):
        return map(lambda longevity, freq: self.n_moment(longevity, freq, 1), self.prdm9_longevity,
                   self.prdm9_frequencies)

    def prdm9_longevity_var(self):
        return map(lambda longevity, freq: self.normalized_var(longevity, freq), self.prdm9_longevity,
                   self.prdm9_frequencies)

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

    def cross_homozygosity(self, lag):
        cross_homozygosity = []
        for index in range(0, len(self.ids) - lag):
            cross_homozygosity.append(0.)
            slice_dict = dict(zip(self.ids[index], self.prdm9_frequencies[index]))
            lag_dict = dict(zip(self.ids[index + lag], self.prdm9_frequencies[index + lag]))
            for key in list(set(slice_dict.keys()) & set(lag_dict.keys())):
                cross_homozygosity[index] += slice_dict[key] * lag_dict[key]
        return np.mean(cross_homozygosity)

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
                 recombination_rate=1.0 * 10 ** -3, fitness_param=1., fitness='linear', scaled=True,
                 alpha_zero=1.0 * 10 ** 4):
        self.mutation_rate_prdm9 = mutation_rate_prdm9  # The rate at which new allele for PRDM9 are created
        self.erosion_rate_hotspot = erosion_rate_hotspot  # The rate at which the hotspots are eroded
        self.recombination_rate = recombination_rate
        self.population_size = float(population_size)  # population size

        self.scaled = scaled
        self.alpha_zero = alpha_zero
        fitness_hash = {"linear": 1, "polynomial": 2, "sigmoid": 3}
        assert fitness in fitness_hash.keys(), "Parameter 'fitness' must be a string: ['linear','sigmoid','polynomial']"
        self.fitness_family = fitness_hash[fitness]
        self.fitness_param = fitness_param
        if self.fitness_family == 1:
            self.fitness_param = 1.

    def unbound_mean_erosion(self, balance):
        rate_out = np.power(self.erosion_rate_hotspot * self.recombination_rate, 1)
        rate_in = np.power(self.fitness_param * self.mutation_rate_prdm9, 1)
        return rate_out / (balance * rate_in)

    def bound_mean_erosion(self, balance):
        rate_out = np.power(self.erosion_rate_hotspot * self.recombination_rate, 1)
        rate_in = np.power(self.fitness_param * self.mutation_rate_prdm9, 1)
        return rate_in / (rate_in + balance * rate_out)

    def estimated_simpson(self, mean_erosion):
        denom = 1 - 2 * mean_erosion - mean_erosion * np.real(lambertw(-np.exp(- 1. / mean_erosion) / mean_erosion))
        return (2 * mean_erosion * self.erosion_rate_hotspot * self.recombination_rate * self.population_size) / (
            self.fitness_param * denom)

    def __str__(self):
        name = "u=%.1e" % self.mutation_rate_prdm9 + \
               "_v=%.1e" % self.erosion_rate_hotspot + \
               "_r=%.1e" % self.recombination_rate + \
               "_n=%.1e" % self.population_size + \
               "_a=%.1e" % self.alpha_zero
        if self.fitness_family == 3 or self.fitness_family == 2:
            name += "_f=%.1e" % self.fitness_param
        return name

    def caption(self):
        caption = "Mutation rate of PRDM9: %.1e. \n" % self.mutation_rate_prdm9 + \
                  "Erosion rate of the hotspots: %.1e. \n" % self.erosion_rate_hotspot + \
                  "Recombination rate at the hotspots: %.1e. \n" % self.recombination_rate + \
                  "Population size: %.1e. \n" % self.population_size + \
                  "Alpha_0: %.1e. \n" % self.alpha_zero
        if self.fitness_family == 3:
            caption += "Inflexion point of the fitness function PRDM9=%.1e. \n" % self.fitness_param
        if self.fitness_family == 2:
            caption += "Exponent of the polynomial fitness function PRDM9=%.1e. \n" % self.fitness_param
        return caption

    def copy(self):
        return copy.copy(self)


class SimulationParams(object):
    def __init__(self, drift=True, linearized=True, discrete=True, color="blue", scaling=100, cut_off=0.01,
                 nbr_of_steps=1000):
        self.scaling = scaling
        self.drift = drift
        self.discrete = discrete
        if self.discrete:
            self.linearized = True
            self.cut_off = cut_off
        else:
            self.linearized = linearized
            self.cut_off = 0
        self.color = color

        self.nbr_of_steps = nbr_of_steps  # Number of steps at which we make computations

    def __str__(self):
        return "scale=%.1e" % self.scaling + \
               "_drift=%s" % self.drift + \
               "_linear=%s" % self.linearized + \
               "_discrete=%s" % self.discrete + \
               "_color=%s" % self.color + \
               "_n=%.1e" % self.nbr_of_steps

    def caption(self):
        caption = "Simulation of %s*Ne generations " % self.scaling + \
                  "\n  Recorded %.1e times. \n" % self.nbr_of_steps
        caption += self.caption_for_attr('discrete')
        caption += self.caption_for_attr('drift')
        caption += self.caption_for_attr('linearized')
        return caption

    def caption_for_attr(self, attr):
        if attr == 'discrete':
            if getattr(self, attr):
                return "  The population size and time is considered DISCRETE. \n"
            else:
                return "  The population size and time is considered CONTINUOUS. \n" + \
                       "  Cut-off for the death of PRDM9=%.1e. \n" % self.cut_off
        elif attr == 'drift':
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
        self.t_max = max(int(self.simu_params.scaling * self.model_params.population_size),
                         self.simu_params.nbr_of_steps)  # Number of generations

        if self.simu_params.discrete:
            self.model = ModelDiscrete(10, model_params, simu_params)
        else:
            self.model = ModelContinuous(10, model_params, simu_params)
        self.data = SimulationData()
        self.generations = []

    def __str__(self):
        return "tmax=%.1e_" % self.t_max + str(self.model_params) + "_" + str(self.simu_params)

    def caption(self):
        return "%.1e generations computed \n \n" % self.t_max + \
               "Model parameters : \n" + self.model_params.caption() + "\n" + \
               "Simulation parameters : \n" + self.simu_params.caption()

    def run(self):
        start_time = time.time()
        step = float(self.t_max) / self.simu_params.nbr_of_steps
        step_t = 0.

        for t in range(self.t_max):
            # Randomly create new alleles of PRDM9

            if self.simu_params.discrete:
                self.model.forward(t=1)

            step_t += 1
            if step_t > step and t / float(self.t_max) > 0.01:
                step_t -= step
                if not self.simu_params.discrete:
                    self.model.forward(t=float(self.simu_params.scaling) / self.simu_params.nbr_of_steps)

                if int(10 * t) % self.t_max == 0:
                    print "Computation at {0}%".format(float(100 * t) / self.t_max)

                self.generations.append(t)
                self.data.store(self.model)

                if time.time() - start_time > 7200:
                    self.t_max = t
                    print "Breaking the loop, time over 7200s"
                    break

        self.save_figure()
        return self

    def plot_histogram(self, lst, x):
        if (not lst) or lst.count(lst[0]) == len(lst):
            plt.hist(lst, color=self.simu_params.color, alpha=0.3)
        else:
            s_density = gaussian_kde(lst)(x)
            plt.plot(x, s_density, color=self.simu_params.color)
            plt.fill_between(x, s_density, np.zeros(100), color=self.simu_params.color, alpha=0.3)
        plt.text(0.5, 0.5, "Mean = {0:.1e}  \nVar = {1:.1e}".format(np.mean(lst), np.var(lst)), fontsize=12,
                 verticalalignment='top')

    def save_figure(self):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
        plt.subplot(331)
        plt.text(0.05, 0.98, self.caption(), fontsize=10, verticalalignment='top')
        theta = np.arange(0.0, 1.0, 0.01)
        plt.plot(theta, self.model.fitness(theta), color=self.simu_params.color)
        plt.title('The fitness function')
        plt.xlabel('x')
        plt.ylabel('w(x)')

        plt.subplot(332)
        plt.plot(self.generations, self.data.simpson_entropy_prdm9(), color=self.simu_params.color)
        plt.plot(self.generations, self.data.shannon_entropy_prdm9(), color='green')
        plt.title('Efficient nbr of PRDM9 alleles over time. \n')
        plt.xlabel('Generation')
        plt.ylabel('Number of alleles')
        plt.yscale('log')

        plt.subplot(333)
        plt.plot(self.generations, self.data.hotspots_erosion_mean(), color=self.simu_params.color)
        plt.title('Mean erosion of the hotspots over time. \n')
        plt.xlabel('Generation')
        plt.ylabel('Mean erosion')
        plt.yscale('log')

        plt.subplot(334)
        plt.hexbin(self.data.flat_erosion(), self.data.flat_fitness(), self.data.flat_longevity(),
                   gridsize=200, bins='log')
        plt.title('PRDM9 fitness vs hotspot erosion')
        plt.xlabel('hotspot erosion')
        plt.ylabel('PRMD9 fitness')

        plt.subplot(335)
        plt.hexbin(self.data.flat_erosion(), self.data.flat_frequencies(), self.data.flat_longevity(),
                   gridsize=200, bins='log')
        plt.title('PRMD9 frequency vs hotspot erosion')
        plt.xlabel('hotspot erosion')
        plt.ylabel('PRMD9 frequency')

        plt.subplot(336)
        lags = np.arange(0, 20, 1)
        plt.plot(lags, map(lambda lag: self.data.cross_homozygosity(lag), lags))
        plt.title('Cross correlation of homozygosity')
        plt.xlabel('Lag')
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


class BatchSimulation(object):
    def __init__(self,
                 model_params,
                 batch_params,
                 axis="null",
                 nbr_of_simulations=20,
                 scale=10 ** 2):
        axis_hash = {"fitness": 0, "mutation": 1, "erosion": 2, "population": 3, "recombination": 4,
                     "alpha": 5, "scaling": 6, "cut_off": 7}
        assert axis in axis_hash.keys(), "Axis must be either 'population', 'mutation', 'erosion'," \
                                         "'recombination', 'fitness', or 'scaling'"
        assert scale > 1, "The scale parameter must be greater than one"
        self.axis = axis_hash[axis]
        self.axis_str = {0: "The fitness inflexion point", 1: "Mutation rate of PRDM9",
                         2: "Erosion rate of the hotspots", 3: "The population size",
                         4: "The recombination rate", 5: "Alpha 0",
                         6: "The scaling factor", 7: "Cut-off"}[self.axis]
        self.scale = scale
        self.axis_range = []
        range_current = 1.
        self.nbr_of_simulations = nbr_of_simulations
        self.model_params = model_params.copy()
        self.batch_params = batch_params.copy()

        self.simulations = {}
        for simu_params in self.batch_params:
            self.simulations[simu_params.color] = []

        effect = self.scale ** (1. / (self.nbr_of_simulations - 1))
        for axis_current in range(self.nbr_of_simulations):
            for simu_params in self.batch_params:
                self.simulations[simu_params.color].append(Simulation(model_params.copy(), simu_params.copy()))
                if self.axis == 7:
                    simu_params.cut_off *= effect
            self.axis_range.append(range_current)
            range_current *= effect
            if self.axis == 0:
                model_params.fitness_param *= effect
            elif self.axis == 1:
                model_params.mutation_rate_prdm9 *= effect
            elif self.axis == 2:
                model_params.erosion_rate_hotspot *= effect
            elif self.axis == 3:
                model_params.population_size *= effect
            elif self.axis == 4:
                model_params.recombination_rate *= effect
            elif self.axis == 5:
                model_params.alpha_zero *= effect
            elif self.axis == 6:
                model_params.mutation_rate_prdm9 /= effect
                model_params.erosion_rate_hotspot /= effect
                model_params.population_size *= effect
                model_params.recombination_rate /= effect

    def caption(self):
        return "Batch of %s simulations. \n" % self.nbr_of_simulations + self.axis_str + \
               "is scaled %.1e times.\n" % self.scale + self.batch_params.caption() + self.model_params.caption()

    def run(self, nbr_of_cpu=4):
        set_dir("/" + self.axis_str + " " + id_generator(8))
        for key, simulations in self.simulations.iteritems():
            if nbr_of_cpu > 1:

                pool = Pool(nbr_of_cpu)
                self.simulations[key] = pool.map(execute, simulations)
            else:
                map(lambda x: x.run(), simulations)
        self.pickle()
        self.save_figure()

    def pickle(self):
        pickle.dump(self, open(self.axis_str + ".p", "wb"))

    def plot_time_series(self, time_series, color, caption):
        mean = map(lambda series: np.mean(series), time_series)
        plt.plot(self.axis_range, mean,
                 color=color)
        sigma = map(lambda series: np.sqrt(np.var(series)), time_series)
        y_max = np.add(mean, sigma)
        y_min = np.subtract(mean, sigma)
        plt.fill_between(self.axis_range, y_max, y_min, color=color, alpha=0.3)
        plt.title('{0} for different {1}'.format(caption, self.axis_str))
        plt.xlabel(self.axis_str)
        plt.ylabel(caption)
        plt.xscale('log')

    def save_figure(self, k=1):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        plt.subplot(321)
        plt.text(0.05, 0.98, self.caption(), fontsize=10, verticalalignment='top')

        for simu_params in self.batch_params:
            simulations = self.simulations[simu_params.color]

            plt.subplot(321)
            theta = np.arange(0.0, 1.0, 0.01)
            for simulation in simulations:
                plt.plot(theta, simulation.model.fitness(theta), color=simu_params.color)

            plt.title('The fitness function')
            plt.xlabel('x')
            plt.ylabel('w(x)')
            models_params = map(lambda sim: sim.model_params, simulations)
            mean_erosion = map(lambda model_param: model_param.bound_mean_erosion(k), models_params)
            params_simpson = map(
                lambda model_param, l: model_param.estimated_simpson(l), models_params, mean_erosion)
            simu_simpson = map(lambda sim: sim.estimated_simpson(), simulations)

            plt.subplot(322)
            self.plot_time_series(map(lambda sim: sim.data.simpson_entropy_prdm9(), simulations), simu_params.color,
                                  'Number of PRDM9 alleles')
            plt.text(0.5, 0.5, "k = %s" % k, fontsize=12, verticalalignment='top')
            plt.plot(self.axis_range, params_simpson, color='black')
            plt.plot(self.axis_range, simu_simpson, color='grey')
            plt.yscale('linear')
            plt.yscale('log')

            plt.subplot(323)

            self.plot_time_series(
                map(lambda sim: np.array(sim.data.hotspots_erosion_bound()), simulations), simu_params.color,
                'Mean Hotspot Erosion')
            plt.text(0.5, 0.5, "k = %s" % k, fontsize=12, verticalalignment='top')
            plt.plot(self.axis_range, mean_erosion, color='orange')
            plt.yscale('linear')

            plt.subplot(324)
            self.plot_time_series(map(lambda sim: sim.data.hotspots_erosion_var(), simulations), simu_params.color,
                                  'Var Hotspot Erosion')
            plt.subplot(325)
            self.plot_time_series(map(lambda sim: sim.data.prdm9_longevity_mean(), simulations), simu_params.color,
                                  'Mean PRDM9 longevity')
            plt.yscale('log')
            plt.subplot(326)
            self.plot_time_series(map(lambda sim: sim.data.prdm9_longevity_var(), simulations), simu_params.color,
                                  'Var PRDM9 longevity')

        plt.tight_layout()

        plt.savefig(self.axis_str + str(k) + '.png')
        plt.clf()
        print 'Simulation computed'
        return self


class Batches(list):
    def save_figure(self, k_range=np.logspace(-1, 1, 5)):
        self.save_erosion(k_range)
        self.save_simpson()

    def save_erosion(self, k_range=np.logspace(-1, 1, 5)):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        for j, batch in enumerate(self):
            plt.subplot(2, 2, j + 1)
            for simu_params in batch.batch_params:
                simulations = batch.simulations[simu_params.color]

                models_params = map(lambda sim: sim.model_params, simulations)

                batch.plot_time_series(
                    map(lambda sim: np.array(sim.data.hotspots_erosion_bound()), simulations), simu_params.color,
                    'Mean Hotspot Erosion')
                for k_value in k_range:
                    plt.text(0.5, 0.5, "k = %s" % k_value, fontsize=12, verticalalignment='top')
                    mean_erosion = map(lambda model_param: model_param.bound_mean_erosion(k_value), models_params)
                    plt.plot(batch.axis_range, mean_erosion, color='orange')
                plt.yscale('linear')

        plt.tight_layout()

        plt.savefig('erosion.png')
        plt.clf()
        print 'Figure computed'
        return self

    def save_simpson(self):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        for j, batch in enumerate(self):
            plt.subplot(2, 2, j + 1)
            for simu_params in batch.batch_params:
                simulations = batch.simulations[simu_params.color]

                simu_simpson = map(lambda sim: sim.estimated_simpson(), simulations)

                batch.plot_time_series(map(lambda sim: sim.data.simpson_entropy_prdm9(), simulations),
                                       simu_params.color,
                                       'Number of PRDM9 alleles')
                plt.plot(batch.axis_range, simu_simpson, color='grey')
                plt.yscale('log')

        plt.tight_layout()

        plt.savefig('simpson.png')
        plt.clf()
        print 'Figure computed'
        return self


if __name__ == '__main__':
    set_dir("/tmp")
    model_parameters = ModelParams(mutation_rate_prdm9=1.0 * 10 ** -8,
                                   erosion_rate_hotspot=1.0 * 10 ** -5,
                                   population_size=10 ** 4,
                                   recombination_rate=1.0 * 10 ** -2,
                                   fitness_param=0.1,
                                   fitness='polynomial',
                                   scaled=False,
                                   alpha_zero=1.)
    batch_parameters = BatchParams(drift=True, linearized=False, discrete=True, color="blue", scaling=100,
                                   cut_off=0.01, nbr_of_steps=1000)
    batch_parameters.append_simu_params(dict(red="linearized"))
    batch_simulation = BatchSimulation(model_parameters, batch_parameters, axis="mutation",
                                       nbr_of_simulations=14, scale=10 ** 4)
    batch_simulation.run(nbr_of_cpu=1)
    '''
    set_dir("/tmp/ns-pickle")
    lst = os.listdir(os.getcwd())
    batches = Batches()
    for pickle_file in lst:
        if pickle_file[-2:] == ".p":
            batch_simulation = pickle.load(open(pickle_file, "rb"))
            batch_simulation.save_figure()
            batches.append(batch_simulation)

    batches.save_figure(np.logspace(-0.5, 0.5, 5))
    '''