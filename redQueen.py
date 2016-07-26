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


# Normalize an array to 1
def sum_to_one(array):
    return np.divide(array, np.sum(array))


# Flatten an array
def flatten(array):
    list(itertools.chain.from_iterable(array))


class Model(object):
    def __init__(self, mutation_rate_prdm9=1.0 * 10 ** -5, mutation_rate_hotspot=1.0 * 10 ** -7,
                 population_size=10 ** 4, recombination_rate=1.0 * 10 ** -3, alpha=1.,
                 fitness='linear', drift=True, linearized=True):
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
        # Mu is the scaled mutation rate
        self.mu = 2 * self.population_size * self.mutation_rate_prdm9
        # Rho is the scaled erosion rate
        self.rho = 4 * self.population_size * self.mutation_rate_hotspot * self.recombination_rate
        self.epsilon = np.sqrt(self.rho / (self.mu * self.alpha))
        assert self.rho < 0.5, "The scaled erosion rate is too large, decrease either the " \
                               "recombination rate, the erosion rate or the population size"
        assert self.rho > 0.0000000001, "The scaled erosion rate is too low, increase either the " \
                                        "recombination rate, the erosion rate or the population size"

    def forward(self):
        self.mutation()
        self.erosion_and_selection()

    def mutation(self):
        # The number of new alleles is poisson distributed
        new_alleles = np.random.poisson(self.mu)
        # Initialize new alleles in the population only if necessary
        if new_alleles > 0:
            self.prdm9_polymorphism -= np.random.multinomial(new_alleles, sum_to_one(self.prdm9_polymorphism))
            self.prdm9_polymorphism = np.append(self.prdm9_polymorphism, np.ones(new_alleles))
            self.hotspots_activity = np.append(self.hotspots_activity, np.ones(new_alleles))
            self.prdm9_longevity = np.append(self.prdm9_longevity, np.zeros(new_alleles))
            self.ids = np.append(self.ids, range(self.id, self.id + new_alleles))
            self.id += new_alleles

    def erosion_and_selection(self):
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
        if self.fitness_family == 3:
            return np.power(x, self.sigmoid_slope) / (
                np.power(x, self.sigmoid_slope) + np.power(self.alpha, self.sigmoid_slope))
        elif self.fitness_family == 2:
            return np.power(x, self.alpha)
        else:
            return x

    def derivative_log_fitness(self, x):
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
        if mean_activity == 0.:
            return 0.
        else:
            return -1 * mean_activity * np.real(lambertw(-np.exp(- 1. / mean_activity) / mean_activity))

    def mean_activity_estimation(self):
        return brentq(lambda x: self.mean_activity_equation(x), 0, 1)

    def mean_activity_equation(self, x):
        return self.derivative_log_fitness(x) * (1 - x) * (1 - self.activity_limit(x)) - (x * self.rho / self.mu)

    def mean_activity_small_load(self):
        return 1 - self.epsilon / np.sqrt(2)

    def activity_limit_small_load(self):
        return 1 - np.sqrt(2) * self.epsilon

    def frequencies_wrt_activity(self, mean_activity, l_limit):
        l = np.linspace(l_limit, 1)
        x = 1. - l + mean_activity * np.log(l)
        x *= self.derivative_log_fitness(mean_activity) / (2 * self.rho)
        return l, np.clip(x, 0., 1.)

    def frequencies_wrt_activity_estimation(self):
        mean_activity = self.mean_activity_estimation()
        l_limit = self.activity_limit(mean_activity)
        return self.frequencies_wrt_activity(mean_activity, l_limit)

    def frequencies_wrt_activity_small_load(self):
        mean_activity = self.mean_activity_small_load()
        l_limit = self.activity_limit_small_load()
        return self.frequencies_wrt_activity(mean_activity, l_limit)

    def prdm9_diversity_estimation(self):
        mean_activity = self.mean_activity_estimation()
        if mean_activity == 1.:
            return self.population_size
        else:
            diff = 1 - 2 * mean_activity + self.activity_limit(mean_activity)
            diversity = 4 * self.rho / (self.derivative_log_fitness(mean_activity) * diff)
            return max(1., diversity)

    def prdm9_diversity_small_load(self):
        return max(1., 12 * self.rho / (self.alpha * self.epsilon**2))
        # return max(1., 12 * self.rho / (self.alpha * self.epsilon**2) * (1 - self.epsilon/np.sqrt(2)))

    def landscape_variance_estimation(self):
        mean_activity = self.mean_activity_estimation()
        diff = 1 - 2 * mean_activity + self.activity_limit(mean_activity)
        landscape = mean_activity * diff * self.derivative_log_fitness(mean_activity) / (4 * self.rho)
        return min(1., landscape)

    def landscape_variance_small_load(self):
        return 1. / self.prdm9_diversity_small_load()
        # landscape = 1. / (12 * self.rho / (self.alpha * self.epsilon**2)) * (1 - self.epsilon/np.sqrt(2))
        # if landscape < 0:
        # landscape = 1.
        # return min(1., landscape)

    def probability_fixation(self, mean_activity):
        if mean_activity == 0.:
            return 1.
        elif mean_activity == 1.:
            return 1. / self.population_size
        else:
            selection = self.derivative_log_fitness(mean_activity) * (1. - mean_activity)
            return (1. - np.exp(-selection)) / (1. - np.exp(-2. * self.population_size * selection))

    def turn_over_estimation(self):
        mean_activity = self.mean_activity_estimation()
        return self.prdm9_diversity_estimation() / (self.mu * self.probability_fixation(mean_activity))

    def turn_over_small_load(self):
        return self.prdm9_diversity_small_load() / (self.mu * self.alpha * (1. - self.mean_activity_small_load()))

    def __str__(self):
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
class DataSnapshots(object):
    def __init__(self):
        self.prdm9_frequencies, self.hotspots_activity, self.prdm9_fitness = [], [], []

        self.nbr_prdm9, self.ids, self.prdm9_longevities = [], [], []

    def store(self, step):
        self.prdm9_frequencies.append(sum_to_one(step.prdm9_polymorphism))
        self.prdm9_fitness.append(step.prdm9_fitness)
        self.hotspots_activity.append(step.hotspots_activity)
        self.ids.append(step.ids)
        self.nbr_prdm9.append(step.prdm9_polymorphism)
        self.prdm9_longevities.append(step.prdm9_longevity)

    # mean over the population for each sample
    def mean_activity_array(self):
        return np.array(map(lambda erosion, freq: self.n_moment(erosion, freq, 1), self.hotspots_activity,
                            self.prdm9_frequencies))

    # mean over the samples (mean over the simulation trajectory)
    def mean_activity(self):
        return np.mean(self.mean_activity_array())

    def prdm9_diversity_array(self):
        return np.array(map(lambda frequencies: 1. / np.sum(np.power(frequencies, 2)), self.prdm9_frequencies))

    # mean over the samples (mean over the simulation trajectory)
    def prdm9_diversity(self):
        return np.mean(self.prdm9_diversity_array())

    def landscape_variance_array(self):
        return np.array(map(lambda erosion, freq: self.n_moment(freq, erosion, 2), self.hotspots_activity,
                            self.prdm9_frequencies))

    # mean over the samples (mean over the simulation trajectory)
    def landscape_variance(self):
        return np.mean(self.landscape_variance_array())

    def turn_over(self):
        return self.dichotomic_search(0.5)

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

    @staticmethod
    def n_moment(x, frequencies, n):
        if n > 1:
            return np.sum((x ** n) * frequencies)
        else:
            return np.sum(x * frequencies)


# a simulation takes an instance of model as a parameter
# creates a DataSnapshots object
# run the model and stores states of the model at regular time intervals into snapshot object

class Simulation(object):
    def __init__(self, model, loops=10):
        self.t_max = 0
        self.nbr_of_steps = 0.
        self.loops = loops
        self.model = model
        self.data = DataSnapshots()
        self.generations = []

    def __str__(self):
        return "tmax=%.1e_" % self.t_max + str(self.model)

    def caption(self):
        return "%.1e generations computed \n \n" % self.t_max + \
               "Model parameters : \n" + self.model.caption()

    def burn_in(self):
        t = 0
        initial_variants = set(self.model.ids)
        while len(initial_variants & set(self.model.ids)) > 0:
            self.model.forward()
            t += 1

        self.nbr_of_steps = max(100, 10000 / len(self.model.ids))
        self.t_max = 10 * (int(max(int(self.loops * t), self.nbr_of_steps)) / 10 + 1)

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

        plt.subplot(312)
        plt.scatter(generations, flatten(self.data.hotspots_activity), color=YELLOW, lw=0)
        plt.plot(self.generations, self.data.mean_activity_array(), color=GREEN, lw=3)
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
        self.save_estimation()
        self.save_phase_plan()
        plt.close('all')
        print str(self)

    def save_estimation(self):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1920 / my_dpi), dpi=my_dpi)
        mean_activity = self.model.mean_activity_estimation()
        theta = np.linspace(0, 1, 100)
        plt.plot(theta, np.array(map(lambda x: self.model.mean_activity_equation(x) + x, theta)), color=BLUE, linewidth=2)
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
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1920 / my_dpi), dpi=my_dpi)

        a = list(itertools.chain.from_iterable(self.data.hotspots_activity))
        b = list(itertools.chain.from_iterable(self.data.prdm9_frequencies))
        c = list(itertools.chain.from_iterable(self.data.prdm9_longevities))
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
    def __init__(self, model, parameter="null", nbr_of_simulations=20, scale=10 ** 2, loops=10):
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
        return "Batch of %s simulations. \n" % self.nbr_of_simulations + self.parameter_name + \
               "is scaled %.1e times.\n" % self.scale + self.model.caption()

    def run(self, nbr_of_cpu=7, directory_id=None):
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

    def save_figure(self):
        map(lambda x: x.save_figure(), self.simulations)

    def pickle(self):
        pickle.dump(self, open(self.parameter_name + ".p", "wb"))

    def plot_series(self, series, color, caption):
        parameter_caption = {"fitness": "The fitness parameter",
                             "mutation": "Mutation rate of PRDM9",
                             "erosion": "Mutation rate of the hotspots",
                             "population": "The population size",
                             "recombination": "The recombination rate of the hotspots"}[self.parameter]
        mean = map(lambda serie: np.mean(serie), series)
        plt.plot(self.parameter_range, mean, color=color, linewidth=2)
        sigma = map(lambda serie: np.sqrt(np.var(serie)), series)
        y_max = np.add(mean, sigma)
        y_min = np.subtract(mean, sigma)
        plt.fill_between(self.parameter_range, y_max, y_min, color=color, alpha=0.3)
        plt.xlabel(parameter_caption)
        plt.ylabel(caption)
        if self.parameter == "fitness" and self.model.fitness_family == 3:
            plt.xscale('linear')
        else:
            plt.xscale('log')


class Batch(list):
    def save_figures(self):
        self.save_figure('mean_activity', 'linear')
        self.save_figure('prdm9_diversity', 'log')
        self.save_figure('landscape_variance', 'log')
        self.save_figure('turn_over', 'log')

    def save_figure(self, summary_statistic="mean_activity", yscale='log'):
        caption = {"mean_activity": "The mean activity of the hotspots",
                   "prdm9_diversity": "The diversity of PRDM9",
                   "landscape_variance": "The hotspots landscape variance",
                   "turn_over": "The turn-over time"}[summary_statistic]
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        for j, batch in enumerate(self):
            plt.subplot(2, 2, j + 1)

            models = map(lambda sim: sim.model, batch.simulations)

            if summary_statistic == 'turn_over':
                lag = map(lambda sim: sim.generations[sim.data.dichotomic_search(0.5)], batch.simulations)
                plt.plot(batch.parameter_range, lag, color=BLUE)
                plt.xscale('log')
            else:
                batch.plot_series(
                    map(lambda sim: np.array(getattr(sim.data, summary_statistic + "_array")()), batch.simulations),
                    BLUE, caption)
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
        pickle.dump(self, open("Batch.p", "wb"))


def load_batch(dir_id):
    set_dir("/tmp/" + dir_id)
    simulation_batch = pickle.load(open("Batch.p", "rb"))
    simulation_batch.save_figures()
    map(lambda x: x.save_figure(), simulation_batch)


def make_batch():
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
                                               nbr_of_simulations=128, scale=10 ** 4, loops=30))
    for simulation_along_parameter in batch:
        simulation_along_parameter.run(nbr_of_cpu=8)
    batch.pickle()
    batch.save_figures()

if __name__ == '__main__':
    # load_batch("0E1F0873")
    make_batch()
