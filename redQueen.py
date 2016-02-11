from scipy.stats.kde import gaussian_kde
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import time
import os


def set_dir(path):
    path = os.getcwd() + path
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    os.chdir(path)


def execute(x):
    return x.run()


# The fitness function, given the erosion of the hotspots
def w(y, inflexion):
    return y ** 4 / (y ** 4 + inflexion ** 4)


# The shannon entropy
def entropy(frequencies):
    return np.exp(np.sum(-frequencies * np.log(frequencies)))


# The fitness function, given the erosion of the hotspots
def n_moment(x, frequencies, n):
    if n > 1:
        return np.sum((x ** n) * frequencies)
    else:
        return np.sum(x * frequencies)


def cv(x, frequencies):
    squared_first_moment = n_moment(x, frequencies, 1) ** 2
    return (n_moment(x, frequencies, 2) - squared_first_moment) / squared_first_moment


class Simulation(object):
    def __init__(self, mutation_rate_prdm9=1.0 * 10 ** -5,
                 erosion_rate_hotspot=1.0 * 10 ** -7,
                 population_size=10 ** 4,
                 inflexion=0.5,
                 neutral=False,
                 scaling=30):
        initial_number_of_alleles = 10
        self.scaling = scaling
        self.inflexion = inflexion
        assert self.inflexion > 0.001, "The inflexion point must be greater than 0.001"
        assert self.inflexion <= 1, "The inflexion point must be lower than 1"

        self.number_of_steps = 10000  # Number of steps at which we make computations
        self.t_max = int(self.scaling * population_size)  # Number of generations
        assert self.t_max > self.number_of_steps, "The scaling factor is too low, there won't" \
                                                  "be enough generation computed"

        self.mutation_rate_prdm9 = mutation_rate_prdm9  # The rate at which new allele for PRDM9 are created
        self.erosion_rate_hotspot = erosion_rate_hotspot  # The rate at which the hotspots are eroded
        self.population_size = float(population_size)  # population size
        self.neutral = neutral  # If the fitness function is neutral

        self.scaled_erosion_rate = self.erosion_rate_hotspot * self.population_size
        assert self.scaled_erosion_rate < 0.1, "The population size is too large for this value of erosion rate"
        assert self.scaled_erosion_rate > 0.0000001, "The population size is too low for this value of erosion rate"

        self.scaled_mutation_rate = self.population_size * self.mutation_rate_prdm9

        self.prdm9_polymorphism = np.ones(initial_number_of_alleles) * self.population_size / initial_number_of_alleles
        self.hotspots_erosion = np.ones(initial_number_of_alleles)
        self.prdm9_longevity = np.zeros(initial_number_of_alleles)
        self.prdm9_high_frequency = np.zeros(initial_number_of_alleles, dtype=bool)

        self.exact_prdm9_frequencies = np.ones(initial_number_of_alleles) / initial_number_of_alleles
        self.exact_hotspots_erosion = np.ones(initial_number_of_alleles)
        self.exact_prdm9_longevity = np.zeros(initial_number_of_alleles)

        self.prdm9_polymorphism_cum = []
        self.hotspots_erosion_cum = []
        self.prdm9_fitness_cum = []
        self.prdm9_high_frequency_cum = []
        self.prdm9_longevity_cum = []

        self.exact_hotspots_erosion_cum = []
        self.exact_prdm9_frequencies_cum = []
        self.exact_prdm9_longevity_cum = []

        self.prdm9_nb_alleles = []
        self.prdm9_entropy_alleles = []
        self.hotspots_erosion_mean = []
        self.hotspots_erosion_cv = []
        self.prdm9_longevity_mean = []
        self.prdm9_longevity_cv = []

        self.exact_prdm9_entropy_alleles = []
        self.exact_hotspots_erosion_mean = []
        self.exact_hotspots_erosion_cv = []
        self.exact_prdm9_longevity_mean = []
        self.exact_prdm9_longevity_cv = []

        self.generations = []

    def __str__(self):
        return "The mutation rate of PRDM9: %.1e" % self.mutation_rate_prdm9 + \
               "\nThe erosion rate of the hotspots : %.1e" % self.erosion_rate_hotspot + \
               "\nThe population size : %.1e" % self.population_size + \
               "\nThe number of generations computed : %.1e" % self.t_max

    def __repr__(self):
        return "The polymorphism of PRDM9: %s" % self.prdm9_polymorphism + \
               "\nThe strength of the hotspots : %s" % self.hotspots_erosion + \
               "\nThe longevity hotspots : %s" % self.prdm9_longevity

    def run(self):
        start_time = time.time()
        step = float(self.t_max) / self.number_of_steps
        step_t = 0.

        for t in range(self.t_max):

            # Randomly create new alleles of PRDM9
            new_alleles = np.random.poisson(2 * self.scaled_mutation_rate)

            if new_alleles > 0:
                self.prdm9_polymorphism -= np.random.multinomial(new_alleles, np.divide(
                        self.prdm9_polymorphism, self.population_size))
                self.prdm9_polymorphism = np.append(self.prdm9_polymorphism,
                                                    np.ones(new_alleles))
                self.prdm9_longevity = np.append(self.prdm9_longevity, np.zeros(new_alleles))
                self.hotspots_erosion = np.append(self.hotspots_erosion, np.ones(new_alleles))
                self.prdm9_high_frequency = np.append(self.prdm9_high_frequency, np.zeros(new_alleles))

                self.remove_extincted()

                #  we should check for selection coefficient and probability of dying immediately
                exact_erosion_mean = np.sum(self.exact_prdm9_frequencies * self.exact_hotspots_erosion)
                s_initial = (1 - exact_erosion_mean) / exact_erosion_mean
                fixation_probability = (1 - np.exp(-s_initial)) / (1 - np.exp(-self.population_size * s_initial))
                fixed = np.sum(np.array(map(lambda x: x < fixation_probability, np.random.uniform(size=new_alleles)),
                                        dtype=bool))
                if fixed > 0:
                    self.exact_prdm9_frequencies *= (1 - float(fixed) / self.population_size)
                    self.exact_prdm9_frequencies = np.append(self.exact_prdm9_frequencies,
                                                             np.ones(fixed) / self.population_size)
                    self.exact_hotspots_erosion = np.append(self.exact_hotspots_erosion, np.ones(fixed))
                    self.exact_prdm9_longevity = np.append(self.exact_prdm9_longevity, np.zeros(fixed))

            # Compute the PRDM9 frequencies for convenience
            prdm9_frequencies = np.divide(self.prdm9_polymorphism, self.population_size)

            # Compute the fitness for each allele
            nb_prdm9_alleles = self.prdm9_polymorphism.size
            if self.neutral:
                fitness_matrix = np.ones([nb_prdm9_alleles, nb_prdm9_alleles])
            else:
                fitness_matrix = np.empty([nb_prdm9_alleles, nb_prdm9_alleles])
                for i in range(nb_prdm9_alleles):
                    for j in range(nb_prdm9_alleles):
                        fitness_matrix[i, j] = (self.hotspots_erosion[i] + self.hotspots_erosion[j]) / 2

            fitness_vector = np.dot(fitness_matrix, prdm9_frequencies)

            step_t += 1
            if step_t > step and t / float(self.t_max) > 0.01:
                step_t -= step
                self.generations.append(t)
                sqrt_population_size = np.sqrt(self.population_size)
                self.prdm9_high_frequency += np.array(
                        map(lambda f: f > sqrt_population_size, self.prdm9_polymorphism), dtype=bool)
                self.prdm9_nb_alleles.append(self.prdm9_polymorphism.size)
                self.prdm9_entropy_alleles.append(entropy(prdm9_frequencies))

                self.hotspots_erosion_mean.append(n_moment(1 - self.hotspots_erosion, prdm9_frequencies, 1))
                self.hotspots_erosion_cv.append(cv(1 - self.hotspots_erosion, prdm9_frequencies))
                self.prdm9_longevity_mean.append(n_moment(self.prdm9_longevity, prdm9_frequencies, 1))
                self.prdm9_longevity_cv.append(cv(self.prdm9_longevity, prdm9_frequencies))
                self.prdm9_polymorphism_cum.extend(prdm9_frequencies)
                self.hotspots_erosion_cum.extend(1 - self.hotspots_erosion)
                self.prdm9_longevity_cum.extend(self.prdm9_longevity)
                self.prdm9_fitness_cum.extend(fitness_vector)
                self.prdm9_high_frequency_cum.extend(self.prdm9_high_frequency)

                self.exact_prdm9_entropy_alleles.append(entropy(self.exact_prdm9_frequencies))
                self.exact_hotspots_erosion_mean.append(
                        n_moment(1 - self.exact_hotspots_erosion, self.exact_prdm9_frequencies, 1))
                self.exact_hotspots_erosion_cv.append(cv(1 - self.exact_hotspots_erosion, self.exact_prdm9_frequencies))
                self.exact_prdm9_longevity_mean.append(
                        n_moment(self.exact_prdm9_longevity, self.exact_prdm9_frequencies, 1))
                self.exact_prdm9_longevity_cv.append(cv(self.exact_prdm9_longevity, self.exact_prdm9_frequencies))
                self.exact_prdm9_frequencies_cum.extend(self.exact_prdm9_frequencies)
                self.exact_hotspots_erosion_cum.extend(1 - self.exact_hotspots_erosion)
                self.exact_prdm9_longevity_cum.extend(self.exact_prdm9_longevity)

                if time.time() - start_time > 720:
                    self.t_max = t
                    print "Breaking the loop, time over 720s"
                    break

            dt = 1 / self.population_size
            exact_erosion_mean = np.sum(self.exact_prdm9_frequencies * self.exact_hotspots_erosion)
            euler_prdm9_frequencies = self.exact_prdm9_frequencies * (
                self.exact_hotspots_erosion - exact_erosion_mean) / (2 * exact_erosion_mean)

            euler_hotspots_erosion = self.scaled_erosion_rate * (
                self.exact_prdm9_frequencies * self.exact_hotspots_erosion)

            self.exact_prdm9_frequencies += euler_prdm9_frequencies
            self.exact_hotspots_erosion -= euler_hotspots_erosion
            # Exponential decay for hotspots erosion
            # hotspots_erosion *= np.exp( - erosion_rate_hotspot * prdm9_frequencies)
            self.hotspots_erosion *= (
                1. - self.scaled_erosion_rate * prdm9_frequencies)

            distribution_vector = fitness_vector * prdm9_frequencies

            # Randomly pick the new generation according to the fitness vector
            self.prdm9_polymorphism = np.random.multinomial(int(self.population_size),
                                                            np.divide(distribution_vector,
                                                                      np.sum(distribution_vector)))

            # Increase the longevity of survivors by 1
            self.prdm9_longevity += 1
            self.exact_prdm9_longevity += 1

            self.remove_extincted()
            self.remove_exact_extincted()

        self.save_figure()
        return self

    def remove_extincted(self):
        # Remove the extinct alleles
        remove_extincted = np.array(map(lambda x: x != 0, self.prdm9_polymorphism), dtype=bool)
        if not remove_extincted.all():
            self.prdm9_polymorphism = self.prdm9_polymorphism[remove_extincted]
            self.hotspots_erosion = self.hotspots_erosion[remove_extincted]
            self.prdm9_longevity = self.prdm9_longevity[remove_extincted]
            self.prdm9_high_frequency = self.prdm9_high_frequency[remove_extincted]

    def remove_exact_extincted(self):
        # Remove the extinct alleles
        cut_off = float(1) / self.population_size
        remove_extincted = np.array(map(lambda x: x > cut_off, self.exact_prdm9_frequencies), dtype=bool)
        if not remove_extincted.all():
            self.exact_prdm9_frequencies = self.exact_prdm9_frequencies[remove_extincted]
            self.exact_hotspots_erosion = self.exact_hotspots_erosion[remove_extincted]
            self.exact_prdm9_longevity = self.exact_prdm9_longevity[remove_extincted]

    def save_figure(self):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        plt.subplot(331)
        plt.text(0.05, 0.98, self, fontsize=14, verticalalignment='top')
        theta = np.arange(0.0, 1.0, 0.01)
        if self.neutral:
            plt.plot(theta, np.ones(theta.size), color='red')
        else:
            vectorized_w = np.vectorize(lambda x: w(x, self.inflexion))
            plt.plot(theta, vectorized_w(theta), color='red')
        plt.title('The fitness function')
        plt.xlabel('x')
        plt.ylabel('w(x)')

        plt.subplot(332)
        plt.plot(self.generations, self.prdm9_entropy_alleles, color='red')
        plt.plot(self.generations, self.exact_prdm9_entropy_alleles, color='blue')
        plt.title('Efficient number of PRDM9 alleles over time \n Stochastic : red | Deterministic : blue')
        plt.xlabel('Generation')
        plt.ylabel('Number of alleles')
        plt.yscale('log')

        plt.subplot(333)
        plt.hexbin(self.prdm9_fitness_cum, self.prdm9_polymorphism_cum, self.prdm9_longevity_cum, gridsize=200,
                   bins='log')
        plt.title('PRMD9 frequency vs PRDM9 fitness')
        plt.xlabel('PRDM9 fitness')
        plt.ylabel('PRMD9 frequency')

        plt.subplot(334)
        plt.hexbin(self.hotspots_erosion_cum, self.prdm9_polymorphism_cum, self.prdm9_longevity_cum, gridsize=200,
                   bins='log')
        plt.title('PRMD9 frequency vs hotspot erosion (Stochastic)')
        plt.xlabel('Erosion')
        plt.ylabel('PRMD9 frequency')

        plt.subplot(335)
        plt.hexbin(self.exact_hotspots_erosion_cum, self.exact_prdm9_frequencies_cum, self.exact_prdm9_longevity_cum,
                   gridsize=200,
                   bins='log')
        plt.title('PRMD9 frequency vs hotspot erosion (Stochastic)')
        plt.xlabel('Erosion')
        plt.ylabel('PRMD9 frequency')

        plt.subplot(336)
        plt.hexbin(self.hotspots_erosion_cum, self.prdm9_fitness_cum, self.prdm9_longevity_cum, gridsize=200,
                   bins='log')
        plt.title('PRDM9 fitness vs hotspot erosion')
        plt.xlabel('hotspot erosion')
        plt.ylabel('PRDM9 fitness')

        plt.subplot(337)
        x = np.linspace(0, 1, 100)
        stochastic_density = gaussian_kde(self.prdm9_polymorphism_cum)(x)
        exact_density = gaussian_kde(self.exact_prdm9_frequencies_cum)(x)
        plt.plot(x, stochastic_density, 'r')
        plt.fill_between(x, stochastic_density, np.zeros(100), color='red', alpha=0.3)
        plt.plot(x, exact_density, 'b')
        plt.fill_between(x, exact_density, np.zeros(100), color='blue', alpha=0.3)
        plt.title('PRDM9 frequencies histogram')
        plt.xlabel('PRDM9 Frequencies')
        plt.ylabel('Frequency')

        plt.subplot(338)
        x = np.linspace(0, 1, 100)
        stochastic_density = gaussian_kde(self.hotspots_erosion_cum)(x)
        exact_density = gaussian_kde(self.exact_hotspots_erosion_cum)(x)
        plt.plot(x, stochastic_density, 'r')
        plt.fill_between(x, stochastic_density, np.zeros(100), color='red', alpha=0.3)
        plt.plot(x, exact_density, 'b')
        plt.fill_between(x, exact_density, np.zeros(100), color='blue', alpha=0.3)
        plt.title('Erosion of the hotspots histogram')
        plt.xlabel('Erosion of the hotspots')
        plt.ylabel('Frequency')

        plt.subplot(339)
        x = np.linspace(1, np.max(np.append(self.prdm9_longevity_cum, self.exact_prdm9_longevity_cum)), 100)
        stochastic_density = gaussian_kde(self.prdm9_longevity_cum)(x)
        exact_density = gaussian_kde(self.exact_prdm9_longevity_cum)(x)
        plt.plot(x, stochastic_density, 'r')
        plt.fill_between(x, stochastic_density, np.zeros(100), color='red', alpha=0.3)
        plt.plot(x, exact_density, 'b')
        plt.fill_between(x, exact_density, np.zeros(100), color='blue', alpha=0.3)
        plt.title('Longevity of PRDM9 alleles histogram')
        plt.xlabel('Longevity')
        plt.ylabel('Frequency')

        plt.tight_layout()

        filename = "u=%.1e" % self.mutation_rate_prdm9 + \
                   "_r=%.1e" % self.erosion_rate_hotspot + \
                   "_n=%.1e" % self.population_size + \
                   "_p=%.1e" % self.inflexion + \
                   "_t=%.1e" % self.t_max
        if self.neutral:
            filename += "neutral"
        plt.savefig(filename + '.png')
        plt.clf()
        print filename
        return filename


class BatchSimulation(object):
    def __init__(self,
                 mutation_rate_prdm9,
                 erosion_rate_hotspot,
                 population_size,
                 inflexion=0.5,
                 axis="",
                 number_of_simulations=20,
                 scale=10 ** 2,
                 neutral=False):
        axis_hash = {"mutation": 0, "erosion": 1, "population": 2, "fitness": 3, "mutation&erosion": 4,
                     "erosion&mutation": 4, "scaling": 5}
        assert axis in axis_hash.keys(), "Axis must be either 'population', 'mutation', 'erosion'," \
                                         "'fitness', 'mutation&erosion', or 'scaling'"
        assert scale > 1, "The scale parameter must be greater than one"
        self.axis = axis_hash[axis]
        self.axis_str = {0: "Mutation rate of PRDM9", 1: "Erosion rate of the hotspots", 2: "The population size",
                         3: "The fitness inflexion point", 4: "Scaled mutation rate and erosion rate",
                         5: "The scaling factor"}[self.axis]

        self.inflexion = inflexion
        self.neutral = neutral  # If the fitness function is neutral
        self.mutation_rate_prdm9 = mutation_rate_prdm9  # The rate at which new allele for PRDM9 are created
        self.erosion_rate_hotspot = erosion_rate_hotspot  # The rate at which the hotspots are eroded
        self.population_size = population_size
        parameters = [self.mutation_rate_prdm9, self.erosion_rate_hotspot, self.population_size, self.inflexion]

        if self.axis == 4 or self.axis == 5:
            self.axis_range = np.logspace(0, np.log10(scale),
                                          num=number_of_simulations)
        elif self.axis == 3:
            self.axis_range = np.linspace(0.01, 1, num=number_of_simulations)
        else:
            self.axis_range = np.logspace(np.log10(parameters[self.axis]), np.log10(parameters[self.axis] * scale),
                                          num=number_of_simulations)

        self.simulations = []

        for axis_current in self.axis_range:
            if self.axis == 4:
                parameters[0] = self.mutation_rate_prdm9 * axis_current
                parameters[1] = self.erosion_rate_hotspot * axis_current
            elif self.axis == 5:
                parameters[0] = self.mutation_rate_prdm9 / axis_current
                parameters[1] = self.erosion_rate_hotspot / axis_current
                parameters[2] = self.population_size * axis_current
            else:
                parameters[self.axis] = axis_current
            self.simulations.append(Simulation(*parameters))

    def run(self, number_of_cpu=4):
        set_dir("/" + self.filename())
        if number_of_cpu > 1:
            pool = Pool(number_of_cpu)
            self.simulations = pool.map(execute, self.simulations)
        else:
            map(lambda x: x.run(), self.simulations)
        self.save_figure()

    def __str__(self):
        return "The mutation rate of PRDM9: %.1e" % self.mutation_rate_prdm9 + \
               "\nThe erosion rate of the hotspots : %.1e" % self.erosion_rate_hotspot + \
               "\nThe population size : %.1e" % self.population_size + \
               "\nBatch along the axis : %s" % self.axis_str

    def save_figure(self):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        plt.subplot(321)
        plt.text(0.05, 0.98, self, fontsize=14, verticalalignment='top')
        theta = np.arange(0.0, 1.0, 0.01)
        if self.neutral:
            plt.plot(theta, np.ones(theta.size), color='red')
        else:
            if self.axis == 3:
                for inflexion in self.axis_range:
                    vectorized_w = np.vectorize(lambda x: w(x, inflexion))
                    plt.plot(theta, vectorized_w(theta), color='red')
            else:
                vectorized_w = np.vectorize(lambda x: w(x, self.inflexion))
                plt.plot(theta, vectorized_w(theta), color='red')
        plt.title('The fitness function')
        plt.xlabel('x')
        plt.ylabel('w(x)')

        plt.subplot(322)
        plt.plot(self.axis_range, map(lambda sim: np.mean(sim.prdm9_nb_alleles), self.simulations),
                 color='blue')
        plt.plot(self.axis_range, map(lambda sim: np.mean(sim.prdm9_entropy_alleles), self.simulations),
                 color='green')
        y_max = map(lambda sim: np.percentile(sim.prdm9_entropy_alleles, 98), self.simulations)
        y_min = map(lambda sim: np.percentile(sim.prdm9_entropy_alleles, 2), self.simulations)
        plt.fill_between(self.axis_range, y_max, y_min, color='green', alpha=0.3)
        plt.title('Number of PRDM9 alleles (blue) and \n efficient number of PRDM9 alleles (green)'
                  '\n for different %s' % self.axis_str)
        plt.xlabel(self.axis_str)
        plt.ylabel('Number of alleles')
        if not self.axis == 3:
            plt.xscale('log')
        plt.yscale('log')

        plt.subplot(323)
        plt.plot(self.axis_range, map(lambda sim: np.mean(sim.hotspots_erosion_mean), self.simulations),
                 color='blue')
        y_max = map(lambda sim: np.percentile(sim.hotspots_erosion_mean, 98), self.simulations)
        y_min = map(lambda sim: np.percentile(sim.hotspots_erosion_mean, 2), self.simulations)
        plt.fill_between(self.axis_range, y_max, y_min, color='green', alpha=0.3)
        plt.title('Mean hotspot erosion \n for different %s' % self.axis_str)
        plt.xlabel(self.axis_str)
        plt.ylabel('Hotspot erosion')
        if not self.axis == 3:
            plt.xscale('log')

        plt.subplot(324)
        plt.plot(self.axis_range, map(lambda sim: np.mean(sim.hotspots_erosion_cv), self.simulations),
                 color='blue')
        y_max = map(lambda sim: np.percentile(sim.hotspots_erosion_cv, 98), self.simulations)
        y_min = map(lambda sim: np.percentile(sim.hotspots_erosion_cv, 2), self.simulations)
        plt.fill_between(self.axis_range, y_max, y_min, color='green', alpha=0.3)
        plt.title('Landscape hotspot erosion \n for different %s' % self.axis_str)
        plt.xlabel('Erosion rate')
        plt.ylabel('Hotspot erosion')
        if not self.axis == 3:
            plt.xscale('log')

        plt.subplot(325)
        plt.plot(self.axis_range, map(lambda sim: np.mean(sim.prdm9_longevity_mean), self.simulations),
                 color='blue')
        y_max = map(lambda sim: np.percentile(sim.prdm9_longevity_mean, 98), self.simulations)
        y_min = map(lambda sim: np.percentile(sim.prdm9_longevity_mean, 2), self.simulations)
        plt.fill_between(self.axis_range, y_max, y_min, color='green', alpha=0.3)
        plt.title('Mean PRDM9 longevity \n for different %s' % self.axis_str)
        plt.xlabel(self.axis_str)
        plt.ylabel('PRDM9 longevity')
        if not self.axis == 3:
            plt.xscale('log')

        plt.subplot(326)
        plt.plot(self.axis_range, map(lambda sim: np.mean(sim.prdm9_longevity_cv), self.simulations),
                 color='blue')
        y_max = map(lambda sim: np.percentile(sim.prdm9_longevity_cv, 98), self.simulations)
        y_min = map(lambda sim: np.percentile(sim.prdm9_longevity_cv, 2), self.simulations)
        plt.fill_between(self.axis_range, y_max, y_min, color='green', alpha=0.3)
        plt.title('Variance of PRDM9 longevity \n for different %s' % self.axis_str)
        plt.xlabel(self.axis_str)
        plt.ylabel('PRDM9 longevity')
        if not self.axis == 3:
            plt.xscale('log')

        plt.tight_layout()

        plt.savefig(self.filename() + '.png')
        plt.clf()
        print 'Simulation computed'
        return self

    def filename(self):
        return self.axis_str + \
               " u=%.1e" % self.mutation_rate_prdm9 + \
               "_e=%.1e" % self.erosion_rate_hotspot + \
               "_n=%.1e" % self.population_size


if __name__ == '__main__':
    set_dir("/tmp")
    batch_simulation = BatchSimulation(mutation_rate_prdm9=1.0 * 10 ** -5,
                                       erosion_rate_hotspot=1.0 * 10 ** -5,
                                       population_size=10 ** 3,
                                       axis='scaling',
                                       number_of_simulations=20,
                                       scale=10 ** 2)
    batch_simulation.run(number_of_cpu=7)
    # simulation = Simulation(mutation_rate_prdm9=5.0 * 10 ** -6,
    #                        erosion_rate_hotspot=2.0 * 10 ** -7,
    #                       population_size=4.0*10 ** 5)
    # simulation.run()
