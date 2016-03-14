from scipy.stats.kde import gaussian_kde
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import time
import os


def execute(x):
    return x.run()


def set_dir(path):
    path = os.getcwd() + path
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    os.chdir(path)


# The shannon entropy
def entropy(frequencies):
    return np.exp(np.sum(map(lambda x: (-x * np.log(x) if x > 0 else 0), frequencies)))


# The fitness function, given the erosion of the hotspots
def sum_to_one(array):
    return np.divide(array, np.sum(array))


# The fitness function, given the erosion of the hotspots
def n_moment(x, frequencies, n):
    if n > 1:
        return np.sum((x ** n) * frequencies)
    else:
        return np.sum(x * frequencies)


def normalized_var(x, frequencies):
    squared_first_moment = n_moment(x, frequencies, 1) ** 2
    return (n_moment(x, frequencies, 2) - squared_first_moment) / squared_first_moment


class SimulationStep(object):
    def __init__(self,
                 number_of_alleles,
                 mutation_rate_prdm9,
                 erosion_rate_hotspot,
                 population_size,
                 recombination_rate,
                 inflexion_point,
                 fitness_family,
                 cut_off,
                 alpha_zero,
                 discrete=True,
                 drift=True,
                 linearized=True):
        self.discrete = discrete
        if discrete:
            self.scaled_mutation_rate = population_size * mutation_rate_prdm9
            self.scaled_erosion_rate = erosion_rate_hotspot * population_size * recombination_rate
            assert self.scaled_erosion_rate < 0.1, "The scaled erosion rate is too large, decrease either the " \
                                                   "recombination rate, the erosion rate or the population size"
            assert self.scaled_erosion_rate > 0.000000001, "The scaled erosion rate is too low, increase either the " \
                                                           "recombination rate, the erosion rate or the population size"
            self.alpha = 1.
            self.population_size = population_size
            self.drift = drift
        else:
            self.cut_off = cut_off
            self.scaled_mutation_rate = population_size * mutation_rate_prdm9
            self.scaled_erosion_rate = population_size * erosion_rate_hotspot
            self.scaled_recombination_rate = population_size * recombination_rate
            self.population_size = 1.0
            self.alpha = alpha_zero
        self.prdm9_polymorphism = sum_to_one(np.random.sample(number_of_alleles)) * population_size
        self.hotspots_erosion = np.random.sample(number_of_alleles)
        self.prdm9_longevity = np.zeros(number_of_alleles)
        self.prdm9_fitness = np.ones(number_of_alleles)
        self.fitness_family = fitness_family
        self.inflexion_point = inflexion_point
        self.linearized = linearized
        self.t = 0

    def fitness(self, x):
        if self.fitness_family == 1:
            matrix = x
        elif self.fitness_family == 2:
            matrix = x ** 2
        else:
            k = 4
            matrix = x ** k / (x ** k + self.inflexion_point ** k)
        return np.power(matrix, self.alpha)

    def coefficient_fitness(self, x):
        if self.fitness_family == 1:
            return self.alpha * 1.0 / (2 * x)
        elif self.fitness_family == 2:
            return self.alpha * 1.0 / x
        else:
            k = 4
            return (self.alpha * k / 2) * self.inflexion_point ** 4 / (x * (x ** k + self.inflexion_point ** k))

    def forward(self, t=1):
        if self.discrete:
            self.discrete_forward()
        else:
            self.continuous_forward(t)

    def continuous_forward(self, t):
        dt = min(1. / (max(self.scaled_erosion_rate * self.scaled_recombination_rate,
                           self.alpha, self.scaled_mutation_rate * self.alpha) * 1), t / 10)
        self.t += t
        current_t = 0
        while current_t < t:
            current_t += dt
            self.continuous_new_alleles(dt)
            prdm9_frequencies = sum_to_one(self.prdm9_polymorphism)
            self.hotspots_erosion -= (self.scaled_erosion_rate * self.scaled_recombination_rate * dt) * (
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

    def continuous_new_alleles(self, dt):
        prdm9_frequencies = sum_to_one(self.prdm9_polymorphism)
        l_bar = float(np.sum(prdm9_frequencies * self.hotspots_erosion))
        s = 2 * self.coefficient_fitness(l_bar) * (1 - l_bar)
        fixed = np.random.poisson(self.scaled_mutation_rate * s * dt)
        if fixed > 0:
            self.prdm9_polymorphism *= (1 - float(fixed) * self.cut_off)
            self.prdm9_polymorphism = np.append(self.prdm9_polymorphism, np.ones(fixed) * 0.001)
            self.hotspots_erosion = np.append(self.hotspots_erosion, np.ones(fixed))
            self.prdm9_longevity = np.append(self.prdm9_longevity, np.zeros(fixed))

    def discrete_new_alleles(self, new_alleles):
        self.prdm9_polymorphism -= np.random.multinomial(new_alleles, sum_to_one(self.prdm9_polymorphism))
        self.prdm9_polymorphism = np.append(self.prdm9_polymorphism, np.ones(new_alleles))
        self.prdm9_longevity = np.append(self.prdm9_longevity, np.zeros(new_alleles))
        self.hotspots_erosion = np.append(self.hotspots_erosion, np.ones(new_alleles))

    def discrete_forward(self):
        new_alleles = np.random.poisson(self.scaled_mutation_rate)
        if new_alleles > 0:
            self.discrete_new_alleles(new_alleles)
        prdm9_frequencies = sum_to_one(self.prdm9_polymorphism)

        self.hotspots_erosion *= np.exp(- self.scaled_erosion_rate * prdm9_frequencies)

        # Compute the fitness for each allele
        if self.fitness_family == 0:
            distribution_vector = prdm9_frequencies
        else:
            if self.linearized:
                l_bar = np.sum(prdm9_frequencies * self.hotspots_erosion)
                self.prdm9_fitness = self.coefficient_fitness(l_bar) * (self.hotspots_erosion - l_bar)
                distribution_vector = prdm9_frequencies + self.prdm9_fitness * prdm9_frequencies
                if np.max(distribution_vector) > 1.:
                    print "Frequencies of PRDM9 went above 1 in discrete forward"
                    distribution_vector = sum_to_one(np.clip(distribution_vector, 0., 1.))
                if np.min(distribution_vector) < 0.:
                    print "Frequencies of PRDM9 went below 0 in discrete forward"
                    distribution_vector = sum_to_one(np.clip(distribution_vector, 0., 1.))
            else:
                fitness_matrix = self.fitness(np.add.outer(self.hotspots_erosion, self.hotspots_erosion) / 2)
                self.prdm9_fitness = np.dot(fitness_matrix, prdm9_frequencies)
                distribution_vector = sum_to_one(self.prdm9_fitness * prdm9_frequencies)

        if self.drift:
            self.prdm9_polymorphism = np.random.multinomial(int(self.population_size), distribution_vector).astype(
                float)
        else:
            self.prdm9_polymorphism = distribution_vector

        self.prdm9_longevity += 1. / self.population_size
        self.remove_dead_prdm9(cut_off=0)

    def remove_dead_prdm9(self, cut_off):
        remove_extincted = np.array(map(lambda x: x > cut_off, self.prdm9_polymorphism), dtype=bool)
        if not remove_extincted.all():
            self.prdm9_polymorphism = self.prdm9_polymorphism[remove_extincted]
            self.hotspots_erosion = self.hotspots_erosion[remove_extincted]
            self.prdm9_longevity = self.prdm9_longevity[remove_extincted]
            self.prdm9_fitness = self.prdm9_fitness[remove_extincted]

    def __repr__(self):
        return "The polymorphism of PRDM9: %s" % self.prdm9_polymorphism + \
               "\nThe strength of the hotspots : %s" % self.hotspots_erosion + \
               "\nThe longevity hotspots : %s" % self.prdm9_longevity


class SimulationData(object):
    def __init__(self):
        self.prdm9_frequencies_cum, self.hotspots_erosion_cum = [], []
        self.prdm9_fitness_cum, self.prdm9_longevity_cum = [], []

        self.prdm9_nb_alleles, self.prdm9_entropy_alleles = [], []
        self.hotspots_erosion_mean, self.hotspots_erosion_var = [], []
        self.prdm9_longevity_mean, self.prdm9_longevity_var = [], []

    def store(self, step):
        prdm9_frequencies = sum_to_one(step.prdm9_polymorphism)
        self.prdm9_frequencies_cum.extend(prdm9_frequencies)
        self.hotspots_erosion_cum.extend(1 - step.hotspots_erosion)
        self.prdm9_fitness_cum.extend(step.prdm9_fitness)
        self.prdm9_longevity_cum.extend(step.prdm9_longevity)

        self.prdm9_nb_alleles.append(prdm9_frequencies.size)
        self.prdm9_entropy_alleles.append(entropy(prdm9_frequencies))
        self.hotspots_erosion_mean.append(n_moment(1 - step.hotspots_erosion, prdm9_frequencies, 1))
        self.hotspots_erosion_var.append(normalized_var(1 - step.hotspots_erosion, prdm9_frequencies))
        self.prdm9_longevity_mean.append(n_moment(step.prdm9_longevity, prdm9_frequencies, 1))
        self.prdm9_longevity_var.append(normalized_var(step.prdm9_longevity, prdm9_frequencies))


class Simulation(object):
    def __init__(self,
                 mutation_rate_prdm9=1.0 * 10 ** -5,
                 erosion_rate_hotspot=1.0 * 10 ** -7,
                 population_size=10 ** 4,
                 recombination_rate=1.0 * 10 ** -3,
                 inflexion_point=0.5,
                 fitness='linear',
                 cut_off=0.01,
                 alpha_zero=1.0 * 10 ** 4,
                 scaling=100):
        initial_number_of_alleles = 10
        self.mutation_rate_prdm9 = mutation_rate_prdm9  # The rate at which new allele for PRDM9 are created
        self.erosion_rate_hotspot = erosion_rate_hotspot  # The rate at which the hotspots are eroded
        self.recombination_rate = recombination_rate
        self.population_size = float(population_size)  # population size
        self.cut_off = cut_off

        self.alpha_zero = alpha_zero
        fitness_hash = {"constant": 0, "linear": 1, "quadratic": 2, "sigmoid": 3}
        assert fitness in fitness_hash.keys(), "Parameter 'fitness' must be a string:  ['linear','constant'," \
                                               "'quadratic','sigmoid'] or a function"
        self.fitness_family = fitness_hash[fitness]
        self.inflexion_point = inflexion_point

        self.scaling = scaling
        self.number_of_steps = 1000  # Number of steps at which we make computations
        self.t_max = max(int(self.scaling * population_size), self.number_of_steps)  # Number of generations

        params_simulations = [[True, True, True], [True, True, False]]
        self.steps = []
        self.data = []
        for params in params_simulations:
            self.steps.append(
                SimulationStep(initial_number_of_alleles, self.mutation_rate_prdm9, self.erosion_rate_hotspot,
                               self.population_size, self.recombination_rate, self.inflexion_point,
                               self.fitness_family, self.cut_off, self.alpha_zero,
                               *params))
            self.data.append(SimulationData())

        self.generations = []

    def __str__(self):
        return "The mutation rate of PRDM9: %.1e" % self.mutation_rate_prdm9 + \
               "\nThe erosion rate of the hotspots : %.1e" % self.erosion_rate_hotspot + \
               "\nThe population size : %.1e" % self.population_size + \
               "\nThe cut-off : %.1e" % self.cut_off + \
               "\nThe number of generations computed : %.1e" % self.t_max

    def run(self):
        start_time = time.time()
        step = float(self.t_max) / self.number_of_steps
        step_t = 0.

        for t in range(self.t_max):
            # Randomly create new alleles of PRDM9

            for simulation_step in (s for s in self.steps if s.discrete):
                simulation_step.forward()

            step_t += 1
            if step_t > step and t / float(self.t_max) > 0.01:
                step_t -= step
                for simulation_step in (s for s in self.steps if not s.discrete):
                    simulation_step.forward(t=float(self.scaling) / self.number_of_steps)

                if int(100 * t) % self.t_max == 0:
                    print "Computation at {0}%".format(float(100 * t) / self.t_max)

                self.generations.append(t)
                for simulation_step, data in zip(self.steps, self.data):
                    data.store(simulation_step)

                if time.time() - start_time > 4 * 7200:
                    self.t_max = t
                    print "Breaking the loop, time over 720s"
                    break

        self.save_figure()
        return self

    def save_figure(self):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
        s_data = self.data[0]
        d_data = self.data[1]
        plt.subplot(331)
        plt.text(0.05, 0.98, self, fontsize=14, verticalalignment='top')
        theta = np.arange(0.0, 1.0, 0.01)
        plt.plot(theta, self.steps[0].fitness(theta), color='red')
        plt.title('The fitness function')
        plt.xlabel('x')
        plt.ylabel('w(x)')

        plt.subplot(332)
        plt.plot(self.generations, s_data.prdm9_entropy_alleles, color='red')
        plt.plot(self.generations, d_data.prdm9_entropy_alleles, color='blue')
        plt.title('Efficient number of PRDM9 alleles over time \n Linearized : red')
        plt.xlabel('Generation')
        plt.ylabel('Number of alleles')
        plt.yscale('log')

        plt.subplot(333)
        plt.hexbin(s_data.hotspots_erosion_cum, s_data.prdm9_fitness_cum, s_data.prdm9_longevity_cum,
                   gridsize=200,
                   bins='log')
        plt.title('PRDM9 fitness vs hotspot erosion (Linearized)')
        plt.ylabel('PRDM9 hotspot erosion')
        plt.xlabel('PRMD9 frequency')

        plt.subplot(334)
        plt.hexbin(s_data.hotspots_erosion_cum, s_data.prdm9_frequencies_cum, s_data.prdm9_longevity_cum,
                   gridsize=200,
                   bins='log')
        plt.title('PRMD9 frequency vs hotspot erosion (Linearized)')
        plt.xlabel('Erosion')
        plt.ylabel('PRMD9 frequency')

        plt.subplot(335)
        plt.hexbin(d_data.hotspots_erosion_cum, d_data.prdm9_frequencies_cum, d_data.prdm9_longevity_cum,
                   gridsize=200,
                   bins='log')
        plt.title('PRMD9 frequency vs hotspot erosion')
        plt.xlabel('Erosion')
        plt.ylabel('PRMD9 frequency')

        plt.subplot(336)
        plt.hexbin(s_data.prdm9_frequencies_cum, s_data.prdm9_fitness_cum, s_data.prdm9_longevity_cum,
                   gridsize=200,
                   bins='log')
        plt.title('PRDM9 fitness vs hotspot erosion (Linearized)')
        plt.xlabel('Hotspot erosion')
        plt.ylabel('PRDM9 fitness')

        plt.subplot(337)
        x = np.linspace(0, 1, 100)
        s_density = gaussian_kde(s_data.prdm9_frequencies_cum)(x)
        d_density = gaussian_kde(d_data.prdm9_frequencies_cum)(x)
        plt.plot(x, s_density, 'r')
        plt.fill_between(x, s_density, np.zeros(100), color='red', alpha=0.3)
        plt.plot(x, d_density, 'b')
        plt.fill_between(x, d_density, np.zeros(100), color='blue', alpha=0.3)
        plt.title('PRDM9 frequencies histogram')
        plt.xlabel('PRDM9 Frequencies')
        plt.ylabel('Frequency')

        plt.subplot(338)
        x = np.linspace(0, 1, 100)
        s_density = gaussian_kde(s_data.hotspots_erosion_cum)(x)
        d_density = gaussian_kde(d_data.hotspots_erosion_cum)(x)
        plt.plot(x, s_density, 'r')
        plt.fill_between(x, s_density, np.zeros(100), color='red', alpha=0.3)
        plt.plot(x, d_density, 'b')
        plt.fill_between(x, d_density, np.zeros(100), color='blue', alpha=0.3)
        plt.title('Erosion of the hotspots histogram')
        plt.xlabel('Erosion of the hotspots')
        plt.ylabel('Frequency')

        plt.subplot(339)
        x = np.linspace(1, np.max(np.append(s_data.prdm9_longevity_cum, s_data.prdm9_longevity_cum)), 100)
        s_density = gaussian_kde(s_data.prdm9_longevity_cum)(x)
        d_density = gaussian_kde(d_data.prdm9_longevity_cum)(x)
        plt.plot(x, s_density, 'r')
        plt.fill_between(x, s_density, np.zeros(100), color='red', alpha=0.3)
        plt.plot(x, d_density, 'b')
        plt.fill_between(x, d_density, np.zeros(100), color='blue', alpha=0.3)
        plt.title('Longevity of PRDM9 alleles histogram')
        plt.xlabel('Longevity')
        plt.ylabel('Frequency')

        plt.tight_layout()

        filename = "u=%.1e" % self.mutation_rate_prdm9 + \
                   "_r=%.1e" % self.erosion_rate_hotspot + \
                   "_n=%.1e" % self.population_size + \
                   "_f=%.1e" % self.inflexion_point + \
                   "_c=%.1e" % self.cut_off + \
                   "_t=%.1e" % self.t_max
        plt.savefig(filename + '.png')
        plt.clf()
        print filename
        return filename


class BatchSimulation(object):
    def __init__(self,
                 mutation_rate_prdm9,
                 erosion_rate_hotspot,
                 population_size,
                 recombination_rate,
                 inflexion=0.5,
                 fitness='sigmoid',
                 axis="",
                 number_of_simulations=20,
                 cut_off=0.01,
                 scale=10 ** 2):
        axis_hash = {"mutation": 0, "erosion": 1, "population": 2, "fitness": 3, "mutation&erosion": 4,
                     "erosion&mutation": 4, "scaling": 5, "cut_off": 6}
        assert axis in axis_hash.keys(), "Axis must be either 'population', 'mutation', 'erosion'," \
                                         "'fitness', 'mutation&erosion', or 'scaling'"
        assert scale > 1, "The scale parameter must be greater than one"
        self.axis = axis_hash[axis]
        self.axis_str = {0: "Mutation rate of PRDM9", 1: "Erosion rate of the hotspots", 2: "The population size",
                         3: "The fitness inflexion point", 4: "Scaled mutation rate and erosion rate",
                         5: "The scaling factor", 6: "Cut-off"}[self.axis]

        self.inflexion = inflexion
        self.fitness = fitness
        self.cut_off = cut_off
        self.recombination_rate = recombination_rate
        self.mutation_rate_prdm9 = mutation_rate_prdm9  # The rate at which new allele for PRDM9 are created
        self.erosion_rate_hotspot = erosion_rate_hotspot  # The rate at which the hotspots are eroded
        self.population_size = population_size
        parameters = [self.mutation_rate_prdm9, self.erosion_rate_hotspot, self.population_size,
                      self.recombination_rate, self.inflexion, self.fitness, self.cut_off]

        if self.axis == 4 or self.axis == 5 or self.axis == 6:
            self.axis_range = np.logspace(0, np.log10(scale),
                                          num=number_of_simulations)
        elif self.axis == 3:
            self.axis_range = np.linspace(0.001, 0.999, num=number_of_simulations)
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
                parameters[3] = self.recombination_rate / axis_current
            elif self.axis == 6:
                parameters[6] = self.cut_off * axis_current
            elif self.axis == 3:
                parameters[4] = axis_current
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
               "\nCut-off starting at : %.1e" % self.cut_off + \
               "\nBatch along the axis : %s" % self.axis_str + \
               "\nDiscrete (blue) \| Continuous (red)"

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
        if not self.axis == 3:
            plt.xscale('log')
        plt.yscale('log')

    def save_figure(self):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        plt.subplot(321)
        plt.text(0.05, 0.98, self, fontsize=14, verticalalignment='top')
        theta = np.arange(0.0, 1.0, 0.01)
        for simulation in self.simulations:
            plt.plot(theta, simulation.steps[0].fitness(theta), color='red')

        plt.title('The fitness function')
        plt.xlabel('x')
        plt.ylabel('w(x)')

        plt.subplot(322)
        self.plot_time_series(map(lambda sim: sim.data[0].prdm9_entropy_alleles, self.simulations), 'blue', '')
        self.plot_time_series(map(lambda sim: sim.data[1].prdm9_entropy_alleles, self.simulations), 'red',
                              'Number of PRDM9 alleles')
        plt.subplot(323)
        self.plot_time_series(map(lambda sim: sim.data[0].hotspots_erosion_mean, self.simulations), 'blue', '')
        self.plot_time_series(map(lambda sim: sim.data[1].hotspots_erosion_mean, self.simulations), 'red',
                              'Mean Hotspot Erosion')
        plt.subplot(324)
        self.plot_time_series(map(lambda sim: sim.data[0].hotspots_erosion_var, self.simulations), 'blue', '')
        self.plot_time_series(map(lambda sim: sim.data[1].hotspots_erosion_var, self.simulations), 'red',
                              'Var Hotspot Erosion')
        plt.subplot(325)
        self.plot_time_series(map(lambda sim: sim.data[0].prdm9_longevity_mean, self.simulations), 'blue', '')
        self.plot_time_series(map(lambda sim: sim.data[1].prdm9_longevity_mean, self.simulations), 'red',
                              'Mean PRDM9 longevity')
        plt.subplot(326)
        self.plot_time_series(map(lambda sim: sim.data[0].prdm9_longevity_var, self.simulations), 'blue', '')
        self.plot_time_series(map(lambda sim: sim.data[1].prdm9_longevity_var, self.simulations), 'red',
                              'Var PRDM9 longevity')

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
                                       erosion_rate_hotspot=1.0 * 10 ** -7,
                                       population_size=10 ** 4,
                                       recombination_rate=1.0 * 10 ** -3,
                                       axis='fitness',
                                       fitness='sigmoid',
                                       number_of_simulations=20,
                                       inflexion=0.10,
                                       cut_off=1. * 10 ** -6,
                                       scale=10 ** 4)
    batch_simulation.run(number_of_cpu=7)
    '''
    simulation = Simulation(mutation_rate_prdm9=1.0 * 10 ** -5,
                            erosion_rate_hotspot=1.0 * 10 ** -7,
                            population_size=10 ** 4,
                            recombination_rate=1.0 * 10 ** -3,
                            inflexion_point=0.1,
                            fitness='sigmoid',
                            cut_off=0.1,
                            alpha_zero=1.0 * 10 ** 4,
                            scaling=100)
    simulation.run()
    '''
