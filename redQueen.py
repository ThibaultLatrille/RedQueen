import numpy as np
import matplotlib.pyplot as plt
import time
import os


# The fitness function, given the erosion of the hotspots
def w(y):
    return y ** 4 / (y ** 4 + 0.5 ** 4)


class Simulation(object):
    def __init__(self, mutation_rate_prdm9=1.0 * 10 ** -5,
                 erosion_rate_hotspot=1.0 * 10 ** -3,
                 population_size=10 ** 4,
                 neutral=False):
        initial_number_of_alleles = 10
        self.number_of_steps = 10000  # Number of steps at which we make computations
        self.t_max = 100 * population_size  # Number of generations
        self.mutation_rate_prdm9 = mutation_rate_prdm9  # The rate at which new allele for PRDM9 are created
        self.erosion_rate_hotspot = erosion_rate_hotspot  # The rate at which the hotspots are eroded
        self.population_size = float(population_size)  # population size
        self.neutral = neutral  # If the fitness function is neutral

        self.prdm9_polymorphism = np.ones(initial_number_of_alleles) * self.population_size / initial_number_of_alleles
        self.hotspots_erosion = np.ones(initial_number_of_alleles)
        self.prdm9_longevity = np.zeros(initial_number_of_alleles)

        self.prdm9_frequencies_mean = np.zeros(self.number_of_steps)
        self.prdm9_longevity_mean = np.zeros(self.number_of_steps)
        self.hotspots_erosion_mean = np.zeros(self.number_of_steps)

        self.prdm9_frequencies_most_frequent = np.zeros(self.number_of_steps)

        self.prdm9_nb_alleles = np.zeros(self.number_of_steps)
        self.generations = (np.arange(self.number_of_steps) + 1) * self.t_max / self.number_of_steps

        self.run()

    def __str__(self):
        return "The mutation rate of PRDM9: %.1e" % self.mutation_rate_prdm9 + \
               "\nThe erosion rate of the hotspots : %.1e" % self.erosion_rate_hotspot + \
               "\nThe population size : %.1e" % self.population_size + \
               "\nThe number of generations computed : %.1e" % self.t_max

    def __repr__(self):
        return "The polymorphism of PRDM9: %s" % self.prdm9_polymorphism + \
               "\nThe strength of the hotspots : %s" % self.hotspots_erosion + \
               "\nThe longevity hotspots : %s" % self.prdm9_longevity

    def slice(self, t):
        self.prdm9_frequencies_mean = self.prdm9_frequencies_mean[:t]
        self.prdm9_longevity_mean = self.prdm9_longevity_mean[:t]
        self.hotspots_erosion_mean = self.hotspots_erosion_mean[:t]

        self.prdm9_frequencies_most_frequent = self.prdm9_frequencies_most_frequent[:t]

        self.prdm9_nb_alleles = self.prdm9_nb_alleles[:t]
        self.generations = self.generations[:t]

    def run(self):
        start_time = time.time()
        # Initiate the vectors
        scaled_erosion_rate = self.erosion_rate_hotspot * self.population_size
        assert scaled_erosion_rate < 0.1, "The populaton size is too large for this value of erosion rate"
        assert scaled_erosion_rate > 0.0000001, "The populaton size is too low for this value of erosion rate"

        scaled_mutation_rate = self.population_size * self.mutation_rate_prdm9
        for t in range(self.t_max):

            # Randomly create new alleles of PRDM9
            new_alleles = np.random.poisson(2 * scaled_mutation_rate)
            if new_alleles > 0:
                self.prdm9_polymorphism -= np.random.multinomial(new_alleles, np.divide(
                        self.prdm9_polymorphism, self.population_size))
                self.prdm9_polymorphism = np.append(self.prdm9_polymorphism,
                                                    np.ones(new_alleles))
                self.prdm9_longevity = np.append(self.prdm9_longevity, np.zeros(new_alleles))
                self.hotspots_erosion = np.append(self.hotspots_erosion, np.ones(new_alleles))

            # Compute the PRDM9 frequencies for convenience
            prdm9_frequencies = np.divide(self.prdm9_polymorphism, self.population_size)

            # Exponential decay for hotspots erosion
            # hotspots_erosion *= np.exp( - erosion_rate_hotspot * prdm9_frequencies)

            self.hotspots_erosion *= (
                1. - scaled_erosion_rate * prdm9_frequencies)

            # Compute the fitness for each allele
            nb_prdm9_alleles = self.prdm9_polymorphism.size
            if self.neutral:
                fitness_matrix = np.ones([nb_prdm9_alleles, nb_prdm9_alleles])
            else:
                fitness_matrix = np.empty([nb_prdm9_alleles, nb_prdm9_alleles])
                for i in range(nb_prdm9_alleles):
                    for j in range(nb_prdm9_alleles):
                        fitness_matrix[i, j] = w(self.hotspots_erosion[i])

            fitness_vector = np.dot(fitness_matrix, prdm9_frequencies) * prdm9_frequencies
            fitness_vector /= np.sum(fitness_vector)

            # Randomly pick the new generation according to the fitness vector
            self.prdm9_polymorphism = np.random.multinomial(int(self.population_size), fitness_vector)

            # Remove the extinct alleles
            extinction = np.array(map(lambda x: x != 0, self.prdm9_polymorphism), dtype=bool)
            self.prdm9_polymorphism = self.prdm9_polymorphism[extinction]
            self.prdm9_longevity = self.prdm9_longevity[extinction]
            self.hotspots_erosion = self.hotspots_erosion[extinction]

            # Increase the longevity of survivors by 1
            self.prdm9_longevity += 1

            if self.number_of_steps * t % self.t_max == 0:
                step = self.number_of_steps * t / self.t_max

                self.prdm9_nb_alleles[step] = self.prdm9_polymorphism.size

                self.prdm9_frequencies_mean[step] = np.mean(self.prdm9_polymorphism) / self.population_size
                self.prdm9_longevity_mean[step] = np.mean(self.prdm9_longevity)
                self.hotspots_erosion_mean[step] = 1 - np.mean(self.hotspots_erosion)

                self.prdm9_frequencies_most_frequent[step] = np.max(self.prdm9_polymorphism) / self.population_size

                if time.time() - start_time > 360:
                    self.t_max = t
                    self.slice(step)
                    break

    def figure(self):
        my_dpi = 96
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

        plt.subplot(331)
        plt.text(0.05, 0.95, self, fontsize=14, verticalalignment='top')
        plt.axis('off')

        plt.subplot(332)
        theta = np.arange(0.0, 1.0, 0.01)
        if self.neutral:
            plt.plot(theta, np.ones(theta.size), color='red')
        else:
            vectorized_w = np.vectorize(w)
            plt.plot(theta, vectorized_w(theta), color='red')
        plt.title('The fitness function')
        plt.xlabel('x')
        plt.ylabel('w(x,0)')

        plt.subplot(333)
        plt.plot(self.generations, self.prdm9_nb_alleles, color='blue')
        plt.title('Number of PRDM9 alleles over time')
        plt.xlabel('Generation')
        plt.ylabel('population_sizeumber of alleles')

        plt.subplot(334)
        plt.plot(self.generations, self.prdm9_frequencies_mean, color='blue')
        plt.plot(self.generations, self.prdm9_frequencies_most_frequent, color='red')
        plt.title('Mean PRDM9 frequencies (blue) and \n Frequency of the most frequent allele (red) over time')
        plt.xlabel('Generation')
        plt.ylabel('Frequency')
        plt.axis([1, self.t_max, 0, 1])

        plt.subplot(335)
        plt.plot(self.generations, self.hotspots_erosion_mean, color='blue')
        plt.title('Mean erosion of the hotspots (blue) over time')
        plt.xlabel('Generation')
        plt.ylabel('Erosion')
        plt.axis([1, self.t_max, 0, 1])

        plt.subplot(336)
        plt.plot(self.generations, self.prdm9_longevity_mean, color='blue')
        plt.title('Mean longevity of PRDM9 alleles over time')
        plt.xlabel('Generation')
        plt.ylabel('Longevity')

        plt.subplot(337)
        plt.hist(self.prdm9_polymorphism / self.population_size, color='red')
        plt.title('PRDM9 frequencies histogram')
        plt.xlabel('PRDM9 Frequencies')
        plt.ylabel('Frequency')

        plt.subplot(338)
        plt.hist(1 - self.hotspots_erosion, color='red')
        plt.title('Erosion of the hotspots histogram')
        plt.xlabel('Erosion of the hotspots')
        plt.ylabel('Frequency')

        plt.subplot(339)
        plt.hist(self.prdm9_longevity, color='red')
        plt.title('Longevity of PRDM9 alleles histogram')
        plt.xlabel('Longevity')
        plt.ylabel('Frequency')

        plt.tight_layout()

    def show(self):
        self.figure()
        plt.show()

    def save_figure(self):
        self.figure()

        filename = "u=%.1e" % self.mutation_rate_prdm9 + \
                   "_r=%.1e" % self.erosion_rate_hotspot + \
                   "_n=%.1e" % self.population_size + \
                   "_t=%.1e" % self.t_max
        if self.neutral:
            filename += "neutral"

        path = os.getcwd() + "/tmp"
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise

        os.chdir(path)
        plt.savefig(filename + '.png')
        print filename
        return filename


simulation = Simulation(1.0 * 10 ** -5, 1.0 * 10 ** -6, 10 ** 4)
simulation.save_figure()
