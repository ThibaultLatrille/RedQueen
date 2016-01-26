import numpy as np
import matplotlib.pyplot as plt


# The fitness function, given the erosion of the hotspots
def w(x, y):
    return (x + y) ** 4 / ((x + y) ** 4 + 0.5 ** 4)


class Simulation(object):
    def __init__(self, mutation_rate_prdm9=1.0 * 10 ** -5,
                 erosion_rate_hotspot=1.0 * 10 ** -3,
                 population_size=2 * 10 ** 4,
                 t_max=30000,
                 neutral=False):
        initial_number_of_alleles = 10
        self.mutation_rate_prdm9 = mutation_rate_prdm9  # The rate at which new allele for PRDM9 are created
        self.erosion_rate_hotspot = erosion_rate_hotspot  # The rate at which the hotspots are eroded
        self.population_size = population_size  # population size
        self.t_max = t_max  # Number of steps
        self.neutral = neutral  # If the fitness function is neutral

        self.prdm9_polymorphism = np.ones(initial_number_of_alleles) * self.population_size / initial_number_of_alleles
        self.hotspots_erosion = np.ones(initial_number_of_alleles)
        self.prdm9_longevity = np.zeros(initial_number_of_alleles)

        self.prdm9_fequencies_mean = np.zeros(t_max)
        self.prdm9_longevity_mean = np.zeros(t_max)
        self.hotspots_erosion_mean = np.zeros(t_max)

        self.prdm9_fequencies_most_frequent = np.zeros(t_max)
        self.prdm9_longevity_most_frequent = np.zeros(t_max)
        self.hotspots_erosion_most_frequent = np.zeros(t_max)

        self.most_frequent_change = []
        self.prdm9_nb_alleles = np.zeros(t_max)
        self.generations = np.arange(t_max) + 1

        self.run()

    def __repr__(self):
        return "The mutation rate of PRDM9: %s" % self.mutation_rate_prdm9 + \
               "\n The erosion rate of the hotspots : %s" % self.hotspots_erosion + \
               "\n The population size : %s" % self.population_size + \
               "\n The number of generations computed : %s" % self.t_max

    def __str__(self):
        return "The polymorphism of PRDM9: %s" % self.prdm9_polymorphism + \
               "\n The strength of the hotspots : %s" % self.hotspots_erosion + \
               "\n The longevity hotspots : %s" % self.prdm9_longevity

    def run(self):
        # Initiate the vectors
        most_frequent_index = -1
        for t in range(self.t_max):

            # Randomly create new alleles of PRDM9
            new_alleles = np.random.poisson(2 * self.population_size * self.mutation_rate_prdm9)
            if new_alleles > 0:
                self.prdm9_polymorphism -= np.random.multinomial(new_alleles, np.divide(
                        self.prdm9_polymorphism, float(self.population_size)))
                self.prdm9_polymorphism = np.append(self.prdm9_polymorphism,
                                                    np.ones(new_alleles))
                self.prdm9_longevity = np.append(self.prdm9_longevity, np.zeros(new_alleles))
                self.hotspots_erosion = np.append(self.hotspots_erosion, np.ones(new_alleles))

            # Compute the PRDM9 frequencies for convenience
            prdm9_frequencies = np.divide(self.prdm9_polymorphism, float(self.population_size))

            # Exponential decay for hotspots erosion
            # hotspots_erosion *= np.exp( - erosion_rate_hotspot * prdm9_frequencies)

            self.hotspots_erosion *= (
                1. - self.erosion_rate_hotspot * prdm9_frequencies)

            # Compute the fitness for each allele
            nb_prdm9_alleles = self.prdm9_polymorphism.size
            if self.neutral:
                fitness_matrix = np.ones([nb_prdm9_alleles, nb_prdm9_alleles])
            else:
                fitness_matrix = np.empty([nb_prdm9_alleles, nb_prdm9_alleles])
                for i in range(nb_prdm9_alleles):
                    for j in range(nb_prdm9_alleles):
                        if i != j:
                            fitness_matrix[i, j] = w(self.hotspots_erosion[i], self.hotspots_erosion[j])
                        else:
                            fitness_matrix[i, j] = w(self.hotspots_erosion[i], 0)

            fitness_vector = np.dot(fitness_matrix, prdm9_frequencies) * prdm9_frequencies
            fitness_vector /= np.sum(fitness_vector)

            # Randomly pick the new generation according to the fitness vector
            self.prdm9_polymorphism = np.random.multinomial(self.population_size, fitness_vector)

            # Remove the extinct alleles
            extinction = np.array(map(lambda x: x != 0, self.prdm9_polymorphism), dtype=bool)
            self.prdm9_polymorphism = self.prdm9_polymorphism[extinction]
            self.prdm9_longevity = self.prdm9_longevity[extinction]
            self.hotspots_erosion = self.hotspots_erosion[extinction]

            # Increase the longevity of survivors by 1
            self.prdm9_longevity += 1
            self.prdm9_nb_alleles[t] = self.prdm9_polymorphism.size

            self.prdm9_fequencies_mean[t] = np.mean(self.prdm9_polymorphism) / self.population_size
            self.prdm9_longevity_mean[t] = np.mean(self.prdm9_longevity)
            self.hotspots_erosion_mean[t] = 1 - np.mean(self.hotspots_erosion)

            if most_frequent_index != np.argmax(self.prdm9_polymorphism):
                self.most_frequent_change.append(t)
                most_frequent_index = np.argmax(self.prdm9_polymorphism)

            self.prdm9_fequencies_most_frequent[t] = self.prdm9_polymorphism[most_frequent_index] / float(
                self.population_size)
            self.prdm9_longevity_most_frequent[t] = self.prdm9_longevity[most_frequent_index]
            self.hotspots_erosion_most_frequent[t] = 1 - self.hotspots_erosion[most_frequent_index]

        self.most_frequent_change.pop(0)

    def plot(self):
        plt.figure(1)
        plt.subplot(331)
        theta = np.arange(0.0, 2.0, 0.01)
        if self.neutral:
            plt.plot(theta, np.ones(theta.size), color='red')
        else:
            vectorized_w = np.vectorize(w)
            plt.plot(theta, vectorized_w(theta, 0), color='red')
        plt.title('The fitness function')
        plt.xlabel('x')
        plt.ylabel('w(x,0)')

        plt.subplot(332)
        plt.plot(self.generations, self.prdm9_nb_alleles, color='blue')
        plt.title('population_sizeumber of PRDM9 alleles over time')
        plt.xlabel('Generation')
        plt.ylabel('population_sizeumber of alleles')

        plt.subplot(334)
        plt.plot(self.generations, self.prdm9_fequencies_mean, color='blue')
        plt.plot(self.generations, self.prdm9_fequencies_most_frequent, color='red')
        plt.axvline(self.most_frequent_change[0])
        plt.title('Mean PRDM9 frequencies (blue) and Frequency of the most frequent allele (red) over time')
        plt.xlabel('Generation')
        plt.ylabel('Frequency')
        plt.axis([1, self.t_max, 0, 1])

        plt.subplot(335)
        plt.plot(self.generations, self.hotspots_erosion_mean, color='blue')
        plt.plot(self.generations, self.hotspots_erosion_most_frequent, color='red')
        plt.title('Mean erosion of the hotspots (blue) and Erosion of the most frequent allele (red) and over time')
        plt.xlabel('Generation')
        plt.ylabel('Erosion')
        plt.axis([1, self.t_max, 0, 1])

        plt.subplot(336)
        plt.plot(self.generations, self.prdm9_longevity_mean, color='blue')
        plt.plot(self.generations, self.prdm9_longevity_most_frequent, color='red')
        plt.title('Mean longevity of PRDM9 alleles (blue) and longevity of the most frequent allele (red) over time')
        plt.xlabel('Generation')
        plt.ylabel('Longevity')

        plt.subplot(337)
        plt.hist(self.prdm9_polymorphism / float(self.population_size), color='red')
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

        plt.show()


simulation = Simulation(1.0 * 10 ** -5, 1.0 * 10 ** -3, 2 * 10 ** 4, 30000, True)
simulation.plot()
