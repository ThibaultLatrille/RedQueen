import numpy as np
import matplotlib.pyplot as plt

N = 2 * 10 ** 4  # population size
K = 10  # The initial number of alleles
t_max = 30000  # Number of steps

# The rate at which new allele for PRDM9 are eroded
mutation_rate_prdm9 = 1.0 * 10 ** -5

# The rate at which the hotspots are degraded
mutation_rate_hotspot = 1.0 * 10 ** -3

# Initiate the vectors
prdm9_polymorphism = np.ones(K) * N / K
hotspots_erosion = np.ones(K)
prdm9_longevity = np.zeros(K)

prdm9_fequencies_mean = np.zeros(t_max)
prdm9_longevity_mean = np.zeros(t_max)
hotspots_erosion_mean = np.zeros(t_max)

prdm9_fequencies_most_frequent = np.zeros(t_max)
prdm9_longevity_most_frequent = np.zeros(t_max)
hotspots_erosion_most_frequent = np.zeros(t_max)

most_frequent_change = []
most_frequent_index = -1

prdm9_nb_alleles = np.zeros(t_max)


# The fitness function, given the erosion of the hotspots
def w(x, y):
    return (x + y) ** 4 / ((x + y) ** 4 + 0.5 ** 4)


print "Initial population size: %s" % N
print "Initial number of alleles : %s" % K

for t in range(t_max):
    # Randomly create new alleles of PRDM9
    new_alleles = np.random.poisson(2 * N * mutation_rate_prdm9)
    if new_alleles > 0:
        prdm9_polymorphism -= np.random.multinomial(new_alleles, prdm9_polymorphism / N)
        prdm9_polymorphism = np.append(prdm9_polymorphism,
                                       np.ones(new_alleles))  # initialize the vector of PRDM9 alleles
        prdm9_longevity = np.append(prdm9_longevity, np.zeros(new_alleles))
        hotspots_erosion = np.append(hotspots_erosion, np.ones(new_alleles))  # initialize the hotspot strenght

    # Compute the PRDM9 frequencies for convenience
    prdm9_frequencies = prdm9_polymorphism / float(N)

    # Exponential decay for hotspots erosion
    # hotspots_erosion *= np.exp( - mutation_rate_hotspot * prdm9_frequencies)
    hotspots_erosion *= (1. - mutation_rate_hotspot * prdm9_frequencies)

    # Compute the fitness for each allele
    nb_prdm9_alleles = prdm9_polymorphism.size
    fitness_matrix = np.empty([nb_prdm9_alleles, nb_prdm9_alleles])
    for i in range(nb_prdm9_alleles):
        for j in range(nb_prdm9_alleles):
            if i != j:
                fitness_matrix[i, j] = w(hotspots_erosion[i], hotspots_erosion[j])
            else:
                fitness_matrix[i, j] = w(hotspots_erosion[i], 0)

    fitness_vector = np.dot(fitness_matrix, prdm9_frequencies) * prdm9_frequencies
    fitness_vector /= np.sum(fitness_vector)

    # Randomly pick the new generation according to the fitness vector
    prdm9_polymorphism = np.random.multinomial(N, fitness_vector)

    # Remove the extinct alleles
    extinction = np.array(map(lambda x: x != 0, prdm9_polymorphism), dtype=bool)
    prdm9_polymorphism = prdm9_polymorphism[extinction]
    prdm9_longevity = prdm9_longevity[extinction]
    hotspots_erosion = hotspots_erosion[extinction]

    # Increase the longevity of survivors by 1
    prdm9_longevity += 1
    prdm9_nb_alleles[t] = prdm9_polymorphism.size

    prdm9_fequencies_mean[t] = np.mean(prdm9_polymorphism) / N
    prdm9_longevity_mean[t] = np.mean(prdm9_longevity)
    hotspots_erosion_mean[t] = 1 - np.mean(hotspots_erosion)

    if most_frequent_index != np.argmax(prdm9_polymorphism):
        most_frequent_change.append(t)
        most_frequent_index = np.argmax(prdm9_polymorphism)

    prdm9_fequencies_most_frequent[t] = prdm9_polymorphism[most_frequent_index] / N
    prdm9_longevity_most_frequent[t] = prdm9_longevity[most_frequent_index]
    hotspots_erosion_most_frequent[t] = 1 - hotspots_erosion[most_frequent_index]

print "Output the result"
print "The polymorphism of PRDM9: %s" % prdm9_polymorphism
print "The strength of the hotspots : %s" % hotspots_erosion
print "The longevity hotspots : %s" % prdm9_longevity

most_frequent_change.pop(0)
generations = np.arange(t_max) + 1

plt.figure(1)
plt.subplot(331)
x = np.arange(0.0, 2.0, 0.01)
vectorized_w = np.vectorize(w)
plt.plot(x, vectorized_w(x, 0), color='red')
plt.title('The fitness function')
plt.xlabel('x')
plt.ylabel('w(x,0)')

plt.subplot(332)
plt.plot(generations, prdm9_nb_alleles, color='blue')
plt.title('Number of PRDM9 alleles over time')
plt.xlabel('Generation')
plt.ylabel('Number of alleles')

plt.subplot(334)
plt.plot(generations, prdm9_fequencies_mean, color='blue')
plt.plot(generations, prdm9_fequencies_most_frequent, color='red')
plt.axvline(most_frequent_change[0])
plt.title('Mean PRDM9 frequencies over time')
plt.xlabel('Generation')
plt.ylabel('Mean frequency')
plt.axis([1, t_max, 0, 1])

plt.subplot(335)
plt.plot(generations, hotspots_erosion_mean, color='blue')
plt.plot(generations, hotspots_erosion_most_frequent, color='red')
plt.title('Mean erosion of the hotspots over time')
plt.xlabel('Generation')
plt.ylabel('Mean strength')
plt.axis([1, t_max, 0, 1])

plt.subplot(336)
plt.plot(generations, prdm9_longevity_mean, color='blue')
plt.plot(generations, prdm9_longevity_most_frequent, color='red')
plt.title('Mean longevity of PRDM9 alleles over time')
plt.xlabel('Generation')
plt.ylabel('Mean longevity')

plt.subplot(337)
plt.hist(prdm9_polymorphism / float(N), color='red')
plt.title('PRDM9 frequencies histogram')
plt.xlabel('PRDM9 Frequencies')
plt.ylabel('Frequency')

plt.subplot(338)
plt.hist(1 - hotspots_erosion, color='red')
plt.title('Erosion of the hotspots histogram')
plt.xlabel('Erosion of the hotspots')
plt.ylabel('Frequency')

plt.subplot(339)
plt.hist(prdm9_longevity, color='red')
plt.title('Longevity of PRDM9 alleles histogram')
plt.xlabel('Longevity')
plt.ylabel('Frequency')

plt.show()
