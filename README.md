### The Red-Queen model of recombination hot-spot evolution: a theoretical investigation.

**Thibault Latrille, Laurent Duret, Nicolas Lartillot**

In humans and many other species, recombination events cluster in narrow hot spots distributed
across the genome, whose location is determined by the Zn-finger protein PRDM9. Hot spots are
not shared between closely related species, suggesting that hot spots are short-lived. To explain
this fast evolutionary dynamics of recombination landscapes, an intra-genomic Red Queen model,
based on the interplay between two antagonistic forces, has been proposed. On the one hand, biased
gene conversion, mediated by double-strand breaks, results in a rapid extinction of hot spots in
the population. On the other hand, the resulting genome-wide depletion of recombination induces
positive selection favoring new Prdm9 alleles recognizing new sequence motifs across the genome
and restoring normal levels of recombination. This Red Queen scenario is currently the reference
model for explaining the fast turnover of recombination landscapes. Thus far, however, it has
not been formalized as a quantitative population-genetic model, fully accounting for the intricate
interplay between biased gene conversion, mutation, selection, demography and genetic diversity
at the PRDM9 locus.

Here, we propose a population-genetic model of the Red Queen dynamic of recombination. This
model was implemented as a Wright-Fisher simulator, allowing exploration of the behaviour of the
model (in terms of the implied mean equilibrium recombination rate, diversity at the PRDM9 locus,
or turnover rate) as a function of the parameters (effective population size, mutation and erosion
rates). In a second step, analytical results, based on self-consistent mean-field approximations,
were derived. These analytical results reproduce the scaling relations observed in the simulations,
offering key insights about the detailed population-genetic mechanisms of the Red Queen model.
Empirical fit of the model to current data from the mouse and humans suggests both a high
mutation rate at PRDM9 and strong biased gene conversion on its targets.

#### Code to reproduce the figures showed in the manuscript
Requirements:
 - Python 3.5
 - numpy
 - scipy
 - matplotlib

```
$ python3 redQueen.py -n 100000
```

optional arguments:
  - -c <nbr of cpu>, --cpu <nbr of cpu>
  
                        Number of CPU available for parallel computing
                        (default: 4)
  - -t <time of simulation>, --time <time of simulation>
  
                        The number of steps of simulations (proportional to
                        time) (default: 50)
  - -w <The wall time>, --wall_time <The wall time>
  
                        The wall time (in seconds) of the whole computation
                        (default: inf)
  - -r <range order of magnitude>, --range <range order of magnitude>
  
                        The order of magnitude to span around the focal
                        parameters (default: 2)
  - -s <number of simulations>, --simulations <number of simulations>
  
                        number of independent simulations to span (default:
                        32)
  - -n <population size>, --population_size <population size>
  
                        number of simulations (default: 100000)
  - -u <Prdm9 mutation rate>, --mutation_rate_prdm9 <Prdm9 mutation rate>
  
                        The mutation rate of Prdm9 (multiplied by 10e-6)
                        (default: 1.0)
  - -v <Hotspots mutation rate>, --mutation_rate_hotspot <Hotspots mutation rate>
  
                        The mutation rate of hotspots (multiplied by 10e-7)
                        (default: 10.0)
  - -f <fitness>, --fitness <fitness>
  
                        The fitness either 'linear', 'sigmoid', 'power' or
                        'poisson' (default: power)
  - -a <alpha>, --alpha <alpha>
  
                        The parameter alpha of the fitness (default: 0.1)
                        
                        - is not used if the fitness is 'linear'.
                        - is the power exponent if the fitness is 'power'.
                        - is the exponential parameter if the fitness is 'poisson'.
                        - is the inflexion point if the fitness is 'sigmoid'. 
  - -d <redraw figures>, --redraw <redraw figures>
  
                        Draw the figures using the file 'Batch.p' (default:
                        False)

