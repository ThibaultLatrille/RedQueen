**The Red-Queen model of recombination hot-spot evolution: a theoretical investigation,**\
_Thibault Latrille, Laurent Duret, Nicolas Lartillot_,\
Phil. Trans. R. Soc. B,\
http://doi.org/10.1098/rstb.2016.0463

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
  
                        Effective population size (default: 100000)
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

