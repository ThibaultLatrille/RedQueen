from redQueen import Model


def tex_f(f):
    if 0.1 < f < 100:
        return "$" + "{0:.2g}".format(f) + "$"
    else:
        base, exponent = "{0:.2e}".format(f).split("e")
        return r"${0} \times 10^{{{1}}}$".format("{0:.2g}".format(float(base)), int(exponent))

parameters = [
    {'n': 10 ** 5, 'u': 3 * 10 ** -6, 'v': 10 ** -7, 'g': 3*10 ** -3, 'f': "power", 'a': 10 ** -4},
    {'n': 10 ** 5, 'u': 3 * 10 ** -6, 'v': 10 ** -7, 'g': 3*10 ** -4, 'f': "power", 'a': 10 ** -4},
    {'n': 10 ** 5, 'u': 3 * 10 ** -7, 'v': 10 ** -7, 'g': 3*10 ** -4, 'f': "power", 'a': 10 ** -4},
    {'n': 10 ** 5, 'u': 3 * 10 ** -6, 'v': 10 ** -7, 'g': 3*10 ** -4, 'f': "power", 'a': 10 ** -5},
    {'n': 10 ** 5, 'u': 10 ** -5, 'v': 10 ** -7, 'g': 10 ** -2, 'f': "poisson", 'a': 5 * 10 ** -2},
    {'n': 10 ** 5, 'u': 10 ** -5, 'v': 10 ** -7, 'g': 10 ** -3, 'f': "poisson", 'a': 5 * 10 ** -2},
    {'n': 10 ** 5, 'u': 10 ** -6, 'v': 10 ** -7, 'g': 10 ** -3, 'f': "poisson", 'a': 5 * 10 ** -2},
    {'n': 10 ** 5, 'u': 10 ** -5, 'v': 10 ** -7, 'g': 10 ** -3, 'f': "poisson", 'a': 1 * 10 ** -2},
]
fitness_param = {"poisson": "$\\beta$", "power": "$\\alpha$"}
fitness_name = {"poisson": "Exponential fitness function", "power": "Power law fitness function"}
table = open("./table.tex", 'w')
table.writelines("\\documentclass[USLetter,5pt]{article}\n"
                 "\\usepackage{adjustbox}\n")
table.writelines("\\newcommand{\\specialcell}[2][c]{%\n\\begin{tabular}[#1]{@{}c@{}}#2\\end{tabular}}\n")
table.writelines("\\begin{document}\n")
for fitness in ["power", "poisson"]:
    heading = ["\\specialcell{Mutation rate of \\\\ PRDM9 ($u$)}",
               "\\specialcell{Erosion rate of \\\\ targets ($v*g$)}",
               "\\specialcell{Fitness \\\\ parameter (" + fitness_param[fitness] + ")}",
               "$\epsilon$",
               "\\specialcell{Mean fraction of \\\\ active targets ($H$)}",
               "\\specialcell{Diversity at \\\\ PRDM9 locus ($D$)}",
               "\\specialcell{Scaled selection \\\\ coefficient ($S$)}",
               "\\specialcell{Turn-over \\\\ time ($T$)}"]

    table.writelines("\\begin{table}[ht]\n\\centering\n\\begin{adjustbox}{width = 1\\textwidth}\n\\small")
    table.writelines("\\begin{tabular}{" + "|c" * len(heading) + "|}\n")
    table.writelines("\\hline\n")
    table.writelines(" & ".join(heading) + "\\\\\n")
    table.writelines("\\hline\n")
    for params in [p for p in parameters if p["f"] == fitness]:
        if params["f"] == "power":
            f_param = params["a"]
        else:
            f_param = 1./params["a"]
        model = Model(mutation_rate_prdm9=params["u"], mutation_rate_hotspot=params["v"],
                      population_size=params["n"], recombination_rate=params["g"], alpha=f_param,
                      fitness=fitness, drift=True, linearized=False, hotspots_variation=True, gamma=1.)
        table.writelines(" & ".join([tex_f(model.mutation_rate_prdm9),
                                     tex_f(model.mutation_rate_hotspot * model.recombination_rate),
                                     tex_f(f_param),
                                     tex_f(model.mutation_rate_hotspot * model.recombination_rate /
                                           model.mutation_rate_prdm9),
                                     tex_f(model.mean_activity_general()),
                                     tex_f(model.prdm9_diversity_general()),
                                     tex_f(model.selective_strength_general()),
                                     tex_f(model.turn_over_general()),
                                     ]) + "\\\\\n")
    table.writelines("\\hline\n")
    table.writelines("\\end{tabular}\n")
    table.writelines("\\end{adjustbox}\n" +
                     "\\caption{" + fitness_name[fitness] + "}\n" +
                     "\\end{table}\n")
table.writelines("\\end{document}\n")
table.close()
