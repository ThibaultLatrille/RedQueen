import os

dir = "./qsub"
i = 0
os.system('rm ' + dir + "/*")
for q, n in [["q1day", 32], ["q1hour", 8]]:
    for fitness in ["polynomial", "poisson"]:
        if fitness == "polynomial":
            alpha_list = [0.0001, 0.001, 0.01]
        else:
            alpha_list = [1, 10]
        for alpha in alpha_list:
            for u, v in [[3.0, 3.0], [10.0, 10.0], [10.0, 1.0]]:
                i += 1
                file_name = "redqueen{}".format(i)
                qsub = open(dir + "/{}.pbs".format(file_name), 'w')
                qsub.writelines("#!/bin/bash\n")
                qsub.writelines("#\n")
                qsub.writelines("#PBS -q {}\n".format(q))
                qsub.writelines("#PBS -l nodes=1:ppn=8,mem=4gb\n")
                qsub.writelines("#PBS -o /pandata/tlatrill/out_err/out{}\n".format(file_name))
                qsub.writelines("#PBS -e /pandata/tlatrill/out_err/err" + file_name + "\n")
                qsub.writelines("cd /panhome/tlatrill/RedQueen\n")
                command = "python3 redQueen.py -u {0} -v {1} -f {2} -a {3} -n {4}".format(u, v, fitness, alpha, n)
                qsub.writelines(command)
                qsub.close()

print('Job completed')
