import os

dir = "./qsub"
i = 0
os.system('rm ' + dir + "/*")
params = ["-u 1.0 -v 10.0 -f power -a 0.01",
          "-u 1.0 -v 10.0 -f poisson -a 10.0",
          "-u 10.0 -v 10.0 -f poisson -a 10.0",
          "-u 10.0 -v 1.0 -f poisson -a 10.0"]
for c, t, s in [[32, 100, 64], [32, 50, 32], [16, 30, 16], [8, 60, 32]]:
    for param in params:
        i += 1
        file_name = "redqueen{}".format(i)
        qsub = open(dir + "/{}.pbs".format(file_name), 'w')
        qsub.writelines("#!/bin/bash\n")
        qsub.writelines("#\n")
        qsub.writelines("#PBS -q q1day\n")
        qsub.writelines("#PBS -l nodes=1:ppn={0},mem=4gb\n".format(c))
        qsub.writelines("#PBS -o /pandata/tlatrill/out_err/out{}\n".format(file_name))
        qsub.writelines("#PBS -e /pandata/tlatrill/out_err/err" + file_name + "\n")
        qsub.writelines("cd /panhome/tlatrill/RedQueen\n")
        command = "python3 redQueen.py {0} -c {1} -t {2} -r 4 -s {3}".format(param, c, t, s)
        qsub.writelines(command)
        qsub.close()

print('Job completed')
