import numpy as np
import sys

numCores = [int(sys.argv[1]), int(sys.argv[2])]
Na = int(sys.argv[3])
aLim = [float(sys.argv[4]),float(sys.argv[5])]
Nb = int(sys.argv[6])
bLim = [float(sys.argv[7]),float(sys.argv[8])]

aValCut = np.linspace(aLim[0],aLim[1],numCores[0]+1)
bValCut = np.linspace(bLim[0],bLim[1],numCores[1]+1)

ind=0
with open('createFolders.txt', 'w') as f:
    for i in range(numCores[0]):
        for j in range(numCores[1]):
            f.write("mkdir part" + "%03d" % (ind+1) +"\n")
            ind = ind+1

ind = 0
with open('runMultiple.txt', 'w') as f:
    f.write("#!/bin/bash"+"\n")                                                                     
    # f.write("#SBATCH -n " + str(int(sys.argv[1])*int(sys.argv[2])) + "   # Number of MPI tasks"  +"\n")   
    f.write("#SBATCH --array=1-" + str(int(sys.argv[1])*int(sys.argv[2])) + "   # job array index"  +"\n")
    f.write("#SBATCH --cpus-per-task 1" +"\n")                               
    #f.write("#SBATCH --nodes 1 " +"\n") 
    #f.write("#SBATCH -p general" +"\n") 
    f.write("#SBATCH --mem 1G" +"\n") 
    f.write("#SBATCH -t 48:00:00        # Time limit " + "\n")     
    f.write("#SBATCH --export=ALL " + "\n")                                          
    f.write("#SBATCH -J qsOptNAE " + "\n")     

    f.write("arrayjob=`cat commands_input.txt | awk -v line=$SLURM_ARRAY_TASK_ID '{if (NR == line) print $0}'`" + "\n")
    f.write('run_command="python optAnhark2optZ.py "' + "\n")
    
    f.write("echo $run_command$arrayjob" + "\n")
    f.write('bash -c "$run_command$arrayjob"' + "\n")

with open('commands_input.txt', 'w') as f:
    for i in range(numCores[0]):
        for j in range(numCores[1]):
            f.write(" " + str(Na) + " " + str(aValCut[i]) + " " + str(aValCut[i+1]) + " " + str(Nb) + " " + str(bValCut[j]) + " " + str(bValCut[j+1])+" /p/QSOpt/NAEopt/optZ/runZn4k2resplus/part"+ "%03d" % (ind+1) +"/ \n")
            ind = ind+1
