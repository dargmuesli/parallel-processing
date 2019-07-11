#!/bin/bash

####### Mail Notify / Job Name / Comment #######
#SBATCH --job-name=BuddyLcm

####### Partition #######
#SBATCH --partition=exec

####### Ressources #######
#SBATCH --time=20
#SBATCH --mem-per-cpu=1000

####### Node Info #######
#SBATCH --exclusive
#SBATCH -n NUMPLACES
#SBATCH --ntasks-per-node=PPERNODE
#SBATCH --distribution=cyclic

####### Output #######
#SBATCH --output=HOME/pv/apgas_seq/out/out.BuddyLcm-placesNUMPLACES-placesPerNodePPERNODE-threadsTHREADS.%j
#SBATCH --error=HOME/pv/apgas_seq/out/err.BuddyLcm-placesNUMPLACES-placesPerNodePPERNODE-threadsTHREADS.%j

####### Script #######

cd HOME/pv/apgas_seq/bin

echo $(scontrol show hostname | paste -d, -s) | tr "," "\n"> hostfile-${SLURM_JOB_ID}
echo $HOSTNAME > hostfile2-${SLURM_JOB_ID}
sed '1,1d' hostfile-${SLURM_JOB_ID} >> hostfile2-${SLURM_JOB_ID}
myfile=$(< hostfile2-${SLURM_JOB_ID})
for ((z=1;z<PPERNODE;z++))
do
     echo -e "$myfile" >> hostfile2-${SLURM_JOB_ID}
done

java -cp .:HOME/pv/apgas_seq/lib/* \
  -Dapgas.places=NUMPLACES \
  -Dapgas.threads=THREADS \
  -Dapgas.hostfile=HOME/pv/apgas_seq/bin/hostfile2-${SLURM_JOB_ID} \
  -Dapgas.launcher=apgas.impl.SrunKasselLauncher \
  -Dapgas.verbose.launcher=VERBOSELAUNCHER \
  groupP.BuddyLcm 16384 10000 123 456 50000 false

rm hostfile-${SLURM_JOB_ID}
rm hostfile2-${SLURM_JOB_ID}
