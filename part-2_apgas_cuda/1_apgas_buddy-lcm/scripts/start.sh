#!/bin/bash

FOO=$(mktemp)
sed "s|NUMPLACES|1|g;
     s|THREADS|16|g;
     s|PPERNODE|1|g;
     s|VERBOSELAUNCHER|false|g;
     s|HOME|${HOME}|g;" < ${HOME}/pv/apgas_seq/scripts/job-commit.sh > ${FOO}

for i in {1..3}
do
    sbatch ${FOO}
done

rm ${FOO}
