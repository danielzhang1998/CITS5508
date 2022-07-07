#!/bin/bash -l
#SBATCH --job-name=GE-conda
#SBATCH --partition=cits5508
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --export=ALL

start_time=$( date )

# To configure GNU Environment for Mothur
module load Anaconda3/2020.11
eval "$(conda shell.bash hook)"

# activate the Python environment for the unit CITS5508
conda activate /group/cits5508/cits5508-2022

# list the environment loaded by the modules.
# Can remove the two lines below if you want.
# module list
# conda list

# Note: SLURM_JOBID is a unique number for every job.
# These are generic variables.

# Below is the Python file that would be run. Replace
# lab05-sample.ipynb by your own file name.
SCRIPT=lab05-sample.ipynb

SCRATCH=$MYSCRATCH/run_conda/$SLURM_JOBID
RESULTS=$MYGROUP/conda_results

###############################################
# Creates a unique directory in the SCRATCH directory for this job to run in.
if [ ! -d $SCRATCH ]; then 
    mkdir -p $SCRATCH 
fi 
echo Working SCRATCH directory is $SCRATCH

###############################################
# Creates a unique directory in your GROUP directory for the results of this job
if [ ! -d $RESULTS ]; then 
     mkdir -p $RESULTS
fi 
echo Results will be stored in $RESULTS/$SLURM_JOBID

#############################################
# Rather than copying the large CIFAR-10 batch files to the $SCRATCH directory
# (where the Python file will be run), we create symbolic links to the data files
# in that directory.

cd ${SLURM_SUBMIT_DIR}
echo "SLURM_SUBMIT_DIR is"
echo ${SLURM_SUBMIT_DIR}

# copy the mothur analysis script to SCRATCH
cp ${SCRIPT} ${SCRATCH}

# go to the /scratch... directory and create symbolic links to the
# files for the CIFAR-10 dataset and a link to Du's data_loader.py.
cd ${SCRATCH}

ln -s /group/cits5508/CIFAR-10/data_batch_1
ln -s /group/cits5508/CIFAR-10/data_batch_2
ln -s /group/cits5508/CIFAR-10/data_batch_3
ln -s /group/cits5508/CIFAR-10/data_batch_4
ln -s /group/cits5508/CIFAR-10/data_batch_5
ln -s /group/cits5508/CIFAR-10/test_batch

ln -s /group/cits5508/Lab05/data_loader.py

# we can delete the line below. It just shows the contents of
# the /scratch... directory before running Python.
ls -al

# now run our Python notebook file
ipython ./${SCRIPT} 


# remove all the symbolic link files (we don't need them)
# (the actual files that they point to are not affected by this
# removal)
/bin/rm data_batch_*  test_batch  data_loader.py

#############################################
# Now move the output produced by the Python notebook file from
# the /scratch... directory to my home directory.
cd $HOME
mv ${SCRATCH} ${RESULTS}

echo "mv ${SCRATCH} ${RESULTS}"
echo "Please see the ${RESULTS} directory for any output"

echo
echo "Mothur MPI job started  at $start_time"
echo "Mothur MPI job finished at `date`"


