#!/bin/bash
#SBATCH -e log/log.err
#SBATCH -o log/log.out
#SBATCH -p kshdnormal02
#SBATCH -N 4
#SBATCH -J LAST_WAV
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH --exclusive
#SBATCH --exclude=j09r3n[15-18],j09r2n19,j09r3n[00-02],e18r1n[00-07],b09r2n[11-14],e07r4n19,e08r1n[00-02],j09r4n[03-06],j13r2n[16-19],b08r3n[14-17],b09r2n[15-18],j09r2n[10-13],e05r2n[14-17],j09r4n[11-18],e06r1n[12-15],e18r3n[03-06],e11r1n[15-19],e11r2n[00-02],e18r3n[07-10],e10r4n[00-03],e18r3n[11-14],e18r2n[06-09],e07r1n[05-08],j09r3n[07-10],e06r1n[16-19],e06r2n[13-16],b05r3n19,b05r4n00,b05r4n01,b05r4n02,f15r3n05,f15r3n06,f15r3n07,f15r3n08,f15r1n11,f16r4n09,f17r4n10,f17r4n11,f16r4n[00-03],f15r3n09,f15r3n10,f15r3n11,f15r3n12,f15r3n04,f17r2n04,f17r2n05,f17r2n06

# the lowercase `n` above does hints SLURM to provide sufficent cpu resources for task, does not launch that many tasks as does `srun`
# but we don't really need all that many cpu cores to train a network, is this overpricing?

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
# module load compiler/rocm/dtk-23.10
module load compiler/rocm/dtk-23.10
source /public/home/qinxy/anaconda3/bin/activate
conda activate photon_avse
export LD_LIBRARY_PATH=/public/home/qinxy/anaconda3/envs/avse/lib:$LD_LIBRARY_PATH


# get the number of assigned hosts by logging host list queried from
# `scontrol` int a temp file with the same filename as the assigned slurm job id
hostfile=./$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}
num_node=$(cat $hostfile|sort|uniq | wc -l)

# topolgy of DCU cards are known in advance as four per node,
# this assumes we always takeover entire nodes
num_DCU=$(($num_node*4))

# this generates the argument for single_prcess.sh, the rendezvous port for process-group construction
# the first line uses `sed` to takeout the first line, which is the first node of assigend hosts,
# this choice is arbitrary, we are in a symmetric cluster
# the second line uses `awk` to single out the first column of the hostname, this seems to be redundant,
# maybe there are cases where `scontrol` does something weird
# in any case, the resultant $dist_url should be something like "e05r4n16", since we uses hsotnames in this setup
nodename=$(cat $hostfile | sed -n "1p")	# one p
dist_url=`echo $nodename | awk '{print $1}'`

# ?? (maybe) delete any previous hostfile-dls?
rm `pwd`/hostfile-dl -f

# move hostfile into a tmp dir
cat $hostfile | sort | uniq > `pwd`/tmp

for i in `cat ./tmp`
do
  echo ${i} slots=4 >> `pwd`/hostfile-dl
done

mpirun -np $num_DCU --allow-run-as-root -hostfile `pwd`/hostfile-dl sh `pwd`/single_process.sh $dist_url
