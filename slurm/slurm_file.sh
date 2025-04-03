#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=128
#SBATCH -p normal
#SBATCH -t 49:00:00
#SBATCH --mail-type all

module add lang/Python/3.9.6-GCCcore-11.2.0

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6382
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"

srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" \
        --port=6380 \
        --node-manager-port=6800 \
        --object-manager-port=6801 \
        --ray-client-server-port=20001 \
       --redis-shard-ports=6802 \
        --min-worker-port=20002 \
        --max-worker-port=29999 \
	--include-dashboard=false \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --block &
    sleep 5
done

python RLS/main.py --scenario_file YOURSCENARIO --file YOURARGFILE


