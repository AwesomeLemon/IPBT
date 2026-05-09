#######################################################
# 1. IPBT

## 1.1 Main experiments

### 1.1.1. RL
for task in humanoid hopper walker pusher; do
  for seed in {0..7}; do
    python run.py --config-name ipbt_rl_0001 server=star04 task=${task} ++general.seed_offset=${seed};
done; done

### 1.1.2. CIFAR-10/CIFAR-C100/Fashion-MNIST
for task in c10 c100 fmnist; do
  for seed in {0..7}; do
    python run.py --config-name ipbt_cls_0001 server=star04 task=${task} ++general.seed_offset=${seed};
done; done

### 1.1.3. TinyImageNet
(for seed in {0..7}; do python run.py --config-name ipbt_timg_0001 server=star04 ++general.seed_offset=${seed}; done) &

## 1.2. Ablation studies

### Use the commands above for the Humanoid/Hopper/CIFAR-10/CIFAR-100, using the following config numbers (instead of 0001):
### (the order is the same as in Fig. 7 and then in the figures in the appendix)

### 0000 - Default IPBT
### 0016 - Stagnation detection of BG-PBT
### 0008 - Shrink-perturb (1.0, 0.0)
### 0009 - Shrink-perturb (0.0, 1.0)
### 0006 - Shrink-perturb (0.4, 0.1)
### 0007 - Shrink-perturb (0.1, 0.1)
### 0014 - Prob. of random weight reinit. 0.0
### 0017 - Meta BO of BG-PBT
### 0013 - Prob. of random HPs reinit. 1.0
### 0012 - Prob. of random HPs reinit. 0.0
### 0010 - Step size: linear increase
### 0011 - Step size: constant
### 0005 - Population multiple 1
### 0004 - Population multiple 3
### (Appendix starts)
### 0022 - Step 0.5%
### 0019 - Step 3.0%
### 0018 - Distillation

#######################################################
# 2. PBT-Zoo: PBT, PB2, PB2-Mix, BG-PBT, FIRE-PBT
# The commands are a bit complicated for two reasons:
# - Setup of BG-PBT has to be changed depending on the step size/task (see Appendix E)
# - FIRE-PBT has separate configs with a special suffix because it needs intermediate evaluations & must have subpopulation sizes specified

process_params() {
  local algo="$1" step="$2"
  local algo_full maybe_fire_suffix

  [ "$algo" == "firepbt" ] && maybe_fire_suffix="_fire" || maybe_fire_suffix=""

  if [ "$algo" == "bgpbt" ]; then
    # rl, c10/100/fmnist, timg
    if [ "$step" == "15e6" ] || [ "$step" == "500" ] || [ "$step" == "1500" ]; then
      algo_full="bgpbt_steps100"
    elif [ "$step" == "50e6" ] || [ "$step" == "16660" ] || [ "$step" == "50000" ]; then
      algo_full="bgpbt_steps3"
    else
      algo_full="$algo"
    fi
  else
    algo_full="$algo"
  fi

  echo "$algo_full $maybe_fire_suffix"
}

## 2.1 CIFAR-10/CIFAR-C100/Fashion-MNIST

### 2.1.1. Everything except BG w/ steps 100
(cfgs=(pbtzoo_cls_0000 pbtzoo_cls_0001 pbtzoo_cls_0002 pbtzoo_cls_0003);
steps=(16660 5000 1660 500);
for task in c10 c100 fmnist; do
  for i in "${!cfgs[@]}"; do
    for algo in pbt pb2rand pb2mix bgpbt firepbt; do
      [ "$algo" == "bgpbt" ] && [ "${steps[$i]}" == "500" ] && continue
      for seed in {0..7}; do
        read algo_full maybe_fire_suffix <<< $(process_params "$algo" "${steps[$i]}")
        python run.py --config-name ${cfgs[$i]} server=star04 task=${task} algo=${algo_full} algo/pop_size@algo=8${maybe_fire_suffix}.yaml algo/t_step@algo=${steps[$i]}${maybe_fire_suffix}.yaml ++general.seed_offset=${seed}
      done;
    done;
  done;
done) &

### 2.1.2. BG w/ steps 100
for task in c10 c100 fmnist; do
  for seed in {0..7}; do
    python run.py --config-name pbtzoo_cls_0003 server=star04 algo=bgpbt_steps100 task=${task} ++algo.reinit_weights_strategy=copy ++general.seed_offset=${seed};
done; done

## 2.2. RL

### 2.2.1. Everything except BG w/ steps 100
(cfgs=(pbtzoo_rl_0000 pbtzoo_rl_0001 pbtzoo_rl_0002 pbtzoo_rl_0003);
steps=(50e6 15e6 5e6 15e5);
for task in hopper humanoid pusher walker; do
  for i in "${!cfgs[@]}"; do
    for algo in pbt pb2rand pb2mix bgpbt firepbt; do
      [ "$algo" == "bgpbt" ] && [ "${steps[$i]}" == "15e5" ] && continue
      for seed in {0..7}; do
        read algo_full maybe_fire_suffix <<< $(process_params "$algo" "${steps[$i]}")
        python run.py --config-name ${cfgs[$i]} server=star04 task=${task} algo=${algo_full} algo/pop_size@algo=8${maybe_fire_suffix}.yaml algo/t_step@algo=${steps[$i]}${maybe_fire_suffix}.yaml ++general.seed_offset=${seed}
      done;
    done;
  done;
done) &

## 2.2.2. BG w/ steps 100
for task in hopper humanoid pusher walker; do
  for seed in {0..7}; do
    python run.py --config-name pbtzoo_rl_0003 server=star04 algo=bgpbt_steps100 task=${task} ++algo.reinit_weights_strategy=distill ++general.seed_offset=${seed};
done; done

## 2.3. TinyImageNet

### 2.3.1. Everything except BG w/ steps 100
(cfgs=(pbtzoo_timg_0000 pbtzoo_timg_0001 pbtzoo_timg_0002 pbtzoo_timg_0003);
steps=(50000 15000 5000 1500);
for i in "${!cfgs[@]}"; do
  for algo in pbt pb2rand pb2mix bgpbt firepbt; do
    [ "$algo" == "bgpbt" ] && [ "${steps[$i]}" == "1500" ] && continue
    for seed in {0..7}; do
      read algo_full maybe_fire_suffix <<< $(process_params "$algo" "${steps[$i]}")
      python run.py --config-name ${cfgs[$i]} server=star04 algo=${algo_full} algo/pop_size@algo=8${maybe_fire_suffix}.yaml algo/t_step@algo=${steps[$i]}${maybe_fire_suffix}.yaml ++general.seed_offset=${seed}
    done;
  done;
done) &

### 2.3.2. BG w/ steps 100
for seed in {0..7}; do
  python run.py --config-name pbtzoo_timg_0003 server=star04 algo=bgpbt_steps100 ++algo.reinit_weights_strategy=copy ++general.seed_offset=${seed};
done;

#######################################################
# 3. Ray Tune: ASHA, Random Search

## 3.1. RL
for task in humanoid_cosine_restart hopper_cosine_restart walker_cosine_restart pusher_cosine_restart; do
  for algo in asha rs; do
    for seed in {0..7}; do
      python run_raytune.py --config-name raytune_rl_0001 server=star04 algo=${algo} task=${task} ++general.seed_offset=${seed};
    done;
  done;
done

## 3.2. CIFAR-10/CIFAR-C100/Fashion-MNIST
for task in c10_cosine_restart c100_cosine_restart fmnist_cosine_restart; do
  for algo in asha rs; do
    for seed in {0..7}; do
      python run_raytune.py --config-name raytune_cls_0001 server=star04 algo=${algo} task=${task} ++general.seed_offset=${seed};
    done;
  done;
done

## 3.3. TinyImageNet
for algo in asha rs; do
  for seed in {0..7}; do
    python run_raytune.py --config-name raytune_timg_0001 server=star04 algo=${algo} task=timg_cosine_restart ++general.seed_offset=${seed};
  done;
done;

#######################################################
# 4. SMAC3

## 4.1. RL
for task in humanoid hopper walker pusher; do
  for seed in {0..7}; do
    python run_smac.py --config-name smac_rl_0001 server=star04 task=${task} ++general.seed_offset=${seed};
done; done

## 4.2. CIFAR-10/CIFAR-C100/Fashion-MNIST
for task in c10 c100 fmnist; do
  for seed in {0..7}; do
    python run_smac.py --config-name smac_cls_0001 server=star04 task=${task} ++general.seed_offset=${seed};
done; done

## 4.3. TinyImageNet
for seed in {0..7}; do
  python run_smac.py --config-name smac_timg_0001 server=star04 ++general.seed_offset=${seed};
done;


#######################################################
# 5. Schedule replay

## 5.1. IPBT - CIFAR-10/CIFAR-C100/Fashion-MNIST

for task in c10 c100 fmnist; do
  for seed in {0..7}; do
    python run.py --config-name replay_cls_ipbt6_seedmatch_0001 server=star04 task=${task} ++general.seed_offset=${seed};
done; done

## 5.2. IPBT - Humanoid

for seed in {0..7}; do 
  python run.py --config-name replay_rl_ipbt6_seedmatch_0001 server=star04 ++general.seed_offset=${seed};
done

## 5.3. ASHA - Humanoid
for seed in {0..7}; do
  python run.py --config-name replay_rl_asha_seedmatch_0001 server=star04 ++general.seed_offset=${seed};
done;