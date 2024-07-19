 #python3 eval_dataset.py dataset=facebook/anli num_shards=32 shard=0,1 tags=anli hydra.launcher.qos=normal hydra.launcher.time=960  -m
 #python3 eval_dataset.py dataset=facebook/anli num_shards=32 shard=$(seq -s, 2 1 5) tags=anli hydra.launcher.qos=m hydra.launcher.time=720 -m
 #python3 eval_dataset.py dataset=facebook/anli num_shards=32 shard=$(seq -s, 6 1 13) tags=anli hydra.launcher.qos=m2 hydra.launcher.time=480 -m
 #python3 eval_dataset.py dataset=facebook/anli num_shards=32 shard=$(seq -s, 14 1 29) tags=anli hydra.launcher.qos=m3 hydra.launcher.time=240 -m
 python3 eval_dataset.py dataset=winogrande num_shards=32 shard=$(seq -s, 0 1 31) subset=winogrande_debiased tags=wino hydra.launcher.qos=scavenger hydra.launcher.time=2880 hydra.launcher.partition=rtx6000 -m
