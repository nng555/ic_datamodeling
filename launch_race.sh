 #python3 eval_dataset.py dataset=ehovy/race subset=high num_shards=32 shard=0,1 tags=race hydra.launcher.qos=normal hydra.launcher.time=960  -m
 #python3 eval_dataset.py dataset=ehovy/race subset=high num_shards=32 shard=$(seq -s, 2 1 5) tags=race hydra.launcher.qos=m hydra.launcher.time=720 -m
 #python3 eval_dataset.py dataset=ehovy/race subset=high num_shards=32 shard=$(seq -s, 6 1 13) tags=race hydra.launcher.qos=m2 hydra.launcher.time=480 -m
 #python3 eval_dataset.py dataset=ehovy/race subset=high num_shards=32 shard=$(seq -s, 14 1 29) tags=race hydra.launcher.qos=m3 hydra.launcher.time=240 -m
 python3 eval_dataset.py dataset=ehovy/race subset=high num_shards=30 shard=$(seq -s, 0 1 29) tags=race hydra.launcher.qos=m4 hydra.launcher.time=120 -m
