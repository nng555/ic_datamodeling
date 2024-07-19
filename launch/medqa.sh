 python3 eval_dataset.py dataset=bigbio/med_qa num_shards=32 shard=0,1 tags=medqa hydra.launcher.qos=normal hydra.launcher.time=960  -m
 python3 eval_dataset.py dataset=bigbio/med_qa num_shards=32 shard=$(seq -s, 2 1 3) tags=medqa hydra.launcher.qos=m hydra.launcher.time=720 -m
 python3 eval_dataset.py dataset=bigbio/med_qa num_shards=32 shard=$(seq -s, 3 1 10) tags=medqa hydra.launcher.qos=m2 hydra.launcher.time=480 -m
 python3 eval_dataset.py dataset=bigbio/med_qa num_shards=32 shard=$(seq -s, 11 1 25) tags=medqa hydra.launcher.qos=m3 hydra.launcher.time=240 -m
 python3 eval_dataset.py dataset=bigbio/med_qa num_shards=32 shard=$(seq -s, 16 1 31) tags=medqa hydra.launcher.qos=m4 hydra.launcher.time=120 -m
 #python3 eval_dataset.py dataset=bigbio/med_qa num_shards=32 shard=$(seq -s, 0 1 31) tags=medqa hydra.launcher.qos=scavenger hydra.launcher.time=2880 -m
