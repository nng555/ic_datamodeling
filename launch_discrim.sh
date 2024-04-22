 #python3 eval_dataset.py dataset=Anthropic/discrim-eval subset=explicit num_shards=32 shard=0,1 tags=discrim hydra.launcher.qos=normal hydra.launcher.time=960  -m
 #python3 eval_dataset.py dataset=Anthropic/discrim-eval subset=explicit num_shards=32 shard=$(seq -s, 2 1 5) tags=discrim hydra.launcher.qos=m hydra.launcher.time=720 -m
 #python3 eval_dataset.py dataset=Anthropic/discrim-eval subset=explicit num_shards=32 shard=$(seq -s, 6 1 13) tags=discrim hydra.launcher.qos=m2 hydra.launcher.time=480 -m
 #python3 eval_dataset.py dataset=Anthropic/discrim-eval subset=explicit num_shards=32 shard=$(seq -s, 14 1 29) tags=discrim hydra.launcher.qos=m3 hydra.launcher.time=240 -m
 python3 eval_dataset.py dataset=Anthropic/discrim-eval subset=explicit num_shards=32 shard=$(seq -s, 30 1 31) tags=discrim hydra.launcher.qos=m4 hydra.launcher.time=120 -m
