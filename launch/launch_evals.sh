python3 eval_dataset.py dataset=Anthropic/discrim-eval subset=explicit mask_p=0.15  tags=discrim hydra.launcher.qos=m hydra.launcher.time=720 -m
python3 eval_dataset.py dataset=ehovy/race mask_p=0.15 subset=high tags=race hydra.launcher.qos=m hydra.launcher.time=720 -m
python3 eval_dataset.py dataset=ucinlp/drop mask_p=0.15 tags=drop hydra.launcher.qos=m hydra.launcher.time=720 -m
python3 eval_dataset.py dataset=stanfordnlp/imdb mask_p=0.15 tags=imdb hydra.launcher.qos=m hydra.launcher.time=720 -m
