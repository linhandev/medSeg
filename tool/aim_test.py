import aim

hyperparam_dict = {"lr": 0.1, "decay": 0.99, "network": "unet"}
aim.set_params(hyperparam_dict, name="params_name")
for _ in range(10):
    aim.track(_, name="test_metric", epoch=_)
