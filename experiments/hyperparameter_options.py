params_pif = {
    "model" : ["ModelAPIF", "ModelBPIF"],
    "batch_size" : [4, 6, 12],
    "patience" : {
        "small" : [12, 16],
        "big" : [8, 12]
        },
    "lr" : [1e-4, 5e-5]
}

params_baseline = {
    "model" : ["ModelA", "ModelB"],
    "batch_size" : [4, 6, 12],
    "patience" : {
        "small" : [12, 16],
        "big" : [8, 12]
        },
    "lr" : [1e-4, 5e-5]
}

params_baseline_2 = {
    "model" : ["ModelA", "ModelB"],
    "batch_size" : [4, 6, 12],
    "patience" : {
        "small" : [12, 16],
        "big" : [8, 12]
        },
    "lr" : [1e-4, 5e-5]
}

params_liu = {
    "model" : ["ModelLiu", "ModelB"],
    "batch_size" : [12, 16, 32],
    "patience" : {
        "small" : [12, 16],
        "big" : [8, 12]
        },
    "lr" : [1e-4, 5e-5]
}