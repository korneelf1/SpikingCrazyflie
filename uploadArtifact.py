import wandb
wandbname='still-valley-183'
files = ['stabilize/sac/policy_snn_actor_Full_State_2024-09-28 02:08:27.464642_slope_10.pth',
        ]
wandb.init(project='SSAC',id=wandbname, resume=True)
for file in files:
    wandb.log_artifact(file, name=wandbname, type='model')
wandb.finish()