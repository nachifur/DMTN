import yaml
import os

# # clear
# flag = os.system("rm -rf checkpoints")
# if flag == 0:
#     print("clear checkpoints success")
# flag = os.system("rm -rf src/__pycache__")
# if flag == 0:
#     print("clear src/__pycache__ success")
    
fr = open("config.yml", 'r')
config = yaml.load(fr, Loader=yaml.FullLoader)
config["DEBUG"] = 1
config["GPU"] = [0]
with open("config.yml", 'w') as f_obj:
    yaml.dump(config, f_obj)
print("in debug mode")