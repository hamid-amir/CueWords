import utils
import torch
import os


################# configs ################
OUTPUT_DIR = "results/"
if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)

MODEL_CHECKPOINTS = ['roberta-base', 'google/gemma-2b', './finetuned-roberta']
DATASET_PATHS = ['data/gender_agreement']
MIN_NUM_CUES = 2
MAX_NUM_CUES = 10
###########################################

for model_ckpt in MODEL_CHECKPOINTS:
  for dataset_path in DATASET_PATHS:
    for num_cues in range(MIN_NUM_CUES, MAX_NUM_CUES+1):
      torch.cuda.empty_cache()
      d2cm = utils.dataset2CMs(model_ckpt, dataset_path, num_cues)
      df = d2cm.retrieveCM()

      # save some useful atributions for the dataframe
      df.attrs["model_checkpoint"] = model_ckpt
      df.attrs["dataset_path"] = dataset_path
      df.attrs["num_cues"] = num_cues

      # dump the dataframe to a pickle file
      file_name = model_ckpt.split('/')[-1] + '_' + dataset_path.split('/')[-1] + '_' + str(num_cues) + '.pkl'
      df.to_pickle(OUTPUT_DIR + file_name)
      print(OUTPUT_DIR + file_name)
      print("="*50)
