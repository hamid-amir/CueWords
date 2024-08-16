import utils
import torch
import os
import argparse


################# configs ################
# MODEL_CHECKPOINTS = ['bert-base-uncased', './finetuned-bert', 'gpt2', './finetuned-gpt2', 'google/gemma-2b', 'roberta-base', './finetuned-roberta']
# DATASET_PATHS = ['data/gender_agreement']
###########################################


def main(model_ckpt, dataset_path, do_ablation):
  OUTPUT_DIR = "results_vp/" if not do_ablation else "results_vp/ablation/"
  if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

  MIN_NUM_CUES = 2
  MAX_NUM_CUES = 6

  for num_cues in range(MIN_NUM_CUES, MAX_NUM_CUES+1):
    torch.cuda.empty_cache()
    d2vp = utils.dataset2VPs(model_ckpt, dataset_path, num_cues)
    df = d2vp.retrieveVP(ablation_study=do_ablation)

    # save some useful atributions for the dataframe
    df.attrs["model_checkpoint"] = model_ckpt
    df.attrs["dataset_path"] = dataset_path
    df.attrs["num_cues"] = num_cues

    # dump the dataframe to a pickle file
    file_name = model_ckpt.split('/')[-1] + '_' + dataset_path.split('/')[-1] + '_' + str(num_cues) + '.pkl'
    df.to_pickle(OUTPUT_DIR + file_name)
    print(OUTPUT_DIR + file_name)
    print("="*50)




if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Extracts value patching scores given a model and a dataset')
  parser.add_argument('-m', '--model_ckpt', type=str, required=True, choices=['bert-base-uncased', './finetuned-bert', 'gpt2', './finetuned-gpt2', 'google/gemma-2b', 'roberta-base', './finetuned-roberta'],
                      help='Model checkpoint to use')
  parser.add_argument('-d', '--data_path', type=str, default='data/gender_agreement', help='Specify the dataset path (default: %(default)s)')
  parser.add_argument('-a', '--ablation', type=bool, default=False, help='Choose if you want to do the ablation study or not')

  args = parser.parse_args()
  main(args.model_ckpt, args.data_path, args.ablation)