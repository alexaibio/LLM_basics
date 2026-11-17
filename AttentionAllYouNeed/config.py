from pathlib import Path


def get_config():
    return {
        "batch_size": 20,
        "num_epochs": 30,  #20,
        "lr": 5**-4,
        "seq_len": 350,
        # Transformer hyperparameters (values from the paper's base model in comments)
        "d_model": 512,     # embedding size                         (paper: 512)
        "N": 1,             # number of encoder and of decoder blocks (paper: 6)
        "h": 8,             # number of attention heads              (paper: 8)
        "d_ff": 2048,       # hidden size of the feed-forward block  (paper: 2048)
        "dropout": 0.1,     #                                        (paper: 0.1)
        "datasource": 'Helsinki-NLP/opus_books',  # HF now requires 'namespace/name'
        "lang_src": "en",
        "lang_tgt": "ru",
        "model_folder": "model_saved",   # model weights + tokenizers
        "model_basename": "transl_model_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/transl_model"
    }


PROJECT_DIR = Path(__file__).resolve().parent   # .../AttentionAllYouNeed


def get_model_folder(config) -> Path:
    # Model weights and tokenizers are saved here.
    # Anchored to the project folder, so it does not depend on the current working directory
    return PROJECT_DIR / config['model_folder']   # .../AttentionAllYouNeed/model_saved


def get_weights_file_path(config, epoch: str):
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(get_model_folder(config) / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_filename = f"{config['model_basename']}*"
    weights_files = list(get_model_folder(config).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])