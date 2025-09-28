from pathlib import Path


def get_config():
    return {
        "batch_size": 64,  # on our model and setup, batch size 64 used about 34gb of vram on an a100 40gb
        # you can use gradient accumulation to simulate larger batches if memory is not enough, tradeoff is longer training time per epoch
        "num_epochs": 50,  # in testing, loss went down to ~3.5 after 20 epochs(took around ~100 minutes), so running up to 50 might help if validation loss keeps improving
        "lr": 10
        ** -4,  # using 1e-4 as starting lr with adam, usually safe, you can adjust with a scheduler if needed
        "seq_len": 350,
        "d_model": 512,
        "datasource": "opus_books",
        "lang_src": "en",
        "lang_tgt": "fr",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
