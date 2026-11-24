# Train b_model_pytorch (Transformer from torch.nn primitives).
# Same steps as train_manual.py; the difference is the masks:
#   train_manual.py uses the masks from BilingualDataset (1 = allowed, combined padding & causal)
#   here they are built from the token ids in the PyTorch convention (True = blocked, separate padding and causal)

import os
import warnings

import torch
import torch.nn as nn
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from b_model_pytorch.model import build_transformer
from config import get_config, get_model_folder, get_weights_file_path, latest_weights_file_path
from dataset.build_dataset import get_ds
from utils import vizualize_model_by_keys


# Define the device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
device = torch.device(device)


def padding_mask(tokens: torch.Tensor, pad_idx: int) -> torch.Tensor:
    # (B, S) token ids --> (B, S) bool, True = [PAD] (attention ignores these keys)
    return tokens == pad_idx


def causal_mask(size: int, device) -> torch.Tensor:
    # (L, L) bool, True above the diagonal = future positions (blocked)
    # same as manual causal_mask, but inverted: there 1 = allowed
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


def greedy_decode(model, source, src_pad_mask, tokenizer_tgt, max_len, device):
    """
    for inference (generation) we have to run encoder only once!
    - compute encoder once, then generate first token by decoder, then add it to decoded input etc
    - generation stops when we reach EOS or max_length
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, src_pad_mask)

    # Initialize the decoder input with the sos token (1, 1) = batch x first token
    decoder_input = torch.tensor([[sos_idx]], dtype=source.dtype, device=device)

    while decoder_input.size(1) < max_len:
        # the generated tokens contain no [PAD], so only the causal mask is needed
        out = model.decode(encoder_output, src_pad_mask, decoder_input, causal_mask(decoder_input.size(1), device))

        # get next token: highest score at the last position
        next_word = model.project(out[:, -1]).argmax(dim=-1, keepdim=True)   # (1, 1)
        decoder_input = torch.cat([decoder_input, next_word], dim=1)

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)   # remove the batch dimension


def train_model(config):

    # Make sure the weights folder exists
    get_model_folder(config).mkdir(parents=True, exist_ok=True)

    # transform text corpus into datasets with encoder_input, decoder_input tensors
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = build_transformer(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    src_pad_idx = tokenizer_src.token_to_id('[PAD]')
    tgt_pad_idx = tokenizer_tgt.token_to_id('[PAD]')

    vizualize_model_by_keys(model)
    print(model)

    writer = SummaryWriter(config['experiment_name'])   # Tensorboard
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    # load checkpoint if config['preload'] is specified
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    # make PAD does not contribute to the loss (labels are target tokens -> target [PAD])
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx, label_smoothing=0.1).to(device)

    ##### TRAIN LOOP
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()               # set model back to a training mode
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)       # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device)       # (B, seq_len)

            # masks in PyTorch convention (the dataset's encoder_mask / decoder_mask are not used)
            src_pad_mask = padding_mask(encoder_input, src_pad_idx)   # (B, seq_len)
            tgt_pad_mask = padding_mask(decoder_input, tgt_pad_idx)   # (B, seq_len)
            tgt_mask = causal_mask(decoder_input.size(1), device)     # (seq_len, seq_len)

            # FORWARD pass: run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, src_pad_mask)                                  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, src_pad_mask, decoder_input, tgt_mask, tgt_pad_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)                                                 # (B, seq_len, vocab_size)

            # Compute the loss
            label = batch['label'].to(device)   # (B, seq_len)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        _run_validation(
            model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
            lambda msg: batch_iterator.write(msg), global_step, writer
        )

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}_{loss.item():5.2f}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


def _run_validation(
        model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device,
        print_msg, global_step, writer, num_examples=2
):
    model.eval()
    count = 0
    src_pad_idx = tokenizer_src.token_to_id('[PAD]')

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except Exception:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():   # disable gradient calculation
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)   # (1, seq_len)

            # check that the batch size is 1 (it should be always 1 for validation ds)
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, padding_mask(encoder_input, src_pad_idx), tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().tolist())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            # print_msg = tqdm's write(): print() would break the progress bar
            print_msg('-' * console_width)
            print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break

    if writer:
        # Compute the char error rate
        cer = torchmetrics.CharErrorRate()(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)

        # Compute the word error rate
        wer = torchmetrics.WordErrorRate()(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)

        # Compute the BLEU metric
        bleu = torchmetrics.BLEUScore()(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()

    # Separate checkpoints and Tensorboard logs from the manual model: the two models name their
    # layers differently, so a manual checkpoint cannot be loaded into this one (and vice versa).
    # The prefix must not start with the manual one ('transl_model_'), or latest_weights_file_path()
    # of the manual model would pick these files up with its 'transl_model_*' pattern.
    config['model_basename'] = 'pytorch_transl_model_'
    config['experiment_name'] = 'runs/transl_model_pytorch'

    train_model(config)

    print()
