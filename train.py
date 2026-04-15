import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import PhoWhisperCTCModel, load_wav_to_mel
from phoneme_set import encode_sequence, PAD_IDX

# ================= DATASET =================
class AudioDataset(Dataset):
    def __init__(self, manifest_path):
        self.data = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        wav = item["audio"]
        phonemes = item["phonemes"]

        mel = load_wav_to_mel(wav)  # (1, T, 80)
        target = torch.tensor(encode_sequence(phonemes), dtype=torch.long)

        return mel.squeeze(0), target


# ================= COLLATE =================
def collate_fn(batch):
    mels, targets = zip(*batch)

    mel_lens = [m.shape[0] for m in mels]
    target_lens = [len(t) for t in targets]

    mels = nn.utils.rnn.pad_sequence(mels, batch_first=True)
    targets = torch.cat(targets)

    return mels, targets, mel_lens, target_lens


# ================= TRAIN =================
def train(num_epochs=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    dataset = AudioDataset("data/manifest.jsonl")
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = PhoWhisperCTCModel(device=device)
    optimizer = torch.optim.Adam(model.ctc_head.parameters(), lr=1e-3)

    ctc_loss = nn.CTCLoss(blank=PAD_IDX, zero_infinity=True)

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        # Progress bar for the current epoch
        progress_bar = tqdm(
            loader,
            desc=f"Epoch [{epoch + 1}/{num_epochs}]",
            unit="batch",
            dynamic_ncols=True,
            colour="cyan",
        )

        for mels, targets, mel_lens, target_lens in progress_bar:
            mels = mels.to(device)

            logits = model(mels)  # (B, T, V)
            log_probs = logits.log_softmax(dim=-1)

            log_probs = log_probs.permute(1, 0, 2)  # (T, B, V)

            input_lengths = torch.tensor([log_probs.size(0)] * log_probs.size(1))
            target_lengths = torch.tensor(target_lens)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar with live batch loss
            progress_bar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss":   f"{total_loss / num_batches:.4f}",
            })

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  ✓ Epoch [{epoch + 1}/{num_epochs}] complete — Avg Loss: {avg_loss:.4f}\n")

    torch.save(model.ctc_head.state_dict(), "checkpoint.pt")
    print("Model saved to checkpoint.pt")


if __name__ == "__main__":
    train()