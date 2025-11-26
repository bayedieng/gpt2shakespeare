import torch
from dataset import Gpt2ShakespeareDatset
from torch.utils.data import DataLoader, random_split
from tqdm import trange
from model import Gpt2Shakespeare

dataset = Gpt2ShakespeareDatset()
train_len = int(0.8 * len(dataset))
test_len = len(dataset) - train_len
batch_size = 8
num_epochs = 3

g = torch.Generator().manual_seed(42)
trainset, testset = random_split(dataset, [train_len, test_len], g)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)


loss_func = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam()


def train():
    for _ in (t := trange(num_epochs)):
        for x, y in trainloader:
            x = x.cuda()
            y = y.cuda()

            optim.zero_grad()
            out = Gpt2Shakespeare(x)
            loss = loss_func(out, y)

            loss.backward()
            optim.step()

            pred_out = torch.argmax(out, dim=1)
            acc = (pred_out == y).float().mean()
            t.set_description(f"Accuracy = {acc.item():.2f} loss = {loss.item():.2f}")
