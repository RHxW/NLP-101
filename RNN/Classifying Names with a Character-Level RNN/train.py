import torch
from torch.utils.data import DataLoader
from torch.optim import SGD

from dataset import NamesClassifyDataset
from rnn import Network


def train(run_test, save_weight, save_path):
    dataset = NamesClassifyDataset()

    # sizes
    input_size = dataset.input_size
    hidden_size = 128
    output_size = dataset.C

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,  # 每个词长度不一样，所以batch_size只能为1
        shuffle=True,
        drop_last=True,
    )

    network = Network(input_size, hidden_size, output_size)

    optimizer = SGD(network.parameters(), lr=1e-3, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    EPOCHS = 2

    for epoch in range(EPOCHS):
        for i, data in enumerate(dataloader):
            name_encoding, class_idx = data
            # print(name_encoding.shape)
            # print(class_idx)
            # exit()
            name_encoding = name_encoding.squeeze(0)
            hidden_init = torch.zeros(1, 1, hidden_size)
            pred = network(name_encoding, hidden_init)
            pred = pred.squeeze(0)
            loss = criterion(pred, class_idx)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if save_weight:
        torch.save(network.state_dict(), save_path)

    if run_test:
        network.eval()
        testset = dataset.testset
        for i in range(len(testset)):
            test_sample


if __name__ == "__main__":
    train(True, True, './weights.pth')
