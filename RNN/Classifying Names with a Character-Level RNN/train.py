import torch
from torch.utils.data import DataLoader
from torch.optim import SGD

from dataset import NamesClassifyDataset
from rnn import Network


def train_test(run_train: bool, train_epochs: int, run_test: bool, save_weight: bool, save_path: str):
    assert run_train or run_test
    if run_train:
        assert 0 < train_epochs

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

    if run_train:
        EPOCHS = train_epochs

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
                # print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if save_weight:
            torch.save(network.state_dict(), save_path)

    if run_test:
        print("-" * 30)
        print("TESTING...")
        print("-" * 30)
        network.eval()
        testset = dataset.testset  # [[name, language],...]
        softmax = torch.nn.Softmax(dim=0)
        T, F = 0, 0
        for i in range(len(testset)):
            test_sample, gt_language = testset[i]
            test_sample_encoding = dataset.LetterTool.line2tensor(test_sample)
            label = dataset.languages.index(gt_language)
            # print(test_sample_encoding.shape)
            # print(label)
            # exit()
            hidden_init = torch.zeros(1, 1, hidden_size)
            pred = network(test_sample_encoding, hidden_init).squeeze()
            pred = softmax(pred)
            pred = torch.argmax(pred).item()
            if pred == label:
                T += 1
            else:
                F += 1
        N = T + F
        acc = T/N
        print("test sample count: ", N)
        print("acc: ", acc)


if __name__ == "__main__":
    train_test(run_train=True, train_epochs=5, run_test=True, save_weight=True, save_path='./weights.pth')
