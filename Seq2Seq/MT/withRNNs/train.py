import random
import torch

from model import EncoderRNN, DecoderRNN
from data import MTDataset

def train(epochs, lang1, lang2, shuffle=True, teacher_forcing_ratio=0.5):
    """

    :param epochs:
    :param lang1:
    :param lang2:
    :param shuffle: 随机采样
    :param teacher_forcing_ratio: 普通模式RNN会使用上一个state的输出作为下一个的输入，teacher_forcing模式会使用gt作为下一个的输入，可以避免训练时前面出现的错误影响后面的预测
    :return:
    """
    file_dir = '/Users/wangyunhang/Desktop/NLP-101/Seq2Seq/MT/'
    dataset = MTDataset(file_dir, lang1, lang2, shuffle)
    SOS_token = dataset.data_converter.SOS_token
    EOS_token = dataset.data_converter.EOS_token
    N = len(dataset)
    input_size = dataset.input_lang.n_words
    hidden_size = 256
    output_size = dataset.output_lang.n_words
    encoder = EncoderRNN(input_size, hidden_size)
    decoder = DecoderRNN(hidden_size, output_size)

    lr = 1e-3
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=lr)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=lr)

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for i in range(N):
            input_tensor, target_tensor = dataset[i]
            encoder_outputs = []
            encoder_hidden = encoder.initHidden()
            loss = 0
            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            for j in range(input_length):
                x = input_tensor[j]
                encoder_output, encoder_hidden = encoder(x, encoder_hidden)
                encoder_outputs.append(encoder_output.squeeze())

            decoder_input = torch.tensor([[SOS_token]])
            decoder_hidden = decoder.initHidden()

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                # 用gt target作为decoder的输入
                for j in range(target_length):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    loss += criterion(decoder_output, target_tensor[j])
                    decoder_input = target_tensor[j]
                    if decoder_input.item() == EOS_token:
                        break
            else:
                # 用预测结果作为decoder的输入
                for j in range(target_length):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    loss += criterion(decoder_output, target_tensor[j])
                    decoder_input = decoder_output
                    if decoder_input.item() == EOS_token:
                        break

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()


if __name__ == '__main__':
    epochs = 2
    lang1 = 'eng'
    lang2 = 'fra'
    train(epochs, lang1, lang2, shuffle=True, teacher_forcing_ratio=0.5)