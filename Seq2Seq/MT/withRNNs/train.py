import os.path
import random
import torch
import datetime

from model import EncoderRNN, DecoderRNN, AttnDecoderRNN
from data import MTDataset


def train(file_dir, epochs, lang1, lang2, save_dir, device, resume=True, resume_file='',decoder_with_attn=False, shuffle=True, teacher_forcing_ratio=0.5, log_interval=100):
    """

    :param epochs:
    :param file_dir:
    :param lang1:
    :param lang2:
    :param save_dir:
    :param device:
    :param resume:
    :param decoder_with_attn:
    :param shuffle: 随机采样
    :param teacher_forcing_ratio: 普通模式RNN会使用上一个state的输出作为下一个的输入，teacher_forcing模式会使用gt作为下一个的输入，可以避免训练时前面出现的错误影响后面的预测
    :param log_interval:
    :return:
    """
    if not os.path.isdir(save_dir):
        return
    if save_dir[-1] != '/':
        save_dir += '/'

    dataset = MTDataset(file_dir, 'train', lang1, lang2, shuffle)
    SOS_token = dataset.data_converter.SOS_token
    EOS_token = dataset.data_converter.EOS_token
    N = len(dataset)
    input_size = dataset.input_lang.n_words
    hidden_size = 256
    output_size = dataset.output_lang.n_words
    encoder = EncoderRNN(input_size, hidden_size).to(device)

    if decoder_with_attn:
        decoder = AttnDecoderRNN(hidden_size, output_size).to(device)
    else:
        decoder = DecoderRNN(hidden_size, output_size).to(device)

    if resume:
        if not os.path.exists(resume_file):
            print('model resume fail')
        else:
            sd = torch.load(resume_file, map_location='cpu')
            encoder.load_state_dict(sd['encoder'])
            decoder.load_state_dict(sd['decoder'])


    lr = 1e-3
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=lr)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=lr)

    criterion = torch.nn.CrossEntropyLoss()

    assert 0 < log_interval < N
    N = 60000

    for epoch in range(epochs):
        for i in range(N):
            input_tensor, target_tensor = dataset[i]
            input_tensor.to(device)
            target_tensor.to(device)
            encoder_outputs = []
            encoder_hidden = encoder.initHidden().to(device)
            loss = 0
            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            for j in range(input_length):
                x = input_tensor[j].to(device)
                encoder_output, encoder_hidden = encoder(x, encoder_hidden)
                encoder_outputs.append(encoder_output.squeeze().to(device))

            decoder_input = torch.tensor([[SOS_token]]).to(device)
            decoder_hidden = encoder_hidden.to(device)

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                # 用gt target作为decoder的输入
                for j in range(target_length):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    loss += criterion(decoder_output.squeeze(0).to(device), target_tensor[j])
                    decoder_input = target_tensor[j]
                    if decoder_input.item() == EOS_token:
                        break
            else:
                # 用预测结果作为decoder的输入
                for j in range(target_length):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    loss += criterion(decoder_output.squeeze(0).to(device), target_tensor[j])
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi
                    if decoder_input.item() == EOS_token:
                        break

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            # log
            if i % log_interval == 0:
                print("epoch: %d, %d/%d, loss: %.4f, %s" % (epoch, i, N, loss, datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")))
        save_path = save_dir + 'weights_simpleDecoder_%s-%s_ep_%d.pth' % (lang1, lang2, epoch)
        torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, save_path)


if __name__ == '__main__':
    file_dir = '/Users/wangyunhang/Desktop/NLP-101/Seq2Seq/MT/data/'
    epochs = 1
    lang1 = 'eng'
    lang2 = 'fra'
    save_dir = './'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    resume = True
    resume_file = ''
    train(file_dir, epochs, lang1, lang2, save_dir, device, resume=resume, resume_file=resume_file, shuffle=True, teacher_forcing_ratio=0.5)
