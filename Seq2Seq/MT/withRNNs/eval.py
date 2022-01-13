import torch

from model import EncoderRNN, DecoderRNN, AttnDecoderRNN
from data import MTDataset


def eval(file_dir, lang1, lang2, weight_path, device, eval_num, decoder_with_attn=False, shuffle=True):
    """

    :param file_dir:
    :param lang1:
    :param lang2:
    :param weight_path:
    :param device:
    :param eval_num:
    :param decoder_with_attn:
    :param shuffle:
    :return:
    """
    dataset = MTDataset(file_dir, 'eval', lang1, lang2, shuffle)
    input_lang = dataset.input_lang
    output_lang = dataset.output_lang
    N = dataset.N
    SOS_token = dataset.data_converter.SOS_token
    EOS_token = dataset.data_converter.EOS_token

    input_size = input_lang.n_words
    hidden_size = 256
    output_size = output_lang.n_words

    encoder = EncoderRNN(input_size, hidden_size).to(device)

    if decoder_with_attn:
        decoder = AttnDecoderRNN(hidden_size, output_size).to(device)
    else:
        decoder = DecoderRNN(hidden_size, output_size).to(device)

    weights = torch.load(weight_path, map_location='cpu')
    encoder.load_state_dict(weights['encoder'])
    decoder.load_state_dict(weights['decoder'])

    assert 1 < eval_num <= N
    for i in range(eval_num):
        input_tensor, target_tensor = dataset[i]
        input_tensor.to(device)
        input_length = len(input_tensor)
        target_tensor.to(device)
        encoder_outputs = []
        encoder_hidden = encoder.initHidden().to(device)
        decoded_words = []

        for j in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
            encoder_outputs.append(encoder_output.squeeze())

        decoder_input = torch.tensor([[SOS_token]]).to(device)
        decoder_hidden = encoder_hidden
        while True:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        print("input: ", dataset.pairs[i][0])
        print("output: ", " ".join(decoded_words))
        print("target: ", dataset.pairs[i][1])
        print("*" * 30)

if __name__ == '__main__':
    file_dir = '/Users/wangyunhang/Desktop/NLP-101/Seq2Seq/MT/data/'
    lang1 = 'eng'
    lang2 = 'fra'
    save_path = './weights_%s-%s_0.pth' % (lang1, lang2)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    eval_num = 10
    eval(file_dir, lang1, lang2, save_path, device, eval_num)