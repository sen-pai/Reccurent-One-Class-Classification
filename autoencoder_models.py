import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import numpy as np


class RNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        n_layers=1,
        hidden_size=100,
        encoder_dropout=0.0,
        bidirectional=False,
    ):
        super(RNNEncoder, self).__init__()

        self.input_size = input_dim
        self.hidden_size = hidden_size
        self.layers = n_layers
        self.dropout = encoder_dropout
        self.bi = bidirectional

        self.encoder_rnn = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=True,
        )

    def encode(self, inputs, input_lengths):
        packed_input = pack_padded_sequence(
            inputs, input_lengths, enforce_sorted=False, batch_first=True
        )

        self.batch_size = inputs.size()[0]
        max_len = int(torch.max(input_lengths).item())

        encoder_outputs, (h_n, c_n) = self.encoder_rnn(packed_input)
        encoder_outputs, _ = pad_packed_sequence(
            encoder_outputs, batch_first=True, total_length=max_len
        )
        # encoder_outputs -> [batch size, max seq lenght, hidden size]
        # h_n -> [1, batch size, hidden size]
        # c_n -> [1, batch size, hidden size]

        if self.bi:
            h_n = h_n.view(1, self.batch_size, self.hidden_size*2)
            c_n = c_n.view(1, self.batch_size, self.hidden_size*2)
            
        return encoder_outputs, h_n, c_n

    def device(self) -> torch.device:
        """Heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        return first_param.device


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        max_seq_len = encoder_outputs.size(1)
        h = hidden.repeat(max_seq_len, 1, 1).transpose(0, 1)
        # h -> [batch size, max seq len, hidden size]
        # encoder_outputs -> [batch size, max seq len, hidden size]
        attn_energies = self.score(h, encoder_outputs)
        # attn_energies -> [batch size, max seq len]
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class BaseDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=1,
        n_layers=1,
        hidden_size=100,
        dropout=0.0,
        bidirectional=False,
        use_r_linear=True,  # adds a MLP after RNN predictions
    ):
        super(BaseDecoder, self).__init__()

        self.input_size = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.layers = n_layers
        self.dropout = dropout
        self.bi = bidirectional
        self.use_r_linear = use_r_linear

    def single_step_deocde(self, prev_decode_output, encoder_outputs, h_i, c_i):
        raise NotImplementedError

    def decode(self, h_n, c_n, lens, encoder_outputs):
        raise NotImplementedError

    def dummy_decoder_input(self, batch_first=True):
        if batch_first:
            dummy_inp = torch.zeros(self.batch_size, 1, self.output_dim)
        else:
            dummy_inp = torch.zeros(1, self.batch_size, self.output_dim)

        return dummy_inp.to(self.device())

    def dummy_output(self, max_len, batch_first=True):

        if batch_first:
            dummy_out = torch.zeros(self.batch_size, max_len, self.output_dim)
        else:
            dummy_out = torch.zeros(max_len, self.batch_size, self.output_dim)

        return dummy_out.to(self.device())

    def device(self) -> torch.device:
        """Heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        return first_param.device


class RNNDecoder(BaseDecoder):
    def __init__(
        self,
        input_dim,
        output_dim=1,
        n_layers=1,
        hidden_size=100,
        dropout=0.0,
        bidirectional=False,
        use_r_linear=True,  # adds a MLP after RNN predictions
    ):
        super().__init__(
            input_dim,
            output_dim,
            n_layers,
            hidden_size,
            dropout,
            bidirectional,
            use_r_linear,
        )

        self.reconstruct_rnn = nn.LSTM(
            self.input_size,
            self.output_dim if not self.use_r_linear else self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=False,
        )

        if self.bi:
            self.attention = Attention(hidden_size*2)
        else:
            self.attention = Attention(hidden_size)
        
        fin_h_size = self.hidden_size * 2 if self.bi else self.hidden_size
        fin_mid_size = int(fin_h_size / 2)
        self.reconstruct_linear = nn.Sequential(
            nn.Linear(fin_h_size * 2, fin_mid_size),
            nn.ReLU(),
            nn.Linear(fin_mid_size, output_dim),
        )

    def single_step_deocde(self, prev_decode_output, encoder_outputs, h_i, c_i):
        # prev_decoder_output -> [1, batch size, 1]
        # encoder_outputs -> [batch size, max seq len, hidden size]
        # h_i -> [1, batch size, hidden size]
        # attention_weights -> [batch size, 1, max seq len]
        if self.bi:
            h_i = h_i.view(1, self.batch_size, self.hidden_size*2)
            c_i = c_i.view(1, self.batch_size, self.hidden_size*2)

        attention_weights = self.attention(h_i[-1], encoder_outputs)

        # print(attention_weights.shape)
        context = attention_weights.bmm(encoder_outputs)
        # context -> [batch size, 1, hidden]
        context = context.transpose(0, 1)

        rnn_input = torch.cat([prev_decode_output, context], 2)

        if self.bi:
            h_i = h_i.view(2*self.layers, -1, self.hidden_size)
            c_i = c_i.view(2*self.layers, -1, self.hidden_size)
            
        output, (h_n, c_n) = self.reconstruct_rnn(rnn_input, (h_i, c_i))

        if self.use_r_linear:
            output = self.reconstruct_linear(torch.cat([output, context], 2))
        return output, h_n, c_n, attention_weights

    def decode(self, h_n, c_n, lens, encoder_outputs):
        
        max_len = int(max(lens))
        # print(max_len)
        self.batch_size = encoder_outputs.size()[0]
        max_encoder_seq_len = encoder_outputs.size()[1]

        inp = self.dummy_decoder_input(batch_first=False)
        final_reconstructed = self.dummy_output(max_len, batch_first=False)

        all_attn = torch.zeros(max_len, self.batch_size, max_encoder_seq_len).to(self.device())
        for i in range(max_len):
            rnn_output, h_n, c_n, attention_weights = self.single_step_deocde(
                inp, encoder_outputs, h_n, c_n
            )

            final_reconstructed[i] = rnn_output
            inp = rnn_output

            all_attn[i] = attention_weights.squeeze(1)

        return final_reconstructed.permute(1, 0, 2), all_attn



class RNNDecoderNoAttn(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=1,
        n_layers=1,
        hidden_size=100,
        dropout=0.0,
        bidirectional=False,
        use_r_linear=True,  # adds a MLP after RNN predictions
    ):
        super().__init__(
            input_dim,
            output_dim,
            n_layers,
            hidden_size,
            dropout,
            bidirectional,
            use_r_linear,
        )

        self.reconstruct_rnn = nn.LSTM(
            self.input_size,
            self.output_dim if not self.use_r_linear else self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=False,
        )
        fin_h_size = self.hidden_size * 2 if self.bi else self.hidden_size

        fin_mid_size = int(fin_h_size / 2)

        self.reconstruct_linear = nn.Sequential(
            nn.Linear(fin_h_size, fin_mid_size),
            nn.ReLU(),
            nn.Linear(fin_mid_size, output_dim),
        )

    def single_step_deocde(self, prev_decode_output, encoder_outputs, h_i, c_i):
        # prev_decoder_output -> [1, batch size, 1]
        # encoder_outputs -> [batch size, max seq len, hidden size]
        # h_i -> [1, batch size, hidden size]
        output, (h_n, c_n) = self.reconstruct_rnn(prev_decode_output, (h_i, c_i))

        if self.use_r_linear:
            output = self.reconstruct_linear(output)

        return output, h_n, c_n

    def decode(self, h_n, c_n, lens, encoder_outputs):
        max_len = int(max(lens))

        self.batch_size = encoder_outputs.size()[0]

        inp = self.dummy_decoder_input(batch_first=False)
        final_reconstructed = self.dummy_output(max_len, batch_first=False)

        for i in range(max_len):
            rnn_output, h_n, c_n = self.single_step_deocde(
                inp, encoder_outputs, h_n, c_n
            )

            final_reconstructed[i] = rnn_output
            inp = rnn_output

        return final_reconstructed.permute(1, 0, 2)


class Seq2SeqAttn(nn.Module):
    def __init__(self, encoder, decoder, attn: bool = True):
        super(Seq2SeqAttn, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attn = attn

    def forward(self, input_seq, encoder_lens, decoder_lens):
        out, h, c = self.encoder.encode(input_seq, encoder_lens)

        if self.attn:
            # print(decoder_lens)
            output, all_attn = self.decoder.decode(h, c, decoder_lens, out)
            return output, all_attn
        else:
            output = self.decoder.decode(h, c, decoder_lens, out)
            return output



if __name__ == "__main__":

    #testing stuff
    from dataloader import ReverseDataset
    from data_utils import pad_collate
    from torch.utils.data import DataLoader

    k = ReverseDataset()

    k_dataloader = DataLoader(
        k,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=pad_collate,
    )

    e = RNNEncoder(input_dim=1, bidirectional=True)
    d= RNNDecoder(input_dim=(e.input_size + e.hidden_size*2), hidden_size= e.hidden_size, bidirectional=True)

    # e = RNNEncoder(input_dim=1, bidirectional=False)
    # d= RNNDecoder(input_dim=(e.input_size + e.hidden_size), hidden_size= e.hidden_size, bidirectional=False)
    

    for i, (x, y, lens) in enumerate(k_dataloader):
        out, h, c = e.encode(x, lens)
        print("encoder out", out.shape)
        print("encoder h", h.shape)
        print("encoder c", c.shape)

        # attn = a(h, out)

        output = d.decode(h, c, lens, out)

        print("y", y)
        print("output ", output)

        break