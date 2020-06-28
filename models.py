import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import functional as tf

#################################################################################
#                                                                               #
#################################################################################

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

#################################################################################
# LSTM Model
#################################################################################
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,
                 dropout=0.0, bidirectional=False, truncated=False):
        super(LSTMModel, self).__init__()
        '''
        input_size – The number of expected features in the input x
        hidden_size – The number of features in the hidden state h
        num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
        bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
        dropout – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
        bidirectional – If True, becomes a bidirectional LSTM. Default: False
        '''
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # Building LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=bidirectional,
                            bias=True)

        # Readout layer
        self.fc = nn.Linear(self.num_directions * hidden_dim, output_dim, bias=True)
        self.relu = nn.ReLU()


        self.truncated = truncated

    def forward(self, x, x_lens=''):
        # x of shape
        # (batch_dim, seq_dim, input_dim)

        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_dim).requires_grad_()).to(device)

        # Initialize cell state with zeros
        c0 = Variable(torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_dim).requires_grad_()).to(device)

        # x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        if self.truncated:
        # # We need to detach as we are doing truncated backpropagation through time (BPTT)
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # # If we don't, we'll backprop all the way to the start even after going through another batch
        else:
            out, (hn, cn) = self.lstm(x, (h0, c0))

        # out, _ = pad_packed_sequence(out, batch_first=True)
        # out has shape (batch_dim, seq_dim, num_directions * hidden_dim)

        out = self.relu(out)
        out = self.fc(out[:, -1, :].squeeze())

        # out has shape (batch_dim, output_dim)
        return out

#################################################################################
#                                                                               #
#################################################################################

#################################################################################
# Encoder with Input Attention and Decoder with Temporal Attention
#################################################################################
class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, seq_len: int, dropout=0.4):
    	"""
    	

    	Parameters
    	----------
    	input_size : int
    		Input features size.
    	hidden_size : int
    		Hidden state size.
    	seq_len : int
    		Sequence length.
    	dropout : float, optional
    		Dropout probability 0<p<1. The default is 0.4.
			

    	"""
		
        super(Encoder, self).__init__()
		

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + seq_len , out_features=1)
        # self.attn_layer = nn.Sequential(
        #     nn.Linear(in_features=2 * hidden_size + seq_len, out_features=hidden_size),
        #     nn.Tanh(),
        #     nn.Linear(in_features=hidden_size, out_features=1)
        #     )

    def init_hidden(self, x, hidden_size: int):
    	"""
    	Initialise hidden state.

    	Parameters
    	----------
    	x : torch.tensor
    		Input of shape (batch_size, seq_len, input_size).
    	hidden_size : int
    		Hidden state size.

    	Returns
    	-------
    	h0 or c0: torch.tensor 
    		Hidden state of shape (1, batch_size, hidden_size).

    	"""
		
        return Variable(torch.zeros(1, x.size(0), hidden_size)).to(device)


    def forward(self, input_data):
    	"""
    	Performs a forward pass.

    	Parameters
    	----------
    	input_data : torch.tensor
    		Input of shape (batch_size, seq_len, input_size).

    	Returns
    	-------
    	input_weighted : torch.tensor
    		Weighted input of same shape as input_data.
    	input_encoded : torch.tensor
    		Encoded input of shape (batch_size, seq_len, hidden_size).

    	"""
		
        # input_data: (batch_size, seq_len, input_size)
        input_weighted = Variable(torch.zeros(input_data.size(0),
                                self.seq_len, self.input_size)).to(device)
        input_encoded = Variable(torch.zeros(input_data.size(0),
                                self.seq_len, self.hidden_size)).to(device)
        hidden = self.init_hidden(input_data, self.hidden_size)  # 1 * batch_size * hidden_size
        cell = self.init_hidden(input_data, self.hidden_size)

        for t in range(self.seq_len):
            # Eqn. 8: concatenate the hidden states with each predictor
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim=2)  # (batch_size, input_size, 2 * hidden_size)
            # Eqn. 8: Get attention weights
            x = x.view(-1, self.hidden_size * 2 + self.seq_len) # (2 * hidden_size, batch_size * input_size)
            x = self.attn_linear(x)  # (batch_size * input_size, 1)

            # Eqn. 9: Softmax the attention weights
            x = x.view(-1, self.input_size) # (batch_size, input_size)
            attn_weights = tf.softmax(x, dim=1) # (batch_size, input_size)

            # Eqn. 10: LSTM
            weighted_input = attn_weights * input_data[:, t, :] # (batch_size, input_size)

            _, (hidden, cell) = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            # (1, batch_size, hidden_size)

            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return input_weighted, input_encoded

class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int,
                 seq_len: int, out_feats=1, dropout=0.4):
    	"""
    	

    	Parameters
    	----------
    	encoder_hidden_size : int
    		Hidden state size of encoder.
    	decoder_hidden_size : int
    		Hidden state size.
    	seq_len : int
    		Sequence length.
    	out_feats : int, optional
    		Output size. The default is 1.
    	dropout : float, optional
    		Dropout probability 0<p<1. The default is 0.4.


    	"""
		
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(encoder_hidden_size, 1)
            )

        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()

    def init_hidden(self, x, hidden_size: int):
		"""
    	Initialise hidden states.

    	Parameters
    	----------
    	x : torch.tensor
    		Input of shape (batch_size, seq_len, x.size(2)).
    	hidden_size : int
    		Hidden state size.

    	Returns
    	-------
    	h0 or c0: torch.tensor 
    		Hidden state of shape (1, batch_size, decoder_hidden_size).

    	"""
        return Variable(torch.zeros(1, x.size(0), hidden_size)).to(device)

    def forward(self, input_encoded, y_history):
    	"""
    	Performs a forward pass.

    	Parameters
    	----------
    	input_encoded : torch.tensor
    		Encoded input from Encoder of shape (batch_size, seq_len, encoder_hidden_size).
    	y_history : torch.tensor
    		Target history of shape (batch_size, seq_len).

    	Returns
    	-------
    	y: torch.tensor
    		Target prediction of shape (batch_size, output_size).

    	"""
		

        # input_encoded: (batch_size, seq_len, encoder_hidden_size)
        # y_history: (batch_size, seq_len)

        # Initialize hidden and cell, (1, batch_size, decoder_hidden_size)
        hidden = self.init_hidden(input_encoded, self.decoder_hidden_size)
        cell = self.init_hidden(input_encoded, self.decoder_hidden_size)
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size)).to(device)

        for t in range(self.seq_len):
            # (batch_size, seq_len, 2 * decoder_hidden_size + encoder_hidden_size)
            x = torch.cat((hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2) #.to(device)

            # Eqn. 12 & 13: softmax on the computed attention weights
            x = x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)

            x = self.attn_layer(x).view(-1, self.seq_len)

            x = tf.softmax(x, dim=1).to(device)  # (batch_size, seq_len)

            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # (batch_size, encoder_hidden_size)

            # Eqn. 15
            y_tilde = self.fc(torch.cat((context, y_history[:, t].reshape(y_history.shape[0],1)), dim=1))  # (batch_size, out_size)
            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters()
            _, (hidden, cell) = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            # hidden  # 1 * batch_size * decoder_hidden_size
            # cell   # 1 * batch_size * decoder_hidden_size

        # Eqn. 22: final output
        return self.fc_final(torch.cat((hidden[0], context), dim=1))

class AttnEncDecLSTM(nn.Module):

    def __init__(self, input_size, output_size, encoder_hidden_size,
                decoder_hidden_size, seq_len, dropout=0.4):
    	"""
    	

    	Parameters
    	----------
		input_size : int
    		Input features size.
		output_size : int, optional
    		Output size.
		encoder_hidden_size : int
    		Hidden state size of encoder.
    	decoder_hidden_size : int
    		Hidden state size.
    	seq_len : int
    		Sequence length.
 	 	dropout : float, optional
    		Dropout probability 0<p<1. The default is 0.4.


    	"""
		
        super(AttnEncDecLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.seq_len = seq_len
        self.encoder = Encoder(input_size=input_size,
                               hidden_size=encoder_hidden_size,
                               seq_len=seq_len,
                               dropout=dropout).to(device)
        self.decoder = Decoder(encoder_hidden_size=encoder_hidden_size,
                            decoder_hidden_size=decoder_hidden_size,
                            seq_len=seq_len,
                            out_feats=output_size,
                            dropout=dropout).to(device)

    def forward(self, x, y_history):
    	"""
    	Performs a forward pass.

    	Parameters
    	----------
    	x : torch.tensor
    		Input of shape (batch_size, seq_len, input_size).
    	y_history : torch.tensor
    		Target history of shape (batch_size, seq_len).

    	Returns
    	-------
    	out : torch.tensor
    		Target prediction of shape (batch_size, output_size).

    	"""
		
        input_weighted, input_encoded = self.encoder(x)
        out = self.decoder(input_encoded, y_history)
        return out
#################################################################################
#                                                                               #
#################################################################################

#################################################################################
# Simple Encoder-Decoder Model
#################################################################################
class SimpleEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.5, num_layers = 1, bidirectional = False):
    	"""
    	

    	Parameters
    	----------
		input_size : int
    		Input features size.
		hidden_size : int
    		Hidden state size.	
		dropout : Tfloat, optional
    		Dropout probability 0<p<1. The default is 0.5.
    	num_layers : int, optional
    		Number of LSTM stacked layers. The default is 1.
    	bidirectional : bool, optional
    		Implements bidirectional LSTM. The default is False.



    	"""
        
		super(SimpleEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.enc_lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, bidirectional=bidirectional,
                            batch_first=True, dropout=dropout)



    def _initiate_hidden(self, x):
    	"""
    	Initialise hidden states.

    	Parameters
    	----------
    	x : torch.tensor
    		Input of shape (batch_size, seq_len, input_size).
 

    	Returns
    	-------
    	h0: torch.tensor 
    		Hidden state of shape (num_directions * num_layers, batch_size, decoder_hidden_size).
    	c0 : torch.tensor
    		Cell state of shape (num_directions * num_layers, batch_size, decoder_hidden_size).

    	"""
		

        # h0, c0 of shape (num_directions)
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size).requires_grad_())
        h0 = h0.to(device)

        # Initialize cell state with zeros
        c0 = Variable(torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size).requires_grad_())
        c0 = c0.to(device)

        return h0, c0

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0, c0 = self._initiate_hidden(x)
        # Pass through the lstm net
        out, (hn, cn) = self.enc_lstm(x, (h0, c0))



        # if self.bidirectional:

        #     out = out.view(x.shape[0], x.shape[1], self.num_directions, self.hidden_size).sum(2).squeeze()
        return (hn, cn)

class SimpleDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=1, dropout=0.5,bidirectional=False):
    	"""
    	

    	Parameters
    	----------
		input_size : int
    		Input features size.
		output_size : int
    		Output size.
		hidden_size : int
    		Hidden state size.	
		dropout : Tfloat, optional
    		Dropout probability 0<p<1. The default is 0.5.
    	num_layers : int, optional
    		Number of LSTM stacked layers. The default is 1.
    	bidirectional : bool, optional
    		Implements bidirectional LSTM. The default is False.
    	


    	"""
        
		super(SimpleDecoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.dec_lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                dropout=dropout,bidirectional=self.bidirectional,
								batch_first=True)

        self.fc_out = nn.Linear(hidden_size * self.num_directions, output_size)


    def forward(self, y_history, hidden, cell):
    	"""
    	

    	Parameters
    	----------
		y_history : torch.tensor
    		Target history of shape (batch_size, seq_len).
    	hidden : torch.tensor
    		Hidden state d0 from encoder of shape
			(num_directions * num_layers, batch_size, decoder_hidden_size).
    	cell : torch.tensor
    		Cell state c0 from encoder of shape
			(num_directions * num_layers, batch_size, decoder_hidden_size).

    	Returns
    	-------
    	prediction : torch.tensor
    		Target prediction of shape.
    	hidden : torch.tensor
    		Hidden states of last layer.
    	cell : torch.tensor
    		Cell states of last layer.

    	"""
		

        output, (hidden, cell) = self.dec_lstm (y_history, (hidden, cell))
        # output = [seq_len, batch_size, hid_dim * n_dir]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]

        # seq_len and n_dir will always be 1 in the decoder
        prediction = self.fc_out(output[:,-1,:])
        # prediction = [batch_size, output_dim]
        return prediction, hidden, cell

class SimpleEncDec(nn.Module):

    def __init__(self,  input_size, output_size, hidden_size, dropout=0.5,				  
				 num_layers = 1, bidirectional = False):
    	"""
    	Simple Encoder-Decoder LSTM. Last hidden and cell states of encoder,
		are input to decoder.

    	Parameters
    	----------
		input_size : int
    		Input features size.
		output_size : int
    		Output size.
		hidden_size : int
    		Hidden state size of encoder and decoder.	
    	
    	dropout : Tfloat, optional
    		Dropout probability 0<p<1. The default is 0.5.
    	num_layers : int, optional
    		Number of LSTM stacked layers. The default is 1.
    	bidirectional : bool, optional
    		Implements bidirectional LSTM. The default is False.


    	"""
		
        
		super(SimpleEncDec, self).__init__()
        self.output_size = output_size
        self.encoder = SimpleEncoder(input_size[0], hidden_size, dropout,
									  num_layers, bidirectional)
        self.decoder = SimpleDecoder(input_size[1], output_size, hidden_size,
									   num_layers, dropout, bidirectional)

    def forward(self, x, y_history):
    	"""
    	

    	Parameters
    	----------
    	x : torch.tensor
    		Input of shape (batch_size, seq_len, input_size).
    	y_history : torch.tensor
    		Targe history of shape (batch_size, seq_len, y_history.size(2)).

    	Returns
    	-------
    	out : torch.tensor
    		Target prediction of shape (batch_size, output_size).

    	"""
		
        h, c = self.encoder(x)

        out = self.decoder(y,h,c)[0]

        return out
#################################################################################
#                                                                               #
#################################################################################
