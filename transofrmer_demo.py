import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

'''
Demo of a transformer to numeric series

Generate syntetic numeric sequences
Encoder-only transformer predict a target_number
target_number - sum of last and second-last number in sequence
'''

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout, input_dim, max_seq_length):
        super(TransformerEncoder, self).__init__()

        # Define the positional encoding layer
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)

        # Create the Transformer decoder layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_encoder_layers
        )

        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Output layer
        self.output_layer = nn.Linear(d_model*max_seq_length, 1)  # Assuming a single output value

    def forward(self, x):
        # Add positional encoding to the input
        x = self.pos_encoder(x)
        # Apply the input embedding layer
        x = self.input_embedding(x)
        # Apply the Transformer decoder
        x = self.transformer_encoder(x)
        x = x.view(x.size(0),-1)
        # Apply the output layer
        x = self.output_layer(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(0.1)

        # Create positional encodings
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()  # Get batch size and sequence length
        # Expand the positional encoding to match the batch size
        pe = self.pe.unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pe  # Add positional encoding to the input
        return self.dropout(x)




input_dim = 1  # Input dimension
d_model = 1  # Dimension of the model
num_encoder_layers = 6
nhead = 1
dim_feedforward = 512
dropout = 0.1

EPOCH = 1000
DATA_SAMPLES = 10000
BATCH_SIZE = 1000
SEQ_LENGTH = 10

if __name__ =='__main__':
    x = torch.rand(DATA_SAMPLES, SEQ_LENGTH, 1)
    target = torch.zeros(DATA_SAMPLES, 1)
    for i in range(DATA_SAMPLES):
        target[i][0] = x[i][-1][0]+x[i][-2][0]
    x_train, x_test, target_train, target_test = train_test_split(x, target, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(x_train, target_train)
    test_dataset = TensorDataset(x_test, target_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    ten_board_writer = SummaryWriter()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Cuda Device: ',device, torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
    print('Current device: ',device)


    model = TransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout, input_dim, SEQ_LENGTH)
    model = model.to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(EPOCH):
        model.train()
        train_total_loss = 0.0
        test_total_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float()) 
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()
        #ten_board_writer.add_scalar('Loss/Train', train_total_loss / len(train_loader), epoch)
        print(f"Epoch {epoch + 1}, Train_Loss: {train_total_loss / len(train_loader)}")


        model.eval()
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            test_total_loss += loss.item() 
        #ten_board_writer.add_scalar('Loss/Test', test_total_loss / len(test_loader), epoch)
        ten_board_writer.add_scalars(
            'Loss', 
            {
                'Train': train_total_loss / len(train_loader), 
                'Test': test_total_loss / len(test_loader)
                },
            epoch
            )

        print(f"Epoch {epoch + 1}, Test_Loss: {test_total_loss / len(test_loader)}")

