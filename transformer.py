import torch
import torch.nn as nn
import os
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import time
import shutil
from sklearn.model_selection import StratifiedShuffleSplit

'''
Transformer for binary classification of flower voltage activity affected/not affected by motions around
'''


OSCILLO_FOLDER = 'dataset/2023-11-29'
BATCH_SIZE = 200
CHECKPOINT_INTERVAL = 100
SAVE_WEIGHTS = 'transformer.pt'
PATCH_LENGHT = 10
PATCH_STRIDE = 5
LR = 10e-6

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, input_dim, max_seq_length):
        super(TransformerEncoder, self).__init__()
        
        # Define the positional encoding layer
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Create the Transformer encoder layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_encoder_layers
        )
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Output layer
        self.output_layer = nn.Linear(d_model*max_seq_length, 1)  
    
    def forward(self, x):
        # Add positional encoding to the input
        x = self.pos_encoder(x)
        # Apply the input embedding layer
        x = self.input_embedding(x.to(self.input_embedding.weight.dtype))
        # Apply the Transformer encoder
        x = self.transformer_encoder(x)
        # Apply the output layer
        x = x.view(x.size(0),-1)
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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

def batch_size_sliding_window(tensor, window_size, stride):
    # Number of patches
    if (tensor.size(1)-window_size) % (window_size-stride) != 0:
        print('Not sliceble for equal parts')
    else:
        num_patches = 1 + (tensor.size(1) - window_size) // stride
    
    # Initialize the result tensor
    result = torch.zeros((tensor.size(0), num_patches, window_size), dtype=tensor.dtype)
    for i in range(num_patches):
        result[:, i] = tensor[:, i * stride:i * stride + window_size]
    return result




if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Cuda Device: ',device, torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
    print('Current device: ',device)




    ten_board_writer = SummaryWriter()
    current_directory = os.getcwd()
    load_path = os.path.join(current_directory, OSCILLO_FOLDER, 'labeled_data.pth')
    load_data = torch.load(load_path)
    x_tensor = load_data['labeled_voltage_v1'].unsqueeze(2)
    label_tensor = load_data['labels']
    unique_classes, counts = torch.unique(label_tensor, return_counts=True)
    print('samples by class:',counts)
    # Identify indices of each class
    class_0_indices = (label_tensor == 0).nonzero(as_tuple=True)[0]
    class_1_indices = (label_tensor == 1).nonzero(as_tuple=True)[0]
    min_class_size = min(len(class_0_indices), len(class_1_indices))
    #Undersampling
    balanced_class_0_indices = class_0_indices[torch.randperm(len(class_0_indices))[:min_class_size]]
    balanced_class_1_indices = class_1_indices[torch.randperm(len(class_1_indices))[:min_class_size]]
    # Combine indices
    balanced_indices = torch.cat((balanced_class_0_indices, balanced_class_1_indices))
    x_tensor = x_tensor[balanced_indices]
    label_tensor = label_tensor[balanced_indices]

    unique_classes, counts = torch.unique(label_tensor, return_counts=True)
    total_samples = label_tensor.size(0)
    weights = total_samples / (2 * counts.float())
    weights = weights / weights.sum()
    print('samples by class undersampling:',counts)
    print('class weights',weights)
    x_tensor = batch_size_sliding_window(x_tensor.squeeze(-1), PATCH_LENGHT, PATCH_STRIDE)
    
   


    # Stratified Split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss.split(x_tensor, label_tensor):
        x_train, x_test = x_tensor[train_index], x_tensor[test_index]
    label_train, label_test = label_tensor[train_index], label_tensor[test_index]



    train_dataset = TensorDataset(x_train, label_train)
    test_dataset = TensorDataset(x_test, label_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print('Train samples:', label_train.shape, 'Test samples:', label_test.shape)


    model = TransformerEncoder(d_model=10, nhead=2, num_encoder_layers=10, dim_feedforward=128, dropout=0.1, input_dim=10, max_seq_length=39)
    model = model.to(device)
    model.train()
    class_weights = torch.tensor([weights[0], weights[1]], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1]/class_weights[0])  # Binary weighted Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=LR)


    if os.path.isfile('transformer.pt'):
        model.load_state_dict(torch.load('transformer.pt'))
        print('weights loaded from file.\n\n')

    epoch=0
    test_samples_number = len(test_dataset)
    while True:
        start_time = time.time()
        epoch +=1
        model.train()
        train_total_loss = 0.0
        test_total_loss = 0.0

        for inputs, labels in train_loader:
            labels = labels.to(torch.float32)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1)) 
            loss.backward()
            train_total_loss += loss.item()
            optimizer.step()
        print(f"Epoch {epoch}, Train_Loss: {train_total_loss / len(train_loader)}")

        model.eval()
        correct_predictions_total = 0
        true_positives_total = 0
        actual_positives_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                labels = labels.to(torch.float32)
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1))
                test_total_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct_predictions = (predictions == labels.view(-1, 1)).float().sum()
                correct_predictions_total +=correct_predictions
                true_positives = ((predictions == 1) & (labels.view(-1, 1) == 1)).float().sum()
                true_positives_total += true_positives
                actual_positives = (labels.view(-1, 1) == 1).float().sum()
                actual_positives_total += actual_positives                 
            print(f"Epoch {epoch}, Test_Loss: {test_total_loss / len(test_loader)}")
            accuracy = correct_predictions_total / test_samples_number
            recall = true_positives_total / actual_positives_total if actual_positives_total > 0 else torch.tensor(0.0)
            print(f"Epoch {epoch}, Test Accuracy: {accuracy.item()}, Test Recall: {recall.item()}")

       
        ten_board_writer.add_scalars(
            'Loss', 
            {
                'Train': train_total_loss / len(train_loader), 
                'Test': test_total_loss / len(test_loader)
                },
            epoch
            )
        #print(f"Epoch {epoch + 1}, Test_Loss: {test_total_loss / len(test_loader)}")
        epoch_time = time.time()-start_time
        print('Epoch train time: ',epoch_time, '\n\n')

        if epoch % CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), 'temp_'+ SAVE_WEIGHTS)
            shutil.move('temp_'+ SAVE_WEIGHTS, SAVE_WEIGHTS)
            print('weights saved')

