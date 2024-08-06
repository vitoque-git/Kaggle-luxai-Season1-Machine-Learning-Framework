
# luxai-2021-ML 

This repository contains the code used to train a machine learning model for the Kaggle Lux AI Season 1 competition. The aim of the competition is to develop agents that can efficiently manage resources and strategize in a simulated environment.
A runnable notebook can be found here:
https://www.kaggle.com/code/vitoque/lux-ai-with-il-decreasing-learning-rate/notebook


## Imitation Learning 

The AI model in this project uses a form of imitation learning, where the model is trained to mimic the behavior of an expert or a set of expert demonstrations. In the context of the Lux AI competition, imitation learning involves training the model on a dataset of game episodes where actions taken by the agents are recorded.

### Dataset Collection 

The dataset consists of observations and corresponding actions from game episodes. These observations include various game states such as unit positions, resources, city tiles, and more. The actions represent the decisions made by the agents in those states.

### Training Process 

The model is trained using supervised learning techniques, where the input is the game state, and the output is the action taken by the expert agent. The loss function measures the difference between the predicted actions by the model and the actual actions taken by the expert. The model parameters are updated to minimize this loss, effectively making the model imitate the expert's behavior.


```python
# Example of action prediction
with torch.set_grad_enabled(phase == 'train'):
    policy = model(states)
    loss = criterion(policy, actions)
    _, preds = torch.max(policy, 1)
```

### Advantages of Imitation Learning 
 
- **Efficiency** : Imitation learning can leverage a large amount of pre-recorded data, making it efficient to train compared to reinforcement learning, which requires extensive interaction with the environment.
 
- **Simplicity** : The approach simplifies the training process as it directly learns from the expert's decisions without the need for complex reward signals.

This type of imitation learning helps in quickly developing a competitive agent by learning from the strategies and actions of expert players, enabling the model to perform well in the Lux AI competition environment.
## Training Methodology 

The training methodology for the Lux AI competition involves several key steps, including data preparation, neural network architecture design, and the training process itself. Below is a detailed explanation of these steps:

### Input for Neural Network 
The `make_input` function prepares the input data for the neural network. It processes the game state observations and converts them into a format suitable for the model. The input includes information about units, city tiles, resources, research points, and more. The data is normalized and arranged into a tensor of shape `(20, 32, 32)`.

```python
def make_input(obs, unit_id):
    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    
    b = np.zeros((20, 32, 32), dtype=np.float32)
    
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        
        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            if unit_id == strs[3]:
                # Position and Cargo
                b[:2, x, y] = (
                    1,
                    (wood + coal + uranium) / 100
                )
            else:
                # Units
                team = int(strs[2])
                cooldown = float(strs[6])
                idx = 2 + (team - obs['player']) % 2 * 3
                b[idx:idx + 3, x, y] = (
                    1,
                    cooldown / 6,
                    (wood + coal + uranium) / 100
                )
        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            idx = 8 + (team - obs['player']) % 2 * 2
            b[idx:idx + 2, x, y] = (
                1,
                cities[city_id]
            )
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 12, 'coal': 13, 'uranium': 14}[r_type], x, y] = amt / 800
        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            b[15 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10
    
    # Day/Night Cycle
    b[17, :] = obs['step'] % 40 / 40
    # Turns
    b[18, :] = obs['step'] / 360
    # Map Size
    b[19, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

    return b
```

### Dataset Preparation 
The `LuxDataset` class handles the dataset preparation, converting observations and samples into a format suitable for PyTorch data loading.

```python
class LuxDataset(Dataset):
    def __init__(self, obses, samples):
        self.obses = obses
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_id, unit_id, action = self.samples[idx]
        obs = self.obses[obs_id]
        state = make_input(obs, unit_id)
        
        return state, action
```

### Neural Network Architecture 

The neural network architecture consists of convolutional layers designed to process the input tensors and predict actions. The model is implemented using PyTorch.


```python
class BasicConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim, 
            kernel_size=kernel_size, 
            padding=(kernel_size[0] // 2, kernel_size[1] // 2)
        )
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h

class LuxNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        self.conv0 = BasicConv2d(20, filters, (3, 3), True)
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 5, bias=False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        p = self.head_p(h_head)
        return p
```

### Training the Model 
The `train_model` function handles the training process. It includes training and validation phases, loss calculation, and model saving based on performance.

```python
def train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.cuda()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            epoch_loss = 0.0
            epoch_acc = 0
            
            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                states = item[0].cuda().float()
                actions = item[1].cuda().long()

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    policy = model(states)
                    loss = criterion(policy, actions)
                    _, preds = torch.max(policy, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(policy)
                    epoch_acc += torch.sum(preds == actions.data)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / data_size

            print(f'LR: {optimizer.param_groups[0]["lr"]} Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
        
        if epoch_acc > best_acc:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, 20, 32, 32))
            traced.save('model.pth')
            print(f'Saved model.pth from epoch {epoch + 1} as it is the most accurate so far: Acc: {epoch_acc:.4f}')
            best_acc = epoch_acc
        
        traced = torch.jit.trace(model.cpu(), torch.rand(1, 20, 32, 32))
        traced.save(f'model_{epoch + 1}.pth')
        
        scheduler.step(epoch_loss)
```

The model is then trained using the provided training and validation data loaders, loss function, optimizer, and scheduler.


```python
# Initialize the model
model = LuxNet()

# Load pre-trained model if available
# model = torch.jit.load('../input/models/model8050.pth')

# Split the data into training and validation sets
train, val = train_test_split(samples, test_size=0.1, random_state=42, stratify=labels)
batch_size = 64

# Create data loaders
train_loader = DataLoader(
    LuxDataset(obses, train), 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=2
)
val_loader = DataLoader(
    LuxDataset(obses, val), 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=2
)
dataloaders_dict = {"train": train_loader, "val": val_loader}

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.05, patience=2)

# Train the model
train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs=15)
```

This methodology ensures that the model is trained effectively, utilizing a combination of convolutional neural networks, data preprocessing, and appropriate training strategies.


## Project Structure 

The repository contains the following files:
 
- `Learn.py`: Script for training the model.
 
- `TestTorch.py`: Script for testing the model.
 
- `get_episodes.py`: Script for extracting episodes from game simulations.
 
- `get_episodes_rl.py`: Script for extracting episodes using reinforcement learning strategies.
 
- `get_episodes_tb.py`: Script for extracting episodes using TensorBoard.
 
- `sort_episodes.py`: Script for sorting the extracted episodes.
 
- `split.py`: Script for splitting the dataset into training and validation sets.
 
- `README.md`: This file.

## Installation 

To set up the environment, ensure you have Python installed and run:


```sh
pip install -r requirements.txt
```
Ensure you have all necessary dependencies listed in the `requirements.txt` file. If the `requirements.txt` file is not provided, manually install the packages used in the scripts (e.g., PyTorch, NumPy, etc.).
## Usage 

### Training the Model 
To train the model, use the `Learn.py` script. You can configure the training parameters directly in the script or by passing arguments (if implemented).

```sh
python Learn.py
```

### Testing the Model 
To test the trained model, use the `TestTorch.py` script.

```sh
python TestTorch.py
```

### Data Preparation 
 
1. **Extract Episodes** : Use the `get_episodes.py`, `get_episodes_rl.py`, or `get_episodes_tb.py` scripts to extract episodes from game simulations.

```sh
python get_episodes.py
```
 
2. **Sort Episodes** : Sort the extracted episodes for better organization.

```sh
python sort_episodes.py
```
 
3. **Split Dataset** : Split the dataset into training and validation sets using the `split.py` script.

```sh
python split.py
```

## Results 

Various models were trained and evaluated. Below are some results for different configurations:
 
- **Model 1 (Base Model)** 
  - Loss: 0.5305 | Accuracy: 0.7862
 
- **Model 2 (Improved Model)** 
  - Loss: 0.4596 | Accuracy: 0.8156

Further results and comparisons are detailed in the training logs.

## Contributing 

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License 

This project is licensed under the MIT License. See the LICENSE file for details.
