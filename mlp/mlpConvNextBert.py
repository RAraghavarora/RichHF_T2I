from datasets import load_dataset
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torchmetrics.regression import MeanSquaredError, R2Score
from scipy.stats import pearsonr, spearmanr

batch_size = 10
num_epochs = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rhf_dataset_train = load_dataset('RAraghavarora/RichHumanFeedback', split='train')
rhf_dataset_val = load_dataset('RAraghavarora/RichHumanFeedback', split='dev')
rhf_dataset_test = load_dataset('RAraghavarora/RichHumanFeedback', split='test')

convnext_dataset_train = load_dataset('appliedml2024/Vision_ConvNext', split='convnext_train')['features']
convnext_dataset_val = load_dataset('appliedml2024/Vision_ConvNext', split='convnext_dev')['features']
convnext_dataset_test = load_dataset('appliedml2024/Vision_ConvNext', split='convnext_test')['features']

textembed_dataset_train = load_dataset('appliedml2024/text_embedding', split='train')
textembed_dataset_val = load_dataset('appliedml2024/text_embedding', split='dev')
textembed_dataset_test = load_dataset('appliedml2024/text_embedding', split='test')

bert_dataset_train = textembed_dataset_train['BERT_text_embedding']
bert_dataset_val = textembed_dataset_val['BERT_text_embedding']
bert_dataset_test = textembed_dataset_test['BERT_text_embedding']

rhf_artifact_train = rhf_dataset_train['overall_score']
rhf_artifact_val = rhf_dataset_val['overall_score']
rhf_artifact_test = rhf_dataset_test['overall_score']


combined_dataset_train = []
combined_dataset_val = []
combined_dataset_test = []
for i in range(0,len(rhf_artifact_train)):
    temp = convnext_dataset_train[i].copy()
    temp = temp + bert_dataset_train[i].copy()
    combined_dataset_train.append(temp)

    if (i < len(rhf_artifact_val)):
        tempval = convnext_dataset_val[i].copy()
        tempval = tempval + bert_dataset_val[i].copy()
        combined_dataset_val.append(tempval)

    if (i < len(rhf_artifact_test)):   
        temptest = convnext_dataset_test[i].copy()
        temptest = temptest + bert_dataset_test[i].copy()
        combined_dataset_test.append(temptest)


combined_dataset_train = torch.Tensor(combined_dataset_train)
combined_dataset_val = torch.Tensor(combined_dataset_val)
combined_dataset_test = torch.Tensor(combined_dataset_test)

rhf_artifact_train = torch.Tensor(rhf_artifact_train)
rhf_artifact_val = torch.Tensor(rhf_artifact_val)
rhf_artifact_test = torch.Tensor(rhf_artifact_test)

print(rhf_artifact_train[0])
train_dataset = TensorDataset(combined_dataset_train, rhf_artifact_train)
val_dataset = TensorDataset(combined_dataset_val, rhf_artifact_val)
test_dataset = TensorDataset(combined_dataset_test, rhf_artifact_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2560,1706) # TODO sweep hidden layer num
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1706,1)
        ### END CODE ###

    def forward(self, inp):
        x = self.fc1(inp)
        x = self.relu(x)
        x = self.fc2(x)

        return x


model = MLP().to(device)

criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


print("starting training")
# Training the Model
for epoch in range(0,num_epochs):
    for i, (vectors, scores) in enumerate(train_loader):
        vectors = Variable(vectors).to(device)
        scores = Variable(scores).to(device)
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(vectors)
        loss = criterion(outputs, scores)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1,
                       len(train_dataset) // batch_size, loss.data.item()))


    model.eval()
    y_list = torch.Tensor()
    y_pred_list = torch.Tensor()
    for i, (vectors, scores) in enumerate(test_loader):
        vectors = Variable(vectors).to(device)
        scores = scores.to(device)
        y_pred = model(vectors)
        scores = torch.Tensor.cpu(scores)
        y_pred = torch.Tensor.cpu(y_pred)
        if i == 0:
            y_list = scores
            y_pred_list = y_pred
        else:
            y_list = torch.cat((y_list, scores), dim=0)
            y_pred_list = torch.cat((y_pred_list, y_pred), dim=0)

    y_pred_list = y_pred_list.squeeze()

    mean_squared_error = MeanSquaredError()
    eval_loss = mean_squared_error(y_pred_list, y_list)
    r2score = R2Score()
    eval_r2 = r2score(y_pred_list, y_list)
    y_pred_list = y_pred_list.detach().numpy()
    y_list = y_list.numpy()
    pearson = pearsonr(y_pred_list, y_list)
    spearman = spearmanr(y_pred_list, y_list)
    print('Loss: %.4f, R2: %.4f, Pearson: %s, Spearman: %s'
                    % (eval_loss, eval_r2, str(pearson), str(spearman)))
