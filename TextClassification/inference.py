import torch
import random
import pandas as pd
from torch import nn
from transformers import BertModel
from transformers import BertTokenizer

# Define the labels
LABELS = {'business': 0,
          'entertainment': 1,
          'sport': 2,
          'tech': 3,
          'politics': 4}

labels_list = list(LABELS)

# Define the BERT classifier
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.out = nn.Linear(768, len(LABELS))

    def forward(self, input_id, mask):
        _, o2 = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        out = self.out(o2)
        return out

def predict(model, tokenizer, text, actual):
    print()
    text_dict = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
    mask = text_dict['attention_mask']
    input_id = text_dict['input_ids']

    # Classify the image and extract the predictions
    logits = model(input_id, mask)
    probabilities = nn.Softmax(dim=-1)(logits)
    sortedProbabilities = torch.argsort(probabilities, dim=-1, descending=True)

    # Get the top prediction and associated probability
    (label, prob) = (labels_list[probabilities.argmax().item()], probabilities.max().item())

    # Display the text, actual label, and predicted label
    print('Text: ' + text)
    print('Actual: ' + actual)
    print('Predict: {}, {:.2f}%'.format(label, prob * 100))

    # Loop over the predictions and display the top-5 predictions and
    # corresponding probabilities to the terminal
    for (i, idx) in enumerate(sortedProbabilities[0, :5]):
        print("{}. {}: {:.2f}%".format(i + 1, labels_list[idx.item()],
                                       probabilities[0, idx.item()] * 100))

# Load the BERT tokenizer
print('[INFO] Loading the BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Load the model and set it to evaluation mode
print('[INFO] Loading the model...')
model = torch.load('bbc_text_model1/BBC-Text_model.pt', map_location=torch.device('cpu'))
model.eval()

# Get the test data (random sample)
p = 0.03
df = pd.read_csv('BBC-Text/test.csv', header=0,
                 skiprows=lambda i: i>0 and random.random() > p)
sample = [(x, y) for x, y in zip(df['category'], df['text'])]

# Start classifying the test texts
print('[INFO] Classifying {} texts...'.format(len(sample)))
for actual, text in sample:
    predict(model, tokenizer, text, actual)