import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW
import streamlit as st
import matplotlib.pyplot as plt
import gdown
import os

# python -m streamlit run front_end.py

class TransformerBinaryClassifier(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', hidden_size=768, num_classes=2):
        super(TransformerBinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        probabilities = self.softmax(logits)
        return probabilities
    
def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        print(outputs)
    return outputs[0].numpy()

def plot_probabilities(probabilities):
    labels = ['User Created', 'AI Generated']
    colors = ['lightgreen', 'lightcoral']
    
    fig, ax = plt.subplots()
    bars = ax.bar(labels, probabilities, color=colors)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Probability (%)')
    ax.set_title('Class Probabilities')

    ax.set_ylim([0, 1])
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
    ax.legend(bars, labels, loc='upper right')

    return fig, ax

def download_model_from_drive(drive_link, destination):
    try:# Check if the file already exists
        if not os.path.exists(destination):
            # File doesn't exist, proceed with the download
            gdown.download(drive_link, destination, quiet=False)
        else:
            # File already exists, skip the download
            print(f"File '{destination}' already exists. Skipping download.")
    except:
        print("Error in the file download step - attempting download again")
        gdown.download(drive_link, destination, quiet=False)

def main():
    st.title("Chat Checker with BERT")

    # Sidebar with navigation links (tabs)
    tab = st.sidebar.radio("Navigation", ["Home", "Project Background", "Model Metrics"])

    # Google Drive load
    drive_link = 'https://drive.google.com/uc?id=1Q0wEE-tH_dhuL6WvjVCUtGU6zO6ZsXDk'
    destination = 'BERT_model/trained_models/bert_binary_classifier.pth'

    # Download the model file if it doesn't exist
    download_model_from_drive(drive_link, destination)

    # local load
    model_path = "BERT_model/trained_models/bert_binary_classifier.pth"
    loaded_model = TransformerBinaryClassifier()
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()

    if tab == "Home":
        st.write("This app is designed to detect if a text is AI generated or not. The model running here utilized the pre-trained BERT model as well as some additional data. The maximum number of words and punctuation marks that will be considered is 512.")

        st. write("Try it out by typing in the box below!")
        user_input = st.text_area("Input Text", "")


        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if st.button("Predict"):
            user_input = [user_input]
            probabilities = predict_sentiment(user_input, loaded_model, tokenizer)

            st.write(f"Sentiment Probabilities: {probabilities}")
            prob_not_generated = probabilities[0]
            prob_generated = probabilities[1]

            if prob_generated > prob_not_generated:
                st.write(f"I am {prob_generated * 100:.2f}% sure this was generated")
            else:
                st.write(f"I am {prob_not_generated * 100:.2f}% sure this was NOT generated")
            fig, ax = plot_probabilities(probabilities)
            st.pyplot(fig)

    elif tab == "Project Background":
        st.title("Project Background")
        st.write("The Chat Checker application is developed to address the growing need for identifying AI-generated text in various online platforms. With the proliferation of AI-generated content, distinguishing between human and AI-generated text has become crucial for maintaining trust and authenticity in communication. This project leverages state-of-the-art natural language processing techniques, utilizing the BERT (Bidirectional Encoder Representations from Transformers) model. The model has been trained on a diverse dataset containing both user-created and AI-generated text to effectively classify and differentiate between the two.")
        
        st.subheader("How the model works")
        st.write("The underlying TransformerBinaryClassifier utilizes the Transformer architecture, specifically BERT (Bidirectional Encoder Representations from Transformers). Transformers leverage self-attention mechanisms, enabling the model to capture contextual dependencies across input sequences. In the case of BERT, attention heads attend to both left and right context, facilitating bidirectional understanding. The input text undergoes tokenization, and embeddings are computed for each token. Stacked transformer layers process these embeddings, extracting hierarchical features. The final pooled output captures contextual information, and a fully connected layer with softmax activation yields class probabilities. Training involves minimizing cross-entropy loss with AdamW optimization, ensuring the model's ability to discern intricate patterns in text data for accurate sentiment classification between \"User Created\" and \"AI Generated.\"")

    elif tab == "Model Metrics":
        st.title("Model Metrics")
        st.write("This is where you will find some metrics once a better model has been trained better.")
        


if __name__ == '__main__':
    main()
