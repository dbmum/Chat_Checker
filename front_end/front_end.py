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
    labels = ['AI Generated','Written by a Real Person']
    colors = ['lightcoral','lightgreen']
    
    fig, ax = plt.subplots()
    bars = ax.bar(labels, probabilities, color=colors)
    
    ax.set_xlabel('')
    ax.set_ylabel('Probability (%)')
    ax.set_title('Probability of Your Text Being Generated by AI')

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

def apply_theme(theme_choice):
    dark = '''
            <style>
                .stApp {
                    background-color: black;
                    color: white; /* Text color for dark mode */
                }

                .stApp * {
                    color: white !important; /* Text color for dark mode */
                }
            </style>
            '''

    light = '''
            <style>
                .stApp {
                    background-color: white;
                    color: black; /* Text color for light mode */
                }

                .stApp * {
                    color: black !important; /* Text color for light mode */
                }
            </style>
            '''

    contrast = '''
            <style>
                .stApp {
                    background-color: #FFD700;
                    color: black; /* Text color for contrast mode */
                }

                .stApp * {
                    color: black !important; /* Text color for contrast mode */
                }
            </style>
            '''

    # Apply the selected theme
    if theme_choice == "Dark":
        st.markdown(dark, unsafe_allow_html=True)
    elif theme_choice == "Light":
        st.markdown(light, unsafe_allow_html=True)
    elif theme_choice == "Constrast":
        st.markdown(contrast, unsafe_allow_html=True)

def load_model():
    """
    load the pretrained weights and return a tokenizer as well.
    This is meant to only be run once per session

    The model provided as well as the link can be replaced with your own versions of these models.
    """
    # Google Drive load
    drive_link = 'https://drive.google.com/file/d/1-0-A0fLH04REvW6wo6ezS6gM-oV-18rM/view?usp=sharing'
    destination = 'BERT_model\995-epoch-0.pth'

    # Download the model file if it doesn't exist
    download_model_from_drive(drive_link, destination)

    # local load
    loaded_model = TransformerBinaryClassifier()
    loaded_model.load_state_dict(torch.load(destination))
    loaded_model.eval()

    return loaded_model

def main():
    st.title("Chat Checker")

    # Sidebar with navigation links (tabs)
    tab = st.sidebar.radio("Navigation", ["Home", "Project Background", "Model Metrics"])

    if "loaded_model" not in st.session_state:
        st.session_state.loaded_model = load_model()


    # Dropdown menu to choose theme
    theme_choice = st.selectbox("Choose a theme", ("Light", "Dark", "Constrast"))

    # Apply the selected theme
    apply_theme(theme_choice)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if tab == "Home":
        st.write("This app is designed to detect if a text is AI generated or not. The model running here utilized the pre-trained BERT model as well as some additional data. The maximum number of words and punctuation marks that will be considered is 512.")

        st. write("Try it out by typing in the box below!")
        user_input = st.text_area("Input Text", "")

        if st.button("Predict"):
            user_input = [user_input]
            probabilities = predict_sentiment(user_input, st.session_state.loaded_model, tokenizer)

            st.write(f"Sentiment Probabilities: {probabilities}")
            prob_not_generated = probabilities[1]
            prob_generated = probabilities[0]

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
        st.write("The underlying model utilizes the Transformer architecture, specifically BERT (Bidirectional Encoder Representations from Transformers). Transformers leverage self-attention mechanisms, enabling the model to capture contextual dependencies across input sequences. In the case of BERT, attention heads attend to both left and right context, facilitating bidirectional understanding. The input text undergoes tokenization, and embeddings are computed for each token. Stacked transformer layers process these embeddings, extracting hierarchical features. The final pooled output captures contextual information, and a fully connected layer with softmax activation yields class probabilities. Training involves minimizing cross-entropy loss with AdamW optimization, ensuring the model's ability to discern intricate patterns in text data for accurate sentiment classification between \"User Created\" and \"AI Generated.\"")

    elif tab == "Model Metrics":
        st.title("Model Metrics")
        st.write("The model running was trained specifically on real and generated news articles.")
        st.write('This weights used in this model got an accuracy score of 99.5% on the test data set that was seperated from the data that the model was trained on.')
        st.write('I chose accuracy as a good metric for this model for several reasons. First, I wanted to prioritize mostly that the model was correct. I trained on a balanced dataset that had the same number of each type of article in it. Secondly, I did not want to skew away from false positives or false negatives specifically. This application is a tool that should be used in your search for truth, but not make up your mind completely about a given source. Lastly, accuracy is a kind of metric that makes sense to people. I think that too many machine learning models aim to confuse or obfuscate their reliability through confusing metrics that do not have clear meaning.')
    st.subheader('Chat Checker Considerations') 
    st.write("Even though this tool will NOT tell you if a given text is factually correct, I think it is important that people can know who the real author is. This tool is trained mostly on news data and may not be able to accuratly predict on other types of text.")
    st.write('This model was trained on generated articles by Chat GPT 3.5 Turbo, and may not detect text from other more modern models.')

if __name__ == '__main__':
    main()
