import torch
from transformers import AutoTokenizer
from faknow.data.process.text_process import TokenizerFromPreTrained
from faknow.model.content_based.mdfend import MDFEND
import warnings
import os
import gdown
import streamlit as st

class NewsClassifier:
    def __init__(self):
        self.max_len = 250
        self.bert = 'dccuchile/bert-base-spanish-wwm-cased'
        self.tokenizer = TokenizerFromPreTrained(self.max_len, self.bert)
        self.domain_num = 11

        # Google Drive file ID for the model
        file_id = '17u8fXwxm5JVWqEJwdcxzea2LhVl0KR5m'
        model_path = 'models/model_10_experts_20_epoch_best.pth'
        # https://drive.google.com/file/d/17u8fXwxm5JVWqEJwdcxzea2LhVl0KR5m/view?usp=sharing
        self.MODEL_SAVE_PATH = self.download_from_gdrive(file_id, model_path)
        self.MDFEND_MODEL = self.load_model()

    @staticmethod
    def download_from_gdrive(file_id, output_path):
        output = os.path.join(os.path.dirname(__file__), output_path)
        
        # Check if the file already exists
        if not os.path.exists(output):
            gdown.download(id=file_id, output=output, quiet=False)
        
        return output

    def load_model(self):
        model = MDFEND(self.bert, self.domain_num)
        model.load_state_dict(torch.load(f=self.MODEL_SAVE_PATH, map_location=torch.device('cpu')))
        model.eval()
        st.write('load model')
        st.write(model.eval())
        return model

    def predict(self, text, domain=None):
        if domain is None:
            warnings.warn('The news domain was not identified. The model accuracy has been reduced.')
            domain = 0
        inputs = self.tokenizer(text)
        st.write('inside MDFEND')
        st.write(tokenizer.max_len)
        st.write(tokenizer.bert)
        st.write(text)
        st.write(inputs)
        with torch.no_grad():
            outputs = self.MDFEND_MODEL(inputs['token_id'], inputs['mask'], torch.tensor(domain))
        st.write('outputs')
        st.write(outputs.item())
        return outputs.item()

