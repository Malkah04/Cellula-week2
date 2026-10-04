import re
import torch

from LSTM.LSTMmodel import LSTMclassifier
from RNN.RNNmodel import RNNclassifier


class TextClassifier:

    def __init__(self, checkpoint_path="lstm_model_checkpoint.pth"):

        self.device = torch.device("cpu")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True
        )

        self.word_to_idx = checkpoint["word_to_idx"]
        self.max_len = checkpoint["max_len"]
        self.embedding_dim = checkpoint["embedding_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.num_classes = checkpoint["num_classes"]

        self.model = LSTMclassifier(
            vocab_size=len(self.word_to_idx),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes
        )
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.model.to(self.device)
        self.model.eval()

    def tokenize(text):
        text = text.lower()    
        tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b|[!?]", text)
        
        return tokens

    def preprocess(self, text):

        if not isinstance(text, str) or not text.strip():
            raise ValueError("input text is required")

        tokens = self.tokenize(text)

        sequence = [
            self.word_to_idx.get(
                token,
                self.word_to_idx["<UNK>"]
            )
            for token in tokens
        ]

        if len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]

        sequence += [self.word_to_idx["<PAD>"]] * (self.max_len - len(sequence))

        return torch.tensor(
            [sequence],
            dtype=torch.long,
            device=self.device
        )

    def predict(self, text):

        input_tensor = self.preprocess(text)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(
                logits,
                dim=1
            )
            predicted_id = int(
                torch.argmax(
                    probabilities,
                    dim=1
                ).item()
            )
            confidence = float(
                probabilities[
                    0,
                    predicted_id
                ].item()
            )

        return {
            "label_id": predicted_id,
            "confidence": confidence
        }


class RNNTextClassifier:

    def __init__(
        self,
        checkpoint_path="rnn_model_checkpoint.pth"
    ):

        self.device = torch.device("cpu")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True
        )

        self.word_to_idx = checkpoint["word_to_idx"]
        self.max_len = checkpoint["max_len"]
        self.embedding_dim = checkpoint["embedding_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.num_classes = checkpoint["num_classes"]

        self.model = RNNclassifier(
            vocab_size=len(self.word_to_idx),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)

        self.model.eval()

    def tokenize(text):
        text = text.lower()    
        tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b|[!?]", text)
        
        return tokens

    def preprocess(self, text):

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "input text is required."
            )
        tokens = self.tokenize(text)

        sequence = [
            self.word_to_idx.get(
                token,
                self.word_to_idx["<UNK>"]
            )
            for token in tokens
        ]

        if len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]

        sequence += [self.word_to_idx["<PAD>"]] * (self.max_len - len(sequence))

        return torch.tensor(
            [sequence],
            dtype=torch.long,
            device=self.device
        )

    def predict(self, text):

        input_tensor = self.preprocess(text)
        with torch.no_grad():
            logits = self.model(input_tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            )
            predicted_id = int(
                torch.argmax(
                    probabilities,
                    dim=1
                ).item()
            )
            confidence = float(
                probabilities[
                    0,
                    predicted_id
                ].item()
            )

        return {
            "label_id": predicted_id,
            "confidence": confidence
        }