from inference import TextClassifier, RNNTextClassifier
from database import save_predictions
from image_captioning import ImageCaptioner


class ToxicityBackend:

    def __init__(self):

        self.lstm_classifier = TextClassifier()
        self.rnn_classifier = RNNTextClassifier()
        self.image_captioner = ImageCaptioner()

        self.label_names = {
            0: "Safe",
            1: "Violent Crimes",
            2: "Non-Violent Crimes",
            3: "unsafe",
            4: "Unknown S-Type",
            5: "Sex-Related Crimes",
            6: "Suicide & Self-Harm",
            7: "Elections",
            8: "Child Sexual Exploitation"
        }

    def get_classifier(self, model_type):

        if model_type == "LSTM":
            return self.lstm_classifier

        if model_type == "RNN":
            return self.rnn_classifier

        raise ValueError(
            f"Unsupported model type: {model_type}"
        )

    def classify(
        self,
        text=None,
        image=None,
        image_path=None,
        model_type="LSTM"
    ):

        # Validate input
        if not text and image is None:
            raise ValueError(
                "Text or image is important"
            )

        # Process text
        original_text = None

        if text and text.strip():
            original_text = text.strip()

        image_caption = None
        if image is not None:image_caption = self.image_captioner.generate_caption(image)
        if not image_caption: raise ValueError("Could not generate a caption for the image")

        text_parts = []
        if original_text:
            text_parts.append(original_text)

        if image_caption:
            text_parts.append(image_caption)

        combined_text = "\n".join(text_parts)

        if not combined_text.strip(): raise ValueError("No text available for classification")

        classifier = self.get_classifier(model_type)
        result = classifier.predict(combined_text)
        label_id = result["label_id"]

        predicted_label = self.label_names.get(
            label_id,
            "Unknown"
        )

        confidence = result["confidence"]

        if original_text and image is not None:
            input_type = "text + image"

        elif original_text:
            input_type = "text"

        else:
            input_type = "image"

        save_predictions(
            input_type=input_type,
            input_text=original_text,
            model_type=model_type,
            image_path=image_path,
            image_caption=image_caption,
            predicted_label=predicted_label,
            confidence=confidence
        )

        return {
            "input_type": input_type,
            "model_type": model_type,
            "input_text": original_text,
            "image_caption": image_caption,
            "combined_text": combined_text,
            "label": predicted_label,
            "label_id": label_id,
            "confidence": confidence
        }