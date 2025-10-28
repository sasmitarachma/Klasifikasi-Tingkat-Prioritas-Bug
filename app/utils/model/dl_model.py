import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.initializers import Orthogonal
from keras.utils import custom_object_scope
import nltk
import re
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class BugPriorityClassifier:
    def __init__(self):
        self.model = None
        self.vocab = None
        self.label_encoder = None
        self.lemmatizer = None
        self.custom_stopwords = None
        self.max_length = 12  # sama dengan Colab
        self.setup_preprocessing()
        self.load_model()
    
    def setup_preprocessing(self):
        """Setup untuk text preprocessing"""
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except:
            pass
        
        self.lemmatizer = WordNetLemmatizer()
        important_words = {'not', 'no', "don't", 'cannot'}
        try:
            self.custom_stopwords = set(stopwords.words('english')) - important_words
        except:
            self.custom_stopwords = set()

    def preprocess_text(self, text):
        """UNIFIED preprocessing function - sama seperti Colab dan debug"""
        if not isinstance(text, str):
            return []

        text = contractions.fix(text)
        text = text.lower()
        text = re.sub(r"#(\w+)", r"\1", text)
        text = re.sub(r"\b\d+\b", "", text)
        text = re.sub(r"[^a-zA-Z\s.-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        words = text.split()
        combined_words = []
        skip_next = False
        
        for i in range(len(words)):
            if skip_next:
                skip_next = False
                continue
            if words[i] in {'not', 'no', "don't", 'cannot'} and i + 1 < len(words):
                combined = words[i] + '_' + words[i + 1]
                combined_words.append(combined)
                skip_next = True
            elif words[i] not in self.custom_stopwords and len(words[i]) > 2:
                combined_words.append(words[i])

        try:
            processed_text = [self.lemmatizer.lemmatize(word) for word in combined_words]
        except:
            processed_text = combined_words

        return processed_text

    def tokens_to_sequence(self, tokens):
        """Convert tokens to sequence IDs"""
        return [self.vocab.get(token, self.vocab.get('<OOV>', 0)) for token in tokens]

    def load_model(self):
        """Load model, vocabulary, dan label encoder"""
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../dl_model'))
        model_path = os.path.join(base_path, 'final_model.keras')
        vocab_path = os.path.join(base_path, 'vocab.pkl')
        label_path = os.path.join(base_path, 'label_encoder.pkl')

        with custom_object_scope({'Orthogonal': Orthogonal()}):
            self.model = load_model(model_path)

        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        with open(label_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

    def predict_priority(self, summaries):
        """Main prediction function - KONSISTEN dengan debug"""
        if isinstance(summaries, str):
            summaries = [summaries]

        # Gunakan preprocessing yang sama
        preprocessed_tokens = [self.preprocess_text(summary) for summary in summaries]
        sequences = [self.tokens_to_sequence(tokens) for tokens in preprocessed_tokens]
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')

        predictions = self.model.predict(padded, verbose=0)
        predicted_class_indices = np.argmax(predictions, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(predicted_class_indices)
        confidence_scores = np.max(predictions, axis=1)

        return predicted_labels.tolist(), confidence_scores.tolist()

    def predict_single(self, summary_text):
        """Single prediction - wrapper untuk predict_priority"""
        labels, scores = self.predict_priority([summary_text])
        return labels[0], scores[0]

    def debug_prediction(self, text):
        """Debug function untuk troubleshooting"""
        print(f"Test sentence: {text}")
        print("="*80)
        
        # 1. CEK PREPROCESSING
        tokens = self.preprocess_text(text)
        print(f"Preprocessed tokens: {tokens}")
        print(f"Number of tokens: {len(tokens)}")
        
        # 2. CEK SEQUENCE CONVERSION
        sequence = self.tokens_to_sequence(tokens)
        print(f"Token to sequence: {sequence}")
        print(f"Sequence length: {len(sequence)}")
        
        # 3. CEK PADDING
        padded = pad_sequences([sequence], maxlen=self.max_length, padding='post')
        print(f"Padded sequence: {padded[0]}")
        print(f"Padded length: {len(padded[0])}")
        print(f"Max length setting: {self.max_length}")
        
        # 4. CEK VOCABULARY MAPPING
        print(f"\nVocabulary mapping for each token:")
        for token in tokens:
            vocab_id = self.vocab.get(token, self.vocab.get('<OOV>', 0))
            print(f"  '{token}' -> {vocab_id}")
        
        # 5. CEK MODEL OUTPUT DETAIL
        prediction_probs = self.model.predict(padded, verbose=0)
        print(f"\nModel output probabilities: {prediction_probs[0]}")
        
        predicted_class_idx = np.argmax(prediction_probs[0])
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = np.max(prediction_probs[0])
        
        print(f"Predicted class index: {predicted_class_idx}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        
        # 6. CEK LABEL ENCODER
        print(f"\nLabel encoder classes: {self.label_encoder.classes_}")
        
        # 7. CEK VOCAB SIZE DAN INFO
        print(f"\nVocabulary info:")
        print(f"Vocab size: {len(self.vocab)}")
        print(f"<OOV> token ID: {self.vocab.get('<OOV>', 'NOT FOUND')}")
        
        # 8. CEK APAKAH ADA TOKEN YANG TIDAK DIKENAL
        unknown_tokens = [token for token in tokens if token not in self.vocab]
        if unknown_tokens:
            print(f"Unknown tokens (mapped to <OOV>): {unknown_tokens}")
        else:
            print("All tokens are in vocabulary")
        
        print("\n" + "="*80)
        return {
            'tokens': tokens,
            'sequence': sequence,
            'padded': padded[0],
            'prediction_probs': prediction_probs[0],
            'predicted_class': predicted_class,
            'confidence': confidence
        }

    def test_consistency(self):
        """Test apakah predict_single dan debug_prediction memberikan hasil sama"""
        test_cases = [
            "Unable to add text . text boxes to any header/footer in Base",
            "CRASH in SfxTabDialogController::ResetHdl(weld::Button &)",
            "LibreOffice comments style formatting is unstable and cumbersome"
        ]
        
        print("TESTING CONSISTENCY:")
        print("="*80)
        
        for i, test_text in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test_text[:50]}...")
            
            # Method 1: predict_single
            pred1, conf1 = self.predict_single(test_text)
            
            # Method 2: debug_prediction (tanpa print detail)
            debug_result = self.debug_prediction(test_text)
            pred2, conf2 = debug_result['predicted_class'], debug_result['confidence']
            
            # Compare
            match = (pred1 == pred2) and (abs(conf1 - conf2) < 0.0001)
            status = "✅ CONSISTENT" if match else "❌ INCONSISTENT"
            
            print(f"predict_single: {pred1} ({conf1:.4f})")
            print(f"debug_prediction: {pred2} ({conf2:.4f})")
            print(f"Status: {status}")
            print("-" * 60)

# Usage example:
if __name__ == "__main__":
    classifier = BugPriorityClassifier()
    
    # Test single prediction
    result = classifier.predict_single("Unable to add text . text boxes to any header/footer in Base")
    print(f"Single prediction: {result}")
    
    # Test consistency
    classifier.test_consistency()