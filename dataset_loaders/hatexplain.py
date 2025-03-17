import numpy as np

class HateSpeechProcessor:
    def __init__(self):
        """
        Initialize the TextProcessor class.

        Args:
            input_texts (list of list of str): Array of arrays containing tokenized text.
            labels (list of dict): List of dictionaries containing 'label' keys with arrays of labels.
            rationales (list of list of float): Array of arrays containing rationale scores.
        """

    def process_input_texts(self, input_texts):
        """
        Join each inner array in input_texts into a single string.

        Returns:
            list of str: Processed input_texts.
        """
        return [" ".join(text_array) for text_array in input_texts]

    def process_labels(self, labels):
        """
        Find the most frequent label for each inner array in labels.

        Returns:
            list of int: Processed labels.
        """
        return [int(np.bincount(label['label']).argmax()) for label in labels]

    def process_rationales(self, rationales):
        """
        Return the rationales as-is or apply any necessary processing.

        Returns:
            list of list of float: Processed rationales.
        """
        processed_rationales = []
        for rationale in rationales:
            if len(rationale) > 0:
                # Compute the union of the current rationale
                union = [any(each) for each in zip(*rationale)]
                union = [int(each) for each in union]
                processed_rationales.append(union)
            else:
                # If the rationale is empty, append it as-is
                processed_rationales.append(rationale)
        return processed_rationales

    def process_dataset(self,input_texts, labels, rationales):
        """
        Process input_texts, labels, and rationales.

        Returns:
            tuple: (processed_input_texts, processed_labels, processed_rationales)
        """
        processed_input_texts = self.process_input_texts(input_texts)
        processed_labels = self.process_labels(labels)
        processed_rationales = self.process_rationales(rationales)
        return processed_input_texts, processed_labels, processed_rationales