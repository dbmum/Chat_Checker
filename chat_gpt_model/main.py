import pandas as pd
import tensorflow as tf
from model.model import binary_classifier_model, default_hparams
import numpy as np

def load_data():
    train = pd.read_csv("data/raw_data/train_v3_drcat_01.csv", quoting=1)
    train = train.dropna(subset=['text'])
    train = train[train.RDizzl3_seven == True].reset_index(drop=True)
    test = pd.read_csv('data/raw_data/train_essays.csv')

    X_train = train.text.values
    y_train = train.label.values

    X_test = test.text.values
    y_test = test.generated.values

    # Assuming X_train is a DataFrame with a 'text' column containing string values
    unique_texts = np.unique(X_train)
    text_to_index = {text: idx for idx, text in enumerate(unique_texts)}

    # Convert 'text' column to integer indices
    X_train_indices = np.array([text_to_index[text] for text in X_train])

    return X_train_indices, y_train, X_test, y_test

def main():
    # Load your data
    X_train, y_train, X_test, y_test = load_data()

    # Model parameters
    hparams = default_hparams()

    # Build the binary classifier model
    model_output = binary_classifier_model(hparams, X_train)

    # Binary cross-entropy loss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train, logits=model_output['logits']))

    # Accuracy metric
    predictions = tf.cast(tf.sigmoid(model_output['logits']) > 0.5, tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_train), tf.float32))

    # Training operation
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    train_op = optimizer.minimize(loss)

    # Training loop (replace with your actual training loop)
    for epoch in range(10):
        # Replace this with your actual mini-batch training logic
        for batch_X, batch_y in zip(X_train['text_index'], y_train):
            with tf.GradientTape() as tape:
                current_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_y, logits=model_output['logits']))
            gradients = tape.gradient(current_loss, model_output.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model_output.trainable_variables))

        current_accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_train), tf.float32))
        print(f'Epoch {epoch + 1}, Loss: {current_loss.numpy()}, Accuracy: {current_accuracy.numpy()}')

    # Evaluation (replace with your actual evaluation logic)
    test_predictions = tf.cast(tf.sigmoid(binary_classifier_model(hparams, X_test)['logits']) > 0.5, tf.float32)
    test_accuracy = tf.reduce_mean(tf.cast(tf.equal(test_predictions, y_test), tf.float32))
    print(f'Test Accuracy: {test_accuracy.numpy()}')

if __name__ == "__main__":
    main()

