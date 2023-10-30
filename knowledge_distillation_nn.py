import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the iris dataset from CSV
data = pd.read_csv('iris_data.csv')
X = data.drop('target', axis=1).values
y = data['target'].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data for better training performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Teacher model
def create_teacher_model(input_shape=(4,)):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(3, activation='softmax')  # 3 classes for the iris dataset
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#Student model
def create_student_model(input_shape=(4,)):
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=input_shape),
        keras.layers.Dense(3, activation='softmax')  # 3 classes for the iris dataset
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def distillation_loss(y_true, y_pred, teacher_preds, temperature=1.0):
    true_loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    soft_loss = keras.losses.kullback_leibler_divergence(tf.nn.softmax(teacher_preds / temperature),
                                                         tf.nn.softmax(y_pred / temperature))
    return true_loss + soft_loss

def train_student_with_distillation(student, teacher, x_train, y_train, temperature=1.0, epochs=5):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for x_batch, y_batch in zip(x_train, y_train):
            x_batch = np.expand_dims(x_batch, axis=0)
            teacher_preds = teacher.predict(x_batch)
            with tf.GradientTape() as tape:
                student_preds = student(x_batch, training=True)
                loss = distillation_loss(y_batch, student_preds, teacher_preds, temperature)
            grads = tape.gradient(loss, student.trainable_variables)
            student.optimizer.apply_gradients(zip(grads, student.trainable_variables))

def main():
    teacher = create_teacher_model()
    teacher.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))  # Train the teacher model using fit
    test_loss_teacher, test_acc_teacher = teacher.evaluate(X_test, y_test, verbose=2)

    student = create_student_model()
    train_student_with_distillation(student, teacher, X_train, y_train, temperature=2.0, epochs=10) #Training with 10 epochs
    test_loss, test_acc = student.evaluate(X_test, y_test, verbose=2)

    print(f"\nTeacher Model - Loss: {test_loss_teacher:.4f}, Accuracy: {test_acc_teacher * 100:.2f}%")
    print(f"Student Model - Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    main()
