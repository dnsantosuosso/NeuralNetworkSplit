import numpy as np
import tensorflow as tf
from tensorflow import keras

#Teacher model: big model that teaches student
def create_teacher_model(input_shape=(32,)):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#Student model: smaller, more efficient model that learns from teacher
def create_student_model(input_shape=(32,)):
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=input_shape),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def distillation_loss(y_true, y_pred, teacher_preds, temperature=1.0):
    # Standard categorical cross-entropy for true labels
    true_loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    # KL divergence for soft labels
    soft_loss = keras.losses.kullback_leibler_divergence(tf.nn.softmax(teacher_preds / temperature),
                                                         tf.nn.softmax(y_pred / temperature))

    return true_loss + soft_loss

def train_student_with_distillation(student, teacher, x_train, y_train, temperature=1.0, epochs=5, batch_size=32):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        num_batches = len(x_train) // batch_size
        
        for step in range(num_batches):
            # Creating batches
            x_batch = x_train[step*batch_size:(step+1)*batch_size]
            y_batch = y_train[step*batch_size:(step+1)*batch_size]
            
            # Compute teacher predictions
            teacher_preds = teacher.predict(x_batch)

            # Train student
            with tf.GradientTape() as tape:
                student_preds = student(x_batch, training=True)
                loss = distillation_loss(y_batch, student_preds, teacher_preds, temperature)

            # Compute and apply gradients
            grads = tape.gradient(loss, student.trainable_variables)
            student.optimizer.apply_gradients(zip(grads, student.trainable_variables))


def main():
    # Generate random training data
    x_train = np.random.rand(1000, 32)
    y_train = np.random.randint(10, size=(1000,))

    # Generate random testing data
    x_test = np.random.rand(200, 32)
    y_test = np.random.randint(10, size=(200,))

    teacher = create_teacher_model()

    # Note: In a real scenario, you'd want to train the teacher model first.
    # For this example, we'll skip the teacher training for simplicity.

    student = create_student_model()

    train_student_with_distillation(student, teacher, x_train, y_train, temperature=2.0, epochs=5)

    # Evaluate teacher model
    teacher_loss, teacher_accuracy = teacher.evaluate(x_test, y_test, verbose=0)
    print(f"Teacher Model - Loss: {teacher_loss:.4f}, Accuracy: {teacher_accuracy*100:.2f}%")

    # Evaluate student model
    student_loss, student_accuracy = student.evaluate(x_test, y_test, verbose=0)
    print(f"Student Model - Loss: {student_loss:.4f}, Accuracy: {student_accuracy*100:.2f}%")



if __name__ == "__main__":
    main()