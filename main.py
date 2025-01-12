#Social_Media_Messages_Filter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_data(file_path):
    raw_data = pd.read_csv(file_path, encoding='latin-1')

    if 'v1' in raw_data.columns and 'v2' in raw_data.columns: #this handles the spam dataset
        raw_data = raw_data[['v1', 'v2']]
        raw_data.columns = ['label', 'message']
        raw_data['label'] = raw_data['label'].map({'spam': 1, 'ham': 0})
    elif 'tagging' in raw_data.columns and 'comments' in raw_data.columns: #this handles the discrimination dataset
        raw_data = raw_data[['tagging', 'comments']]
        raw_data.columns = ['label', 'message']
    else:
        raise ValueError("Unsupported dataset format. Expected columns like ['v1', 'v2'] or ['tagging', 'comments'].")

    return raw_data

def preprocess_data(raw_data):
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(raw_data["message"])
    y = raw_data["label"]
    return x, y, vectorizer

def train_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=33)
    model = MultinomialNB()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    return model, x_test, y_test, y_predict

def evaluate_model(y_test, y_predict):
    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))

def main():
    #file_path = '/Users/yaoshihui/Desktop/CPMP3106_Final_Project/spam.csv'
    file_path = '/Users/yaoshihui/Desktop/CPMP3106_Final_Project/Suspicious_Communication.csv'
    raw_data = load_data(file_path)
    X, y, vectorizer = preprocess_data(raw_data)
    model, X_test, y_test, y_predict = train_model(X, y)
    evaluate_model(y_test, y_predict)

    print("\nInput a message to check if it should be filtered or not (type 'exit' to quit):")
    while True:
        user_input = input("Enter your message: ").strip()
        if user_input.lower() == 'exit':
            print("Enjoy Internet and stay safe! Bye!")
            break

        user_features = vectorizer.transform([user_input])
        prediction = model.predict(user_features)[0]

        print(f"Message: '{user_input}' should {'be filtered' if prediction == 1 else 'NOT be filtered'}\n")

main()
