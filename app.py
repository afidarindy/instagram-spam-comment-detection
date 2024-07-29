from flask import Flask, render_template, request, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secret key
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# user data for login
users = {'afida@udb.com': {'password': '123'}}

class User(UserMixin):
    def __init__(self, email):
        self.id = email

@login_manager.user_loader
def load_user(email):
    if email in users:
        return User(email)
    return None

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        if email in users and users[email]['password'] == password:
            user = User(email)
            login_user(user)
            return redirect(url_for('upload_file'))
        else:
            return "Invalid email or password"
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            try:
                # Try reading the file with UTF-8 encoding
                try:
                    df = pd.read_csv(file, encoding='utf-8', delimiter=';',  on_bad_lines='skip')
                except UnicodeDecodeError:
                    # Fallback to another encoding if UTF-8 fails
                    df = pd.read_csv(file, encoding='ISO-8859-1', delimiter=';',  on_bad_lines='skip')
                
                # Handle missing values
                df = df.dropna(subset=['komentar'])  # Optionally, use this instead: df['komentar'] = df['komentar'].fillna('')
                
                missing_cols = [col for col in ['caption', 'komentar', 'label'] if col not in df.columns]
                if missing_cols:
                    return f"Dataset is missing columns: {', '.join(missing_cols)}. Required columns are 'caption', 'komentar', and 'label'."

                vectorizer = TfidfVectorizer(stop_words='english')
                X = vectorizer.fit_transform(df['komentar'])
                y = df['label']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                models = {
                    'Random Forest': RandomForestClassifier(),
                    'KNN': KNeighborsClassifier(),
                    'SVM': SVC()
                }

                results = {}

                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred)
                    cr = classification_report(y_test, y_pred, output_dict=True)

                    # results[name] = {
                    #     'accuracy': accuracy,
                    #     'recall': cr['accuracy']['recall'],
                    #     'precision': cr['accuracy']['precision'],
                    #     'fscore': cr['accuracy']['f1-score'],
                    #     'confusion_matrix': plot_confusion_matrix(cm, title=f'{name} Confusion Matrix'),
                    #     'report': cr
                    # }
                    # Safely extract metrics
                    recall = cr.get('weighted avg', {}).get('recall', 0)
                    precision = cr.get('weighted avg', {}).get('precision', 0)
                    fscore = cr.get('weighted avg', {}).get('f1-score', 0)

                    results[name] = {
                        'accuracy': accuracy,
                        'recall': recall,
                        'precision': precision,
                        'fscore': fscore,
                        'confusion_matrix': plot_confusion_matrix(cm, title=f'{name} Confusion Matrix'),
                        'report': cr
                    }

                return render_template('results.html', results=results)
            except Exception as e:
                return f"An error occurred: {e}"

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
