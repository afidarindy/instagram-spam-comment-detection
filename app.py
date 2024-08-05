from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
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
import os
import pdfkit
import pickle
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    user_level = db.Column(db.String(50), nullable=False)

class Record(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    time = db.Column(db.DateTime, default=datetime.utcnow)
    dataset_file = db.Column(db.String(150), nullable=False)
    result_file = db.Column(db.String(150), nullable=False)
    highest_accuracy = db.Column(db.Float, nullable=False)
    algorithm_name = db.Column(db.String(100), nullable=False)

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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        user = User.query.filter_by(email=email).first()
        if user and user.password == password:
            login_user(user)
            if user.user_level == 'guest':
                return redirect(url_for('history'))
            return redirect(url_for('upload_file'))
        else:
            return "Invalid email or password"
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Debugging in upload_file route
@app.route('/', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            try:
                df = pd.read_csv(file, delimiter=';', encoding='utf-8')

                if not all(col in df.columns for col in ['caption', 'komentar', 'label']):
                    return "Dataset must contain 'caption', 'komentar', and 'label' columns."

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
                highest_accuracy = 0
                best_model_name = ""

                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    if accuracy > highest_accuracy:
                        highest_accuracy = accuracy
                        best_model_name = name

                    cm = confusion_matrix(y_test, y_pred)
                    cr = classification_report(y_test, y_pred, output_dict=True)

                    results[name] = {
                        'accuracy': accuracy,
                        'recall': cr['weighted avg']['recall'],
                        'precision': cr['weighted avg']['precision'],
                        'fscore': cr['weighted avg']['f1-score'],
                        'confusion_matrix': plot_confusion_matrix(cm, title=f'{name} Confusion Matrix'),
                        'report': cr
                    }

                # Save results to a file on server
                result_file_path = os.path.join('results', f'result_{datetime.now().timestamp()}.pkl')
                with open(result_file_path, 'wb') as f:
                    pickle.dump(results, f)

                # Save results metadata to session
                session['result_file'] = result_file_path
                session['dataset_file'] = file.filename
                session['highest_accuracy'] = highest_accuracy
                session['best_model_name'] = best_model_name

                return render_template('results.html', results=results)
            except Exception as e:
                return f"An error occurred: {e}"

    return render_template('upload.html')

@app.route('/save_results', methods=['POST'])
@login_required
def save_results():
    try:
        # Debug prints
        print("Save results called.")

        # Retrieve session data
        result_file_path = session.get('result_file')
        dataset_file_path = session.get('dataset_file')
        highest_accuracy = session.get('highest_accuracy')
        best_model_name = session.get('best_model_name')

        if not result_file_path or not dataset_file_path:
            return "No result file or dataset file found in session.", 400

        results_html = render_template('results.html', results=session.get('results'))
        print("Generated results HTML.")

        pdf = pdfkit.from_string(results_html, False)
        print("Generated PDF.")

        pdf_file_path = os.path.join('results', f'result_{datetime.now().timestamp()}.pdf')
        with open(pdf_file_path, 'wb') as f:
            f.write(pdf)
        print(f"Saved PDF to {pdf_file_path}.")

        record = Record(
            dataset_file=dataset_file_path,
            result_file=pdf_file_path,
            highest_accuracy=highest_accuracy,
            algorithm_name=best_model_name
        )
        db.session.add(record)
        db.session.commit()
        print("Saved record to database.")

        return '', 200
    except Exception as e:
        print(f"An error occurred: {e}")
        return str(e), 500

@app.route('/history')
@login_required
def history():
    records = Record.query.all()
    if not records:
        return "No data available"
    return render_template('history.html', records=records)

if __name__ == "__main__":
    app.run(debug=True)  # Atau app.run() untuk mode non-debug
