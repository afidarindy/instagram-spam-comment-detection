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

                # Save results to session
                session['results'] = results
                session['dataset_file'] = file.filename
                session['highest_accuracy'] = highest_accuracy
                session['best_model_name'] = best_model_name

                # Debug prints
                print("Results:", results)
                print("Session data:", session)

                return redirect(url_for('result_page'))
            except Exception as e:
                return f"An error occurred: {e}"

    return render_template('upload.html')

@app.route('/result')
@login_required
def result_page():
    results = session.get('results')
    if results:
        print("Rendering results with data:", results)
        return render_template('results.html', results=results)
    else:
        print("No data available in session")
        return "No data available"

@app.route('/save_results', methods=['POST'])
@login_required
def save_results():
    try:
        # Save the result page as PDF
        results_html = render_template('results.html', results=session.get('results'))
        pdf = pdfkit.from_string(results_html, False)

        # Save PDF to a file
        result_file_path = os.path.join('results', f'result_{datetime.utcnow().timestamp()}.pdf')
        with open(result_file_path, 'wb') as f:
            f.write(pdf)

        # Save dataset file
        dataset_file_path = os.path.join('datasets', session.get('dataset_file'))

        # Save record to database
        record = Record(
            dataset_file=dataset_file_path,
            result_file=result_file_path,
            highest_accuracy=session.get('highest_accuracy'),
            algorithm_name=session.get('best_model_name')
        )
        db.session.add(record)
        db.session.commit()

        return '', 200
    except Exception as e:
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