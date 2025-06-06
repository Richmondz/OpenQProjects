# C:\Users\fishe\OneDrive\Documents\Project\OpenQProjects\Business Consulting\auth_routes.py
from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, current_user
from models import User, Session
from forms import RegistrationForm, LoginForm

auth_bp = Blueprint('auth_bp', __name__, template_folder='templates')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        session = Session()
        new_user = User(username=form.username.data, email=form.email.data)
        new_user.set_password(form.password.data)
        session.add(new_user)
        session.commit()
        session.close()
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('auth_bp.login'))
    return render_template('register.html', title='Register', form=form)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        session = Session()
        user = session.query(User).filter_by(email=form.email.data).first()
        session.close()
        if user and user.check_password(form.password.data):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Login Unsuccessful. Please check email and password.', 'danger')
    return render_template('login.html', title='Login', form=form)

@auth_bp.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))