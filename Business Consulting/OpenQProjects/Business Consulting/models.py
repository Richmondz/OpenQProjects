# C:\Users\fishe\OneDrive\Documents\Project\OpenQProjects\Business Consulting\models.py
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from flask_login import UserMixin
from flask_bcrypt import Bcrypt

bcrypt = Bcrypt()
DATABASE_FILE = 'ai_cache.db'
engine = create_engine(f'sqlite:///{DATABASE_FILE}')
Session = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base, UserMixin):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

class AICache(Base):
    __tablename__ = 'ai_cache'
    prompt_text = Column(Text, primary_key=True)
    response_text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    Base.metadata.create_all(engine)