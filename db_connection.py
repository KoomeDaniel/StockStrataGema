import smtplib
from email.mime.text import MIMEText
import hashlib
import random
import json
import os
import math
import plotly
import ast
import pandas as pd
import imghdr
import string
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import pdfkit
import pdfgen
import tempfile
import configparser
from sqlalchemy import or_
from tools import tablestyle
from fpdf import FPDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.pagesizes import landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.units import inch
from io import BytesIO
from yahoo_fin import stock_info as si
from newspaper import Article
from newspaper import build
from newsfetch.news import newspaper
from datetime import datetime,date,timedelta
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from jinja2 import Environment, FileSystemLoader
from scrape.gainers import bsegainers
from scrape.losers import bselosers
from flask import Flask, render_template, request, redirect, url_for, flash, session,send_file,jsonify,make_response,send_from_directory,current_app
from bs4 import BeautifulSoup
from flask_sqlalchemy import SQLAlchemy
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from flask_wtf import FlaskForm
from flask_mail import Mail, Message
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from sqlalchemy.exc import SQLAlchemyError
from stock_prediction import StockAnalyzer
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash,check_password_hash
from itsdangerous import URLSafeTimedSerializer, SignatureExpired
from email.mime.text import MIMEText
from io import BytesIO
from pandas import Timestamp
from sqlalchemy.dialects.postgresql import JSON
from wtforms import StringField, SubmitField,PasswordField
from wtforms.validators import DataRequired, EqualTo, Email
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import Table, Paragraph, Spacer, BaseDocTemplate, Frame, PageTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus.flowables import Image, Spacer, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from flask_caching import Cache
from functools import lru_cache

cache = Cache(config={'CACHE_TYPE': 'simple'})
app = Flask(__name__)
stock_analyzer = None
cache.init_app(app)
# Set the secret key for session security
app.config['SECRET_KEY'] = 'your_secret_key'

# Flask-Login configuration
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# MySQL and SQLAlchemy configuration

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/user'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Suppress a warning
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Automatically reload templates during development
db = SQLAlchemy(app)

config = configparser.ConfigParser()

# Read the configuration file
config.read('config.cfg')

my_mail = config.get('DEFAULT', 'my_mail')
my_smtp_password = config.get('DEFAULT', 'smtp_password')

def datetimeformat(value, format='%Y-%m-%d'):
    # Check if value is a Timestamp object
    if isinstance(value, pd.Timestamp):
        # If it is, convert it to a string
        return value.strftime(format)
    else:
        try:
            # Try to parse the value as a date
            datetime_obj = datetime.strptime(value, format)
            # If value is a valid date string, format it
            return datetime_obj.strftime(format)
        except ValueError:
            # If value is not a valid date string, return it as is
            return value
        except TypeError:
            # If value is a datetime.date object, convert it to a string
            if isinstance(value, date):
                return value.strftime(format)
def safe_json_loads(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print(f"JSON string: {s}")
        pos = s.find('}{')
        if pos != -1:
            s = s[:pos+1] + ',' + s[pos+1:]
            return json.loads(s)
        else:
            raise

# Add the function to jinja filters under the name 'datetimeformat'
app.jinja_env.filters['datetimeformat'] = datetimeformat
class Role(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

user_roles = db.Table('user_roles',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id')),
    db.Column('role_id', db.Integer, db.ForeignKey('role.id'))
)
class Watchlist(db.Model):
    __tablename__ = 'watchlist'
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    security_ticker = db.Column(db.String(8), db.ForeignKey('security.ticker'), primary_key=True)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    verification_code = db.Column(db.String(8), nullable=True)
    verification_attempts = db.Column(db.Integer, default=3)
    is_verified = db.Column(db.Boolean, default=False)
    profile_image = db.Column(db.LargeBinary, nullable=True)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    watchlist_tickers = db.relationship('Security', secondary='watchlist', backref='watchlist', lazy='dynamic')
    watchlist = db.relationship('Watchlist', backref='user', lazy='dynamic')
    def __repr__(self):
        return f'{self.first_name},{self.last_name},{self.email},{self.username},{self.date_created},{self.last_login}'

    @property
    def is_active(self):
        # Check if the user's account is verified
        return self.is_verified
    roles = db.relationship('Role', secondary=user_roles, backref=db.backref('user', lazy='dynamic'))

    def save_image(self, file):
        # Ensure the file is an allowed format and not too large
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
            file_data = file.read()

            # Ensure the file size is within the limit
            if len(file_data) > 1 * 1024 * 1024:  # 1MB limit
                raise ValueError('File size exceeds 1MB')

            # Delete the existing image, if any
            if self.profile_image:
                self.delete_image()

            # Update the user's profile image with the binary data
            self.profile_image = file_data
            db.session().commit()
        else:
            raise ValueError('Invalid file format. Please upload a PNG, JPG, or JPEG image.')

    def delete_image(self):
        # Delete the existing image file, if any
        self.profile_image = None
        db.session().commit()

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    mobile_number = db.Column(db.String(20))
    email_subject = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('feedback', lazy='dynamic'))

class Security(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, nullable=False)
    ticker = db.Column(db.String(8), unique=True, nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    current_news = db.relationship('news', backref='security', lazy='dynamic')
    company_information = db.relationship('company_information', backref='security', lazy='dynamic')
    eod_data_daily = db.relationship('daily__price', backref='security', lazy='dynamic')
    financials = db.relationship('financial', backref='security', lazy='dynamic')

    def __repr__(self):
        return f'{self.name},{self.ticker},{self.date_created},{self.last_updated}'

class company_information(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    business_summary = db.Column(db.Text, nullable=True)
    employees = db.Column(db.Integer, nullable=True)
    industry = db.Column(db.String(32), nullable=True)
    sector = db.Column(db.String(32), nullable=True)
    website = db.Column(db.String(64), nullable=True)
    security_ticker = db.Column(db.String(8), db.ForeignKey('security.ticker'), nullable=False)
    def __repr__(self):
        return f'{self.business_summary},{self.employees},{self.industry},{self.sector},{self.website},{self.security_ticker}'

class news(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    header = db.Column(db.String(200), nullable=False)
    link = db.Column(db.String(500), nullable=False)
    source = db.Column(db.String(100), nullable=False)
    date_retrieved = db.Column(db.DateTime, nullable=False)
    security_ticker = db.Column(db.String(50), db.ForeignKey('security.ticker'), nullable=False)
    def __repr__(self):
        return f'{self.header},{self.link},{self.source},{self.date_retrieved},{self.security_ticker}'
  
class daily__price(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    open_price = db.Column(db.Float, nullable=False)
    close_price = db.Column(db.Float, nullable=False)
    adjusted_close_price = db.Column(db.Float, nullable=False)
    low_price = db.Column(db.Float, nullable=False)
    high_price = db.Column(db.Float, nullable=False)
    daily_volume = db.Column(db.Integer, nullable=False)
    security_ticker = db.Column(db.String(8), db.ForeignKey('security.ticker'), nullable=False)
    timeframe = db.Column(db.String(10), nullable=False)  # Added timeframe column

    def __repr__(self):
        return f'{self.date},{self.open_price},{self.close_price},{self.adjusted_close_price},{self.low_price},{self.high_price},{self.daily_volume},{self.security_ticker},{self.timeframe}'




class financial(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    cashflow_statement = db.Column(JSON, nullable=False)
    balance_sheet = db.Column(JSON, nullable=False)
    income_statement = db.Column(JSON, nullable=False)
    security_ticker = db.Column(db.String(8), db.ForeignKey('security.ticker'), nullable=False)



class UserSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = User

class SecuritySchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Security

class NewsSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = news

class CompanyInformationSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = company_information

class DailyPriceSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = daily__price


class FinancialSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = financial
# Ensure that Flask-Login's user loader function is defined
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def generate_random_password(length=10):
    """Generate a random password of a given length"""
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for i in range(length))

def generate_token(user):
    # Assuming 'SECRET_KEY' is configured in your Flask app
    serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    return serializer.dumps({'user_id': user.id})

def convert_to_dict(item):
    if isinstance(item, dict):
        return item
    elif isinstance(item, str):
        # Convert string to a dictionary in a way that makes sense for your data
        return {'description': item}
    elif isinstance(item, (int, float)):
        # Convert integer/float to a dictionary with a generic key
        return {'value': item}
    else:
        # Handle other types if needed
        return {'unknown_data_type': str(item)}

def is_valid_stock(symbol):
    try:
        # Attempt to fetch historical market data for the given symbol
        stock_data = yf.Ticker(symbol).history(period="1d")
        # If the DataFrame is not empty, symbol is valid
        if not stock_data.empty:
            return True
        else:
            return False
    except Exception as e:
        # If an error occurs (e.g., invalid symbol), return False
        print(f"Error occurred: {e}")
        return False

def send_email(subject, message, to_email):
    # Gmail SMTP server settings
    smtp_server = 'smtp.gmail.com'
    smtp_port = 465  # Use port 587 for TLS

    smtp_username = my_mail  # Replace with your Gmail email
    smtp_password = my_smtp_password # Replace with your Gmail password
    
    # Create a text/plain message
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = smtp_username
    msg['To'] = ', '.join(to_email)

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(smtp_username, smtp_password)  # Login to the SMTP server
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

def send_feedback_email(full_name, email, mobile_number, email_subject, message):
    subject = f"Feedback: {email_subject}"
    body = f"Full Name: {full_name}\nEmail: {email}\nMobile Number: {mobile_number}\n\nMessage:\n{message}"

    to_email = [my_mail]  # Replace with your email address or a designated feedback inbox
    email_sent = send_email(subject, body, to_email)

    if email_sent:
        print('Feedback received! Thank you.')
    else:
        print('Error sending feedback email.')

def get_company_name(ticker):
    info = yf.Ticker(ticker).info
    company_name = info.get('longName', '')
    return company_name

def search_bar_data(search_query):
  """
  Get the results (tickers) for the search bar results  
      
  Returns:
      search_bar: Json format of the search bar results 
  """
  search_bar = list()
  with open('search_bar_data.txt') as f:
      for line in f:
          data = ast.literal_eval(line)
          # Add a condition to filter the data based on the search_query
          if 'ticker' in data and search_query.lower() in data['ticker'].lower():
              search_bar.append(data)
          else:
            pass
  search_bar = json.dumps(search_bar)
  return search_bar


def create_admin():
    with app.app_context():
        admin_role = Role.query.filter_by(name='admin').first()

        if not admin_role:
            admin_role = Role(name='admin')
            db.session.add(admin_role)
            db.session.commit()

        admin_email = 'Koomedaniel045@gmail.com'
        admin = User.query.filter_by(email=admin_email).first()
        password = 'Koome@045'
        hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
        if admin:
            # Update the existing admin with new values
            admin.password = hashed_password
            admin.is_verified = True
            # Update other attributes as needed

            db.session.commit()
            print("Admin updated successfully.")
        else:
            # Create a new admin
            admin = User(
                first_name='Daniel',
                last_name='Koome',
                username='admin',
                email=admin_email,
                password=hashed_password,
                is_verified=True  # Set is_verified to True here
            )
            admin.roles.append(admin_role)
            db.session.add(admin)
            db.session.commit()
            print("Admin created successfully.")

# Call the create_admin function to seed the admin
create_admin()

def get_company_info(ticker):
    security = Security.query.filter_by(ticker=ticker).first()

    # If the Security record doesn't exist, create it
    if not security:
        security = Security(ticker=ticker, name=ticker)  # Add other necessary fields
        db.session.add(security)
        db.session.commit()

    info = yf.Ticker(ticker).info
    print('info:', info)

    # Check if the company information already exists
    existing_company_info = company_information.query.filter_by(security_ticker=ticker).first()

    if not existing_company_info:
        company_info = company_information(
            business_summary=info.get('longBusinessSummary'),
            employees=info.get('fullTimeEmployees'),
            industry=info.get('industry'),
            sector=info.get('sector'),
            website=info.get('website'),
            security_ticker=ticker
        )
        db.session.add(company_info)
        db.session.commit()
        return company_info
    else:
        return existing_company_info

def get_stock_stats(ticker):
    # Fetch the stock data using yfinance
    stock = yf.Ticker(ticker)

    # Fetch the statistics data for the stock
    stats_info = stock.info

    # Prepare the data
    stats_data = {
    'Fiscal Year End': stats_info.get('nextFiscalYearEnd'),
    'Most Recent Quarter': stats_info.get('mostRecentQuarter'),
    'Profit Margin': stats_info.get('profitMargins'),
    'Operating Margin': stats_info.get('operatingMargins'),
    'Return on Assets': stats_info.get('returnOnAssets'),
    'Return on Equity': stats_info.get('returnOnEquity'),
    'Beta': stats_info.get('beta'),
    '52-Week Change': stats_info.get('52WeekChange'),
    'Dividend Date': stats_info.get('dividendDate'),
    'ExDividend Date': stats_info.get('exDividendDate'),
    'LastSplit Date': stats_info.get('lastSplitDate'),
    'Forward Annual Dividend Rate': stats_info.get('forwardEps'),
    'Forward Annual Dividend Yield': stats_info.get('forwardPE'),
    'Trailing Annual Dividend Rate': stats_info.get('trailingAnnualDividendRate'),
    'Trailing Annual Dividend Yield': stats_info.get('trailingAnnualDividendYield'),
    'Payout Ratio': stats_info.get('payoutRatio'),
    'Market Capitalization': stats_info.get('marketCap'),
    'Enterprise Value': stats_info.get('enterpriseValue'),  # Updated from enterpriseToRevenue
    'Trailing PE': stats_info.get('trailingPE'),  # Updated from trailingPE
    'Forward PE': stats_info.get('forwardPE'),
    'PEG Ratio': stats_info.get('pegRatio'),
    'Dividend Date': stats_info.get('lastDividendDate'),
    'Price to Sales': stats_info.get('priceToSalesTrailing12Months'),
    'Price to Book': stats_info.get('priceToBook'),
    'Enterprise to EBITDA': stats_info.get('enterpriseToEbitda'),
    'Enterprise to Revenue': stats_info.get('enterpriseToRevenue'),
    'Shares Outstanding': stats_info.get('sharesOutstanding'),
    'Float': stats_info.get('floatShares'),
    '% Held by Insiders': stats_info.get('heldPercentInsiders'),
    '% Held by Institutions': stats_info.get('heldPercentInstitutions'),
    'Shares Short': stats_info.get('sharesShort'),
    'Short% of Float': stats_info.get('shortPercentOfFloat'),
    'Short% of Shares Outstanding': stats_info.get('sharesPercentSharesOut'),
    'Short Ratio': stats_info.get('shortRatio')

}
    return stats_data

def get_news_info(ticker):
    stock = yf.Ticker(ticker)
    # Get recent news headlines
    news_data = stock.news
    news_info_list = []
    for news_item in news_data:
        print("providerPublishTime:", news_item['providerPublishTime'])
        published_date = datetime.fromtimestamp(news_item['providerPublishTime'])
        print("published_date:", published_date)

        # Check if the news information already exists
        existing_news_info = news.query.filter_by(link=news_item['link'], date_retrieved=published_date).first()

        if not existing_news_info:
            news_info = news(
                header=news_item['title'],
                link=news_item['link'],
                source=news_item['publisher'],
                date_retrieved=published_date,
                security_ticker=ticker
            )
            db.session.add(news_info)
            news_info_list.append(news_info)
    db.session.commit()
    return news_info_list

def get_daily_price(ticker, timeframe):
    today = datetime.today().strftime('%Y-%m-%d')
    data = si.get_data(ticker, start_date='2005-01-01', end_date=today)

    # Resample data based on the timeframe
    if timeframe == 'Weekly':
        data = data.resample('5D').last()
    elif timeframe == 'Monthly':
        data = data.resample('M').last()
    elif timeframe == 'Yearly':
        data = data.resample('Y').last()

    daily_prices = []
    for index, row in data.iterrows():
        # Check if the data already exists in the database
        existing_price = db.session.query(daily__price).filter_by(date=index, security_ticker=ticker, timeframe=timeframe).first()
        if not existing_price:
            daily_price = daily__price(
                date = index,
                open_price = row['open'],
                close_price = row['close'],
                adjusted_close_price = row['adjclose'],
                low_price = row['low'],
                high_price = row['high'],
                daily_volume = row['volume'],
                security_ticker=ticker,
                timeframe=timeframe
            )
            db.session.add(daily_price)
            daily_prices.append(daily_price)
    db.session.commit()
    return daily_prices

def get_financial_info(ticker):
    stock = yf.Ticker(ticker)
    cashflow = stock.cashflow
    balance = stock.balance_sheet
    income = stock.financials

    # Convert the DataFrames to JSON
    cashflow_json = cashflow.to_json()
    balance_json = balance.to_json()
    income_json = income.to_json()

    # Check if the financial information already exists
    existing_financial_info = financial.query.filter_by(security_ticker=ticker).first()

    if not existing_financial_info:
        financial_info = financial(
            cashflow_statement=cashflow_json,
            balance_sheet=balance_json,
            income_statement=income_json,
            security_ticker=ticker
        )
        db.session.add(financial_info)
        db.session.commit()    
        return financial_info
    else:
        return existing_financial_info


def calculate_indicators(start_date, end_date, ticker):
    try:
        # Create a Ticker object for the specified stock symbol
        stock = yf.Ticker(ticker)

        # Get historical daily price data
        data = stock.history(start=start_date, end=end_date)

        # Calculate ATR
        atr = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close']).average_true_range()

        # Calculate EMAs
        ema5 = EMAIndicator(close=data['Close'], window=5).ema_indicator()
        ema21 = EMAIndicator(close=data['Close'], window=21).ema_indicator()
        ema63 = EMAIndicator(close=data['Close'], window=63).ema_indicator()
        ema126 = EMAIndicator(close=data['Close'], window=126).ema_indicator()
        ema252 = EMAIndicator(close=data['Close'], window=252).ema_indicator()

        # Calculate RSI
        rsi = RSIIndicator(close=data['Close']).rsi()

        # Calculate Bollinger Bands
        bb = BollingerBands(close=data['Close'])
        middle_bb = bb.bollinger_mavg()
        upper_bb = bb.bollinger_hband()
        lower_bb = bb.bollinger_lband()

        # Calculate MACD
        macd = MACD(close=data['Close'])
        macd_line = macd.macd()
        signal_line = macd.macd_signal()
        macd_histogram = macd.macd_diff()
        # Calculate Daily Movement
        daily_movement = data['Close'].diff().ewm(span=10).mean()
        # Combine all indicators into a single DataFrame
        tech_ind_data = pd.concat([data['Close'], atr, ema5, ema21, ema63, ema126, ema252, rsi, middle_bb, upper_bb, lower_bb, macd_line, signal_line, macd_histogram, daily_movement], axis=1)
        tech_ind_data.columns = ['Close', 'ATR', 'EMA_5', 'EMA_21', 'EMA_63', 'EMA_126', 'EMA_252', 'RSI', 'BB_20MA', 'BB_UpperBands', 'BB_LowerBands', 'MACD', 'Signal_Line', 'MACD_Histogram', 'Daily_Movement']
        # Convert the DataFrame to a dictionary
        tech_ind_data_dict = tech_ind_data.to_dict('index')

        # Return the dictionary
        return tech_ind_data_dict
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        return None

def user_exists(user_email):
    # Query the User model to check if a user with the given email exists
    user = User.query.filter_by(email=user_email).first()
    return user
def _get_end_date():
    # Get the current date
    end_date = datetime.now()
    return end_date

def ticker_exists(ticker):
    # Query the Watchlist model to check if a ticker exists
    ticker_data = Watchlist.query.filter_by(security_ticker=ticker).first()
    return ticker_data
def _add_new_data(user_email, ticker):
    # Check if the user exists
    user = user_exists(user_email)
    if not user:
        print(f"No user found with email {user_email}")
        return False

    # Check if the ticker already exists in the user's watchlist
    existing_watchlist_entry = Watchlist.query.filter_by(user_id=user.id, security_ticker=ticker).first()
    if existing_watchlist_entry:
        print(f"{ticker} is already in the watchlist.")
        return False

    # Get the stock information from Yahoo Finance
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
    except Exception as e:
        print(f"Error getting data for ticker {ticker}: {str(e)}")
        return False

    # Check if the stock information is valid
    if not info:
        return False

    # Create a new watchlist entry
    entry = Watchlist.query.filter_by(user_id=user.id, security_ticker=ticker).first()
    if not entry:
        # Create a new watchlist entry
        entry = Watchlist(user_id=user.id, security_ticker=ticker)

        # Add the entry to the database
        db.session().add(entry)
        db.session().commit()

    # Add the ticker to the user's watchlist
    user.watchlist.append(entry)
    db.session().commit()

    return True

def get_user_watchlist(user_email):
    """
    Get the watchlist of the user

    Parameters:
        user_email: the email of the user 
    """

    user = user_exists(user_email)
    if user:
        all_tickers = user.watchlist_tickers.all()
        output_ticker_data = list() 
        for ticker in all_tickers:
            # Query the CompanyInfo model to get the company information
            company_info = company_information.query.filter_by(security_ticker=ticker.ticker).first()
            name=get_company_name(ticker.ticker)
            # Query the DailyPrice model to get the latest daily price data
            daily_price = daily__price.query.filter_by(security_ticker=ticker.ticker, timeframe=timeframe).order_by(daily__price.id.desc()).first()


            output_ticker_data.append([ticker.ticker, name, company_info.sector, company_info.industry, daily_price.close_price, daily_price.daily_volume])
            
    return output_ticker_data


def add_user_watchlist(user_email, ticker):
  """
  Add ticker to the watchlist of the user 

  Parameters:
    user_email: the email of the user 
    ticker: ticker symbol to add to watchlist 
  """

  ticker_data = ticker_exists(ticker)
  end = _get_end_date()
  if not ticker_data:
    added_data = _add_new_data(user_email, ticker)
    if not added_data:
      return added_data
    
    db.session.commit()
    
  ticker_data = ticker_exists(ticker)
  ticker_list = ticker_data.user

  if user_exists(user_email) not in ticker_list:
    print(ticker, 'NOT IN WATCHLIST')
    ticker_data.user.append(user_exists(user_email))
  
  db.session.commit()

  return True


def delete_user_watchlist(user_email, ticker):
  """
  Remove ticker from the watchlist of the user 

  Parameters:
    user_email: the email of the user 
    ticker: ticker symbol to add to watchlist 
  """
  user = user_exists(user_email)
  ticker_data = ticker_exists(ticker)
  if ticker_data and user:
    print(ticker, "REMOVING FROM WATCHLIST")
    user.watchlist.remove(ticker_data)
    db.session.commit()
    return True 
  
  return False

def make_plot(chart_data=None, security_ticker=None):
    if chart_data is None:
        # Query the database for chart data based on the provided security ticker
        if security_ticker:
            chart_data = db.session.query(daily__price.date, daily__price.close_price, daily__price.adjusted_close_price).filter_by(security_ticker=security_ticker, timeframe='Daily').all()
        else:
            raise ValueError("Please provide a valid security ticker.")

    df = pd.DataFrame(chart_data, columns=['Date', 'close', 'adj_close'])
    df['Date'] = df['Date'].apply(lambda d: d.strftime(r'%Y-%m-%d'))
    
    data = [
        go.Scatter(
            x=df['Date'],
            y=df['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', dash='solid')  # Solid line for Close Price
        ),
        # Add other traces or customize the plot as needed
    ]
    
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

class VerificationForm(FlaskForm):
    verification_code = StringField('Verification Code', validators=[DataRequired()])
    submit = SubmitField('Verify')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')

class ForgotPasswordForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')


global quote
quote='AAPL'
results_dict = {}

@app.route('/')
def home():
    global  stock_analyzer, results_dict
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    print('processing signup')
    if request.method == 'POST':
        print(request.form)
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        existing_user = User.query.filter(or_(User.username == username, User.email == email)).first()

        if existing_user:
            error_message = "Username already exists. Please choose a different one."
            flash('Username already exists. Please choose a different one.', 'error')
            return render_template('index.html', username_error=True, error_message=error_message)
        
        hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
        verification_code = ''.join(str(random.randint(0, 9)) for _ in range(8))

        user_data = {
            'first_name': first_name,
            'last_name': last_name,
            'username': username,
            'email': email,
            'password': hashed_password,
            'verification_code': verification_code
        }

        with open('user_data.json', 'w') as f:
            json.dump(user_data, f)
        print("User Data:", user_data)
        subject = "Please verify your account"
        message = f'Hello, {username}, Please use the following code to verify your account: {verification_code}'
        to_email = [email]  # Removed extra brackets
        email_sent = send_email(subject, message, to_email)
        if email_sent:
            flash('User registration successful! Verification code sent to your email.', 'success')
            print('User registration successful! Verification code sent to your email.', 'success')
            print('verification:',verification_code)
        else:
            flash('User registration successful! However, we were unable to send a verification code to your email.', 'error')
            print('User registration successful! However, we were unable to send a verification code to your email.', 'error')
        
        return redirect(url_for('verify'))

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    form = VerificationForm()
    try:
        if form.validate_on_submit():
            entered_code = form.verification_code.data

            with open('user_data.json', 'r') as f:
                user_data = json.load(f)

            if user_data['verification_code'] == entered_code:
                new_user = User(
                    first_name=user_data['first_name'],
                    last_name=user_data['last_name'],
                    username=user_data['username'],
                    email=user_data['email'],
                    password=user_data['password'],
                    verification_code=user_data['verification_code'],
                    is_verified=True
                )

                db.session().add(new_user)
                db.session().commit()

                flash('Account verified successfully. You can now login.', 'success')
                return render_template('index.html')

            flash('Incorrect verification code. Please try again.', 'danger')

    except Exception as e:
        print("An error occurred:", e)

    return render_template('verify.html', form=form)

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            # Generate a unique token
            s = URLSafeTimedSerializer(app.config['SECRET_KEY'])
            token = s.dumps(user.email, salt='password-reset-salt')

            # Generate a single-use password
            temp_password = generate_random_password()
            hashed_temp_password = hashlib.sha256(temp_password.encode()).hexdigest()
            user.password = hashed_temp_password
            db.session.commit()

            # Send an email to the user with the single-use password
            send_email('Reset Your Password', f'Your temporary password is: {temp_password}', [user.email])
            session['email'] = user.email
            flash('Check your email for your temporary password', 'info')
            return redirect(url_for('validate_temp_password'))
        else:
            flash('Email not found', 'warning')
    return render_template('Password.html', title='Reset Password', form=form)

@app.route('/validate_temp_password', methods=['GET', 'POST'])
def validate_temp_password():
    email = session.get('email')
    if request.method == 'POST':
        temp_password = request.form.get('temp_password')
        user = User.query.filter_by(email=email).first()
        if user:
            hashed_temp_password = hashlib.sha256(temp_password.encode()).hexdigest()
            if user.password == hashed_temp_password:
                print('correct password')
                token = generate_token(user)
                return redirect(url_for('reset_password', token=token))
            else:
                flash('Invalid password', 'danger')
        else:
            print("No user found with email:", email)
    else:
        print("Request method is not POST")
    return render_template('verify2.html')


@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    s = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    try:
        email = s.loads(token, salt='password-reset-salt', max_age=3600)
    except SignatureExpired:
        flash('The password reset link is expired.', 'warning')
        return redirect(url_for('forgot_password'))

    form = ResetPasswordForm()
    try:
        if form.validate_on_submit():
            user = User.query.filter_by(email=email).first()
            if user:
                # Hash the new password and save it to the database
                user.password = generate_password_hash(form.password.data)
                db.session.commit()

                flash('Your password has been updated!', 'success')
                return redirect(url_for('login'))
            else:
                flash('Invalid user.', 'warning')
                return redirect(url_for('forgot_password'))
    except Exception as e:
        # Log the error for debugging purposes
        print("Error in reset password:", str(e))
        flash('An error occurred while processing your request.', 'error')
    return render_template('Reset.html', title='Reset Password', form=form)


@app.route('/login', methods=['POST'])
def login():
    Username = request.form.get('LOGIN_username')
    Password = request.form.get('Password')
    print("Form data:", request.form) 
    print("LOGIN_username:", Username)
    print("password_candidate:", Password)
    
        
    user = User.query.filter_by(username=Username).first()

    if user:
        stored_password = user.password
        print("stored_password (hashed password from the database):", stored_password)

        Hashed_password = hashlib.sha256(Password.encode('utf-8')).hexdigest()

        if Hashed_password == stored_password:
            print("Entered Password:", Password)
            print("Hashed Password (verification):", Hashed_password)
            print("Login successful for user:", Username)
            print("Login successful for user:", Username) 

            login_user(user)

 # Ensure data is up-to-date
            top_gainers=bsegainers()
            top_losers = bselosers()

            return render_template('symbol.html', top_gainers=top_gainers, top_losers=top_losers, current_user=current_user)
        else:
            # Incorrect password
            print("Incorrect password for user:", Username)
            return render_template('index.html', login_error=True, error_message="Incorrect password")
    else:
        # User not found
        print("User not found:", Username)
        return render_template('index.html', login_error=True, error_message="User not found")
@app.route('/insertintotable', methods=['POST'])
def insertintotable():
    global results_dict, quote,timeframe
    try:
        print("Handling form submission in insertintotable route")
        quote = request.form['nm']
        timeframe = request.form['timeframe']
        if is_valid_stock(quote):
            stock_analyzer = StockAnalyzer(quote, timeframe)
            results_dict = stock_analyzer.process_data()
        else:
            print("Invalid stock symbol. Rendering symbol.html with error message.")
            return render_template('symbol.html', error="Invalid stock symbol.")



        if results_dict:
            print("Redirecting to overview.html")
            return redirect(url_for('overview'))
        else:
            print("Error processing data. Rendering symbol.html")
            return render_template('symbol.html', error="Error processing data.")
    except Exception as e:
        print("Error in insertintotable:", str(e))
        return render_template('symbol.html', error=str(e))

@app.route('/index')
def index():
    top_gainers=bsegainers()
    top_losers = bselosers()

    return render_template('symbol.html', top_gainers=top_gainers, top_losers=top_losers, current_user=current_user)


@lru_cache(maxsize=None)
def get_overview(quote, timeframe):
    try:
        HOME_TITLE = 'Stock Dashboard'
        print('quote:', quote)
        try:
            get_company_info(ticker=quote)
            company_info = company_information.query.filter_by(security_ticker=quote).first()
            print('company_info:', company_info)
        except Exception as e:
            print("Error fetching company info:", str(e))
            return None, str(e)

        # Fetch daily price information from the database
        try:
            get_daily_price(ticker=quote, timeframe='Daily')
            get_daily_price(ticker=quote, timeframe=timeframe)
            daily_price = daily__price.query.filter_by(security_ticker=quote, timeframe=timeframe).all()
        except Exception as e:
            print("Error fetching daily price:", str(e))
            return None, str(e)

        # Fetch news information from yahoo_fin
        try:
            get_news_info(ticker=quote)
            News_info = news.query.filter_by(security_ticker=quote).all()
            print('News_info:', News_info)
        except Exception as e:
            print("Error fetching news info:", str(e))
            return None, str(e)

        # Prepare the general information
        try:
            gen_info = {
                'sector': company_info.sector,
                'industry': company_info.industry,
                'employees': company_info.employees,
                'website': company_info.website,
                'business_summary': company_info.business_summary
            }
        except Exception as e:
            print("Error preparing general info:", str(e))
            return None, str(e)

        # Prepare the news information
        try:
            curr_ticker_news = [(item.header, item.link) for item in News_info]
        except Exception as e:
            print("Error preparing news info:", str(e))
            return None, str(e)
        print('getting company name')
        comp_name=get_company_name(quote)
        print('making plot')
        plot_data = make_plot(security_ticker=quote)
        return (HOME_TITLE, comp_name, quote, plot_data, gen_info, curr_ticker_news), None
    except Exception as e:
        print("Error in overview:", str(e))
        return None, str(e)

@app.route('/overview', methods=['GET'])
def overview():
    global quote, timeframe
    data, error = get_overview(quote, timeframe)
    if error:
        return render_template('symbol.html', error=error)
    else:
        HOME_TITLE, comp_name, quote, plot_data, gen_info, curr_ticker_news = data
        return render_template('overview.html', title=HOME_TITLE, comp_name=comp_name, comp_ticker=quote, plot=plot_data, gen_info=gen_info, curr_ticker_news=curr_ticker_news,error="Error processing data.")


@app.route('/chartdata')
def chartdata():
    try:
        global quote, timeframe
        CHART_DATA_TITLE = 'Chart Data for ' + timeframe + ' data'

        daily_prices = daily__price.query.filter_by(security_ticker=quote, timeframe=timeframe).all()
        # Prepare the data for the table
        eod_data = [[price.date, price.open_price, price.high_price, price.low_price, price.close_price, price.daily_volume, price.adjusted_close_price] for price in daily_prices]
        comp_name=get_company_name(quote)
        
        # Render the table with the fetched data
        return render_template('chartdata.html', title=CHART_DATA_TITLE, comp_name=comp_name,comp_ticker=quote,eod_data=eod_data)

    except Exception as e:
        print("Error in chartdata:", str(e))
        return jsonify(error=str(e))

@app.route('/parse_date', methods=['POST'])
def parse_date():
    global quote, timeframe
    dates_to_parse = request.get_json(force=True)  
    
    if dates_to_parse['start_date'] == "reset_dates":
        daily_prices = daily__price.query.filter_by(security_ticker=quote, timeframe=timeframe).all()
        
        eod_data = [[price.date, price.open_price, price.high_price, price.low_price, price.close_price, price.daily_volume, price.adjusted_close_price] for price in daily_prices]
        return render_template('eoddata.html', eod_data=eod_data)
  
    start_date = datetime.strptime(dates_to_parse['start_date'][:10], "%Y-%m-%d").date() 
    end_date = datetime.strptime(dates_to_parse['end_date'][:10], "%Y-%m-%d").date()
    # Query the database for daily prices within the date range
    daily_prices = daily__price.query.filter(daily__price.date.between(start_date, end_date), daily__price.security_ticker==quote, daily__price.timeframe==timeframe).all()

    # Prepare the data for the table
    eod_data = [[price.date, price.open_price, price.high_price, price.low_price, price.close_price, price.daily_volume, price.adjusted_close_price] for price in daily_prices]

    # Convert the list of lists to a DataFrame
    df = pd.DataFrame(eod_data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjClose'])

    # Filter the DataFrame based on the date range
    temp_data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # Convert the filtered data to a list of lists
    output = temp_data.values.tolist()
    return render_template('eoddata.html', eod_data=output)

@app.route('/technicalindicators')
def techindicators():
    TECHNICAL_INDICATORS_TITLE = 'Technical Indicators'
    global quote
    # Call your function to get the company name and ticker
    comp_ticker = quote
    comp_name = get_company_name(comp_ticker)
    # Set the start_date and end_date to represent a one-year period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    # Call your function to calculate the technical indicators
    tech_ind_data = calculate_indicators(start_date=start_date,end_date=end_date,ticker=comp_ticker)
    print('tech_ind_data1:', tech_ind_data)
    print('redirecting to technicalindicators.html')
    return render_template('technicalindicators.html', title=TECHNICAL_INDICATORS_TITLE, comp_name=comp_name, comp_ticker=comp_ticker, tech_ind_data=tech_ind_data)

@app.template_filter('is_dict')
def is_dict_filter(value):
    return isinstance(value, dict)


@app.route('/financials', methods=['GET'])
def financials():
    global quote
    FINANCIALS_TITLE = 'Financials'
    
    finance = get_financial_info(quote)
    
    # Fetch the financial information for the quote from the database
    financial_info = financial.query.filter_by(security_ticker=quote).first()

    if financial_info is not None:
        # Convert the JSON strings back to DataFrames
        cashflow = pd.read_json(financial_info.cashflow_statement)
        balance_sheet = pd.read_json(financial_info.balance_sheet)
        income_statement = pd.read_json(financial_info.income_statement)

        cashflow_dict = cashflow.to_dict(orient='index')
        balance_sheet_dict = balance_sheet.to_dict(orient='index')
        income_statement_dict = income_statement.to_dict(orient='index')        

        stats_data = {
            'cashflow': cashflow_dict,
            'balance': balance_sheet_dict,
            'income': income_statement_dict
        }
        print('stats_data.keys():', stats_data.keys())
        print("Cashflow:")
        print(stats_data['cashflow'].keys())
        print('stats_data[cashflow]:', stats_data['cashflow'])
        print("Balance:")
        print(stats_data['balance'].keys())
        print('stats_data[balance]:', stats_data['balance'])
        print("Income:")
        print(stats_data['income'].keys())
        print('stats_data[income]:', stats_data['income'])

    comp_name = get_company_name(quote)
    
    
    # Render the template with the fetched data
    return render_template('financials.html', title=FINANCIALS_TITLE, comp_name=comp_name, comp_ticker=quote, fin_data=stats_data)

@app.route('/statistics', methods=['GET'])
def statistics():
    global quote
    STATISTICS_TITLE = 'Statistics'
    # Fetch the statistics data for the quote using yfinance
    stats_data = get_stock_stats(quote)
    comp_name=get_company_name(quote)
    
    # Render the template with the fetched data
    return render_template('statistics.html', title=STATISTICS_TITLE, comp_name=comp_name, comp_ticker=quote, stats_data=stats_data)

@app.route('/marketnews')
def marketnews():
    # Get the news data from different sources
    google_news = build('https://www.google.com/news')
    marketwatch_news = build('https://www.marketwatch.com/')
    business_insider_news = build('https://www.businessinsider.com/')
    financial_post_news = build('https://www.financialpost.com/')

    # Extract the headlines and links
    google_news_data = [(article.title, article.url) for article in google_news.articles]
    marketwatch_news_data = [(article.title, article.url) for article in marketwatch_news.articles]
    business_insider_news_data = [(article.title, article.url) for article in business_insider_news.articles]
    financial_post_news_data = [(article.title, article.url) for article in financial_post_news.articles]

    return render_template('marketnews.html', 
                           google_news_data=google_news_data, 
                           marketwatch_news_data=marketwatch_news_data, 
                           business_insider_news_data=business_insider_news_data, 
                           financial_post_news_data=financial_post_news_data)
@app.route('/parse_date_tech_ind', methods=['POST'])
@login_required
def parse_date_tech_ind():
    global quote
    data = request.get_json()

    # Define default start_date and end_date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    if 'start_date' in data and data['start_date'] != "reset_dates":
        start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')

    if 'end_date' in data and data['end_date'] != "reset_dates":
        end_date = datetime.strptime(data['end_date'], '%Y-%m-%d')

    # Call your calculate_indicators function here with the specified date range
    tech_ind_data = calculate_indicators(start_date=start_date, end_date=end_date, ticker=quote)
    print('tech_ind_data:', tech_ind_data)

    tech_ind_data = {str(key): value for key, value in tech_ind_data.items()}
    
    print('redirecting to techind.html')
    return render_template('techind.html', tech_ind_data=tech_ind_data)

@app.errorhandler(404)
def not_found(e):
  return render_template('error.html')

@app.route('/results', methods=['GET'])
def results():
    global results_dict,timeframe
    try:
        if results_dict:
            # Get recommendation
            idea, decision = results_dict['idea'], results_dict['decision']

            # Gauge chart setup
            plot_bgcolor = "#def"
            quadrant_colors = [plot_bgcolor, "#f25829", "#f2a529", "#eff229", "#85e043", "#2bad4e"]
            quadrant_text = ["", "<b>Strongly Buy</b>", "<b>Buy</b>", "<b>Hold</b>", "<b>Sell</b>", "<b>Strongly Sell</b>"]
            n_quadrants = len(quadrant_colors) - 1

            # Map decision to a value between 0 and 100
            decision_mapping = {
                "STRONGLY BUY": 90,
                "BUY": 70,
                "HOLD": 50,
                "SELL": 30,
                "STRONGLY SELL": 10
            }
            latest_rsi = decision_mapping.get(decision, 50)

            min_value = 0
            max_value = 100
            hand_length = np.sqrt(2) / 4
            hand_angle =  360 * (-latest_rsi/2 - min_value) / (max_value - min_value) - 180
            print("Latest RSI:", latest_rsi)
            print("Hand Angle:", hand_angle)
            # Create gauge chart
            fig = go.Figure(
                data=[
                    go.Pie(
                        values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
                        rotation=90,
                        hole=0.5,
                        marker_colors=quadrant_colors,
                        text=quadrant_text,
                        textinfo="text",
                        hoverinfo="skip",
                        sort=False
                    ),
                ],
                layout=go.Layout(
                    showlegend=False,
                    margin=dict(b=0,t=10,l=10,r=10),
                    width=400,
                    height=400,
                    paper_bgcolor=plot_bgcolor,
                    annotations=[
                        go.layout.Annotation(
                            text=f"<b>Stock Recommendation:</b><br>{decision}",
                            x=0.5, xanchor="center", xref="paper",
                            y=0.25, yanchor="bottom", yref="paper",
                            showarrow=False,
                            font=dict(size=14)
                        )
                    ],
                    shapes=[
                        go.layout.Shape(
                            type="circle",
                            x0=0.48, x1=0.52,
                            y0=0.48, y1=0.52,
                            fillcolor="#333",
                            line_color="#333",
                        ),
                        go.layout.Shape(
                            type="line",
                            x0=0.5, x1=0.5 + hand_length * np.cos(np.radians(hand_angle)),
                            y0=0.5, y1=0.5 + hand_length * np.sin(np.radians(hand_angle)),
                            line=dict(color="#333", width=4)
                        )
                    ]
                )
            )
            
            gauge_chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return render_template('Results.html', Result=results_dict,gauge_chart_json=gauge_chart_json ,timeframe=timeframe)
        else:
            # If results_dict is not available, you might want to handle it accordingly
            return render_template('error.html', error="No data available.")
    except Exception as e:
        print("Error in results route:", str(e))
        return render_template('error.html', error=str(e))

import yfinance as yf

@app.route('/get_price', methods=['POST'])
def get_price():
    global quote
    # Fetch the historical price data for the quote for the last month
    data = yf.download(quote, period="1mo")

    # Get the most recent traded day and the day before it
    current_price = data.iloc[-1]['Close']
    previous_price = data.iloc[-2]['Close']

    # Calculate the price difference as a percentage
    price_difference = ((current_price - previous_price) / previous_price) * 100

    # Format the price difference as a string with a '+' or '-' sign
    price_difference_str = f"{'+-'[price_difference < 0]}{abs(price_difference):.2f}%"
    res = jsonify(new_price=current_price, new_price_diff=price_difference_str)
    return res


@app.route('/get_daily_price_csv')
@login_required
def get_daily_price_csv():
    # Fetch the quote from the session
    global quote, timeframe
    start_date = request.args.get('start_date', default=None)
    end_date = request.args.get('end_date', default=None)

    # Convert the dates to datetime objects if they are not None and not empty strings
    if start_date is not None and start_date != '':
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if end_date is not None and end_date != '':
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    # If no dates are provided, default to the last year's data
    if start_date is None or end_date is None:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=365)


    # Query the database for daily prices within the date range
    daily_prices = daily__price.query.filter(daily__price.date.between(start_date, end_date), daily__price.security_ticker==quote, daily__price.timeframe==timeframe).all()

    # Prepare the data for the CSV
    data1 = [[price.date, price.open_price, price.high_price, price.low_price, price.close_price, price.daily_volume, price.adjusted_close_price] for price in daily_prices]
    df = pd.DataFrame(data1, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjClose'])

    # Ensure the 'outputs' directory exists
    os.makedirs('outputs', exist_ok=True)

    # Save the DataFrame to a CSV file
    csv_path = os.path.join('outputs', f'{quote}_daily_price.csv')
    df.to_csv(csv_path, index=False)

    # Use Flask's send_file function to send the file to the user
    return send_file(csv_path,
                     mimetype='text/csv',
                     as_attachment=True,
                     download_name='daily_price.csv')

@app.route('/get_tech_ind_csv')
@login_required
def get_tech_ind_csv():
    global quote

    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Convert the dates to datetime objects if they are not None and not empty strings
    if start_date is not None and start_date != '':
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if end_date is not None and end_date != '':
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    # If no dates are provided, default to the last year's data
    if start_date is None or end_date is None:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=365)

    os.makedirs('outputs', exist_ok=True)
    print("DOWNLOADING TECHNICAL INDICATORS CSV")
    tech_ind_data = calculate_indicators(start_date=start_date,end_date=end_date,ticker=quote)
    tech_ind_data_df = pd.DataFrame(tech_ind_data)
    # Convert tech_ind_data to a list of lists
    data = [['Date', 'Close', 'ATR', 'EMA5', 'EMA21', 'EMA63', 'EMA126', 'EMA252', 'RSI', 'Middle BB', 'Upper BB', 'Lower BB', 'MACD', 'Signal Line', 'MACD Histogram', 'Daily Movement']]
    for date, values in tech_ind_data.items():
        row = [date.strftime('%Y-%m-%d')]
        for indicator in ['Close', 'ATR', 'EMA_5', 'EMA_21', 'EMA_63', 'EMA_126', 'EMA_252', 'RSI', 'BB_20MA', 'BB_UpperBands', 'BB_LowerBands', 'MACD', 'Signal_Line', 'MACD_Histogram', 'Daily Movement EMA']:
            row.append(values.get(indicator, ''))
        data.append(row)

    # Create a DataFrame from data
    df = pd.DataFrame(data[1:], columns=data[0])

    # Write DataFrame to CSV
    csv_path = os.path.join('outputs', f'{quote}_technical_indicators.csv')
    df.to_csv(csv_path, index=False)

    return send_file(csv_path,
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name='technical_indicators.csv')


@app.route('/eoddata', methods=['GET'])
def eoddata():
    # Fetch the quote from the session
    global quote,timeframe
    daily_prices = daily__price.query.filter_by(security_ticker=quote, timeframe=timeframe).all()

    # Prepare the data for the table
    eod_data = [[price.date, price.open_price, price.high_price, price.low_price, price.close_price, price.daily_volume, price.adjusted_close_price] for price in daily_prices]

    # Render the template with the fetched data
    return render_template('eoddata.html', eod_data=eod_data)
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response
@app.route('/submit_feedback', methods=['POST'])
@login_required
def submit_feedback():
    if request.method == 'POST':
        full_name = current_user.username
        email = current_user.email
        mobile_number = request.form.get('mobile_number')
        email_subject = request.form.get('email_subject')
        message = request.form.get('message')

        feedback = Feedback(
            full_name=full_name,
            email=email,
            mobile_number=mobile_number,
            email_subject=email_subject,
            message=message
        )

        # Associate feedback with the currently logged-in user (if any)
        if current_user.is_authenticated:
            feedback.user = current_user

        db.session().add(feedback)
        db.session().commit()
        send_feedback_email(full_name, email, mobile_number, email_subject, message)
        flash('Thank you for your feedback!', 'success')
        return render_template('symbol.html', current_user=current_user)   
@app.route('/upload-image', methods=['POST'])
@login_required
def upload_image():
    # Check if the post request has the file part
    if 'user-image' not in request.files:
        return 'No file part'

    file = request.files['user-image']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return 'No selected file'

    try:
        # Convert the image to binary format
        file_data = file.read()

        # Update the user's profile image
        current_user.profile_image = file_data
        db.session().commit()

        top_gainers=bsegainers()
        top_losers = bselosers()

        return render_template('symbol.html', top_gainers=top_gainers, top_losers=top_losers, current_user=current_user)
    except Exception as e:
        print(f"Error uploading image: {str(e)}")
        return 'Error uploading image'
@app.route('/display-image')
@login_required
def display_image():
    # Get the user's profile image binary data
    img_data = current_user.profile_image

    if img_data is None:
        # If the user does not have a profile image, serve the default image
        return send_file('static/images/user-icon.png', mimetype='image/png')

    # Convert the binary data to a BytesIO object
    img = BytesIO(img_data)

    # Determine the image format based on the magic bytes
    image_format = imghdr.what(img)

    if image_format not in ['jpeg', 'jpg', 'png']:
        # If the format is not supported, serve the default image
        return send_file('static/images/user-icon.png', mimetype='image/png')

    # Return the image data in response
    return send_file(img, mimetype=f'image/{image_format}')


@app.route('/report')
def report():
    global quote,results_dict,timeframe
    try:
            if results_dict:
                Result=results_dict
                # Your existing code here...
                # Initialize
                os.makedirs('assets', exist_ok=True)
                COMPANY_NAME                = "StockStrataGema<br />COMPANY" # Use <br /> to break lines
                AUTHOR                      = "Kinoti Koome Daniel"
                AUTHOR_EMAIL                = "Koomedaniel045@gmail.com"

                DATE                        = datetime.now().strftime("%Y-%m-%d")
                DATE_FILENAME               = datetime.now().strftime("%Y%m%d")
                FILE_NAME                   = "output.pdf"

                ASSETS_FOLDER               = os.path.join(os.getcwd(), "assets")
                FONT                        = os.path.join(ASSETS_FOLDER, "NotoSansKR[wght].ttf")
                
                pdfmetrics.registerFont(TTFont('NotoSansKR', FONT))
                
                # Add fonts and initialize PDF template
                class MyDocTemplate(BaseDocTemplate):
                    def __init__(self, filename, **kwargs):
                        BaseDocTemplate.__init__(self, filename, **kwargs)
                        frame_x             = (A4[0] - 6*inch) / 2
                        template            = PageTemplate('normal', [Frame(frame_x, 1*inch, 6*inch, 10*inch)])
                        self.addPageTemplates(template)

                    def afterFlowable(self, flowable):
                        "Registers TOC entries."
                        if flowable.__class__ == Paragraph:
                            text            = flowable.getPlainText()
                            self.notify('TOCEntry', (text, self.page))

                doc                         = MyDocTemplate(FILE_NAME,
                                                            pagesize=A4,
                                                            title=f"Report_{DATE_FILENAME}",
                                                            author=f"{AUTHOR} ({AUTHOR_EMAIL})"
                                                        )

                # Set header
                styles                      = getSampleStyleSheet()
                styles.add(ParagraphStyle(name='Center', alignment=1, fontSize=14, spaceAfter=20, textColor=colors.black, fontWeight='bold'))

                left_style                  = ParagraphStyle(name='Left', parent=styles['Normal'], alignment=0)
                center_style                = ParagraphStyle(name='Center', parent=styles['Normal'], alignment=1, fontSize=14, spaceAfter=20, textColor=colors.black, fontWeight='bold', leading=18)
                right_style                 = ParagraphStyle(name='Right', parent=styles['Normal'], alignment=2, fontSize=7, leading=10)

                header                      = [
                                                [
                                                    Paragraph(COMPANY_NAME, left_style),
                                                    "", 
                                                    Paragraph("<b>STOCK MARKET<br />DAILY REPORT</b>", center_style),
                                                    "", 
                                                    Paragraph(f"{AUTHOR}<br />{AUTHOR_EMAIL}<br />{DATE}", right_style)
                                                ]
                                            ]

                header_table                = Table(header, colWidths=[1.5*inch, 0.5*inch, 3*inch, 0.5*inch, 1.5*inch])
                header_table.setStyle(tablestyle.header_style)

                story = [header_table, Spacer(1, 0.3*inch)]

                # Continue with the rest of your code...
                # ...
                #------------------------------------------------------------
                # Get indices, and ficc data
                print("********** Getting indices data **********")
                # table_indices_data          = get_daily_price(ticker=quote, timeframe=timeframe)

                #------------------------------------------------------------

                #------------------------------------------------------------
                # Generate indices intraday change and main sectors change charts
                

                # Use the images from the static folder
                # Get the absolute path to the static folder
                static_folder = os.path.join(current_app.root_path, 'static')

                # Define the width and height for the images
                width, height = 2.4*inch, 1.6*inch

                # Use the images from the static folder
                image_path = os.path.join(static_folder, 'Trends.png')
                Trends_graph = Image(image_path, width, height)

                image_path1 = os.path.join(static_folder, 'ARIMA.png')
                ARIMA_graph = Image(image_path1, width, height)

                image_path2 = os.path.join(static_folder, 'LSTM.png')
                LSTM_graph = Image(image_path2, width, height)

                image_path3 = os.path.join(static_folder, 'LR.png')
                LR_graph = Image(image_path3, width, height)

                # Define the subheadings
                subheading_style = ParagraphStyle(name='Subheading', parent=styles['Normal'], alignment=1, fontSize=12, spaceAfter=10, textColor=colors.black, fontWeight='bold')

                # Define the images and their subheadings
                images_and_subheadings = [
                    ('Trends', Trends_graph),
                    ('ARIMA', ARIMA_graph),
                    ('LSTM', LSTM_graph),
                    ('LR', LR_graph)
                ]

                # Arrange the images and subheadings in a 2x2 layout
                table_graph_data = []
                for i in range(0, len(images_and_subheadings), 2):
                    image_row = []
                    subheading_row = []
                    for j in range(2):
                        if i + j < len(images_and_subheadings):
                            subheading, image = images_and_subheadings[i + j]
                            image_row.extend([image])
                            subheading_row.extend([Paragraph(subheading, subheading_style)])
                            if j < 1:  # Add space between images in the same row
                                image_row.append(Spacer(0.1*inch, 0))
                                subheading_row.append(Spacer(0.1*inch, 0))
                    table_graph_data.append(image_row)
                    table_graph_data.append(subheading_row)

                table_graph = Table(table_graph_data, colWidths=[2.4*inch, 0.1*inch, 2.4*inch])



                #------------------------------------------------------------
                # Get economic calendar data
                print("********** Getting economic calendar data **********")
                
                #------------------------------------------------------------

                # Define the data for the tables
                data_pred_error = [
                    ["Prediction/Error", "ARIMA", "LSTM", "LR"],
                    ["Prediction", Result["arima_pred"], Result["lstm_pred"], Result["lr_pred"]],
                    ["Error", Result["error_arima"], Result["error_lstm"], Result["error_lr"]]
                ]

                data_stock_info = [
                    ["Attribute", "Value"],
                    ["Open", Result["open_s"]],
                    ["High", Result["high_s"]],
                    ["Low", Result["low_s"]],
                    ["Close", Result["close_s"]],
                    ["Adjusted Close", Result["adj_close"]],
                    ["Volume", Result["vol"]]
                ]
                #------------------------------------------------------------
                # Make tables and set styles
                ADDITIONAL_SPACE            = 0.3 * inch
                TOTAL_TABLE_WIDTH           = 6 * inch - 0.1 * inch - ADDITIONAL_SPACE
                SINGLE_TABLE_WIDTH          = TOTAL_TABLE_WIDTH / 2
                COLUMN_WIDTH                = SINGLE_TABLE_WIDTH / 6
                TABLE_ROW_HEIGHT            = 0.15 * inch
                SPACER_WIDTH                = 0.1 * inch

                def create_table(data, column_widths, row_heights, styles):
                    table                   = Table(data, column_widths, row_heights)
                    table.setStyle(styles)
                    return table

                # Indices and FICC tables
                print('creating table indices')
                column_widths_overview      = [COLUMN_WIDTH] * 7
                # table_indices               = create_table(table_indices_data, column_widths_overview, [TABLE_ROW_HEIGHT] * len(table_indices_data), tablestyle.common_style + tablestyle.overview_style)

                print('creating table_pred_error')
                if data_pred_error and data_stock_info:
                    table_pred_error = create_table(data_pred_error, [COLUMN_WIDTH, COLUMN_WIDTH, COLUMN_WIDTH, COLUMN_WIDTH], [TABLE_ROW_HEIGHT] * len(data_pred_error), tablestyle.header_style + tablestyle.common_style)
                    table_stock_info = create_table(data_stock_info, [SINGLE_TABLE_WIDTH, SINGLE_TABLE_WIDTH], [TABLE_ROW_HEIGHT] * len(data_stock_info), tablestyle.header_style + tablestyle.common_style)

                else:
                    # If the data is empty, handle the case appropriately
                    return "Data for tables is empty. Cannot create tables.", 500



                #------------------------------------------------------------

                #------------------------------------------------------------
                # Add tables and divider lines then build PDF
                story.append(table_pred_error)
                story.append(Spacer(1, 0.3*inch))
                story.append(table_stock_info)
                story.append(Spacer(1, 0.3*inch))
                story.append(table_graph)
                story.append(PageBreak())
                #------------------------------------------------------------
                # Generate the PDF report
                output_dir = 'outputs'
                os.makedirs(output_dir, exist_ok=True)
                pdf_path = os.path.join(output_dir, FILE_NAME)
                
                doc = MyDocTemplate(pdf_path, pagesize=A4, title=f"Report_{DATE_FILENAME}", author=f"{AUTHOR} ({AUTHOR_EMAIL})")
                doc.build(story)
                
                return send_file(pdf_path, mimetype='application/pdf', as_attachment=True, download_name=FILE_NAME)
    except Exception as e:
                return str(e), 500

@app.route('/search', methods=['GET'])
def search():
    # Get the search query from the form data
    search_query = request.args.get('search-bar-ticker')

    # Call the search_bar_data function with the search query
    search_results = search_bar_data(search_query)

    # Return the results as JSON
    return jsonify(search_results)

@app.route('/watchlist')
@login_required
def watchlist():
  WATCHLIST_TITLE = 'Your Watchlist'

  user_watchlist = get_user_watchlist(current_user.email)
  return render_template('watchlist.html', title=WATCHLIST_TITLE, user_watchlist=user_watchlist)



from flask_login import current_user, login_required

@app.route('/add_to_watchlist', methods=['POST'])
@login_required
def add_to_watchlist():
    # Use current_user.email to get the email of the current logged-in user
    user_email = current_user.email
    
    # Use request.get_json() to get the symbol from the request data
    request_symbol = request.get_json(force=True)

    # Add the symbol to the watchlist for the current user
    result = _add_new_data(user_email, request_symbol['symbol'])

    if not result:
        print("NO DATA")
        return "False"

    # Get the updated user watchlist
    user_watchlist = get_user_watchlist(user_email)

    # Render the template with the updated watchlist
    return render_template('watchlistdata.html', user_watchlist=user_watchlist)


@app.route('/delete_from_watchlist', methods=['POST'])
def delete_from_watchlist():
  request_symbol = request.get_json(force=True)
  result = delete_user_watchlist(current_user.email, request_symbol['symbol'])
  print('SENDING DELETE WATCHLIST DATA', request_symbol['symbol'])
  return jsonify({'message':"TRUE"})

@app.route('/logout')
def logout():
  logout_user()
  return redirect(url_for('home')) 
if __name__ == '__main__':
    with app.app_context():
        # Create the database tables
        db.create_all()
        print("Connection to the database successful.")
    
    # Run the Flask application
    app.run(debug=False)
