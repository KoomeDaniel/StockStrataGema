# Stockstratagema

This is a machine learning stock prediction system utilizing ARIMA,LSTM and Linear Regression.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

1. Download and install XAMPP from [here](https://www.apachefriends.org/index.html).

### Installing

A step by step series of examples that tell you how to get a development environment running:

1. After installing XAMPP, open the XAMPP Control Panel.
2. Start the Apache and MySQL services.

### Installing

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Run the following command to install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
4. Open XAMPP control panel and start the MySQL and Apache module.
5. Create a new database named "user" in the MySQL module of XAMPP.

### Configuration
Open the config.cfg and enter your email address and the email app key in the respective variable.
    ```
    my_mail = your_email    ||
    smtp_password = your_email_key
    ```

The database connection is set to default configurations. If you have changed the SQL configuration, adjust the Flask configurations in the `db_connection.py` file to connect to the database according to the changes made.
    ```
    mysql://root:@localhost/user
    ```

### Running the Application

1. The tables will be created when the code is run.
2. To run the Flask application, navigate to the project directory and run the following command:
    ```
    python db_connection.py
    ```

## Built With

* [Flask](https://flask.palletsprojects.com/en/2.0.x/) - The web framework used
* [XAMPP](https://www.apachefriends.org/index.html) - PHP & Perl development environment
* [MySQL](https://www.mysql.com/) - Database

## Authors

* **Kinoti Daniel** - *Initial work* - [KoomeDaniel](https://github.com/KoomeDaniel)

## License

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
