# Disaster Response
One of the biggest concerns of hundreds of families is lack of immediate responses from the government during disasters. The fear of being forgotten on the rooftop of an apartment or left abandoned in the middle of a forest fires is the scariest nightmare for those who are affected by natural disasters.

Disaster response organizations receive millions of messages following a disaster and because different organizations take care of different parts of the problem, there needs to be a way of directing messages to the appropriate organization so that they can respond to the problem accordingly.

This web application was built to classify disaster messages so that an emergency professional would know which organization to send the message to. 

## Data
The [data](https://appen.com/datasets/combined-disaster-response-data/) contains 26,248 labeled messages that were sent during past disasters around the world, such as a 2010 earthquake in Haiti and a 2012 super-storm (Sandy) in the U.S..
Each message is labeled as 1 or more of the following 36 categories: <br />

<pre>
'related', 'request', 'offer', 'aid_related', 
'medical_help', 'medical_products',
'search_and_rescue', 'security', 'military', 
'child_alone', 'water', 'food', 'shelter', 
'clothing', 'money', 'missing_people', 'refugees', 
'death', 'other_aid', 'infrastructure_related', 
'transport', 'buildings', 'electricity', 'tools', 
'hospitals', 'shops', 'aid_centers', 
'other_infrastructure', 'weather_related', 
'floods', 'storm', 'fire', 'earthquake', 'cold', 
'other_weather', 'direct_report'
</pre>

None of the messages in the dataset were labeled as `child_alone` so this category was removed altogether before building the classifier, leaving 35 categories to classify.


## Getting Started

#### Setup

1. Run the following command to install the project requirements:
    `pip install -r requirements.txt`

2. Run the following commands to set up the database and model:

    - To run the ETL pipeline that cleans and stores the data:
        `python data/process_data.py data/disaster_messages.csv data/
        disaster_categories.csv data/messages.db`
    - To run the ML pipeline that trains and saves the classifier:
        `python models/train_classifier.py data/messages.db models/classifier.pkl`

3. Navigate to the project's `app/` directory in the terminal

4. Run the following command to run the web app:
    `python run.py`

5. Navigate to http://127.0.0.1:3001/ in the browser

#### Files

- `app/`
    - `run.py` - This script runs the Flask web application and renders the web pages in the `templates/` directory
    - `templates/`
        - `index.html` - Home page of the website, which contains (1) an input field to enter a message to classify and (2) a data dashboard that summarizes the data that the classifier was trained on
        - `result.html` - Result page of the website, which displays the 35 classification results of the message that was entered into the input field
        
- `data/`
    - `process_data.py` - This script runs the ETL pipeline, which imports data from both csv files, merges and cleans the data, and loads it into a SQLite database
    - `disaster_messages.csv` - Text data, which includes the original text, translated English text, and the message genre (how the message was received)
    - `disaster_categories.csv` - Target labels, which includes binary values to indicate which of the 36 categories each message is labeled as

- `models/`
    - `train_classifier.py` - This script runs the machine learning pipeline, which imports the clean data from the database created by the `process_data.py` script, splits that data into a training and test set, instantiates and trains the model (described above) on the training set, evaluates the model on the test set, and saves the model as a pickle file
    
- `notebooks/`
    - `etl_pipeline.ipynb` - This notebook shows the code exploration for the `process_data.py` script
    - `ml_pipeline.ipynb` - This notebook shows the code exploration for the `train_classifier.py` script
    - `dashboard_visuals.ipynb` - This notebook shows the code exploration for the dashboard visualizations on the home page
    
- `requirements.txt` - list of required Python packages

- `pip_env.txt` - list of pip-installed packages after pip-installing the `requirements.txt` file

#### Interface - Home Page

![home page](images/home.png)

#### Interface - Result Page

![result page](images/result.png)
