import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from flask import send_file
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

df0 = pd.read_pickle('/Users/jeanphilippepetit-frere/Concordia bootcamp/final_project/df_og')

gender_list = ['Male','Female']
df = pd.read_pickle('/Users/jeanphilippepetit-frere/Concordia bootcamp/final_project/dataframe_predictions')

pixel_data = np.load('/Users/jeanphilippepetit-frere/Concordia bootcamp/final_project/X_test_array.npy')

app = Flask(__name__)
@app.route('/')
def plot_png():

    random_face = np.random.choice(len(df))
    
    age = df['age'][random_face]
    age_prediction = round(df['age_pred'][random_face])
    gender = gender_list[df['gender'][random_face]]
    gender_prediction = gender_list[round(df['gender_pred'][random_face])]
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title('Age: {0}, Age Pred: {1}, Sex: {2}, Sex Pred: {3}'.format(age, age_prediction, gender,gender_prediction))
    
    axis.imshow(pixel_data[random_face])
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
   
    return Response(output.getvalue(), mimetype='image/png')
@app.route('/data_gender')
def get_vizualization():
    fig,ax=plt.subplots(figsize=(6,6))
    ax=sns.set(style="darkgrid")
    sns.countplot(df0['gender'])
    output = io.BytesIO()

    FigureCanvas(fig).print_png(output)
    
    return Response(output.getvalue(), mimetype='image/png')
@app.route('/data_age')
def get_viz():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title('age distribution')
    counts = df0['age'].value_counts()
    axis.bar(counts.index,counts.values)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
if __name__ == "__main__":
    app.run(debug=False)