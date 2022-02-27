# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + _kg_hide-input=true jupyter={"source_hidden": true} papermill={"duration": 0.063313, "end_time": "2021-11-28T21:24:30.876004", "exception": false, "start_time": "2021-11-28T21:24:30.812691", "status": "completed"} tags=[]
# change font and background color

from IPython.core.display import HTML

def apply_styling_changes():
    styles = open("../input/additional-files/style.css", "r").read()
    return HTML("<style>"+styles+"</style>")
apply_styling_changes()

# + [markdown] id="-rV9fPOVvSP3" papermill={"duration": 0.038095, "end_time": "2021-11-28T21:24:30.953377", "exception": false, "start_time": "2021-11-28T21:24:30.915282", "status": "completed"} tags=[]
# <p><center>
#   <img width="800" id="top" src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/Home.PNG" alt="Homepage of the Data Heroines notebook - Saving the world through data">
#     </center></p>
#
# <center><cite>Source: Image from the authors.</cite></center>

# + [markdown] id="-ruKnYDKZ3du" papermill={"duration": 0.038192, "end_time": "2021-11-28T21:24:31.029571", "exception": false, "start_time": "2021-11-28T21:24:30.991379", "status": "completed"} tags=[]
# ***
# ***
#
# <h1 style="color:#9d4edd; font-size:80px; margin-bottom:0px"><center>🦸‍♀️ Data Heroines 🦸‍♀️</center></h1>
# <p><center style="color:#404040; font-family:bangers; font-size:30px; margin-top:0px">Saving the world through data</center></p>
#
# ***
# ***

# + [markdown] id="nOblqcXQJxx7" papermill={"duration": 0.039312, "end_time": "2021-11-28T21:24:31.106901", "exception": false, "start_time": "2021-11-28T21:24:31.067589", "status": "completed"} tags=[]
# ***
#
# <center><h1 style="color:#5a189a ;  font-family:Bangers">Table of Contents</h1></center>
#
# ***
#
# <ul class="nav flex-column" style="font-size:22px">
#     <li class="nav-item">
#     <a class="nav-link" style="font-family:Bangers ; color:#9d4edd"  href="#begin"><b style="color:#5a189a">Prologue:</b> The Beginning of a Great Adventure</a>
#     </li>
#     <li class="nav-item">
#     <a class="nav-link" style="font-family:Bangers ; color:#9d4edd"  href="#chapter1"><b style="color:#5a189a">Chapter 1:</b> Their Tools and Their Pets</a>
#     </li>
#   <li class="nav-item">
#     <a class="nav-link" style="font-family:Bangers ; color:#9d4edd"  href="#chapter2"><b style="color:#5a189a">Chapter 2:</b> The Data and a Preview of the Heroine's Powers</a>
#   </li>
#     <ul class="nav flex-column">
#         <li class="nav-item"><a class="nav-link" style="font-family:Bangers ; color:#9d4edd"  href="#chapter21"><b style="color:#5a189a">Chapter 2.1:</b> A Detour for Data Cleaning</a></li></ul>
#    <li class="nav-item">
#     <a class="nav-link" style="font-family:Bangers ; color:#9d4edd"  href="#chapter3"><b style="color:#5a189a">Chapter 3:</b> The Adventure is Gaining Traction</a>
#   </li>
#   <li class="nav-item">
#     <a class="nav-link" style="font-family:Bangers ; color:#9d4edd"  href="#chapter4"><b style="color:#5a189a">Chapter 4:</b> The Enemy Straight Ahead</a>
#   </li>
#   <li class="nav-item">
#     <a class="nav-link"  style="font-family:Bangers ; color:#9d4edd"  href="#end" > <b style="color:#5a189a">Epilogue:</b> All's Well That Ends Well</a>
#   </li>
#    <li class="nav-item">
#     <a class="nav-link"  style="font-family:Bangers ; color:#5a189a"  href="#references" ><b>References</b></a>
#   </li>
#        <li class="nav-item">
#     <a class="nav-link"  style="font-family:Bangers ; color:#5a189a"  href="#authors" ><b>Authors</b></a>
#   </li>
# </ul>
#
#
# authors

# + [markdown] id="1tXmvWlqg46I" papermill={"duration": 0.038971, "end_time": "2021-11-28T21:24:31.185162", "exception": false, "start_time": "2021-11-28T21:24:31.146191", "status": "completed"} tags=[]
# Differently from other projects, we won't compare men and women, we will only take a look at the incredible women that makes the field of data.
#
# <div style="color:white; display:fill; border-radius:5px; background-color:#9d4edd; font-size:110%; letter-spacing:0.5px">
#         <p style="padding: 10px; color:white;">
#             This notebook is intended as a celebration of the amazing women in the field of data: our Data Heroines.
#         </p>
# </div>
#   
#   To summarize, our main character, the heroine <span style="color:#C40028">Datana Scientistus</span>, accompanied by her friends <span style="color:#2667C3">Datana Analystus</span>, <span style="color:#0393E3">Datana Enginerus</span> and <span style="color:#51B14B">Machina Learnerum</span>, will take us in a journey through data, focusing on the women that keep this field alive! But wait! There's a twist and they encounter a horrible enemy: Covid-19. What will our heroines do to save the world? Let's find out!
#
#
# <p><center>
#   <img width="800" src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/meet.PNG" alt="">
#     </center></p>
#
#
#
# <div style="color:white; display:fill; border-radius:5px; background-color:#9d4edd; font-size:110%; letter-spacing:0.5px">
#         <p style="padding: 10px; color:white;">
#             The story is supposed to be told in a comic book fashion, with the charts and storytelling resembling their artwork. We hope you like it and have fun!
#         </p>
# </div>
#

# + [markdown] id="EmmkdVGvKA-1" papermill={"duration": 0.097238, "end_time": "2021-11-28T21:24:31.320797", "exception": false, "start_time": "2021-11-28T21:24:31.223559", "status": "completed"} tags=[]
# ***
# <center>
# <h1 style="color:#9d4edd;font-family:Bangers" id="begin"><b style="color:#5a189a">Prologue: </b>The Beginning of a Great Adventure</h1></center>
#
# ***

# + [markdown] id="6ZEkyDeKFjeN" papermill={"duration": 0.037747, "end_time": "2021-11-28T21:24:31.397966", "exception": false, "start_time": "2021-11-28T21:24:31.360219", "status": "completed"} tags=[]
# Our journey begins in the long past year of 2019 BC (Before Covid). <span style="color:#C40028">Datana Scientistus</span>, our young heroine, is eagerly waiting to begin her adventure with her friends, an exploration of the data they have. They have obtained, firsthand, information about the years 2020 and 2021, and they will compare it to the 2019 data. Little do they know what is about to happen…
#
# First, she sets out to think:
#
# <p><center>
#   <img width="1000" src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/EP1.PNG" alt="">
#     </center></p>
#
# <p><center>
#   <img width="1000" src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/EP2.PNG" alt="">
#     </center></p>
#    
#
#
# <span style="color:#C40028">Datana Scientistus</span>, hearing her friend's questions, gets very excited for their journey ahead and immediately begins the data exploration. After thanking her great friend, they go get prepared and call their other friends:  <span style="color:#0393E3">Datana Enginerus</span> and <span style="color:#51B14B">Machina Learnerum</span>.
#
# <p><center>
#   <img width="1000" src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/EP3.PNG" alt="">
#     </center></p>
#     
# <p><center>
#   <img width="1000" src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/EP4.PNG" alt="">
#     </center></p>    
#
# <span style="color:#51B14B">Machina Learnerum</span> says that she has read somewhere that:
#
# <blockquote><p>
#     
# Historically science has been characterized as an activity remarkably male and even so women participated, little by little, in this evolution. From the moment science formalized itself and became a study item in universities, the participation of women has become quite restricted since female participation in educational institutions was not allowed. According to Schiebinger [1] there are thoughts that women only managed to become scientists from the 20th century onwards.
# <br>
# In the area of Computer Science the same situation has occurred: Female participation in this area has its first records in the 19th century [2].
# <br>    
# The first prominent name is that of English mathematician Ada Lovelace, who became known as the first female programmer of history. Another important woman in the field of computing is the American Grace Murray Hopper, who worked on the programming of the Mark I computer series, one of the first digital computers. In the 40s, ENIAC was created, considered the first computer of the computer age, and six women were part of the Corps Female Emergency Volunteer, and had a hard job doing ballistic calculations.
# <br>    
# However, the permanence of women in the fields of knowledge traditionally linked to female identity, such as Psychology, Linguistics, Nutrition, Social Work, Speech Therapy, Home Economics and Nursing, which refer to gender roles linked to donation, care and maternity. Areas of knowledge such as Astronomy, Mathematics, Engineering, Computer Science and Physics are the areas with the least participation of the women [3].
# <br>    
# In this way, researchers at universities around the world have asked why gender inequality happens in the areas of Science and Technology. A recent study shows that the number of girls residing in the United States who intend to enroll in a Computer Science course dropped from 28% in 1995 to 13% in 2008. At Stanford University, researchers and professors are aware of the problem and acquisition, since 2008, looking to make the course more attractive to the female audience. Since then the percentage of women increased from 12.5% (2008) to 21% in 2013 [4].
#
# </p></blockquote>
#
# After telling that to her friends, they want to pay an homage to the great women who dare to be active in a field in which they are a minority.
#
# Now, this is the tools they will use to help in their task...

# + [markdown] id="TK8LC2bxa_LI" papermill={"duration": 0.038108, "end_time": "2021-11-28T21:24:31.474520", "exception": false, "start_time": "2021-11-28T21:24:31.436412", "status": "completed"} tags=[]
# ***
# <center>
# <h1 style="color:#9d4edd;font-family:Bangers" id="chapter1"> <b style="color:#5a189a">Chapter 1:</b> Their Tools and Their Pets</h1></center>
#
# ***

# + [markdown] id="bmW6-NMbJlEz" papermill={"duration": 0.038008, "end_time": "2021-11-28T21:24:31.550602", "exception": false, "start_time": "2021-11-28T21:24:31.512594", "status": "completed"} tags=[]
# To accompany our heroines in this perilous journey they have the help of their pets: the Python 🐍 and the Pandas! 🐼

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _kg_hide-input=true _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" id="0rBTqSUyNIH-" jupyter={"source_hidden": true} papermill={"duration": 3.116786, "end_time": "2021-11-28T21:24:34.706265", "exception": false, "start_time": "2021-11-28T21:24:31.589479", "status": "completed"} tags=[]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# + _kg_hide-input=true id="4YeuugRabPIr" jupyter={"source_hidden": true} papermill={"duration": 0.045518, "end_time": "2021-11-28T21:24:34.790870", "exception": false, "start_time": "2021-11-28T21:24:34.745352", "status": "completed"} tags=[]
colors = [
          '#e0aaff',   #2019 
          '#9d4edd',   #2020
          '#5a189a'    #2021
]

# + [markdown] id="Jd65Z4jJbC9K" papermill={"duration": 0.038395, "end_time": "2021-11-28T21:24:34.868081", "exception": false, "start_time": "2021-11-28T21:24:34.829686", "status": "completed"} tags=[]
# ***
# <center>
# <h1 style="color:#9d4edd ;  font-family:Bangers" id="chapter2"><b style="color:#5a189a">Chapter 2:</b> The Data and a Preview of the Heroine's Powers</h1></center>
#
# ***

# + [markdown] id="QWIJNhlXG7z3" papermill={"duration": 0.03842, "end_time": "2021-11-28T21:24:34.945092", "exception": false, "start_time": "2021-11-28T21:24:34.906672", "status": "completed"} tags=[]
# Now they are ready to start!
#
# To begin the adventure, it's always best to see what we have to work with. <span style="color:#9d4edd">Datana Analystus</span> will do that by taking a closer look at the data.
#

# + _kg_hide-input=true jupyter={"source_hidden": true} papermill={"duration": 3.875511, "end_time": "2021-11-28T21:24:38.861242", "exception": false, "start_time": "2021-11-28T21:24:34.985731", "status": "completed"} tags=[]
# survey data
responses_2021 = pd.read_csv("../input/kaggle-survey-2021/kaggle_survey_2021_responses.csv", skiprows=[1], low_memory=False)
responses_2020 = pd.read_csv("../input/kaggle-survey-2020/kaggle_survey_2020_responses.csv", skiprows=[1], low_memory=False)
responses_2019 = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv", skiprows=[1], low_memory=False)

# + [markdown] papermill={"duration": 0.040216, "end_time": "2021-11-28T21:24:38.940229", "exception": false, "start_time": "2021-11-28T21:24:38.900013", "status": "completed"} tags=[]
# Since we'll be taking a look at the women participation, the heroines will select only this part of the data.

# + _kg_hide-input=true id="FvvbBdFE5Hw6" jupyter={"source_hidden": true} papermill={"duration": 0.096712, "end_time": "2021-11-28T21:24:39.075862", "exception": false, "start_time": "2021-11-28T21:24:38.979150", "status": "completed"} tags=[]
responses_2021_women = responses_2021[responses_2021.Q2 == 'Woman']
responses_2020_women = responses_2020[responses_2020.Q2 == 'Woman']
responses_2019_women = responses_2019[responses_2019.Q2 == 'Female']

# + [markdown] id="WzRbF7MbeQJk" papermill={"duration": 0.039188, "end_time": "2021-11-28T21:24:39.154432", "exception": false, "start_time": "2021-11-28T21:24:39.115244", "status": "completed"} tags=[]
# ***
# <center>
# <h1 style="color:#9d4edd;font-family:Bangers" id="chapter21"><b style="color:#5a189a">Chapter 2.1:</b> A Detour for Data Cleaning</h1></center>
#
# ***

# + [markdown] papermill={"duration": 0.038235, "end_time": "2021-11-28T21:24:39.230946", "exception": false, "start_time": "2021-11-28T21:24:39.192711", "status": "completed"} tags=[]
# <span style="color:#9d4edd">Datana Enginerus</span> is the first to see that the data that they possess has to be cleaned (a lot). That's what they'll do now!

# + [markdown] id="ITSqfyUWeh4y" papermill={"duration": 0.038301, "end_time": "2021-11-28T21:24:39.307791", "exception": false, "start_time": "2021-11-28T21:24:39.269490", "status": "completed"} tags=[]
# <h2 style="color:#9d4edd">Cleaning the Salary Information</h2>

# + [markdown] id="5myWWXZ9NIIG" papermill={"duration": 0.039045, "end_time": "2021-11-28T21:24:39.385602", "exception": false, "start_time": "2021-11-28T21:24:39.346557", "status": "completed"} tags=[]
# In 2019 and 2020 the salary "limit" is "> 500,000". In 2021 it's ">1,000,000". They'll change the 2021 values to match the previous years.

# + _kg_hide-input=true _kg_hide-output=true id="S0XO7HpMeUp_" jupyter={"source_hidden": true} papermill={"duration": 0.235377, "end_time": "2021-11-28T21:24:39.659396", "exception": false, "start_time": "2021-11-28T21:24:39.424019", "status": "completed"} tags=[]
responses_2021_women.replace(['$500,000-999,999', '>$1,000,000', '300,000-499,999'],
                             ['> 500,000','> 500,000','300,000-500,000'], inplace=True)

responses_2021_women['Q25'] = responses_2021_women['Q25'].str.replace('$', '', regex=False)
responses_2020_women['Q24'] = responses_2020_women['Q24'].str.replace('$', '', regex=False)
responses_2019_women['Q10'] = responses_2019_women['Q10'].str.replace('$', '', regex=False)

# + _kg_hide-input=true id="N5gBVvaQekEs" jupyter={"source_hidden": true} papermill={"duration": 0.251866, "end_time": "2021-11-28T21:24:39.950695", "exception": false, "start_time": "2021-11-28T21:24:39.698829", "status": "completed"} tags=[]
#2021
salaries_2021 = {
    '0-49,999': responses_2021_women[responses_2021_women.Q25 == '0-999'].Q25.value_counts()[0]+
               responses_2021_women[responses_2021_women.Q25 == '1,000-1,999'].Q25.value_counts()[0]+
               responses_2021_women[responses_2021_women.Q25 == '2,000-2,999'].Q25.value_counts()[0]+
               responses_2021_women[responses_2021_women.Q25 == '3,000-3,999'].Q25.value_counts()[0]+
               responses_2021_women[responses_2021_women.Q25 == '4,000-4,999'].Q25.value_counts()[0]+
               responses_2021_women[responses_2021_women.Q25 == '5,000-7,499'].Q25.value_counts()[0]+
               responses_2021_women[responses_2021_women.Q25 == '7,500-9,999'].Q25.value_counts()[0]+
               responses_2021_women[responses_2021_women.Q25 == '10,000-14,999'].Q25.value_counts()[0]+
               responses_2021_women[responses_2021_women.Q25 == '15,000-19,999'].Q25.value_counts()[0]+
               responses_2021_women[responses_2021_women.Q25 == '20,000-24,999'].Q25.value_counts()[0]+
               responses_2021_women[responses_2021_women.Q25 == '25,000-29,999'].Q25.value_counts()[0]+
               responses_2021_women[responses_2021_women.Q25 == '30,000-39,999'].Q25.value_counts()[0]+
               responses_2021_women[responses_2021_women.Q25 == '40,000-49,999'].Q25.value_counts()[0],
    '50,000-99,999': responses_2021_women[responses_2021_women.Q25 == '50,000-59,999'].Q25.value_counts()[0]+
                     responses_2021_women[responses_2021_women.Q25 == '60,000-69,999'].Q25.value_counts()[0]+
                     responses_2021_women[responses_2021_women.Q25 == '70,000-79,999'].Q25.value_counts()[0]+
                     responses_2021_women[responses_2021_women.Q25 == '80,000-89,999'].Q25.value_counts()[0]+
                     responses_2021_women[responses_2021_women.Q25 == '90,000-99,999'].Q25.value_counts()[0],
    '100,000-149,999': responses_2021_women[responses_2021_women.Q25 == '100,000-124,999'].Q25.value_counts()[0]+
                       responses_2021_women[responses_2021_women.Q25 == '125,000-149,999'].Q25.value_counts()[0],
    '150,000-199,999': responses_2021_women[responses_2021_women.Q25 == '150,000-199,999'].Q25.value_counts()[0],
    '>200,000': responses_2021_women[responses_2021_women.Q25 == '200,000-249,999'].Q25.value_counts()[0]+
                responses_2021_women[responses_2021_women.Q25 == '250,000-299,999'].Q25.value_counts()[0]+
                responses_2021_women[responses_2021_women.Q25 == '300,000-500,000'].Q25.value_counts()[0]+
                responses_2021_women[responses_2021_women.Q25 == '> 500,000'].Q25.value_counts()[0]
}

salaries_2021 = pd.DataFrame.from_dict(salaries_2021, orient = 'index')
salaries_2021 = salaries_2021.reset_index()
salaries_2021.columns = ['Salary', '2021']

#2020
salaries_2020 = {
    '0-49,999': responses_2020_women[responses_2020_women.Q24 == '0-999'].Q24.value_counts()[0]+
               responses_2020_women[responses_2020_women.Q24 == '1,000-1,999'].Q24.value_counts()[0]+
               responses_2020_women[responses_2020_women.Q24 == '2,000-2,999'].Q24.value_counts()[0]+
               responses_2020_women[responses_2020_women.Q24 == '3,000-3,999'].Q24.value_counts()[0]+
               responses_2020_women[responses_2020_women.Q24 == '4,000-4,999'].Q24.value_counts()[0]+
               responses_2020_women[responses_2020_women.Q24 == '5,000-7,499'].Q24.value_counts()[0]+
               responses_2020_women[responses_2020_women.Q24 == '7,500-9,999'].Q24.value_counts()[0]+
               responses_2020_women[responses_2020_women.Q24 == '10,000-14,999'].Q24.value_counts()[0]+
               responses_2020_women[responses_2020_women.Q24 == '15,000-19,999'].Q24.value_counts()[0]+
               responses_2020_women[responses_2020_women.Q24 == '20,000-24,999'].Q24.value_counts()[0]+
               responses_2020_women[responses_2020_women.Q24 == '25,000-29,999'].Q24.value_counts()[0]+
               responses_2020_women[responses_2020_women.Q24 == '30,000-39,999'].Q24.value_counts()[0]+
               responses_2020_women[responses_2020_women.Q24 == '40,000-49,999'].Q24.value_counts()[0],
    '50,000-99,999': responses_2020_women[responses_2020_women.Q24 == '50,000-59,999'].Q24.value_counts()[0]+
                     responses_2020_women[responses_2020_women.Q24 == '60,000-69,999'].Q24.value_counts()[0]+
                     responses_2020_women[responses_2020_women.Q24 == '70,000-79,999'].Q24.value_counts()[0]+
                     responses_2020_women[responses_2020_women.Q24 == '80,000-89,999'].Q24.value_counts()[0]+
                     responses_2020_women[responses_2020_women.Q24 == '90,000-99,999'].Q24.value_counts()[0],
    '100,000-149,999': responses_2020_women[responses_2020_women.Q24 == '100,000-124,999'].Q24.value_counts()[0]+
                       responses_2020_women[responses_2020_women.Q24 == '125,000-149,999'].Q24.value_counts()[0],
    '150,000-199,999': responses_2020_women[responses_2020_women.Q24 == '150,000-199,999'].Q24.value_counts()[0],
    '>200,000': responses_2020_women[responses_2020_women.Q24 == '200,000-249,999'].Q24.value_counts()[0]+
                responses_2020_women[responses_2020_women.Q24 == '250,000-299,999'].Q24.value_counts()[0]+
                responses_2020_women[responses_2020_women.Q24 == '300,000-500,000'].Q24.value_counts()[0]+
                responses_2020_women[responses_2020_women.Q24 == '> 500,000'].Q24.value_counts()[0]
}

salaries_2020 = pd.DataFrame.from_dict(salaries_2020, orient = 'index')
salaries_2020 = salaries_2020.reset_index()
salaries_2020.columns = ['Salary', '2020']

#2019
salaries_2019 = {
    '0-4,999': responses_2019_women[responses_2019_women.Q10 == '0-999'].Q10.value_counts()[0]+
               responses_2019_women[responses_2019_women.Q10 == '1,000-1,999'].Q10.value_counts()[0]+
               responses_2019_women[responses_2019_women.Q10 == '2,000-2,999'].Q10.value_counts()[0]+
               responses_2019_women[responses_2019_women.Q10 == '3,000-3,999'].Q10.value_counts()[0]+
               responses_2019_women[responses_2019_women.Q10 == '4,000-4,999'].Q10.value_counts()[0]+
               responses_2019_women[responses_2019_women.Q10 == '5,000-7,499'].Q10.value_counts()[0]+
               responses_2019_women[responses_2019_women.Q10 == '7,500-9,999'].Q10.value_counts()[0]+
               responses_2019_women[responses_2019_women.Q10 == '10,000-14,999'].Q10.value_counts()[0]+
               responses_2019_women[responses_2019_women.Q10 == '15,000-19,999'].Q10.value_counts()[0]+
               responses_2019_women[responses_2019_women.Q10 == '20,000-24,999'].Q10.value_counts()[0]+
               responses_2019_women[responses_2019_women.Q10 == '25,000-29,999'].Q10.value_counts()[0]+
               responses_2019_women[responses_2019_women.Q10 == '30,000-39,999'].Q10.value_counts()[0]+
               responses_2019_women[responses_2019_women.Q10 == '40,000-49,999'].Q10.value_counts()[0],
    '50,000-99,999': responses_2019_women[responses_2019_women.Q10 == '50,000-59,999'].Q10.value_counts()[0]+
                     responses_2019_women[responses_2019_women.Q10 == '60,000-69,999'].Q10.value_counts()[0]+
                     responses_2019_women[responses_2019_women.Q10 == '70,000-79,999'].Q10.value_counts()[0]+
                     responses_2019_women[responses_2019_women.Q10 == '80,000-89,999'].Q10.value_counts()[0]+
                     responses_2019_women[responses_2019_women.Q10 == '90,000-99,999'].Q10.value_counts()[0],
    '100,000-149,999': responses_2019_women[responses_2019_women.Q10 == '100,000-124,999'].Q10.value_counts()[0]+
                       responses_2019_women[responses_2019_women.Q10 == '125,000-149,999'].Q10.value_counts()[0],
    '150,000-199,999': responses_2019_women[responses_2019_women.Q10 == '150,000-199,999'].Q10.value_counts()[0],
    '>200,000': responses_2019_women[responses_2019_women.Q10 == '200,000-249,999'].Q10.value_counts()[0]+
                responses_2019_women[responses_2019_women.Q10 == '250,000-299,999'].Q10.value_counts()[0]+
                responses_2019_women[responses_2019_women.Q10 == '300,000-500,000'].Q10.value_counts()[0]+
                responses_2019_women[responses_2019_women.Q10 == '> 500,000'].Q10.value_counts()[0]
}

salaries_2019 = pd.DataFrame.from_dict(salaries_2019, orient = 'index')
salaries_2019 = salaries_2019.reset_index()
salaries_2019.columns = ['Salary', '2019']

# All
salaries = pd.DataFrame()
salaries['Salary'] = salaries_2021['Salary']
salaries['2021'] = salaries_2021['2021']
salaries['2020'] = salaries_2020['2020']
salaries['2019'] = salaries_2019['2019']

# + [markdown] id="tsl-hXOjgwTp" papermill={"duration": 0.038337, "end_time": "2021-11-28T21:24:40.029012", "exception": false, "start_time": "2021-11-28T21:24:39.990675", "status": "completed"} tags=[]
# <h2 style="color:#9d4edd">Cleaning the Country Information</h2>

# + [markdown] id="B0FN4INVgyrv" papermill={"duration": 0.038765, "end_time": "2021-11-28T21:24:40.106684", "exception": false, "start_time": "2021-11-28T21:24:40.067919", "status": "completed"} tags=[]
# <p><center>
#   <img width="800" src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/EP5.PNG" alt="">
#     </center></p>

# + _kg_hide-input=true id="Ik8Ojc2qgzyE" jupyter={"source_hidden": true} papermill={"duration": 0.117375, "end_time": "2021-11-28T21:24:40.263403", "exception": false, "start_time": "2021-11-28T21:24:40.146028", "status": "completed"} tags=[]
#2021
countries_2021 = {
    'India': responses_2021_women[responses_2021_women.Q3 == 'India'].Q3.value_counts()[0],
    'USA': responses_2021_women[responses_2021_women.Q3 == 'United States of America'].Q3.value_counts()[0],
    'Egypt': responses_2021_women[responses_2021_women.Q3 == 'Egypt'].Q3.value_counts()[0],
    'UK*': responses_2021_women[responses_2021_women.Q3 == 'United Kingdom of Great Britain and Northern Ireland'].Q3.value_counts()[0],
    'Nigeria': responses_2021_women[responses_2021_women.Q3 == 'Nigeria'].Q3.value_counts()[0]
}

countries_2021 = pd.DataFrame.from_dict(countries_2021, orient = 'index')
countries_2021 = countries_2021.reset_index()
countries_2021.columns = ['Country', '2021']

#2020
countries_2020 = {
    'India': responses_2020_women[responses_2020_women.Q3 == 'India'].Q3.value_counts()[0],
    'USA': responses_2020_women[responses_2020_women.Q3 == 'United States of America'].Q3.value_counts()[0],
    'Egypt': responses_2020_women[responses_2020_women.Q3 == 'Egypt'].Q3.value_counts()[0],
    'UK*': responses_2020_women[responses_2020_women.Q3 == 'United Kingdom of Great Britain and Northern Ireland'].Q3.value_counts()[0],
    'Nigeria': responses_2020_women[responses_2020_women.Q3 == 'Nigeria'].Q3.value_counts()[0]
}

countries_2020 = pd.DataFrame.from_dict(countries_2020, orient = 'index')
countries_2020 = countries_2020.reset_index()
countries_2020.columns = ['Country', '2020']

#2019
countries_2019 = {
    'India': responses_2019_women[responses_2019_women.Q3 == 'India'].Q3.value_counts()[0],
    'USA': responses_2019_women[responses_2019_women.Q3 == 'United States of America'].Q3.value_counts()[0],
    'Egypt': responses_2019_women[responses_2019_women.Q3 == 'Egypt'].Q3.value_counts()[0],
    'UK*': responses_2019_women[responses_2019_women.Q3 == 'United Kingdom of Great Britain and Northern Ireland'].Q3.value_counts()[0],
    'Nigeria': responses_2019_women[responses_2019_women.Q3 == 'Nigeria'].Q3.value_counts()[0]
}

countries_2019 = pd.DataFrame.from_dict(countries_2019, orient = 'index')
countries_2019 = countries_2019.reset_index()
countries_2019.columns = ['Country', '2019']

# All
countries = pd.DataFrame()
countries['Country'] = countries_2021['Country']
countries['2021'] = countries_2021['2021']
countries['2020'] = countries_2020['2020']
countries['2019'] = countries_2019['2019']

# + [markdown] id="3vcnI1cWg7O3" papermill={"duration": 0.040345, "end_time": "2021-11-28T21:24:40.343817", "exception": false, "start_time": "2021-11-28T21:24:40.303472", "status": "completed"} tags=[]
# <h2 style="color:#9d4edd">Cleaning the Professional Information</h2>

# + _kg_hide-input=true id="F76O2lcNiNKR" jupyter={"source_hidden": true} papermill={"duration": 0.158684, "end_time": "2021-11-28T21:24:40.541643", "exception": false, "start_time": "2021-11-28T21:24:40.382959", "status": "completed"} tags=[]
#2021
professions_2021 = {
    'DS': responses_2021_women[responses_2021_women.Q5 == 'Data Scientist'].Q5.value_counts()[0],
    'DA': responses_2021_women[responses_2021_women.Q5 == 'Data Analyst'].Q5.value_counts()[0],
    'SE': responses_2021_women[responses_2021_women.Q5 == 'Software Engineer'].Q5.value_counts()[0],
    'RS': responses_2021_women[responses_2021_women.Q5 == 'Research Scientist'].Q5.value_counts()[0],
    'MLE': responses_2021_women[responses_2021_women.Q5 == 'Machine Learning Engineer'].Q5.value_counts()[0],
    'BA': responses_2021_women[responses_2021_women.Q5 == 'Business Analyst'].Q5.value_counts()[0],
    'DE': responses_2021_women[responses_2021_women.Q5 == 'Data Engineer'].Q5.value_counts()[0],
    'PM': responses_2021_women[responses_2021_women.Q5 == 'Program/Project Manager'].Q5.value_counts()[0]+
           responses_2021_women[responses_2021_women.Q5 == 'Product Manager'].Q5.value_counts()[0],
    'Stats': responses_2021_women[responses_2021_women.Q5 == 'Statistician'].Q5.value_counts()[0],
    'DBA': responses_2021_women[responses_2021_women.Q5 == 'DBA/Database Engineer'].Q5.value_counts()[0],
    'DR/A': responses_2021_women[responses_2021_women.Q5 == 'Developer Relations/Advocacy'].Q5.value_counts()[0]
}

professions_2021 = pd.DataFrame.from_dict(professions_2021, orient = 'index')
professions_2021 = professions_2021.reset_index()
professions_2021.columns = ['Job_Title', '2021']

#2020
professions_2020 = {
    'DS': responses_2020_women[responses_2020_women.Q5 == 'Data Scientist'].Q5.value_counts()[0],
    'DA': responses_2020_women[responses_2020_women.Q5 == 'Data Analyst'].Q5.value_counts()[0],
    'SE': responses_2020_women[responses_2020_women.Q5 == 'Software Engineer'].Q5.value_counts()[0],
    'RS': responses_2020_women[responses_2020_women.Q5 == 'Research Scientist'].Q5.value_counts()[0],
    'MLE': responses_2020_women[responses_2020_women.Q5 == 'Machine Learning Engineer'].Q5.value_counts()[0],
    'BA': responses_2020_women[responses_2020_women.Q5 == 'Business Analyst'].Q5.value_counts()[0],
    'DE': responses_2020_women[responses_2020_women.Q5 == 'Data Engineer'].Q5.value_counts()[0],
    'PM': responses_2020_women[responses_2020_women.Q5 == 'Product/Project Manager'].Q5.value_counts()[0],
    'Stats': responses_2020_women[responses_2020_women.Q5 == 'Statistician'].Q5.value_counts()[0],
    'DBA': responses_2020_women[responses_2020_women.Q5 == 'DBA/Database Engineer'].Q5.value_counts()[0],
    'DR/A': 0
}

professions_2020 = pd.DataFrame.from_dict(professions_2020, orient = 'index')
professions_2020 = professions_2020.reset_index()
professions_2020.columns = ['Job_Title', '2020']

#2019
professions_2019 = {
    'DS': responses_2019_women[responses_2019_women.Q5 == 'Data Scientist'].Q5.value_counts()[0],
    'DA': responses_2019_women[responses_2019_women.Q5 == 'Data Analyst'].Q5.value_counts()[0],
    'SE': responses_2019_women[responses_2019_women.Q5 == 'Software Engineer'].Q5.value_counts()[0],
    'RS': responses_2019_women[responses_2019_women.Q5 == 'Research Scientist'].Q5.value_counts()[0],
    'MLE': 0,
    'BA': responses_2019_women[responses_2019_women.Q5 == 'Business Analyst'].Q5.value_counts()[0],
    'DE': responses_2019_women[responses_2019_women.Q5 == 'Data Engineer'].Q5.value_counts()[0],
    'PM': responses_2019_women[responses_2019_women.Q5 == 'Product/Project Manager'].Q5.value_counts()[0],
    'Stats': responses_2019_women[responses_2019_women.Q5 == 'Statistician'].Q5.value_counts()[0],
    'DBA': responses_2019_women[responses_2019_women.Q5 == 'DBA/Database Engineer'].Q5.value_counts()[0],
    'DR/A': 0
}

professions_2019 = pd.DataFrame.from_dict(professions_2019, orient = 'index')
professions_2019 = professions_2019.reset_index()
professions_2019.columns = ['Job_Title', '2019']

# All
professions = pd.DataFrame()
professions['Job_Title'] = professions_2021['Job_Title']
professions['2021'] = professions_2021['2021']
professions['2020'] = professions_2020['2020']
professions['2019'] = professions_2019['2019']

# + [markdown] id="JKFgYRxBn9wb" papermill={"duration": 0.038183, "end_time": "2021-11-28T21:24:40.618391", "exception": false, "start_time": "2021-11-28T21:24:40.580208", "status": "completed"} tags=[]
# <h2 style="color:#9d4edd">Cleaning the Education Information</h2>

# + _kg_hide-input=true id="UJEFS76ioDJ6" jupyter={"source_hidden": true} papermill={"duration": 0.13132, "end_time": "2021-11-28T21:24:40.787843", "exception": false, "start_time": "2021-11-28T21:24:40.656523", "status": "completed"} tags=[]
#2021
education_2021 = {
    "Master": responses_2021_women[responses_2021_women.Q4 == "Master’s degree"].Q4.value_counts()[0],
    "Bachelor": responses_2021_women[responses_2021_women.Q4 == "Bachelor’s degree"].Q4.value_counts()[0],
    "Doctorate": responses_2021_women[responses_2021_women.Q4 == "Doctoral degree"].Q4.value_counts()[0],
    "Some college": responses_2021_women[responses_2021_women.Q4 == 'Some college/university study without earning a bachelor’s degree'].Q4.value_counts()[0],
    "Prof* Doctorate": responses_2021_women[responses_2021_women.Q4 == "Professional doctorate"].Q4.value_counts()[0],
    "High School": responses_2021_women[responses_2021_women.Q4 == "No formal education past high school"].Q4.value_counts()[0]
}

education_2021 = pd.DataFrame.from_dict(education_2021, orient = 'index')
education_2021 = education_2021.reset_index()
education_2021.columns = ['Education', '2021']

#2020
education_2020 = {
    "Master": responses_2020_women[responses_2020_women.Q4 == "Master’s degree"].Q4.value_counts()[0],
    "Bachelor": responses_2020_women[responses_2020_women.Q4 == "Bachelor’s degree"].Q4.value_counts()[0],
    "Doctorate": responses_2020_women[responses_2020_women.Q4 == "Doctoral degree"].Q4.value_counts()[0],
    "Some college": responses_2020_women[responses_2020_women.Q4 == 'Some college/university study without earning a bachelor’s degree'].Q4.value_counts()[0],
    "Prof* Doctorate": responses_2020_women[responses_2020_women.Q4 == "Professional degree"].Q4.value_counts()[0],
    "High School": responses_2020_women[responses_2020_women.Q4 == "No formal education past high school"].Q4.value_counts()[0]
}

education_2020 = pd.DataFrame.from_dict(education_2020, orient = 'index')
education_2020 = education_2020.reset_index()
education_2020.columns = ['Education', '2020']

#2019
education_2019 = {
    "Master": responses_2019_women[responses_2019_women.Q4 == "Master’s degree"].Q4.value_counts()[0],
    "Bachelor": responses_2019_women[responses_2019_women.Q4 == "Bachelor’s degree"].Q4.value_counts()[0],
    "Doctorate": responses_2019_women[responses_2019_women.Q4 == "Doctoral degree"].Q4.value_counts()[0],
    "Some college": responses_2019_women[responses_2019_women.Q4 == 'Some college/university study without earning a bachelor’s degree'].Q4.value_counts()[0],
    "Prof* Doctorate": responses_2019_women[responses_2019_women.Q4 == "Professional degree"].Q4.value_counts()[0],
    "High School": responses_2019_women[responses_2019_women.Q4 == "No formal education past high school"].Q4.value_counts()[0]
}

education_2019 = pd.DataFrame.from_dict(education_2019, orient = 'index')
education_2019 = education_2019.reset_index()
education_2019.columns = ['Education', '2019']

# All
education = pd.DataFrame()
education['Education'] = education_2021['Education']
education['2021'] = education_2021['2021']
education['2020'] = education_2020['2020']
education['2019'] = education_2019['2019']

# + [markdown] id="rALPeTe6sVNC" papermill={"duration": 0.039145, "end_time": "2021-11-28T21:24:40.866886", "exception": false, "start_time": "2021-11-28T21:24:40.827741", "status": "completed"} tags=[]
# <h2 style="color:#9d4edd">Cleaning the Information About the Number of Employees</h2>

# + _kg_hide-input=true id="uDBgYIYQsXAQ" jupyter={"source_hidden": true} papermill={"duration": 0.115346, "end_time": "2021-11-28T21:24:41.020776", "exception": false, "start_time": "2021-11-28T21:24:40.905430", "status": "completed"} tags=[]
#2021
employees_2021 = {
    '0-49': responses_2021_women[responses_2021_women.Q21 == '0-49 employees'].Q21.value_counts()[0],
    '50-249': responses_2021_women[responses_2021_women.Q21 == '50-249 employees'].Q21.value_counts()[0],
    '250-999': responses_2021_women[responses_2021_women.Q21 == '250-999 employees'].Q21.value_counts()[0],
    '1000-9,999': responses_2021_women[responses_2021_women.Q21 == '1000-9,999 employees'].Q21.value_counts()[0],
    '>10,000': responses_2021_women[responses_2021_women.Q21 == '10,000 or more employees'].Q21.value_counts()[0]
}

employees_2021 = pd.DataFrame.from_dict(employees_2021,
                                                 orient = 'index')

employees_2021 = employees_2021.reset_index()
employees_2021.columns = ['Employees', '2021']

#2020
employees_2020 = {
    '0-49': responses_2020_women[responses_2020_women.Q20 == '0-49 employees'].Q20.value_counts()[0],
    '50-249': responses_2020_women[responses_2020_women.Q20 == '50-249 employees'].Q20.value_counts()[0],
    '250-999': responses_2020_women[responses_2020_women.Q20 == '250-999 employees'].Q20.value_counts()[0],
    '1000-9,999': responses_2020_women[responses_2020_women.Q20 == '1000-9,999 employees'].Q20.value_counts()[0],
    '>10,000': responses_2020_women[responses_2020_women.Q20 == '10,000 or more employees'].Q20.value_counts()[0]
}
employees_2020 = pd.DataFrame.from_dict(employees_2020,
                                        orient = 'index')

employees_2020 = employees_2020.reset_index()
employees_2020.columns = ['Employees', '2020']

#2019
responses_2019_women.Q6 = responses_2019_women.Q6.str.replace('> 10,000 employees', '10,000 or more employees')
employees_2019 = {
    '0-49': responses_2019_women[responses_2019_women.Q6 == '0-49 employees'].Q6.value_counts()[0],
    '50-249': responses_2019_women[responses_2019_women.Q6 == '50-249 employees'].Q6.value_counts()[0],
    '250-999': responses_2019_women[responses_2019_women.Q6 == '250-999 employees'].Q6.value_counts()[0],
    '1000-9,999': responses_2019_women[responses_2019_women.Q6 == '1000-9,999 employees'].Q6.value_counts()[0],
    '>10,000': responses_2019_women[responses_2019_women.Q6 == '10,000 or more employees'].Q6.value_counts()[0]
}

employees_2019 = pd.DataFrame.from_dict(employees_2019,
                                        orient = 'index')

employees_2019 = employees_2019.reset_index()
employees_2019.columns = ['Employees', '2019']

# All
employees = pd.DataFrame()
employees['Employees'] = employees_2021['Employees']
employees['2021'] = employees_2021['2021']
employees['2020'] = employees_2020['2020']
employees['2019'] = employees_2019['2019']

# + [markdown] id="2s00mHqY-n4r" papermill={"duration": 0.038057, "end_time": "2021-11-28T21:24:41.097932", "exception": false, "start_time": "2021-11-28T21:24:41.059875", "status": "completed"} tags=[]
# <h2 style="color:#9d4edd">Cleaning the Information About Learning Platforms</h2>

# + [markdown] id="jcozak4P-qSq" papermill={"duration": 0.038472, "end_time": "2021-11-28T21:24:41.175777", "exception": false, "start_time": "2021-11-28T21:24:41.137305", "status": "completed"} tags=[]
# <p><center>
#   <img width="800" src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/EP6.PNG" alt="Homepage of the Data Heroines notebook - Saving the world through data">
#     </center></p>

# + _kg_hide-input=true id="WFcSgff6-ryD" jupyter={"source_hidden": true} papermill={"duration": 0.072771, "end_time": "2021-11-28T21:24:41.287044", "exception": false, "start_time": "2021-11-28T21:24:41.214273", "status": "completed"} tags=[]
#2021
learning_platforms_2021 = {
    'Coursera': responses_2021_women.Q40_Part_1.value_counts()[0],
    'edX': responses_2021_women.Q40_Part_2.value_counts()[0],
    'Kaggle': responses_2021_women.Q40_Part_3.value_counts()[0],
    'DataCamp': responses_2021_women.Q40_Part_4.value_counts()[0],
    'Udacity': responses_2021_women.Q40_Part_6.value_counts()[0],
    'Udemy': responses_2021_women.Q40_Part_7.value_counts()[0],
    'LinkedIn': responses_2021_women.Q40_Part_8.value_counts()[0],
    'Uni. Courses': responses_2021_women.Q40_Part_10.value_counts()[0],
}

learning_platforms_2021 = pd.DataFrame.from_dict(learning_platforms_2021,
                                                 orient = 'index')

learning_platforms_2021 = learning_platforms_2021.reset_index()
learning_platforms_2021.columns = ['Platforms', '2021']

#2020
learning_platforms_2020 = {
    'Coursera': responses_2020_women.Q37_Part_1.value_counts()[0],
    'edX': responses_2020_women.Q37_Part_2.value_counts()[0],
    'Kaggle Learn Courses': responses_2020_women.Q37_Part_3.value_counts()[0],
    'DataCamp': responses_2020_women.Q37_Part_4.value_counts()[0],
    'Udacity': responses_2020_women.Q37_Part_6.value_counts()[0],
    'Udemy': responses_2020_women.Q37_Part_7.value_counts()[0],
    'LinkedIn': responses_2020_women.Q37_Part_8.value_counts()[0],
    'Uni. Courses': responses_2020_women.Q37_Part_10.value_counts()[0],
}

learning_platforms_2020 = pd.DataFrame.from_dict(learning_platforms_2020,
                                                 orient = 'index')

learning_platforms_2020 = learning_platforms_2020.reset_index()
learning_platforms_2020.columns = ['Platforms', '2020']

#2019
learning_platforms_2019 = {
    'Coursera': responses_2019_women.Q13_Part_2.value_counts()[0],
    'edX': responses_2019_women.Q13_Part_3.value_counts()[0],
    'Kaggle Learn Courses': responses_2019_women.Q13_Part_6.value_counts()[0],
    'DataCamp': responses_2019_women.Q13_Part_4.value_counts()[0],
    'Udacity': responses_2019_women.Q13_Part_1.value_counts()[0],
    'Udemy': responses_2019_women.Q13_Part_8.value_counts()[0],
    'LinkedIn': responses_2019_women.Q13_Part_9.value_counts()[0],
    'Uni. Courses': responses_2019_women.Q13_Part_10.value_counts()[0],
}

learning_platforms_2019 = pd.DataFrame.from_dict(learning_platforms_2019,
                                                 orient = 'index')

learning_platforms_2019 = learning_platforms_2019.reset_index()
learning_platforms_2019.columns = ['Platforms', '2019']
learning_platforms_2019

#All
learning_platforms = pd.DataFrame()
learning_platforms['Platforms'] = learning_platforms_2021['Platforms']
learning_platforms['2021'] = learning_platforms_2021['2021']
learning_platforms['2020'] = learning_platforms_2020['2020']
learning_platforms['2019'] = learning_platforms_2019['2019']
learning_platforms = learning_platforms.sort_values(by='2021', ascending = False)

# + [markdown] id="bM2ehBFA_ECn" papermill={"duration": 0.038192, "end_time": "2021-11-28T21:24:41.364670", "exception": false, "start_time": "2021-11-28T21:24:41.326478", "status": "completed"} tags=[]
# <h2 style="color:#9d4edd">Cleaning the Information About Most Used Languages</h2>

# + [markdown] id="1srRE_V1_xEy" papermill={"duration": 0.038238, "end_time": "2021-11-28T21:24:41.443285", "exception": false, "start_time": "2021-11-28T21:24:41.405047", "status": "completed"} tags=[]
# <p><center>
#   <img width="800" src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/EP8.png" alt="">
#     </center></p>

# + _kg_hide-input=true id="1lAVcufg_2f7" jupyter={"source_hidden": true} papermill={"duration": 0.076908, "end_time": "2021-11-28T21:24:41.558806", "exception": false, "start_time": "2021-11-28T21:24:41.481898", "status": "completed"} tags=[]
# 2021
language_use_2021 = {
    'Python': responses_2021_women.Q7_Part_1.value_counts()[0],
    'R': responses_2021_women.Q7_Part_2.value_counts()[0],
    'SQL': responses_2021_women.Q7_Part_3.value_counts()[0],
    'C': responses_2021_women.Q7_Part_4.value_counts()[0],
    'C++': responses_2021_women.Q7_Part_5.value_counts()[0],
    'Java': responses_2021_women.Q7_Part_6.value_counts()[0],
    'Javascript': responses_2021_women.Q7_Part_7.value_counts()[0],
    'Bash': responses_2021_women.Q7_Part_10.value_counts()[0],
    'MATLAB': responses_2021_women.Q7_Part_11.value_counts()[0],
    'Other': responses_2021_women.Q7_OTHER.value_counts()[0],
}

language_use_2021 = pd.DataFrame.from_dict(language_use_2021,
                                                 orient = 'index')

language_use_2021 = language_use_2021.reset_index()
language_use_2021.columns = ['Programming_Language', '2021']

# 2020
language_use_2020 = {
    'Python': responses_2020_women.Q7_Part_1.value_counts()[0],
    'R': responses_2020_women.Q7_Part_2.value_counts()[0],
    'SQL': responses_2020_women.Q7_Part_3.value_counts()[0],
    'C': responses_2020_women.Q7_Part_4.value_counts()[0],
    'C++': responses_2020_women.Q7_Part_5.value_counts()[0],
    'Java': responses_2020_women.Q7_Part_6.value_counts()[0],
    'Javascript': responses_2020_women.Q7_Part_7.value_counts()[0],
    'Bash': responses_2020_women.Q7_Part_10.value_counts()[0],
    'MATLAB': responses_2020_women.Q7_Part_11.value_counts()[0],
    'Other': responses_2020_women.Q7_OTHER.value_counts()[0],
}

language_use_2020 = pd.DataFrame.from_dict(language_use_2020,
                                                 orient = 'index')

language_use_2020 = language_use_2020.reset_index()
language_use_2020.columns = ['Programming_Language', '2020']

#2019
language_use_2019 = {
    'Python': responses_2019_women.Q18_Part_1.value_counts()[0],
    'R': responses_2019_women.Q18_Part_2.value_counts()[0],
    'SQL': responses_2019_women.Q18_Part_3.value_counts()[0],
    'C': responses_2019_women.Q18_Part_4.value_counts()[0],
    'C++': responses_2019_women.Q18_Part_5.value_counts()[0],
    'Java': responses_2019_women.Q18_Part_6.value_counts()[0],
    'Javascript': responses_2019_women.Q18_Part_7.value_counts()[0],
    'Bash': responses_2019_women.Q18_Part_9.value_counts()[0],
    'MATLAB': responses_2019_women.Q18_Part_10.value_counts()[0],
    'Other': responses_2019_women.Q18_Part_12.value_counts()[0],
}

language_use_2019 = pd.DataFrame.from_dict(language_use_2019,
                                                 orient = 'index')

language_use_2019 = language_use_2019.reset_index()
language_use_2019.columns = ['Programming_Language', '2019']

# All
language_use = pd.DataFrame()
language_use['Programming_Language'] = language_use_2021['Programming_Language']
language_use['2021'] = language_use_2021['2021']
language_use['2020'] = language_use_2020['2020']
language_use['2019'] = language_use_2019['2019']
language_use = language_use.sort_values(by='2021', ascending = False)

# + [markdown] id="dIRNxQrK_4Vk" papermill={"duration": 0.039136, "end_time": "2021-11-28T21:24:41.637620", "exception": false, "start_time": "2021-11-28T21:24:41.598484", "status": "completed"} tags=[]
# <h2 style="color:#9d4edd">Cleaning the Information About Most Recommended Languages</h2>

# + [markdown] id="RdvKAq5WQ96v" papermill={"duration": 0.039414, "end_time": "2021-11-28T21:24:41.716825", "exception": false, "start_time": "2021-11-28T21:24:41.677411", "status": "completed"} tags=[]
# <p><center>
#   <img width="800" src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/EP7.PNG" alt="Homepage of the Data Heroines notebook - Saving the world through data">
#     </center></p>

# + _kg_hide-input=true id="NVfl2W6ZRAGN" jupyter={"source_hidden": true} papermill={"duration": 0.159701, "end_time": "2021-11-28T21:24:41.916997", "exception": false, "start_time": "2021-11-28T21:24:41.757296", "status": "completed"} tags=[]
# 2021
recommended_2021 = {
    'Python': responses_2021_women[responses_2021_women.Q8 == 'Python'].Q8.value_counts()[0],
    'R': responses_2021_women[responses_2021_women.Q8 == 'R'].Q8.value_counts()[0],
    'SQL': responses_2021_women[responses_2021_women.Q8 == 'SQL'].Q8.value_counts()[0],
    'C': responses_2021_women[responses_2021_women.Q8 == 'C'].Q8.value_counts()[0],
    'C++': responses_2021_women[responses_2021_women.Q8 == 'C++'].Q8.value_counts()[0],
    'Java': responses_2021_women[responses_2021_women.Q8 == 'Java'].Q8.value_counts()[0],
    'Javascript': responses_2021_women[responses_2021_women.Q8 == 'Javascript'].Q8.value_counts()[0],
    'Bash': responses_2021_women[responses_2021_women.Q8 == 'Bash'].Q8.value_counts()[0],
    'MATLAB': responses_2021_women[responses_2021_women.Q8 == 'MATLAB'].Q8.value_counts()[0],
    'Other': responses_2021_women[responses_2021_women.Q8 == 'Other'].Q8.value_counts()[0],
}

recommended_2021 = pd.DataFrame.from_dict(recommended_2021, orient = 'index')
recommended_2021 = recommended_2021.reset_index()
recommended_2021.columns = ['Language', '2021']

# 2020
recommended_2020 = {
    'Python': responses_2020_women[responses_2020_women.Q8 == 'Python'].Q8.value_counts()[0],
    'R': responses_2020_women[responses_2020_women.Q8 == 'R'].Q8.value_counts()[0],
    'SQL': responses_2020_women[responses_2020_women.Q8 == 'SQL'].Q8.value_counts()[0],
    'C': responses_2020_women[responses_2020_women.Q8 == 'C'].Q8.value_counts()[0],
    'C++': responses_2020_women[responses_2020_women.Q8 == 'C++'].Q8.value_counts()[0],
    'Java': responses_2020_women[responses_2020_women.Q8 == 'Java'].Q8.value_counts()[0],
    'Javascript': responses_2020_women[responses_2020_women.Q8 == 'Javascript'].Q8.value_counts()[0],
    'Bash': responses_2020_women[responses_2020_women.Q8 == 'Bash'].Q8.value_counts()[0],
    'MATLAB': responses_2020_women[responses_2020_women.Q8 == 'MATLAB'].Q8.value_counts()[0],
    'Other': responses_2020_women[responses_2020_women.Q8 == 'Other'].Q8.value_counts()[0],
}

recommended_2020 = pd.DataFrame.from_dict(recommended_2020, orient = 'index')
recommended_2020 = recommended_2020.reset_index()
recommended_2020.columns = ['Language', '2020']

#2019
recommended_2019 = {
    'Python': responses_2019_women[responses_2019_women.Q19 == 'Python'].Q19.value_counts()[0],
    'R': responses_2019_women[responses_2019_women.Q19 == 'R'].Q19.value_counts()[0],
    'SQL': responses_2019_women[responses_2019_women.Q19 == 'SQL'].Q19.value_counts()[0],
    'C': responses_2019_women[responses_2019_women.Q19 == 'C'].Q19.value_counts()[0],
    'C++': responses_2019_women[responses_2019_women.Q19 == 'C++'].Q19.value_counts()[0],
    'Java': responses_2019_women[responses_2019_women.Q19 == 'Java'].Q19.value_counts()[0],
    'Javascript': responses_2019_women[responses_2019_women.Q19 == 'Javascript'].Q19.value_counts()[0],
    'Bash': responses_2019_women[responses_2019_women.Q19 == 'Bash'].Q19.value_counts()[0],
    'MATLAB': responses_2019_women[responses_2019_women.Q19 == 'MATLAB'].Q19.value_counts()[0],
    'Other': responses_2019_women[responses_2019_women.Q19 == 'Other'].Q19.value_counts()[0],
}

recommended_2019 = pd.DataFrame.from_dict(recommended_2019, orient = 'index')
recommended_2019 = recommended_2019.reset_index()
recommended_2019.columns = ['Language', '2019']

# All
recommended = pd.DataFrame()
recommended['Language'] = recommended_2021['Language']
recommended['2021'] = recommended_2021['2021']
recommended['2020'] = recommended_2020['2020']
recommended['2019'] = recommended_2019['2019']
recommended = recommended.sort_values(by='2021', ascending = False)

# + [markdown] id="krWXDRHFbMz3" papermill={"duration": 0.039691, "end_time": "2021-11-28T21:24:41.996028", "exception": false, "start_time": "2021-11-28T21:24:41.956337", "status": "completed"} tags=[]
# ***
#
# <center><h1 style="color:#9d4edd;font-family:Bangers" id="chapter3"> <b style="color:#5a189a">Chapter 3:</b> The Adventure is Gaining Traction</h1></center>
#
# ***

# + [markdown] id="zqhuLUGQJdPN" papermill={"duration": 0.039202, "end_time": "2021-11-28T21:24:42.074958", "exception": false, "start_time": "2021-11-28T21:24:42.035756", "status": "completed"} tags=[]
# As the [old books of Zen](https://www.python.org/dev/peps/pep-0020/) said: "Simple is better than complex", so our heroines will try their best to keep her visualizations as simple as possible while still conveying the right information.

# + [markdown] id="ktxZflMwHGgg" papermill={"duration": 0.04075, "end_time": "2021-11-28T21:24:42.155439", "exception": false, "start_time": "2021-11-28T21:24:42.114689", "status": "completed"} tags=[]
# Their first finding in this adventure is related to the following: how many women took the survey in each year? Let's explore!

# + [markdown] papermill={"duration": 0.038553, "end_time": "2021-11-28T21:24:42.233338", "exception": false, "start_time": "2021-11-28T21:24:42.194785", "status": "completed"} tags=[]
# <h2 style="color:#9d4edd ;  font-family:Bangers">Did the number of women answering the survey increase, decrease or stay about the same?</h2>

# + _kg_hide-input=true jupyter={"source_hidden": true} papermill={"duration": 0.252347, "end_time": "2021-11-28T21:24:42.525900", "exception": false, "start_time": "2021-11-28T21:24:42.273553", "status": "completed"} tags=[]
survey_takers = []
data_files = [responses_2021_women, responses_2020_women, responses_2019_women]

for i, data_files in enumerate(data_files):
    survey_takers.append([(2021-i), len(data_files)])
survey_takers = pd.DataFrame(survey_takers, columns=['year', 'value'])

fig = go.Figure(go.Bar (x = survey_takers['year'].sort_values(ascending = True),
                        y = survey_takers['value'].sort_values(ascending = True),
                        marker=dict(color=colors, line=dict(color='black', width=2))))

fig.update_layout(
    title="""How Many Women Took The Survey?
            <br>Or the <i><b style="color:#5a189a">Heroine Ascension</b></i>""",
    title_font_size=22,
    font_size=13,
    plot_bgcolor='white',
    legend_x=0.99, legend_xanchor='right', legend_font_size=18,
    hovermode='x',
    bargap=0.1,
    width=800, height=500)

fig.update_xaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black',
    type = 'category', title = 'Year')
fig.update_yaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black',
    title = 'Number of Women')

fig.add_layout_image(
    dict(
        source="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/logo_data.png",
        xref="paper", yref="paper",
        x=1.13, y=-0.25,
        sizex=0.3, sizey=0.3,
        xanchor="right", yanchor="bottom"))

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.13, y=-0.3,
    text="""<i style='color:#A9A9A9'>Source: Kaggle Survey from 2019 to 2021</i>""",
    align='left', arrowcolor='white')

fig.show()

# + [markdown] id="lteRGTXwHytK" papermill={"duration": 0.042852, "end_time": "2021-11-28T21:24:42.629643", "exception": false, "start_time": "2021-11-28T21:24:42.586791", "status": "completed"} tags=[]
# From this, our heroine sees that in 2019 and 2020 the number of people stayed relatively the same, with an increase in the number of survey takers in 2021. What could have happened to influece that?
#
# Going forward, they want to compare the salary distribution against the years. Their adventure is picking up steam!

# + [markdown] id="-JONdUg1DqKk" papermill={"duration": 0.050991, "end_time": "2021-11-28T21:24:42.730501", "exception": false, "start_time": "2021-11-28T21:24:42.679510", "status": "completed"} tags=[]
# <h2 style="color:#9d4edd ;  font-family:Bangers">Did we have a significant change in the distribution of women per country in the dataset?</h2>

# + _kg_hide-input=true id="xH29Mpp5E66C" jupyter={"source_hidden": true} outputId="c2bbdf6c-d8a5-4414-b4ab-a259e3b12a02" papermill={"duration": 0.093354, "end_time": "2021-11-28T21:24:42.866989", "exception": false, "start_time": "2021-11-28T21:24:42.773635", "status": "completed"} tags=[]
fig = go.Figure(data=[
    go.Bar(name='2019', x=countries.Country, y=countries['2019'],
           marker=dict(color=colors[0], line=dict(color='black', width=2))),
    go.Bar(name='2020', x=countries.Country, y=countries['2020'], 
           marker=dict(color=colors[1], line=dict(color='black', width=2))),
    go.Bar(name='2021', x=countries.Country, y=countries['2021'],
           marker=dict(color=colors[2], line=dict(color='black', width=2)))
])

fig.update_layout(
    title="""Where Are All The Women at?
            <br>Or the <i><b style="color:#5a189a">The World is Female</b></i>""",
    title_font_size=22,
    font_size=13,
    plot_bgcolor='white',
    legend_x=0.99, legend_xanchor='right', legend_font_size=18,
    hovermode='x',
    bargap=0.15,
    width=800, height=500
)

fig.update_xaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black',
    title='Country')
fig.update_yaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black',
    title='Number of Women')

fig.add_layout_image(
    dict(
        source="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/logo_data.png",
        xref="paper", yref="paper",
        x=1.13, y=-0.25,
        sizex=0.3, sizey=0.3,
        xanchor="right", yanchor="bottom"))

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.11, y=-0.3,
    text="""<i style='color:#A9A9A9'>Source: Kaggle Survey from 2019 to 2021</i>""",
    align='left', arrowcolor='white')

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.17, y=-0.24,
    text="""<i style='color:#A9A9A9'>*United Kingdom of Great Britain and Northern Ireland</i>""",
    align='left', arrowcolor='white')

fig.show()

# + [markdown] papermill={"duration": 0.041113, "end_time": "2021-11-28T21:24:42.972702", "exception": false, "start_time": "2021-11-28T21:24:42.931589", "status": "completed"} tags=[]
# <span style="color:#51B14B">Machina Learnerum</span> notes that:
#
# <blockquote>
#     When we look at women working with data by country, it’s possible to observe that most women are from India and secondly from the United States. It seems like India has a progressive growth between 2019 and 2021. In India the number of women employed in the Information technology (IT) industry has seen a rapid increase over the past 10 years, with more than 30% of employees now being female. On the other hand, the stagnation or decline in the participation of women in IT in many Western countries is happening. According to NASSCOM's Women and IT Scorecard – India, a study, undertaken with the UK's Open University, has shown that in India women represented 46.8% of the postgraduates in IT and computing during the academic year 2014-2015. Surprisingly, this is more than double the rate seen in the UK! 😯
# </blockquote>
#
# "Interesting findings!", the girls said in unison.

# + [markdown] id="Ld0Ns7hNu-bO" papermill={"duration": 0.041106, "end_time": "2021-11-28T21:24:43.055716", "exception": false, "start_time": "2021-11-28T21:24:43.014610", "status": "completed"} tags=[]
# <h2 style="color:#9d4edd ;  font-family:Bangers">What about salaries, anything we should look out for?</h2>

# + _kg_hide-input=true _kg_hide-output=false id="BcKTLiU691oP" jupyter={"source_hidden": true} outputId="43e12e6f-cfff-435f-e080-fa457996fc53" papermill={"duration": 0.108514, "end_time": "2021-11-28T21:24:43.206952", "exception": false, "start_time": "2021-11-28T21:24:43.098438", "status": "completed"} tags=[]
fig = go.Figure(data=[
    go.Bar(name='2019', x=salaries.Salary, y=salaries['2019'],
           marker=dict(color=colors[0], line=dict(color='black', width=2))),
    go.Bar(name='2020', x=salaries.Salary, y=salaries['2020'],
           marker=dict(color=colors[1], line=dict(color='black', width=2))),
    go.Bar(name='2021', x=salaries.Salary, y=salaries['2021'],
           marker=dict(color=colors[2], line=dict(color='black', width=2)))
])

fig.update_layout(
    title="""Salary Range for Women in the Data Industry
            <br>Or the <i><b style="color:#5a189a">Heroine's Salary</b></i>""",
    title_font_size=22,
    font_size=13,
    plot_bgcolor='white',
    legend_x=0.99, legend_xanchor='right', legend_font_size=18,
    hovermode='x',
    bargap=0.1,
    width=800, height=500)

fig.update_xaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black', title = 'Salary Range')
fig.update_yaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black', title = 'Count')

fig.add_layout_image(
    dict(
        source="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/logo_data.png",
        xref="paper", yref="paper",
        x=1.13, y=-0.25,
        sizex=0.3, sizey=0.3,
        xanchor="right", yanchor="bottom"))

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.13, y=-0.3,
    text="""<i style='color:#A9A9A9'>Source: Kaggle Survey from 2019 to 2021</i>""",
    align='left', arrowcolor='white')

fig.show()

# + [markdown] papermill={"duration": 0.046575, "end_time": "2021-11-28T21:24:43.330738", "exception": false, "start_time": "2021-11-28T21:24:43.284163", "status": "completed"} tags=[]
# The girls begin to wonder and ponder about this chart:
#
# <blockquote>
#     Unfortunately, in the Data Industry, most women earn the lowest salary range:  0 dollars to 49,999 dollars per year. However, in 2021 a considerable increase can be observed. It is important to highlight that according to the U.S. Department of Labor, in 2019, women’s annual earnings were 82.3% of men’s, and this gap was even wider for women of color. Black women were paid 63 percent of what non-Hispanic white men were paid, according to the U.S. Census. So, when will this pay gap change?
#     </blockquote>

# + [markdown] id="1qaxc2YjHvmI" papermill={"duration": 0.042022, "end_time": "2021-11-28T21:24:43.418468", "exception": false, "start_time": "2021-11-28T21:24:43.376446", "status": "completed"} tags=[]
# <h2 style="color:#9d4edd ;  font-family:Bangers">What job titles do women hold? And education level?</h2>

# + [markdown] papermill={"duration": 0.041112, "end_time": "2021-11-28T21:24:43.501038", "exception": false, "start_time": "2021-11-28T21:24:43.459926", "status": "completed"} tags=[]
# "What do the initials below mean?"
#
# <table >
#   <tr>
#     <th style="color:#9d4edd ;  font-family:Bangers; font-size: 25px"><center>Initials</center></th>
#     <th style="color:#9d4edd ;  font-family:Bangers; font-size: 25px"><center>Meaning</center></th>
#   </tr>
#   <tr>
#       <td><center><b style="font-size: 15px">DS</b></center></td>
#         <td><center>Data Scientist</center></td>
#   </tr>
#   <tr>
#     <td><center><b style="font-size: 15px">DA</b></center></td>
#     <td><center>Data Analyst</center></td>
#   </tr>
#    <tr>
#     <td><center><b style="font-size: 15px">SE</b></center></td>
#     <td><center>Software Engineer</center></td>
#   </tr>
#    <tr>
#     <td><center><b style="font-size: 15px">RS</b></center></td>
#     <td><center>Research Scientist</center></td>
#   </tr>
#   <tr>
#     <td><center><b style="font-size: 15px">MLE</b></center></td>
#     <td><center>Machine Learning Engineer</center></td>
#   </tr>
#   <tr>
#     <td><center><b style="font-size: 15px">BA</b></center></td>
#     <td><center>Business Analyst</center></td>
#   </tr>
#   <tr>
#     <td><center><b style="font-size: 15px">DE</b></center></td>
#     <td><center>Data Engineer</td>
#   </tr>
#   <tr>
#     <td><center><b style="font-size: 15px">PM</b></center></td>
#     <td><center>Product/Project/Program Manager</center></td>
#   </tr>
#   <tr>
#     <td><center><b style="font-size: 15px">Stats</b></center></td>
#     <td><center>Statistician</center></td>
#   </tr>
#   <tr>
#     <td><center><b style="font-size: 15px">DBA</b></center></td>
#     <td><center>DBA/Database Engineer</center></td>
#   </tr>
#   <tr>
#     <td><center><b style="font-size: 15px">DR/A</b></center></td>
#     <td><center>Developer Relations/Advocacy</center></td>
#   </tr>
# </table>

# + _kg_hide-input=true id="DSINuvyGH-ZP" jupyter={"source_hidden": true} outputId="0c6b1428-84f6-4133-f360-19a416a890cc" papermill={"duration": 0.077135, "end_time": "2021-11-28T21:24:43.619334", "exception": false, "start_time": "2021-11-28T21:24:43.542199", "status": "completed"} tags=[]
fig = go.Figure(data=[
    go.Bar(name='2019', x=professions.Job_Title, y=professions['2019'],
           marker=dict(color=colors[0], line=dict(color='black', width=2))),
    go.Bar(name='2020', x=professions.Job_Title, y=professions['2020'],
           marker=dict(color=colors[1], line=dict(color='black', width=2))),
    go.Bar(name='2021', x=professions.Job_Title, y=professions['2021'],
           marker=dict(color=colors[2], line=dict( color='black', width=2)))
])

fig.update_layout(
    title="""Job Title Held by Women from 2019 to 2021
            <br>Or the <i><b style="color:#5a189a">Hard-Working Heroine</b></i>""",
    title_font_size=22,
    font_size=13,
    plot_bgcolor='white',
    legend_x=0.99, legend_xanchor='right', legend_font_size=18,
    hovermode='x',
    bargap=0.15,
    width = 800, height=500
)

fig.update_xaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black', title = 'Job Title')
fig.update_yaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black', title = 'Count')

fig.add_layout_image(
    dict(
        source="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/logo_data.png",
        xref="paper", yref="paper",
        x=1.13, y=-0.25,
        sizex=0.3, sizey=0.3,
        xanchor="right", yanchor="bottom"))

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.13, y=-0.3,
    text="""<i style='color:#A9A9A9'>Source: Kaggle Survey from 2019 to 2021</i>""",
    align='left', arrowcolor='white')

fig.show()

# + [markdown] papermill={"duration": 0.041598, "end_time": "2021-11-28T21:24:43.703591", "exception": false, "start_time": "2021-11-28T21:24:43.661993", "status": "completed"} tags=[]
# <span style="color:#C40028">Datana Scientistus</span> says happilly:
#
# <blockquote>
#     Data science is booming!!! Most women work as Data Scientists and Data Analysts. As we can see, in the graph, in 2021 there was a significant increase in both positions. Back in 2014, Harvard Business Review dubbed data science as “the sexiest job of the 21st century.” Some years later, in a 2020 report, global management consulting firm BCG reported that women make up just 15-22% of the workforce in data science. Finally, in 2021, Glassdoor ranked it number two for best jobs, with a median base salary of US$113,736.
#     </blockquote>
#     
# <span style="color:#2667C3">Datana Analystus</span> completed:
#
# <blockquote>
# All of this makes this field very enticing!
# </blockquote>

# + _kg_hide-input=true id="XdIBL8ojV1u_" jupyter={"source_hidden": true} outputId="037de9ad-f27f-4ec6-db3d-9a06056267e9" papermill={"duration": 0.076663, "end_time": "2021-11-28T21:24:43.822069", "exception": false, "start_time": "2021-11-28T21:24:43.745406", "status": "completed"} tags=[]
fig = go.Figure(data=[
    go.Bar(name='2019', x=education.Education, y=education['2019'],
           marker=dict(color=colors[0], line=dict(color='black', width=2))),
    go.Bar(name='2020', x=education.Education, y=education['2020'],
           marker=dict(color=colors[1], line=dict(color='black', width=2))),
    go.Bar(name='2021', x=education.Education, y=education['2021'],
           marker=dict(color=colors[2], line=dict(color='black', width=2)))
])

fig.update_layout(
    title="""Education by Women from 2019 to 2021
            <br>Or the <i><b style="color:#5a189a">Educated Heroine</b></i>""",
    title_font_size=22,
    font_size=13,
    plot_bgcolor='white',
    legend_x=0.99, legend_xanchor='right', legend_font_size=18,
    hovermode='x',
    bargap=0.15,
    width=800, height=500)

fig.update_xaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black', title='Degree')
fig.update_yaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black', title='Count')

fig.add_layout_image(
    dict(
        source="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/logo_data.png",
        xref="paper", yref="paper",
        x=1.13, y=-0.25,
        sizex=0.3, sizey=0.3,
        xanchor="right", yanchor="bottom"))

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.13, y=-0.29,
    text="""<i style='color:#A9A9A9'>*Professional Doctorate</i>
            <br><i style='color:#A9A9A9'>Source: Kaggle Survey from 2019 to 2021</i>""",
    align='left', arrowcolor='white')

fig.show()

# + [markdown] papermill={"duration": 0.042049, "end_time": "2021-11-28T21:24:43.906458", "exception": false, "start_time": "2021-11-28T21:24:43.864409", "status": "completed"} tags=[]
# <span style="color:#0393E3">Datana Enginerus</span> discuss that:
#
# <blockquote>
# When we check out the education by women from 2019 to 2021, there is a pleasant surprise! Most women have a Master's degree. In second place we have women with a Bachelor's degree and in third are women with a Doctorate degree. In both degrees, there is an increase in 2021.
# <br>    
# <br>
# It is important to highlight that a report about women in higher education, launched in 2021 by the UNESCO International Institute for Higher Education in Latin America and the Caribbean (IESALC), reveals that women’s educational attainment have soared: tripled globally between 1995 and 2018! In addition, the female’s enrollment in higher education occurred in 74% of the countries with data as well as in all regions, women are overrepresented!!!     
# </blockquote>
#    
#    "So women are highly educated and have lower wages in this industry? Something's not right!", they all agreed.

# + [markdown] id="NLjJkTW1LFHR" papermill={"duration": 0.043614, "end_time": "2021-11-28T21:24:43.994623", "exception": false, "start_time": "2021-11-28T21:24:43.951009", "status": "completed"} tags=[]
# <h2 style="color:#9d4edd ;  font-family:Bangers">Do women tend to work in big or small companies?</h2>

# + _kg_hide-input=true id="NsUoUmf1L3Tq" jupyter={"source_hidden": true} outputId="5190c8f7-f174-443b-b8aa-15748c1b4b09" papermill={"duration": 0.075296, "end_time": "2021-11-28T21:24:44.114301", "exception": false, "start_time": "2021-11-28T21:24:44.039005", "status": "completed"} tags=[]
fig = go.Figure(data=[
    go.Bar(name='2019', x=employees.Employees, y=employees['2019'],
           marker=dict(color=colors[0], line=dict(color='black', width=2))),
    go.Bar(name='2020', x=employees.Employees, y=employees['2020'],
           marker=dict(color=colors[1], line=dict(color='black', width=2))),
    go.Bar(name='2021', x=employees.Employees, y=employees['2021'],
           marker=dict(color=colors[2], line=dict(color='black', width=2)))
])

fig.update_layout(
    title="""Big or Small Companies? Where do Most Women Work at?
            <br>Or the <i><b style="color:#5a189a">Working Class Heroine</b></i>""",
    title_font_size=22,
    font_size=13,
    plot_bgcolor='white',
    legend_x=0.99, legend_xanchor='right', legend_font_size=18,
    hovermode='x',
    bargap=0.15,
    width=800, height=500
)

fig.update_xaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black',
    title='Number of Employees')
fig.update_yaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black',
    title='Count')

fig.add_layout_image(
    dict(
        source="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/logo_data.png",
        xref="paper", yref="paper",
        x=1.13, y=-0.25,
        sizex=0.3, sizey=0.3,
        xanchor="right", yanchor="bottom"))

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.13, y=-0.3,
    text="""<i style='color:#A9A9A9'>Source: Kaggle Survey from 2019 to 2021</i>""",
    align='left', arrowcolor='white')

fig.show()

# + [markdown] papermill={"duration": 0.043252, "end_time": "2021-11-28T21:24:44.201311", "exception": false, "start_time": "2021-11-28T21:24:44.158059", "status": "completed"} tags=[]
# <span style="color:#51B14B">Machina Learnerum</span>, seeing this chart, observes that:
#
# <blockquote>
# Most women work in small companies (0-49 employees). As you can see in 2021 we have an increase across all sizes of business. 
# Surprisingly, according to Fobers, based on recruitment data, the companies appear to be actually hiring proportionally more women per qualified candidate than men, in the IT area.
# </blockquote>
#
# "Let's hope it keeps going that way!", the girls said hopeful.

# + [markdown] id="Ss71-YQTK5IB" papermill={"duration": 0.043751, "end_time": "2021-11-28T21:24:44.288055", "exception": false, "start_time": "2021-11-28T21:24:44.244304", "status": "completed"} tags=[]
# <h2 style="color:#9d4edd ;  font-family:Bangers">What are we going to find out about learning plataforms?</h2>

# + _kg_hide-input=true id="torf5Rl-V0AR" jupyter={"source_hidden": true} outputId="4beeba4f-0414-4602-d6f1-2c98e33bfb8d" papermill={"duration": 0.074909, "end_time": "2021-11-28T21:24:44.405850", "exception": false, "start_time": "2021-11-28T21:24:44.330941", "status": "completed"} tags=[]
fig = go.Figure(data=[
    go.Bar(name='2019', x=learning_platforms.Platforms, y=learning_platforms['2019'],
           marker=dict(color=colors[0], line=dict(color='black', width=2))),
    go.Bar(name='2020', x=learning_platforms.Platforms, y=learning_platforms['2020'],
           marker=dict(color=colors[1], line=dict(color='black', width=2))),
    go.Bar(name='2021', x=learning_platforms.Platforms, y=learning_platforms['2021'],
            marker=dict(color=colors[2], line=dict(color='black', width=2)))
])

fig.update_layout(
    title="""Learning Platforms Used by Women from 2019 to 2021
        <br>Or the <i><b style="color:#5a189a">Intelligent Heroine</b></i>""",
    title_font_size=22,
    font_size=13,
    plot_bgcolor='white',
    legend_x=0.99, legend_xanchor='right', legend_font_size=18,
    hovermode='x',
    bargap=0.15,
    width=800, height=500)

fig.update_xaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black',
    title='Platform')
fig.update_yaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black',
    title='Count') 

fig.add_layout_image(
    dict(
        source="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/logo_data.png",
        xref="paper", yref="paper",
        x=1.13, y=-0.25,
        sizex=0.3, sizey=0.3,
        xanchor="right", yanchor="bottom"))

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.13, y=-0.3,
    text="""<i style='color:#A9A9A9'>Source: Kaggle Survey from 2019 to 2021</i>""",
    align='left', arrowcolor='white')

fig.show()

# + [markdown] papermill={"duration": 0.043224, "end_time": "2021-11-28T21:24:44.492748", "exception": false, "start_time": "2021-11-28T21:24:44.449524", "status": "completed"} tags=[]
# <span style="color:#C40028">Datana Scientistus</span> stopped for a second to think about this chart and concluded that:
#
# <blockquote>
#     When we discuss learning platforms used by women from 2019 to 2021: Coursera is the most used! In second place we have Kaggle, and Udemy in third. Probably, Coursera is the most used because of the fact that many programs charge a monthly fee, so the faster you finish, the more money you save. Besides that, financial assistance is available in a bunch of courses and this platform offers some free certification courses. And, of course, the available courses are from the best universities in the world. Meanwhile, Kaggle offers a no-setup, customizable Jupyter Notebook environment. You can access free GPUs and a huge repository of community published data and code. In addition, it is possible to work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.
# </blockquote>
#
# "Those are very interesting points about the platforms", <span style="color:#51B14B">Machina Learnerum</span> added.

# + [markdown] id="1lm2bOu_6wyW" papermill={"duration": 0.042993, "end_time": "2021-11-28T21:24:44.579277", "exception": false, "start_time": "2021-11-28T21:24:44.536284", "status": "completed"} tags=[]
# <h2 style="color:#9d4edd ;  font-family:Bangers">And about the most used and the most recommended languages? I've heard rumors about some kind of Python's supremacy.</h2>

# + _kg_hide-input=true id="2TPskje531eL" jupyter={"source_hidden": true} outputId="b5d84de4-a06b-4519-85e0-645318265262" papermill={"duration": 0.075043, "end_time": "2021-11-28T21:24:44.697607", "exception": false, "start_time": "2021-11-28T21:24:44.622564", "status": "completed"} tags=[]
fig = go.Figure(data=[
    go.Bar(name='2019', x=language_use.Programming_Language, y=language_use['2019'],
           marker=dict(color=colors[0], line=dict(color='black', width=2))),
    go.Bar(name='2020', x=language_use.Programming_Language, y=language_use['2020'],
           marker=dict(color=colors[1], line=dict(color='black', width=2))),
    go.Bar(name='2021', x=language_use.Programming_Language, y=language_use['2021'],
           marker=dict(color=colors[2], line=dict(color='black', width=2)))
])

fig.update_layout(
    title="""Most Used Languages by Women from 2019 to 2021
            <br>Or the <i><b style="color:#5a189a">Python Supremacy</b></i>""",
    title_font_size=22,
    font_size=13,
    plot_bgcolor='white',
    legend_x=0.99, legend_xanchor='right', legend_font_size=18,
    hovermode='x',
    bargap=0.15,
    width=800, height=500)

fig.update_xaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black',
    title= 'Language')
fig.update_yaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black',
    title= 'Count')

fig.add_layout_image(
    dict(
        source="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/logo_data.png",
        xref="paper", yref="paper",
        x=1.13, y=-0.25,
        sizex=0.3, sizey=0.3,
        xanchor="right", yanchor="bottom"))

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.13, y=-0.3,
    text="""<i style='color:#A9A9A9'>Source: Kaggle Survey from 2019 to 2021</i>""",
    align='left', arrowcolor='white')

fig.show()

# + [markdown] papermill={"duration": 0.044468, "end_time": "2021-11-28T21:24:44.786671", "exception": false, "start_time": "2021-11-28T21:24:44.742203", "status": "completed"} tags=[]
# <span style="color:#2667C3">Datana Analystus</span> says excitedly:
#
# <blockquote>
# What a supremacy, huh? Python was the most used language by women from 2019 to 2021.  Python is the second-most-popular tool, followed by SQL and R language when it comes to data science and analytics. Python is one of the languages that is witnessing incredible growth and popularity year by year. In 2017, StackOverflow calculated that Python would beat all other programming languages by 2020 as it has become the fastest-growing programming language in the world.
#  </blockquote>
#
# <blockquote> "Why is Python so popular? Do any of you girls know?", <span style="color:#C40028">Datana Scientistus</span> instigated.</blockquote>
#          
# <span style="color:#51B14B">Machina Learnerum</span> was ready to answer:
#          
#  <blockquote>
# The reasons for this supremacy:
# Python continues to impress its new users and entry-level programmers with ease of functions, because it is a very simple language to understand;
# It is bound to be popular with multiple applications across different fields. Which fields? Data science, web development, systems automation and administration, mapping and geography, mathematical computing, finance and trading, game development, application scripting… 
# Furthermore, Python has so many excellent libraries (SciPy, Sckit, Tensor flow, Numpy, Pandas, Flask, Beautiful Soup…) that are able to boost development. 
# </blockquote>
#
#

# + _kg_hide-input=true id="DXdyNf9T0ocy" jupyter={"source_hidden": true} outputId="03d12876-6844-440f-8fe9-37415bf81e48" papermill={"duration": 0.077773, "end_time": "2021-11-28T21:24:44.908963", "exception": false, "start_time": "2021-11-28T21:24:44.831190", "status": "completed"} tags=[]
fig = go.Figure(data=[
    go.Bar(name='2019', x=recommended.Language, y=recommended['2019'],
           marker=dict(color=colors[0], line=dict(color='black', width=2))),
    go.Bar(name='2020', x=recommended.Language, y=recommended['2020'],
           marker=dict(color=colors[1], line=dict(color='black', width=2))),
    go.Bar(name='2021', x=recommended.Language, y=recommended['2021'],
          marker=dict(color=colors[2], line=dict(color='black', width=2)))
])

fig.update_layout(
    title="""Most Recommended Languages by Women from 2019 to 2021
            <br>Or the <i><b style="color:#5a189a">Python Supremacy, Part II</b></i>""",
    title_font_size=22,
    font_size=13,
    plot_bgcolor='white',
    legend_x=0.99, legend_xanchor='right', legend_font_size=18,
    hovermode='x',
    bargap=0.15,
    width=800, height=500)
fig.update_xaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black',
    title='Language')
fig.update_yaxes(
    mirror=True, ticks='outside', showline=True, linewidth=3, linecolor='black',
    title='Count')

fig.add_layout_image(
    dict(
        source="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/logo_data.png",
        xref="paper", yref="paper",
        x=1.13, y=-0.25,
        sizex=0.3, sizey=0.3,
        xanchor="right", yanchor="bottom"))

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.13, y=-0.3,
    text="""<i style='color:#A9A9A9'>Source: Kaggle Survey from 2019 to 2021</i>""",
    align='left', arrowcolor='white')

fig.show()

# + [markdown] papermill={"duration": 0.046296, "end_time": "2021-11-28T21:24:45.001117", "exception": false, "start_time": "2021-11-28T21:24:44.954821", "status": "completed"} tags=[]
# <span style="color:#0393E3">Datana Enginerus</span> was ready to conclude that:
#
# <blockquote>
# Since the above question was not a multiple choice one, we can really see how much does Python matter for the data world. And no surprises here: the most recommended languages are fully linked to the languages that women most used between 2019 and 2021. So, again we can appreciate the supremacy of Python! Well, previously all the reasons for this language to be so used were highlighted, so it's evident: <b style="color:#9d4edd">GO LEARN PYTHON!</b>
# </blockquote>
#
# The heroines all laughed at this 😂

# + [markdown] id="Ju8eRpXBfR1n" papermill={"duration": 0.044771, "end_time": "2021-11-28T21:24:45.090610", "exception": false, "start_time": "2021-11-28T21:24:45.045839", "status": "completed"} tags=[]
# ***
# <center>
# <h1 style="color:#9d4edd;font-family:Bangers" id="chapter4"><b style="color:#5a189a">Chapter 4:</b> The Enemy Straight Ahead</h1></center>
#
# ***

# + [markdown] id="SnYZS93xIRu0" papermill={"duration": 0.044499, "end_time": "2021-11-28T21:24:45.179909", "exception": false, "start_time": "2021-11-28T21:24:45.135410", "status": "completed"} tags=[]
# Now, to wrap it up, our heroines set out to look at how Covid-19, humanity's greatest enemy, affects our world and to see if it had any impact on the data observed since 2019 BC (Before Covid). Or is it too soon to tell?
#
# First, <span style="color:#C40028">Datana Scientistus</span> takes a look at the evolution of Covid-19 through time: OH NO!!!

# + [markdown] papermill={"duration": 0.045186, "end_time": "2021-11-28T21:24:45.272159", "exception": false, "start_time": "2021-11-28T21:24:45.226973", "status": "completed"} tags=[]
# <p><center>
#   <img width="800" src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/Screenshot_1.png" alt="">
#     </center></p>
#     
# <center><cite>Source: <a href="https://covid19.who.int/">WHO Coronavirus (COVID-19) Dashboard</a></cite></center>

# + [markdown] papermill={"duration": 0.044756, "end_time": "2021-11-28T21:24:45.363749", "exception": false, "start_time": "2021-11-28T21:24:45.318993", "status": "completed"} tags=[]
# From this, she sees that indeed the impact it has in the world is gigantic. Overnight, we had to adjust our whole lives to fit in this new scenario. Can we say that the increase in the number of people working from home was a major factor for the increase of women in the data workspace? Maybe [5], but from the data alone it's hard to draw a conclusion.
#
# Will this enemy ever be defeated?
#
# <p><center>
#   <img width="800" src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/Screenshot_2.png" alt="">
#     </center></p>
#     
# <center><cite>Source: <a href="https://www.bloomberg.com/graphics/covid-vaccine-tracker-global-distribution/">More Than 7.9 Billion Shots
# Given: Covid-19 Tracker</a></cite></center>
#
# <span style="color:#51B14B">Machina Learnerum</span> thinks so! With the increase in the number of vaccinated people, the light at the end of the tunnel is looking brighter everyday.

# + [markdown] papermill={"duration": 0.045491, "end_time": "2021-11-28T21:24:45.457250", "exception": false, "start_time": "2021-11-28T21:24:45.411759", "status": "completed"} tags=[]
# ***
# <center>
# <h1 style="color:#9d4edd ;  font-family:Bangers" id="end"><b style="color:#5a189a">Epilogue:</b> All's Well That Ends Well</h1></center>
#
# ***

# + [markdown] papermill={"duration": 0.044497, "end_time": "2021-11-28T21:24:45.546672", "exception": false, "start_time": "2021-11-28T21:24:45.502175", "status": "completed"} tags=[]
# 🦸‍♀️ After their great and fullfiling adventure, our heroines were ready to confirm: <b style="color:#9d4edd">Data Girls Rule the World!</b>
#
# 🦸‍♀️ And it is evident that women in the data industry should be more well compensated, since they are highly educated .
#
# 🦸‍♀️ All of this shows that we have a lot to improve, but we are slowly getting there!
#
# 🦸‍♀️ Even after the increase in the Covid-19 cases and deaths, it's easy to see that women never gave up and the number of women in the data industry is increasing each year, so let's congratulate all our amazing heroines in this world!
#
# 🦸‍♀️ Now our heroines will rest after such a extensive adventure that they could all learn something from!

# + [markdown] papermill={"duration": 0.044655, "end_time": "2021-11-28T21:24:45.637312", "exception": false, "start_time": "2021-11-28T21:24:45.592657", "status": "completed"} tags=[]
# <p><center>
#   <img width="800" src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/thanks2.PNG" alt="">
#     </center></p>

# + [markdown] papermill={"duration": 0.045452, "end_time": "2021-11-28T21:24:45.727873", "exception": false, "start_time": "2021-11-28T21:24:45.682421", "status": "completed"} tags=[]
# ***
#
# <center>
# <h1 style="color:#5a189a;font-family:Bangers" id="references">References</h1></center>
#
# ***

# + [markdown] papermill={"duration": 0.045196, "end_time": "2021-11-28T21:24:45.819333", "exception": false, "start_time": "2021-11-28T21:24:45.774137", "status": "completed"} tags=[]
# - Images from:
#
#     https://br.freepik.com/vetores-gratis/superhero-popular-character-isometric-icons-set_4268120.htm#page=1&position=3&from_view=detail#&position=3&from_view=detail
#     
#     https://br.freepik.com/vetores-gratis/conjunto-de-icones-de-acoes-super-mulher_4265872.htm#page=1&query=hero%20female&position=1&from_view=search
#     
# - [1]  SCHIEBINGER, L. O feminismo mudou a ciência? São Paulo: EDUSC, 2001.
# - [2] MARÇULA, M.; BENINI, F. Pio Armando. Informática – Conceitos e aplicações. São Paulo: Editora Érica, 2014.
# - [3] SILVA, F.; RIBEIRO, P. A inserção das mulheres na ciência. Revista Linhas Críticas, Brasília, v. 18, n. 35, p. 171-191, 2012.
# - [4] BROWN, K,V. More women in Computer Science classes. Available at: http://www.sfgate.com/education/article/Tech-shift-More-women-in-computerscience-classes-5243026.php#page-2
# - [5] [Future of Work: Is the new normal of ‘Working from Home’ a boon for women?](https://yourstory.com/2021/03/future-of-work-new-normal-working-women-in-tech-advantage/amp)
# - [It's 2021 and women STILL make 82 cents for every dollar earned by a man](https://www.nbcnews.com/know-your-value/feature/it-s-2021-women-still-make-82-cents-every-dollar-ncna1261755)
# - [Women in Tech: India Leads the Way](https://go.451research.com/women-in-tech-india-employment-trends.html)
# - [Data science is booming. So where are the women?](https://www.theglobeandmail.com/business/article-data-science-is-booming-so-where-are-the-women/)
# - [These Programming Languages Have The Most (And Fewest) Female Coders](https://www.forbes.com/sites/alexkonrad/2014/08/12/these-programming-languages-have-the-most-and-fewest-female-coders/?sh=3afeed09355c)
# - [UNESCO IESALC report asserts that gender inequality in higher education remains a universal issue](https://www.iesalc.unesco.org/en/2021/03/08/unesco-iesalc-report-asserts-that-gender-inequality-in-higher-education-remains-a-universal-issue/)
# - [StackOverflow: Developer survey results 2017](https://insights.stackoverflow.com/survey/2017)
#
#
#
#
#
#

# + [markdown] papermill={"duration": 0.046582, "end_time": "2021-11-28T21:24:45.911551", "exception": false, "start_time": "2021-11-28T21:24:45.864969", "status": "completed"} tags=[]
# ***
#
# <center>
# <h1 style="color:#5a189a;font-family:Bangers" id="authors">Authors</h1></center>
#
# ***

# + [markdown] papermill={"duration": 0.046039, "end_time": "2021-11-28T21:24:46.004242", "exception": false, "start_time": "2021-11-28T21:24:45.958203", "status": "completed"} tags=[]
# <center><div class="row">
#   <div class="col-md-4">
#     <div class="card">
#       <div class="card-body" style="width: 10rem;">
#           <a href="https://www.linkedin.com/in/marivaldotorres/"><img src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/Cah.png" class="card-img-top" alt="..."></a>
#         <center><h5 class="card-title" style="color:#9d4edd; font-family:Bangers">Carolina Dias</h5></center>
# <center><a href="https://www.linkedin.com/in/carodias/"><img src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/linkedin-logo.png" style="width: 20px;" alt="..."></a></center>
#       </div>
#     </div>
#   </div>
#   <div class="col-md-4">
#     <div class="card">
#       <div class="card-body" style="width: 10rem;">
#                    <a href="https://www.linkedin.com/in/marivaldotorres/"><img src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/juh.png" class="card-img-top" alt="..."></a>
#           <center><h5 class="card-title" style="color:#9d4edd; font-family:Bangers">Marivaldo Torres Junior</h5></center>
#         <center><a href="https://www.linkedin.com/in/marivaldotorres/"><img src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/linkedin-logo.png" style="width: 20px;" alt="..."></a></center>
#       </div>
#     </div>
#   </div>
#     <div class="col-md-4">
#     <div class="card">
#       <div class="card-body" style="width: 10rem;">
#        <a href="https://www.linkedin.com/in/marivaldotorres/"><img src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/val.png" class="card-img-top" alt="..."></a>
#           <center><h5 class="card-title" style="color:#9d4edd; font-family:Bangers">Valquíria Alencar</h5></center>
#                 <center><a href="https://www.linkedin.com/in/valquiria-alencar/"><img src="https://raw.githubusercontent.com/DadosNus/Kaggle-survey/main/img/linkedin-logo.png" style="width: 20px;" alt="..."></a></center>
#       </div>
#     </div>
#   </div>
# </div></center>

# + [markdown] papermill={"duration": 0.045158, "end_time": "2021-11-28T21:24:46.094367", "exception": false, "start_time": "2021-11-28T21:24:46.049209", "status": "completed"} tags=[]
# <center><a href="#top" style="color:#9d4edd ;  font-family:Bangers">To the top </a></center>
