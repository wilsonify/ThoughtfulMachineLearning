FROM public.ecr.aws/lambda/python:3.10
RUN yum install libgomp -y
RUN yum groupinstall "Development Tools" -y
COPY . /var/task
RUN pip install pycaret
RUN pip install shap>=0.38.0
RUN pip install interpret>=0.2.7
RUN pip install umap-learn>=0.5.2
RUN pip install pandas-profiling>=3.1.0
RUN pip install explainerdashboard>=0.3.8
RUN pip install Flask==2.2.3
RUN pip install bokeh<3.0.0
RUN pip install autoviz>=0.1.36
RUN pip install fairlearn==0.7.0
RUN pip install deepchecks>=0.9.2
RUN pip install xgboost>=1.1.0
RUN pip install catboost>=0.23.2
RUN pip install kmodes>=0.11.1
RUN pip install mlxtend>=0.19.0
RUN pip install statsforecast>=0.5.5
RUN pip install scikit-learn-intelex>=2023.0.1
RUN pip install tune-sklearn>=0.2.1
RUN pip install ray[tune]>=1.0.0
RUN pip install hyperopt>=0.2.7
RUN pip install optuna>=3.0.0
RUN pip install scikit-optimize>=0.9.0
RUN pip install mlflow>=1.24.0,<2.0.0
RUN pip install gradio>=2.8.10
RUN pip install boto3>=1.24.56
RUN pip install fastapi>=0.75.0
RUN pip install uvicorn>=0.17.6
RUN pip install m2cgen>=0.9.0
RUN pip install evidently>=0.1.45.dev0,<0.3
RUN pip install fugue>=0.8.0
RUN pip install flask>=2.0.3
RUN pip install matplotlib>=3.3.0
RUN pip install scikit-plot>=0.3.7
RUN pip install yellowbrick>=1.4
RUN pip install plotly>=5.0.0
RUN pip install kaleido>=0.2.1
RUN pip install schemdraw==0.15
RUN pip install plotly-resampler>=0.8.3.1
RUN pip install statsmodels>=0.12.1
RUN pip install sktime>=0.16.1,!=0.17.1,!=0.17.2,!=0.18.0
RUN pip install tbats>=1.1.3
RUN pip install pmdarima>=1.8.0,!=1.8.1,<3.0.0

EXPOSE 8080
CMD [ "s01_wine_scrape/__main__.lambda_handler" ]
