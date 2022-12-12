# Deploying a Retail Sales Prediction Model with Streamlit

![Preview](https://miro.medium.com/max/748/1*DvxQV-pqDKW2gFkuyWXvNw.png)

Preview of the app

Accurate sales prediction is crucial for any retail business, and machine learning can provide accurate forecasts to inform planning and decision making. In this article, I demonstrate how to use Streamlit, a great framework for building interactive web apps, to deploy a sales prediction model.

I will begin from exporting the relevant items to be used, setting up your environment, importing the items, building your interface, and completing the backend.

## 1.0 Introduction

**1.1 Streamlit  
**Streamlit is an open-source Python app framework for building and deploying web-based data science applications. It allows users to create and share interactive data analyses and machine learning tools without having to write complex web application code. It does all this with a  **_Python script._**

**1.2 Why deployment?  
**Deployment is the last stage of the CRISP-DM, the framework used to guide  [_this Retail Sales Prediction project_](https://github.com/KOdoi-OJ/Retail_Sales_Prediction_App_with_Streamlit). Building something as potentially useful as a sales prediction app and leaving it on your computer or as a repo on your GitHub limits its use. Because, what then was the benefit of building it? This is one of the reasons why the sales prediction model I built is being built into an app and made available for use by others.

# 2.0 The Process

**2.1 Workflow overview  
**As indicated earlier, the workflow may be summarized as follows:

-   Export ML items
-   Set up environment
-   Import ML items
-   Build interface
-   Set up the backend to process inputs and display outputs
-   Deploy

**2.2 Toolkit Export  
**The process begins with exporting the key items used during your modelling process from your notebook. The toolkit typically includes the encoder, scaler, model, and pipeline (if used). For ease of access, these items may be put together in a dictionary and exported. In this case, Pickle will be used for the exports, so it must first be imported.

# Import Pickle  
import pickle

The dictionary can then be created and exported with pickle as shown below;

![](https://miro.medium.com/max/496/1*8ViNBopOY5WcfIP-Ey8cJA.png)

Create a dictionary of ML items and export with Pickle

Note that the values in the dictionary are the names of the variables that represent my encoder, scaler, and model. And the names of the output file can be changed as desired.

Since your workflow likely used specific libraries and modules, they also have to be exported with the help of the OS library into a text file called  _requirements_:

# Import OS  
import os

After importing OS, you may then export the requirements with:

# Exporting the requirements  
requirements = "\n".join(f"{m.__name__}=={m.__version__}" for m in globals().values() if getattr(m, "__version__", None))  
  
with open("requirements.txt", "w") as f:  
    f.write(requirements)

Other things being equal, this should be the last major action you take in your notebook. Next up is VSCode.

**2.3 Setting up your environment  
**This step involves creating the folder or repository for your app. You may want to create a  _resources_ folder to hold the items you have exported from your notebook. The  _requirements_ file should be at the root of your repository or main folder.

To prevent any conflicts with your variables, you may use the following code to create a virtual environment, activate it in your terminal, and install the requirements in your  _requirements file_:

# Create and activate virtual environment  
python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt

**2.4 Importing ML Items  
**Be sure to switch to the workspace of the virtual environment. You’ll know it is active when you have  _(venv)_ preceding the path to your current working directory in the terminal.

Next is to define a function to load your ML items. In my case, I used the code below which has a default value for the file path:

# Function to import the Machine Learning toolkit  
@st.cache(almlow_output_mutation=True)  
def load_ml_toolkit(relative_path):  
    with open(relative_path, "rb") as file:  
        loaded_object = pickle.load(file)  
    return loaded_object

I’m sure you’ve notice the use of  _@st.cache_, it is used to store the function in cache so that the script doesn’t have to recreate the function each time a change is made.  
You may then instantiate the toolkit and each of the items you exported;

# Loading the toolkit  
loaded_toolkit = load_ml_toolkit(r"streamlit_src\ML_toolkit")  
  
# Instantiating the elements of the Machine Learning Toolkit  
mm_scaler = loaded_toolkit["scaler"]  
dt_model = loaded_toolkit["model"]  
oh_encoder = loaded_toolkit["encoder"]

**2.5 Building your interface  
**From there you build your interface using the components provided by Streamlit. The most common ones you’re most likely going to use are:

-   _st.container()_: to define a container (read box) to keep other components and keep your work organized
-   _st.columns(n)_: to define columns in your workspace. Replace n with the number of columns you want to create
-   _st.sidebar_: for a sidebar
-   _st.date_input()_: to receive date inputs
-   _st.selectbox()_: for a dropdown box
-   _st.number_input()_: for number inputs
-   _st.radio()_: for a radio
-   _st.checkbox()_: for a checkbox
-   _st.expander()_: for an expander
-   _st.form():_ to create a form to receive inputs from users

**2.6 Setting up the backend  
**After building the interface as you desire, you may then set up your backend to receive inputs, process them, and return outputs to the user. Here, the workflow must be same as in your notebook, and is typically:  _Inputs -> Encoding -> Scaling -> Predicting -> Returning predictions._

In my case, I received the inputs using Streamlit components and assigned them to variables. I then converted the variables into a dictionary, then a DataFrame, and processed them accordingly, as may be seen in the notebook. You can use  _st.write()_ at various stages to display the output of your app. Run the app using

> streamlit run “app_name.py”

_Change “app_name.py” to your app name and path._

To ease the effects of changes in real time, set the app to  **_“Always rerun”_** so that changes reflect in real time.

**2.7 Deployment  
**To deploy, visit  [_https://streamlit.io/cloud_](https://streamlit.io/cloud)_,_ sign in and connect your GitHub (if you haven’t). You can then select new app and the repo of the app for deployment.

# 3.0 Final Notes

Thank you for reading this far, I hope the article was helpful to you.