# Retail_Sales_Prediction_App_with_Streamlit

## Introduction

A retail sales prediction app based on a machine learning model and built with Streamlit. This project is more like a quest to discover how to embeded it into a web app with a user-friendly interface, in this case, [Streamlit](https://streamlit.io/). The objective is to have an interface that makes it easier for users to interact with an ML model, regardless of their level of knowledge in machine learning.

## Process Description

The process begins with exporting the necessary items from the notebook, building an interface that works correctly, importing the necessary items for modelling, and then writing the code to process inputs. The process can therefore be summarized as:

- Export machine learning items from notebook,
- Import the machine learning items into the app script,
- Build the interface,
- Write backend code to process inputs,
- Pass values through the interface,
- Recover these values in backend,
- Apply the necessary processing,
- Submit the processed values to the ML model to make the predictions,
- Process the predictions obtained and display them on the interface.

## Installation

To setup and run this project you need to have [`Python3`](https://www.python.org/) installed on your system. Then you can clone this repo. At the repo's root, use the code from below which applies:

- Windows:

        python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

- Linux & MacOs:

        python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

    **NB:** For MacOs users, please install `Xcode` if you have an issue.

You can then run the app (still at the repository root) with:

      streamlit run streamlit_project/basic_demo/app.py

Ideally, it should open your browser with the app in a new tab, if it doesn't, type this address in your browser:

      http://localhost:8501

## Screenshots

<table>
    <tr>
        <th> Streamlit Retail Prediction App </th>
        <th> App Interface with predictions </th>
    </tr>
    <tr>
        <td><img src= "screenshots\App_interface.png" /></td>
        <td><img src= "screenshots\App_with_prediction.png" /></td>
    </tr>
</table>

## Contact Information

- [Kwame Otchere](https://kodoi-oj.github.io/)
- [![Twitter](https://img.shields.io/twitter/follow/kwameoo_?style=social)](https://twitter.com/kwameoo_)
