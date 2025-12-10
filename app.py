{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "57f0ac63-f2dc-4189-9de1-00550601d16b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n",
    "@st.cache_resource\n",
    "def load_pipeline():\n",
    "# Load Model\n",
    "model = joblib.load(\"personality_model.pkl\")\n",
    " with open('final_feature_list.txt', 'r') as f:\n",
    "        feature_columns = [line.strip() for line in f if line.strip()]\n",
    "    return model, feature_columns\n",
    "\n",
    "model, feature_columns = load_pipeline\n",
    "\n",
    "# Streamlit Layout\n",
    "\n",
    "# Page settings\n",
    "st.set_page_config(page_title=\"Personality Prediction\", layout=\"centered\")\n",
    "st.title(\" Personality Type Prediction App\")\n",
    "st.markdown(\"Provide to predict if a person is an introvert or an extrovert\")\n",
    "\n",
    "#User Input Form\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
