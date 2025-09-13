This is a GPT-style implementation for generating SQL from natural language.

Please download the dataset and results from here: https://drive.google.com/drive/folders/1-ovKYyDiTdEIJZuGSR337RbiGxF5Agvs?usp=drive_link
Place the data folder 'sql_data' in your folder, and you can run tinyGPT_v*.py and tinyGPT_v*_base.py to reproduce the results.

Due to the small scale of this model, it performs better if the input is of the format [context]+context+[question]+question.
In tinyGPT_v2.py, I also implemented a function "Natrual_language_chat" that can handle any arbitrary user input; however, the output usually is not very good.
