import os
with open(os.path.join('C:/Users/farha/Documents/GitHub/TextSentimentAnalysis/','Procfile'), "w") as file1:
    toFile = 'web: sh setup.sh && streamlit run app.py'
    
file1.write(toFile)

# python heroku_run.py