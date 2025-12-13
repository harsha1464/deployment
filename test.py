import joblib
model = joblib.load('project/LogisticRegression_model.pkl')

prediction = model.predict([[1,45,354,476,5,6,7,8]])

print(prediction)