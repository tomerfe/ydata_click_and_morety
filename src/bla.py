import joblib

model = joblib.load(r"C:\Users\Golds\.cookiecutters\serj-ds-skeleton\{{cookiecutter.project_name}}\ydata_click_and_morety\models\meta_learner_with_features.joblib")
params = model.get_params()

for key, value in params.items():
    print(f"{key}: {value}")