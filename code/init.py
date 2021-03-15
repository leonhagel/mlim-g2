import Utils
from DataLoader import *
from FeatureCreator import *
from Model import *

config = Utils.read_json('../config.json')

# data loading
dataloader = DataLoader(config)
dataset = dataloader.get_dataset()


# feature creation
feature_creator = FeatureCreator(dataset, config)
model_data = feature_creator.get_model_data()


# train-test split and model fitting
model = Model(model_data)
X_train, y_train, X_test, y_test = model.train_test_split(config)
model.fit(X_train, y_train)


# generate predictions and compute score
y_hat = model.predict(X_train)
log_loss_score = model.log_loss_score(y_train, y_hat)


# generate output --> optimal coupons here!
print(f'log loss score is {log_loss_score}')