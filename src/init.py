import os
import Utils
from DataLoader import DataLoader
from FeatureCreator import FeatureCreator
from Model import Model
from CouponCreator import CouponCreator

# PREPARING THE ENVIRONMENT AND LOADING THE CONFIG

os.chdir('./src')
config = Utils.read_json('../config.json')
Utils.print_json(config)


# DATA LOADING
dataloader = DataLoader(config)
dataset = dataloader.get_dataset()


# FEATURE CREATION
feature_creator = FeatureCreator(dataset, config)
model_data = feature_creator.get_model_data()
model_data


# MODELING - Train-Test-Split
model = Model(model_data)
X_train, y_train, X_test, y_test = model.train_test_split(config)


# FITTING THE MODEL
model.fit(X_train, y_train)
y_hat = model.predict(X_train)
log_loss_score = model.log_loss_score(y_train, y_hat)
print(f'log loss scores on the train data: \t{log_loss_score}')


# CREATING THE FINAL OUTPUT
coupon_creator = CouponCreator(model)
X_template = X_test.copy()
X_template['discount'] = None
X_template['substitue_discount'] = 0
coupons = coupon_creator.get_top_coupons(config['model']['discounts'], X_template, n_coupons=config['model']['n_coupons'])
print(coupons)


# STORING THE FINAL OUTPUT
coupons.to_parquet(config['output']['path'] + 'final_output.parquet')
