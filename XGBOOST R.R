library(rsample)      # data splitting 
library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # a ?ava-based platform
library(pdp)          # model visualization
library(ggplot2)      # model visualization
library(lime)         # model visualization
library(vtreat)
Bike = read.csv("C:/機器學習/df_0907.csv",header = T,stringsAsFactors = FALSE,fileEncoding = 'utf-8',nrows = 100000)
Bike[,c("sno", "Hr", "holiday", "school_off", "rain","Temperature",
           "RH","WS","Precp","MRT_Dist","School_Dist","rent_count")]


train_sub = sample(nrow(Bike),7/10*nrow(Bike))
train_data = Bike[train_sub,]
test_data = Bike[-train_sub,]
list <- list(train_data,test_data)


library(Matrix)
#define predictor and response variables in training set
train_x = data.matrix(train_data[c("sno", "Hr", "holiday", "school_off", "rain","Temperature","RH","WS","Precp","MRT_Dist","School_Dist")])
train_y = train_data[c("rent_count")]

#define predictor and response variables in testing set
test_x = data.matrix(test_data[c("sno", "Hr", "holiday", "school_off", "rain","Temperature","RH","WS","Precp","MRT_Dist","School_Dist")])
test_y = test_data[c("rent_count")]


require(xgboost)


xgb_train = xgb.DMatrix(data = as.matrix(train_data[c("sno", "Hr", "holiday", "school_off", "rain","Temperature","RH","WS","Precp","MRT_Dist","School_Dist")]),
                     label = train_data$rent_count)
xgb_test = xgb.DMatrix(data = as.matrix(test_data[c("sno", "Hr", "holiday", "school_off", "rain","Temperature","RH","WS","Precp","MRT_Dist","School_Dist")]),
                    label = test_data$rent_count)



#defining a watchlist
watchlist = list(train=xgb_train, test=xgb_test)


#fit XGBoost model and display training and testing data at each iteartion
model = xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, nrounds = 100)


#define final model
model_xgboost = xgboost(data =xgb_train, max.depth = 3, nrounds = 86, verbose = 0)

summary(model_xgboost)


#use model to make pedictions on test data
pred_y <- predict(model_xgboost, xgb_test)

mse = mean((test_y - pred_y)^2)
mae = caret::MAE(test_y, pred_y)
rmse = caret::RMSE(test_y, pred_y)

cat("MSE: ", mse, "MAE: ", mae, " RMSE: ", rmse)



