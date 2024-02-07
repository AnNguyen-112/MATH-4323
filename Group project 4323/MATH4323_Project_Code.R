####################THIS PART IS DEALT WITH KNN MODEL#########################
#DATA PREPROCESSING
library(class)
library(caret)
library(ggplot2)
#Importing data
spotify.2023 <- read.csv("C:/Users/Welcome/Desktop/MATH 4323/Project/spotify-2023.csv")
spotify_data = spotify.2023
spotify_data$streams = as.numeric(spotify_data$streams)
spotify_data$in_deezer_playlists = as.numeric(spotify_data$in_deezer_playlists)
spotify_data$in_shazam_charts = as.numeric(spotify_data$in_shazam_charts)
spotify_data = spotify_data[!is.na(spotify_data$streams), ]
spotify_data = spotify_data[!is.na(spotify_data$in_deezer_playlists), ]
spotify_data = spotify_data[!is.na(spotify_data$in_shazam_charts), ]
#Dropping the songs with little to no streams
spotify_data = spotify_data[spotify_data$streams > 1000000, ]
#Checking if the data has any missing values:
any(is.na(spotify_data))
#Create a response variable that determines whether a song is
#a standout song or not based on the stream variable
spotify_data$standout_song = ifelse(spotify_data$streams > median(spotify_data$streams), 1, 0)

#Correlation check:
spotify_data_numeric <- spotify_data[sapply(spotify_data, is.numeric)]
# Calculate Pearson correlation coefficient for numeric variables
correlation <- cor(spotify_data_numeric, use = "complete.obs")
streams_index <- which(names(spotify_data_numeric) == "streams")
streams_correlations = correlation[streams_index, ]
# Convert to a data frame for easy viewing and sorting
correlation_table <- data.frame(Variable = names(streams_correlations), 
                                Correlation = as.numeric(streams_correlations))

# Sort the table by the absolute value of the correlation in descending order
correlation_table <- correlation_table[order(-abs(correlation_table$Correlation)), ]

# Print the table
print(correlation_table)


#Exclude some of the variables that are closely related to the streams variable
spotify_subset1 <- spotify_data_numeric[, !(names(spotify_data_numeric) %in% c('streams', 'in_apple_charts',
                                                                               'in_spotify_charts', 'in_deezer_charts',
                                                                               'in_shazam_charts'))]
#PERFORMING KNNs:
set.seed(1) # for reproducibility

# Splitting spotify_subset1
train_index_1 <- createDataPartition(spotify_subset1$standout_song, p = .8, list = FALSE)
train_set_1 <- spotify_subset1[train_index_1, ]
test_set_1 <- spotify_subset1[-train_index_1, ]

#Applying KNN
#Tuning for K for spotify_subset1
tune_grid = expand.grid(k = 1:20)
fit.knn_1 <- train(as.factor(standout_song) ~ ., data = train_set_1, method = "knn", 
                   tuneGrid = tune_grid, trControl = trainControl(method = "cv", number = 10))
#Choosing the best K
best_k1 = fit.knn_1$bestTune
# For spotify_subset1
knn.pred_1 <- knn(train = train_set_1[, -ncol(train_set_1)], test = test_set_1[, -ncol(test_set_1)],
                  cl = train_set_1$standout_song, k = best_k1)
table(knn.pred_1, test_set_1$standout_song)
print(mean(knn.pred_1 != test_set_1$standout_song))
#Applying KNN with scaled data:
set.seed(1)
train_set_standardized_1 = scale(train_set_1[ ,-16])
test_set_standardized_1 = scale(test_set_1[, -16],
                                center = attr(train_set_standardized_1, "scaled:center"),
                                scale = attr(train_set_standardized_1, "scaled:scale"))
y.train = train_set_1$standout_song
y.test = test_set_1$standout_song
for(k_value in c(1, 3, 5, 10))
{
  set.seed(1)
  knn.pred <- knn(train = train_set_standardized_1, test = test_set_standardized_1,
                  cl = y.train, k = k_value)
  print(table(knn.pred, y.test))
  print(mean(knn.pred != y.test))
}
#Performing KNN model on the full data set
set.seed(1)
pred = predict(fit.knn_1, spotify_subset1)
mean(pred != spotify_subset1$standout_song)
spotify_subset1$prediction = as.factor(pred)
#Plots for the graphs
ggplot(spotify_subset1, aes_string(x = "in_apple_playlists", y = "in_spotify_playlists", color = "prediction")) +
  geom_point(alpha = 0.6) +
  labs(title = "KNN Model Predictions",
       x = "in_apple_playlists", y = "in_spotify_playlists", color = "Prediction") +
  theme_minimal()
ggplot(spotify_subset1, aes_string(x = "speechiness_.", y = "danceability_.", color = "prediction")) +
  geom_point(alpha = 0.6) +
  labs(title = "KNN Model Predictions",
       x = "speechiness", y = "danceability", color = "prediction") +
  theme_minimal()

####################THIS PART IS DEALT WITH SVM MODEL#########################

  #####library(readr)
library(e1071)
install.packages("caret")
library(caret)


Spotify <- read_csv("spotify-2023.csv")


# Calculate the mean of the Stream variable
# Convert "streams" to numeric, replacing any non-numeric values with NA
Spotify$streams <- as.numeric(as.character(Spotify$streams))

# Check for NAs after the conversion
sum(is.na(Spotify$streams))

# Remove rows with NAs
Spotify <- Spotify[!is.na(Spotify$streams), ]

# Now calculate the mean
mean_stream <- median(Spotify$streams)
print(mean_stream)



# Create the standOut_song variable

Spotify$standout_song <- ifelse(Spotify$streams > mean_stream, 1, 0)


####Eliminating variable can directly influence the result of classification
# Create a new dataset excluding specified columns
Spotify_fixed_model <- Spotify[, 
                                !(names(Spotify) %in% c("streams", 
                                                        "in_spotify_charts", "in_deezer_charts",
                                                        "in_shazam_charts","in_apple_charts","track_name", 
                                                        "artist(s)_name", "key", "mode"))]


###Scaling the data set
# Identify numeric columns
numeric_columns <- sapply(Spotify_fixed_model, is.numeric)

# Remove the column to be excluded from scaling
# Remove the column to be excluded from scaling
numeric_columns <- names(numeric_columns)[names(numeric_columns) != "standout_song"]
numeric_columns

# Create a scaled version of the numeric columns (excluding the specified column)
Spotify_fixed_model[numeric_columns] <- scale(Spotify_fixed_model[numeric_columns])

# Print the scaled dataset
Spotify_fixed_model

####divide dataset to 80% training 20% testing


n <- nrow(Spotify_fixed_model)
RNGkind(sample.kind = "Rounding")
set.seed(4323)
train <- sample(1:n, round(0.8 * n))
train_data <- Spotify_fixed_model[train,]
test_data <- Spotify_fixed_model[-train,]




###svm model
#Radial
set.seed(4323)
tune.out=tune(svm,
                as.factor(standout_song)~.,data= train_data,
                kernel="radial",
                ranges=list(cost=c(0.1,1,10,100,1000),
                            gamma=c(0.5,1,2,3,4)))
summary(tune.out)

best.radial.model <- tune.out$best.model
summary(best.radial.model)

#validation part
svm.best.radialmodel <- predict(best.radial.model,test_data)
svm.best.radialmodel
table(true=test_data$standout_song,
      pred=svm.best.radialmodel)

#test misclassification?
#      pred
#true  0  1
#   0 65 24
#   1 29 72

# test misclassification: (29 + 24) / (190) = 0.2789

###Linear
set.seed(4323)
tune.out = tune(svm,
                as.factor(standout_song)~.,data= train_data,
                kernel="linear",
                ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100)))
summary(tune.out)

best.linear.model <- tune.out$best.model
summary(best.linear.model)
svm.best.linearmodel <- predict(best.linear.model,test_data)
table(true=test_data$standout_song,
      pred=svm.best.linearmodel)

#test misclassification ?
#pred
#true  0  1
#   0 77 12
#   1 17 84
# test misclassification: (17 + 12) / (190) = 0.152632

###Polynomial
set.seed(4323)
tune.out = tune(svm,
                as.factor(standout_song)~.,data= train_data,
                kernel="polynomial",
                ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100), 
                              degree = c(2, 3, 4)))
summary(tune.out)

best.polynomial.model <- tune.out$best.model
summary(best.polynomial.model)
svm.best.polynomialmodel <- predict(best.polynomial.model,test_data)
table(true=test_data$standout_song,
      pred=svm.best.polynomialmodel)

#test misclassification ?
#    pred
#true  0  1
#   0 76 13
#   1 26 75
# test misclassification: (26+ 13) / 190 = 0.2052632



###sigmoid
set.seed(4323)
tune.out <- tune(svm,
                 as.factor(standout_song) ~ ., data = train_data,
                 kernel = "sigmoid",
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100),
                               gamma = c(0.1, 0.5, 1, 2, 3)))

summary(tune.out)
best.sigmoid.model <- tune.out$best.model
summary(best.sigmoid.model)
svm.best.sigmoidmodel <- predict(best.sigmoid.model,test_data)
table(true=test_data$standout_song,
      pred=svm.best.sigmoidmodel)

#test misclassification
#     pred
#true  0  1
#   0 85  4
#   1 38 63
# test misclassification: (38 + 4) / 190 = 0.22

#=> In all svm method, the svm Linear result in lowest error, so 
# I will test it one more time with only three variable: standout_song, 


#Linear model with only 3 variables: standout_song, in_sportify_playlists, in_apple_playlists

# train_data_3_variables <- train_data[c("standout_song", "in_spotify_playlists", "in_apple_playlists")]
# View(train_data_3_variables)
# test_data_3_variables <- test_data[,c("standout_song", "in_spotify_playlists", "in_apple_playlists")]
# View(train_data_3_variables)

Spotify_fixed_model_3_variables <- Spotify_fixed_model[c("standout_song", 
                                                         "in_spotify_playlists", 
                                                         "in_apple_playlists")]

set.seed(4323)
tune.out.2 = tune(svm,
                as.factor(standout_song)~.,data= Spotify_fixed_model_3_variables,
                kernel="linear",
                ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100)))
summary(tune.out.2)

best.linear.model.2 <- tune.out.2$best.model
summary(best.linear.model.2)


#testing validation
svm.best.linearmodel.2 <- predict(best.linear.model.2,Spotify_fixed_model_3_variables)
table(true=Spotify_fixed_model_3_variables$standout_song,
      pred=svm.best.linearmodel.2)


#test misclassification
#       pred
# true   0   1
#    0 433  43
#    1  98 378
#test misclassification: (98 + 43) / 952 = 0.1481092


#for testing set
plot(best.linear.model.2,Spotify_fixed_model_3_variables, 
     main = "SVM classification Plot", 
     col = c("white", "yellow"))



  