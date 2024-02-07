library(readr)
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

####devide dataset to 80% training 20% testing


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


