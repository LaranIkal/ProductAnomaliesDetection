# Load libraries
library( h2o )

h2o.init( nthreads = -1, max_mem_size = "5G", port = 6666 )
h2o.removeAll() ## Removes the data from the h2o cluster in preparation for our final model.

# Reading our data file
allData = read.csv( "secom.data", sep = " ", header = FALSE, encoding = "UTF-8" )

# fixing the data set, there are a lot of NaN records
if( dim(na.omit(allData))[1] == 0 ){
  for( colNum in 1:dim( allData )[2]   ){
    
    # Get valid values from the actual column values
    ValidColumnValues = allData[,colNum][!is.nan( allData[, colNum] )]
    
    # Check each value in the actual active column.
    for( rowNum in 1:dim( allData )[1]   ){
      
      cat( "Processing row:", rowNum, ", Column:", colNum, "Data:", allData[rowNum, colNum], "\n" )
      
      if( is.nan( allData[rowNum, colNum] ) ) {
        
        # Assign random valid value to our row,column with NA value
        allData[rowNum, colNum] = ValidColumnValues[ floor( runif(1, min = 1, max = length( ValidColumnValues ) ) ) ]
      }
    }
  }
}

# spliting all data, the fiirst 90% for training and the rest 10% for testing our model.
trainingData = allData[1:floor(dim(allData)[1]*.9),]
testingData = allData[(floor(dim(allData)[1]*.9)+1):dim(allData)[1],]

# Convert the training dataset to H2O format.
trainingData_hex = as.h2o( trainingData, destination_frame = "train_hex" )

# Set the input variables
featureNames = colnames(trainingData_hex)

# Creating the first model version.
trainingModel = h2o.deeplearning( x = featureNames, training_frame = trainingData_hex
                                  , model_id = "Station1DeepLearningModel"
                                  , activation = "Tanh"
                                  , autoencoder = TRUE
                                  #, reproducible = FALSE
                                  , reproducible = TRUE
                                  , l1 = 1e-5
                                  , ignore_const_cols = FALSE
                                  , seed = 1234
                                  , hidden = c( 400, 200, 400 ), epochs = 50 )


# Getting the anomaly with training data to set the min MSE( Mean Squared Error )
# value before setting a record as anomally
trainMSE = as.data.frame( h2o.anomaly( trainingModel
                                       , trainingData_hex
                                       , per_feature = FALSE ) )

# Check the first 30 descendent sorted trainMSE records to see our outliers
head( sort( trainMSE$Reconstruction.MSE , decreasing = TRUE ), 30)

# 0.020288603 0.017976305 0.012772556 0.011556780 0.010143009 0.009524983 0.007363854 
# 0.005889714 0.005604329 0.005189614[11] 0.005185285 0.005118595 0.004639442 0.004497609
# 0.004438342 0.004419993 0.004298936 0.003961503 0.003651326 0.003426971 0.003367108
# 0.003169319 0.002901914 0.002852006 0.002772110 0.002765924 0.002754586 0.002748887 
# 0.002619872 0.002474702

# Ploting the errors of reconstructing our training data, to have a graphical view
# of our data reconstruction errors
plot( sort( trainMSE$Reconstruction.MSE ), main = 'Reconstruction Error', ylab = "MSE Value." )

# Seeing the chart and the first 30 decresing sorted MSE records, we can decide .01 
# as our min MSE before setting a record as anomally, because we see Just a few 
# records with two decimals greater than zero and we can set those as outliers.
# This value is something you must decide for your data.

# Updating trainingData data set with reconstruction error < .01
trainingDataNew = trainingData[ trainMSE$Reconstruction.MSE < .01, ]

h2o.removeAll() ## Remove the data from the h2o cluster in preparation for our final model.

# Convert our new training data frame to H2O format.
trainingDataNew_hex = as.h2o( trainingDataNew, destination_frame = "train_hex" )

# Creating the final model.
trainingModelNew = h2o.deeplearning( x = featureNames, training_frame = trainingDataNew_hex
                                     , model_id = "Station1DeepLearningModel"
                                     , activation = "Tanh"
                                     , autoencoder = TRUE
                                     #, reproducible = FALSE
                                     , reproducible = TRUE
                                     , l1 = 1e-5
                                     , ignore_const_cols = FALSE
                                     , seed = 1234
                                     , hidden = c( 400, 200, 400 ), epochs = 50 )




# Check our testing data for anomalies.

# Convert our testing data frame to H2O format.
testingDataH2O = as.h2o( testingData, destination_frame = "test_hex" )

# Getting anomalies found in testing data.
testMSE = as.data.frame( h2o.anomaly( trainingModelNew
                                      , testingDataH2O
                                      , per_feature = FALSE ) )

# Binding our data.
testingData = cbind( MSE = testMSE$Reconstruction.MSE , testingData )

anomalies = testingData[ testingData$MSE >= .01,  ]

if( dim(anomalies)[1] > 0 ){
  cat( "Anomalies detected in the sample data, station needs maintenance." )
}

