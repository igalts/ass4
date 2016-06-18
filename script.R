library(readr)
library(tm)
library(party)
library(SnowballC)
library(caret)
library(Metrics)
install.packages("tm")

setwd("C:\\downloads\\leser 4\\bu")

train <- read_csv("train.csv")
test  <- read_csv("test.csv")

train$median_relevance <- factor(train$median_relevance)

#Preprocess the train data
temp <- tm_map(Corpus(VectorSource(train$query)), tolower)
temp <- tm_map(temp, removePunctuation)
temp <- tm_map(temp, removeNumbers)
temp <- tm_map(temp, stripWhitespace)
temp <- tm_map(temp, removeWords, stopwords("english"))
temp <- tm_map(temp, stemDocument, language = "english")
temp <- tm_map(temp, PlainTextDocument)
temp <- data.frame(query = unlist(sapply(temp, `[`, "content")),
                   stringsAsFactors = F)
train$product_title <- factor(temp$query)


temp <- tm_map(Corpus(VectorSource(train$product_title)), tolower)
temp <- tm_map(temp, removePunctuation)
temp <- tm_map(temp, removeNumbers)
temp <- tm_map(temp, stripWhitespace)
temp <- tm_map(temp, removeWords, stopwords("english"))
temp <- tm_map(temp, stemDocument, language = "english")
temp <- tm_map(temp, PlainTextDocument)
temp <-
  data.frame(product_title = unlist(sapply(temp, `[`, "content")),
             stringsAsFactors = F)
train$product_title <- factor(temp$product_title)

temp <-
  tm_map(Corpus(VectorSource(train$product_description)), tolower)
temp <- tm_map(temp, removePunctuation)
temp <- tm_map(temp, removeNumbers)
temp <- tm_map(temp, stripWhitespace)
temp <- tm_map(temp, removeWords, stopwords("english"))
temp <- tm_map(temp, stemDocument, language = "english")
temp <- tm_map(temp, PlainTextDocument)
temp <-
  data.frame(product_description = unlist(sapply(temp, `[`, "content")),
             stringsAsFactors = F)
train$product_description <- factor(temp$product_description)

#Preprocess the test data as well
temp <- tm_map(Corpus(VectorSource(test$query)), tolower)
temp <- tm_map(temp, removePunctuation)
temp <- tm_map(temp, removeNumbers)
temp <- tm_map(temp, stripWhitespace)
temp <- tm_map(temp, removeWords, stopwords("english"))
temp <- tm_map(temp, stemDocument, language = "english")
temp <- tm_map(temp, PlainTextDocument)
temp <- data.frame(query = unlist(sapply(temp, `[`, "content")),
                   stringsAsFactors = F)
test$query <- factor(temp$query)


temp <- tm_map(Corpus(VectorSource(test$product_title)), tolower)
temp <- tm_map(temp, removePunctuation)
temp <- tm_map(temp, removeNumbers)
temp <- tm_map(temp, stripWhitespace)
temp <- tm_map(temp, removeWords, stopwords("english"))
temp <- tm_map(temp, stemDocument, language = "english")
temp <- tm_map(temp, PlainTextDocument)
temp <-
  data.frame(product_title = unlist(sapply(temp, `[`, "content")),
             stringsAsFactors = F)
test$product_title <- factor(temp$product_title)

temp <- tm_map(Corpus(VectorSource(test$product_description)), tolower)
temp <- tm_map(temp, removePunctuation)
temp <- tm_map(temp, removeNumbers)
temp <- tm_map(temp, stripWhitespace)
temp <- tm_map(temp, removeWords, stopwords("english"))
temp <- tm_map(temp, stemDocument, language = "english")
temp <- tm_map(temp, PlainTextDocument)
temp <-
  data.frame(product_description = unlist(sapply(temp, `[`, "content")),
             stringsAsFactors = F)
test$product_description <- factor(temp$product_description)

levels(train$query) <- union(levels(train$query), levels(test$query))
levels(train$product_title) <- union(levels(train$product_title), levels(test$product_title))
levels(train$product_description) <- union(levels(train$product_description), levels(test$product_description))
levels(test$query) <- union(levels(train$query), levels(test$query))
levels(test$product_title) <- union(levels(train$product_title), levels(test$product_title))
levels(test$product_description) <- union(levels(train$product_description), levels(test$product_description))

inTraining <- sample(1:nrow(train),  .75*nrow(train))
training <- train[ inTraining,]
testing  <- train[-inTraining,]

library(party)
library(SnowballC)
library(caret)
gc()
model <- train(median_relevance ~ query+product_title+product_description, data = training,
               method = "repeatedcv",
               repeats = 3,
              trControl = trainControl(classProbs = F))
results <- predict(model, newdata = testing)
library(Metrics)
ScoreQuadraticWeightedKappa(testing$median_relevance, results, 1, 4)

results <- predict(model, newdata = test)
Newsubmission = data.frame(id=test$id, prediction = results)
write.csv(Newsubmission,"model.csv",row.names=F) 