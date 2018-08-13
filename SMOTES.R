

save.image()
# install.packages("caret")
require(caret) # The god father
#install.packages("doSNOW")
require(doSNOW) # for parallel computing
# install.packages("Amelia")
require(Amelia) # FOR MISSMAP
# install.packages("corrplot")
require(corrplot)


# distributing computation among # of cores
getDoParWorkers()
getDoParRegistered()
getDoParName()
getDoParVersion()
cl <- makeCluster(spec=4, type="SOCK")
registerDoSNOW(cl) 
stopCluster(cl)


# importing dataset
dataset <- read.csv("default_of_credit_card_clients.csv",header = T)
str(dataset)

#data preprocessing
dataset <- dataset[,-1]
dataset$default_payment_next_month <- dataset$default.payment.next.month 
dataset$default.payment.next.month<- NULL

# encoding the dependent variable as factor

make.names(dataset$default_payment_next_month)
dataset$default_payment_next_month <- factor(dataset$default_payment_next_month,levels=c("0","1"),labels=c("X0","X1"))
levels(dataset$default_payment_next_month) # SHOULD MATCH WITH  make.names(dataset$default_payment_next_month)


###############################
# Exploratoty data analysis
###############################

missmap(dataset)
any(is.na(dataset))
str(dataset)
summary(dataset)


#checking for correlation
round(cor(dataset[,-25]),2)
# correlation among bill amount
round(cor(dataset[,12:17]),2)
plot(dataset[,-25])

#correaltion among pay amount.Seems acceptable
round(cor(dataset[,-25]),2)
round(cor(dataset[,18:23]),2)
plot(dataset[,-25])

#correlation among pay
round(cor(dataset[,6:11]),2)
plot(dataset[,6:8])
cor(dataset[,6:8])


corr_values <- cor(dataset[,-24])# gives the value of correlation matrix. lies between -1 to 1
corr_plot <- corrplot(corr_values,method = "color",tl.cex=0.5,tl.col="black",title="Correlation Matrix")


# Visualization
require(ggplot2)
pl<- ggplot(data=dataset,aes(x=LIMIT_BAL,y=default_payment_next_month))+
  geom_jitter(aes(color=factor(EDUCATION)),size=0.1,alpha=0.4)+
  ggtitle("Classification of Default Card Holder by Balance Limit & Education")+
  xlab("Balance Limit")+ylab("Default Payment")

pl+facet_grid(EDUCATION~.)

ql <- ggplot(data=dataset,aes(x=SEX))+
  geom_bar(aes(fill=SEX))


###########################################
# Machine learning 
###########################################


# spiltting the dataset
in_training <- createDataPartition(dataset$default_payment_next_month,p=0.70,list=FALSE)
training_set <- dataset[in_training,]
test_set <- dataset[-in_training,]

# check for level ratio
prop.table(table(training_set$default_payment_next_month))
prop.table(table(test_set$default_payment_next_month))


########################
#K_FOLD CROSS VALIDATION
########################

set.seed(12345)
# folds<- createMultiFolds(training_set$alive,k=10,times=10) 
Control <- trainControl(method="repeatedcv",number=10,repeats = 10,classProbs = T,summaryFunction = twoClassSummary)# for CALCULATING ROC classProbS=T IS REQUIRED


# use tuneLength or tunegrid parameter to find the optimum performance of each model

set.seed(12345)
model_bayes_1 <- train(form=default_payment_next_month~.,data=training_set,
                       trControl=Control,tuneLength=5,method="naive_bayes",
                       metric="ROC",preProcess=c("center", "scale","pca","corr","YeoJohnson")) #library naivebayes.can also use metric="ROC"
plot(model_bayes_1)


###########################
# SMOTE
##########################
table(training_set$default_payment_next_month)
prop.table((table(training_set$default_payment_next_month)))
new_training_set <- SMOTE(form=default_payment_next_month~.,data=training_set,perc.over = 100)

# lets train model on new dataset
set.seed(12345)
model_bayes_2 <- train(form=default_payment_next_month~.,data=new_training_set,
                       trControl=Control,tuneLength=5,method="naive_bayes",
                       metric="ROC",preProcess=c("center", "scale","pca","corr","YeoJohnson")) #library naivebayes.can also use metric="ROC"
plot(model_bayes_2)

####prediction with both models
pred_1 <- predict(model_bayes_1,newdata = test_set[,-24])
pred_2 <- predict(model_bayes_2,newdata = test_set[,-24])
table(test_set[,24],pred_1)
table(test_set[,24],pred_2)